#!/usr/bin/env python3
"""
Semantic RAG Pipeline
====================

Main orchestrator for the semantic graph traversal RAG system.
Handles all phases from data acquisition to evaluation with intelligent caching.
"""

import os
import sys
import yaml
import logging
import platform
import psutil
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch

# Import engines
from wiki import WikiEngine
from chunking import ChunkEngine
from models import EmbeddingEngine
from similarity import SimilarityEngine
from retrieval import RetrievalEngine
from datasets import DatasetEngine


class SemanticRAGPipeline:
    """Main pipeline orchestrator for semantic RAG system."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the pipeline with configuration."""
        self.config_path = config_path
        self.config = None
        self.experiment_id = None
        self.logger = None
        self.device = None
        self.start_time = None

        # Data storage for pipeline phases
        self.articles = []
        self.corpus_stats = {}
        self.chunks = []
        self.chunk_stats = {}
        self.embeddings = {}  # Dict[model_name, List[ChunkEmbedding]]
        self.embedding_stats = {}
        self.similarities = {}  # Dict[model_name, similarity_data]
        self.similarity_stats = {}
        self.retrieval_engine = None  # RetrievalEngine instance
        self.retrieval_stats = {}
        self.dataset = []  # List[EvaluationQuestion]
        self.dataset_stats = {}

    def pipe(self) -> Dict[str, Any]:
        """
        Main pipeline execution function.

        Returns:
            Dictionary containing experiment results and metadata
        """
        try:
            # Phase 1: Setup & Initialization
            self._phase_1_setup_and_initialization()

            # Phase 2: Data Acquisition
            if self.config['execution']['mode'] in ['full_pipeline', 'data_only']:
                if 'data_acquisition' not in self.config['execution']['skip_phases']:
                    self._phase_2_data_acquisition()
                else:
                    self.logger.info("⏭️  Skipping Phase 2: Data Acquisition")

            # Phase 3: Embedding Generation
            if self.config['execution']['mode'] in ['full_pipeline', 'embedding_only']:
                if 'embedding_generation' not in self.config['execution']['skip_phases']:
                    self._phase_3_embedding_generation()
                else:
                    self.logger.info("⏭️  Skipping Phase 3: Embedding Generation")

            # Phase 4: Similarity Matrix Construction
            if self.config['execution']['mode'] in ['full_pipeline']:
                if 'similarity_matrices' not in self.config['execution']['skip_phases']:
                    self._phase_4_similarity_matrices()
                else:
                    self.logger.info("⏭️  Skipping Phase 4: Similarity Matrix Construction")

            # Phase 5: Retrieval Graph Construction
            if self.config['execution']['mode'] in ['full_pipeline']:
                if 'retrieval_graphs' not in self.config['execution']['skip_phases']:
                    self._phase_5_retrieval_graphs()
                else:
                    self.logger.info("⏭️  Skipping Phase 5: Retrieval Graph Construction")

            # Phase 6: Dataset Generation
            if self.config['execution']['mode'] in ['full_pipeline']:
                if 'dataset_generation' not in self.config['execution']['skip_phases']:
                    self._phase_6_dataset_generation()
                else:
                    self.logger.info("⏭️  Skipping Phase 6: Dataset Generation")

            # TODO: Add remaining phases
            # self._phase_7_rag_evaluation()
            # etc.

            self.logger.info("Pipeline completed successfully!")
            return {
                "experiment_id": self.experiment_id,
                "status": "completed",
                "execution_time": datetime.now() - self.start_time
            }

        except Exception as e:
            if self.logger:
                self.logger.error(f"Pipeline failed: {str(e)}")
            else:
                print(f"Pipeline failed: {str(e)}")
            raise

    def _phase_1_setup_and_initialization(self):
        """Phase 1: Setup & Initialization"""
        print("🚀 Starting Semantic RAG Pipeline")
        print("=" * 50)

        # Record start time
        self.start_time = datetime.now()

        # 1. Load config.yaml
        self._load_config()

        # 2. Initialize experiment tracker
        self._initialize_experiment_tracker()

        # 3. Create output directories
        self._create_output_directories()

        # 4. Initialize logging
        self._initialize_logging()

        # 5. Check system resources
        self._check_system_resources()

        # 6. Detect and configure device (M1 Mac compatible)
        self._detect_and_configure_device()

        # 7. Validate config parameters
        self._validate_config()

        self.logger.info(f"Phase 1 completed - Experiment ID: {self.experiment_id}")

    def _phase_2_data_acquisition(self):
        """Phase 2: Data Acquisition"""
        self.logger.info("🌐 Starting Phase 2: Data Acquisition")

        # Initialize WikiEngine
        wiki_engine = WikiEngine(self.config, self.logger)

        # Check if we should force recompute
        force_recompute = 'data' in self.config['execution'].get('force_recompute', [])

        if force_recompute:
            self.logger.info("🔄 Force recompute enabled - will fetch fresh articles")
            # Temporarily disable caching
            original_use_cached = self.config['wikipedia']['use_cached_articles']
            self.config['wikipedia']['use_cached_articles'] = False

        try:
            # Acquire articles
            wiki_path = Path(self.config['directories']['data']) / "wiki.json"
            articles = wiki_engine.acquire_articles(wiki_path)

            if not articles:
                raise RuntimeError("No articles were successfully acquired")

            # Get and log corpus statistics
            stats = wiki_engine.get_corpus_statistics()
            self.logger.info("📊 Corpus Statistics:")
            self.logger.info(f"   Articles: {stats['total_articles']:,}")
            self.logger.info(f"   Sentences: {stats['total_sentences']:,}")
            self.logger.info(f"   Words: {stats['total_words']:,}")
            self.logger.info(f"   Avg sentences/article: {stats['avg_sentences_per_article']:.1f}")
            self.logger.info(f"   Avg words/sentence: {stats['avg_words_per_sentence']:.1f}")

            # Store articles in pipeline for later phases
            self.articles = articles
            self.corpus_stats = stats

            self.logger.info("✅ Phase 2 completed successfully")

        except Exception as e:
            self.logger.error(f"❌ Phase 2 failed: {e}")
            raise

        finally:
            # Restore original caching setting if we changed it
            if force_recompute:
                self.config['wikipedia']['use_cached_articles'] = original_use_cached

    def _phase_3_embedding_generation(self):
        """Phase 3: Embedding Generation"""
        self.logger.info("🧠 Starting Phase 3: Embedding Generation")

        # Check if we have articles from Phase 2
        if not self.articles:
            self.logger.warning("No articles available from Phase 2. Loading from cache...")
            # Try to load articles from cache
            wiki_engine = WikiEngine(self.config, self.logger)
            wiki_path = Path(self.config['directories']['data']) / "wiki.json"
            if wiki_path.exists():
                self.articles = wiki_engine._load_cached_articles(wiki_path)
            else:
                raise RuntimeError("No articles available and no cache found. Please run Phase 2 first.")

        try:
            # Step 1: Create chunks using ChunkEngine
            self.logger.info("✂️  Creating chunks from articles")
            chunk_engine = ChunkEngine(self.config, self.logger)
            chunks = chunk_engine.create_chunks(self.articles)
            
            if not chunks:
                raise RuntimeError("No chunks were created from articles")
            
            # Get and log chunk statistics
            chunk_stats = chunk_engine.get_chunking_statistics(chunks)
            self.logger.info("📊 Chunk Statistics:")
            self.logger.info(f"   Total chunks: {chunk_stats['total_chunks']:,}")
            self.logger.info(f"   From articles: {chunk_stats['total_articles']:,}")
            self.logger.info(f"   Avg chunks/article: {chunk_stats['avg_chunks_per_article']:.1f}")
            self.logger.info(f"   Avg words/chunk: {chunk_stats['chunk_length_stats']['mean_words']:.1f}")
            self.logger.info(f"   Avg sentences/chunk: {chunk_stats['sentence_count_stats']['mean_sentences']:.1f}")
            
            # Store chunks in pipeline
            self.chunks = chunks
            self.chunk_stats = chunk_stats
            
            # Step 2: Generate embeddings using EmbeddingEngine
            self.logger.info("✨ Generating embeddings for all models")
            embedding_engine = EmbeddingEngine(self.config, self.logger)
            
            # Check if we should force recompute embeddings
            force_recompute = 'embeddings' in self.config['execution'].get('force_recompute', [])
            
            embeddings = embedding_engine.generate_embeddings(chunks, force_recompute=force_recompute)
            
            if not embeddings:
                raise RuntimeError("No embeddings were generated")
            
            # Get and log embedding statistics
            embedding_stats = embedding_engine.get_embedding_statistics(embeddings)
            self.logger.info("📈 Embedding Statistics:")
            for model_name, stats in embedding_stats.items():
                self.logger.info(f"   {model_name}:")
                self.logger.info(f"      Chunks: {stats['total_chunks']:,}")
                self.logger.info(f"      Dimensions: {stats['embedding_dimension']}")
                self.logger.info(f"      Mean norm: {stats['mean_norm']:.3f}")
                self.logger.info(f"      Std norm: {stats['std_norm']:.3f}")
            
            # Store embeddings in pipeline
            self.embeddings = embeddings
            self.embedding_stats = embedding_stats
            
            self.logger.info("✅ Phase 3 completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Phase 3 failed: {e}")
            raise

    def _phase_4_similarity_matrices(self):
        """Phase 4: Similarity Matrix Construction"""
        self.logger.info("🕰 Starting Phase 4: Similarity Matrix Construction")

        # Check if we have embeddings from Phase 3
        if not self.embeddings:
            self.logger.warning("No embeddings available from Phase 3. Loading from cache...")
            # Try to load embeddings from cache
            embedding_engine = EmbeddingEngine(self.config, self.logger)
            # We'd need to reload chunks and embeddings here
            raise RuntimeError("No embeddings available and no cache loading implemented yet. Please run Phase 3 first.")

        try:
            # Initialize SimilarityEngine
            self.logger.info("🔗 Initializing similarity engine")
            similarity_engine = SimilarityEngine(self.config, self.logger)
            
            # Check if we should force recompute similarities
            force_recompute = 'similarities' in self.config['execution'].get('force_recompute', [])
            
            # Compute similarity matrices
            self.logger.info(f"🎯 Computing similarity matrices for {len(self.embeddings)} models")
            similarities = similarity_engine.compute_similarity_matrices(self.embeddings, force_recompute=force_recompute)
            
            if not similarities:
                raise RuntimeError("No similarity matrices were computed")
            
            # Get and log similarity statistics
            similarity_stats = similarity_engine.get_similarity_statistics(similarities)
            self.logger.info("📊 Similarity Matrix Statistics:")
            for model_name, stats in similarity_stats.items():
                self.logger.info(f"   {model_name}:")
                self.logger.info(f"      Total chunks: {stats['total_chunks']:,}")
                self.logger.info(f"      Intra-document connections: {stats['connections']['intra_document']:,}")
                self.logger.info(f"      Inter-document connections: {stats['connections']['inter_document']:,}")
                self.logger.info(f"      Total connections: {stats['connections']['total']:,}")
                self.logger.info(f"      Sparsity ratio: {stats['sparsity_ratio']:.6f}")
                self.logger.info(f"      Memory usage: {stats['memory_usage_mb']:.1f} MB")
                self.logger.info(f"      Computation time: {stats['computation_time']:.2f}s")
                
                if 'matrix_stats' in stats:
                    matrix_stats = stats['matrix_stats']
                    self.logger.info(f"      Matrix shape: {matrix_stats['shape']}")
                    self.logger.info(f"      Non-zero entries: {matrix_stats['nnz']:,}")
                    self.logger.info(f"      Matrix density: {matrix_stats['density']:.8f}")
            
            # Store similarities in pipeline
            self.similarities = similarities
            self.similarity_stats = similarity_stats
            
            self.logger.info("✅ Phase 4 completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Phase 4 failed: {e}")
            raise

    def _phase_5_retrieval_graphs(self):
        """Phase 5: Retrieval Graph Construction"""
        self.logger.info("🕰 Starting Phase 5: Retrieval Graph Construction")

        # Check if we have similarities from Phase 4
        if not self.similarities:
            self.logger.warning("No similarity matrices available from Phase 4. Loading from cache...")
            # We'd need to reload similarities here - for now, throw error
            raise RuntimeError("No similarity matrices available. Please run Phase 4 first.")
        
        # Check if we have embeddings for retrieval
        if not self.embeddings:
            self.logger.warning("No embeddings available from Phase 3. Loading from cache...")
            raise RuntimeError("No embeddings available. Please run Phase 3 first.")

        try:
            # Initialize RetrievalEngine
            self.logger.info("🎯 Initializing retrieval engine")
            retrieval_engine = RetrievalEngine(
                self.config, 
                self.embeddings, 
                self.similarities, 
                self.logger
            )
            
            # Get retrieval statistics
            retrieval_stats = retrieval_engine.get_retrieval_statistics()
            self.logger.info("📊 Retrieval Engine Statistics:")
            self.logger.info(f"   Algorithm: {retrieval_stats['algorithm']}")
            self.logger.info(f"   Models available: {retrieval_stats['models_available']}")
            for model, count in retrieval_stats['total_chunks_per_model'].items():
                self.logger.info(f"   {model}: {count:,} chunks")
            
            if 'semantic_traversal_config' in retrieval_stats:
                config = retrieval_stats['semantic_traversal_config']
                self.logger.info(f"   Traversal config:")
                self.logger.info(f"      Max hops: {config['max_hops']}")
                self.logger.info(f"      Num anchors: {config['num_anchors']}")
                self.logger.info(f"      Similarity threshold: {config['similarity_threshold']}")
                self.logger.info(f"      Max results: {config['max_results']}")
            
            # Store retrieval engine in pipeline
            self.retrieval_engine = retrieval_engine
            self.retrieval_stats = retrieval_stats
            
            self.logger.info("✅ Phase 5 completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Phase 5 failed: {e}")
            raise

    def _phase_6_dataset_generation(self):
        """Phase 6: Dataset Generation"""
        self.logger.info("📊 Starting Phase 6: Dataset Generation")

        # Check if we have embeddings from Phase 3
        if not self.embeddings:
            self.logger.warning("No embeddings available from Phase 3. Loading from cache...")
            raise RuntimeError("No embeddings available. Please run Phase 3 first.")

        try:
            # Initialize DatasetEngine
            self.logger.info("📈 Initializing dataset engine")
            dataset_engine = DatasetEngine(
                self.config, 
                self.embeddings, 
                self.logger
            )
            
            # Check if we should force recompute datasets
            force_recompute = 'datasets' in self.config['execution'].get('force_recompute', [])
            
            # Generate evaluation dataset
            self.logger.info(f"📝 Generating evaluation dataset")
            dataset = dataset_engine.generate_dataset(force_recompute=force_recompute)
            
            if not dataset:
                raise RuntimeError("No questions were generated for the dataset")
            
            # Get and log dataset statistics
            dataset_stats = dataset_engine.get_dataset_statistics(dataset)
            self.logger.info("📊 Dataset Generation Statistics:")
            self.logger.info(f"   Total questions: {dataset_stats['total_questions']:,}")
            
            if 'by_generation_method' in dataset_stats:
                self.logger.info(f"   By generation method:")
                for method, count in dataset_stats['by_generation_method'].items():
                    self.logger.info(f"      {method}: {count:,}")
            
            if 'by_question_type' in dataset_stats:
                self.logger.info(f"   By question type:")
                for q_type, count in dataset_stats['by_question_type'].items():
                    self.logger.info(f"      {q_type}: {count:,}")
            
            if 'by_expected_advantage' in dataset_stats:
                self.logger.info(f"   By expected advantage:")
                for advantage, count in dataset_stats['by_expected_advantage'].items():
                    self.logger.info(f"      {advantage}: {count:,}")
            
            if 'by_difficulty' in dataset_stats:
                self.logger.info(f"   By difficulty:")
                for difficulty, count in dataset_stats['by_difficulty'].items():
                    self.logger.info(f"      {difficulty}: {count:,}")
            
            if 'question_length_stats' in dataset_stats:
                length_stats = dataset_stats['question_length_stats']
                self.logger.info(f"   Question length: avg={length_stats['mean']:.0f}, min={length_stats['min']}, max={length_stats['max']}")
            
            # Show sample questions
            self.logger.info(f"   Sample questions:")
            for i, question in enumerate(dataset[:3]):
                self.logger.info(f"      {i+1}. [{question.question_type}] {question.question_text}")
                self.logger.info(f"         Expected advantage: {question.expected_advantage}")
            
            # Store dataset in pipeline
            self.dataset = dataset
            self.dataset_stats = dataset_stats
            
            self.logger.info("✅ Phase 6 completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Phase 6 failed: {e}")
            raise

    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            config_path = Path(self.config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)

            print(f"✅ Config loaded from {config_path}")

        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            raise

    def _initialize_experiment_tracker(self):
        """Initialize experiment tracking with unique ID."""
        # Generate experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        self.experiment_id = f"{self.config['experiment']['name']}_{timestamp}_{short_uuid}"

        print(f"🔬 Experiment ID: {self.experiment_id}")

    def _create_output_directories(self):
        """Create all necessary output directories."""
        directories = self.config['directories']

        base_dirs = [
            directories['data'],
            directories['embeddings'],
            directories['visualizations'],
            directories['logs']
        ]

        # Create subdirectories
        subdirs = [
            f"{directories['embeddings']}/raw",
            f"{directories['embeddings']}/similarities",
            f"{directories['embeddings']}/cross_document",
            f"{directories['visualizations']}/experiments",
            f"{directories['data']}/datasets",
            f"{directories['data']}/experiments"
        ]

        all_dirs = base_dirs + subdirs

        for dir_path in all_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Create experiment-specific directories
        exp_viz_dir = Path(directories['visualizations']) / "experiments" / self.experiment_id
        exp_data_dir = Path(directories['data']) / "experiments" / self.experiment_id

        exp_viz_dir.mkdir(parents=True, exist_ok=True)
        exp_data_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Created directory structure")

    def _initialize_logging(self):
        """Initialize logging configuration."""
        log_config = self.config['logging']

        # Create logger
        self.logger = logging.getLogger('semantic_rag_pipeline')
        self.logger.setLevel(getattr(logging, log_config['level']))

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler
        if log_config['log_to_console']:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if log_config['log_to_file']:
            log_file = Path(self.config['directories']['logs']) / f"{self.experiment_id}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Log initial info
        self.logger.info(f"Logging initialized for experiment: {self.experiment_id}")
        self.logger.info(f"Config: {self.config['experiment']}")

        print(f"📝 Logging initialized")

    def _check_system_resources(self):
        """Check system resources and warn if insufficient."""
        # Check available memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)
        max_memory_gb = self.config['system']['max_memory_gb']

        if available_gb < max_memory_gb:
            warning_msg = f"⚠️  Low memory: {available_gb:.1f}GB available, {max_memory_gb}GB recommended"
            print(warning_msg)
            if self.logger:
                self.logger.warning(warning_msg)
        else:
            print(f"💾 Memory check: {available_gb:.1f}GB available ✅")

        # Check disk space
        disk = psutil.disk_usage('.')
        free_gb = disk.free / (1024 ** 3)
        min_disk_gb = self.config['system']['min_disk_space_gb']

        if free_gb < min_disk_gb:
            error_msg = f"❌ Insufficient disk space: {free_gb:.1f}GB free, {min_disk_gb}GB required"
            print(error_msg)
            if self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            print(f"💽 Disk space check: {free_gb:.1f}GB free ✅")

        # Log system info
        if self.logger:
            self.logger.info(f"System: {platform.system()} {platform.machine()}")
            self.logger.info(f"Python: {sys.version}")
            self.logger.info(f"Memory: {available_gb:.1f}GB available")
            self.logger.info(f"Disk: {free_gb:.1f}GB free")

    def _detect_and_configure_device(self):
        """Detect and configure computation device (M1 Mac compatible)."""
        device_config = self.config['system']['device']

        if device_config == "auto":
            # Auto-detect best available device
            if torch.cuda.is_available():
                self.device = "cuda"
                device_name = torch.cuda.get_device_name()
                print(f"🖥️  Using CUDA device: {device_name}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
                print(f"🍎 Using MPS device (Apple Silicon)")
            else:
                self.device = "cpu"
                print(f"💻 Using CPU device")
        else:
            # Use specified device
            self.device = device_config
            print(f"🎯 Using specified device: {self.device}")

        # Validate device availability
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device specified but not available")
        elif self.device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise RuntimeError("MPS device specified but not available")

        # Log device info
        if self.logger:
            self.logger.info(f"Device configured: {self.device}")
            if self.device == "cuda":
                self.logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
                self.logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")
        
        # Update config with resolved device so other components can use it
        self.config['system']['device'] = self.device

    def _validate_config(self):
        """Validate configuration parameters."""
        try:
            # Validate required sections
            required_sections = ['experiment', 'system', 'directories', 'logging']
            for section in required_sections:
                if section not in self.config:
                    raise ValueError(f"Missing required config section: {section}")

            # Validate model names (basic check)
            for model in self.config['models']['embedding_models']:
                if not isinstance(model, str) or len(model) == 0:
                    raise ValueError(f"Invalid embedding model: {model}")

            # Validate numeric parameters
            if self.config['models']['embedding_batch_size'] <= 0:
                raise ValueError("embedding_batch_size must be positive")

            if self.config['chunking']['window_size'] <= 0:
                raise ValueError("window_size must be positive")

            # Validate execution mode
            valid_modes = ['full_pipeline', 'data_only', 'embedding_only', 'evaluation_only', 'visualization_only']
            if self.config['execution']['mode'] not in valid_modes:
                raise ValueError(f"Invalid execution mode. Must be one of: {valid_modes}")

            print(f"✅ Configuration validated")
            if self.logger:
                self.logger.info("Configuration validation passed")

        except Exception as e:
            error_msg = f"Configuration validation failed: {e}"
            print(f"❌ {error_msg}")
            if self.logger:
                self.logger.error(error_msg)
            raise


def main():
    """Main entry point for the pipeline."""
    try:
        # Initialize and run pipeline
        pipeline = SemanticRAGPipeline()
        results = pipeline.pipe()

        print("\n" + "=" * 50)
        print("🎉 Pipeline completed successfully!")
        print(f"📋 Experiment ID: {results['experiment_id']}")
        print(f"⏱️  Execution time: {results['execution_time']}")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()