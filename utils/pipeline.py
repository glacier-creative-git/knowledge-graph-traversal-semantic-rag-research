#!/usr/bin/env python3
"""
Enhanced Multi-Granularity Semantic RAG Pipeline
===============================================

Main orchestrator for the enhanced semantic graph traversal RAG system.
Handles multi-granularity embedding generation, similarity computation, and knowledge graph construction.
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

from utils.knowledge_graph import KnowledgeGraphBuilder
# Import engines (updated for multi-granularity)
from wiki import WikiEngine
from chunking import ChunkEngine
from models import MultiGranularityEmbeddingEngine
from similarity import SimilarityEngine
from retrieval import create_retrieval_engine
from knowledge_graph import KnowledgeGraph
from extraction import ThemeExtractionEngine

class SemanticRAGPipeline:
    """Enhanced main pipeline orchestrator for multi-granularity semantic RAG system."""

    def __init__(self, config_path: str = "/Users/eric/Documents/Development/semantic-rag-chunking-research/config.yaml"):
        """Initialize the pipeline with configuration."""
        self.config_path = config_path
        self.config = None
        self.experiment_id = None
        self.logger = None
        self.device = None
        self.start_time = None

        # Enhanced data storage for multi-granularity pipeline phases
        self.articles = []
        self.corpus_stats = {}
        self.chunks = []
        self.chunk_stats = {}
        
        # Multi-granularity embeddings: Dict[model_name, Dict[granularity_type, List[Embedding]]]
        self.embeddings = {}  
        self.embedding_stats = {}
        
        # Multi-granularity similarities: Dict[model_name, similarity_data]
        self.similarities = {}  
        self.similarity_stats = {}
        
        # Enhanced knowledge graph
        self.knowledge_graph = None  # MultiGranularityKnowledgeGraph instance
        self.kg_stats = {}
        self.retrieval_engine = None  # RetrievalEngine instance
        self.retrieval_stats = {}
        self.questions = []  # List[EvaluationQuestion] from QuestionEngine
        self.question_stats = {}

        # Phase 5: Theme extraction storage (entity extraction removed)
        self.theme_data = {}
        self.theme_stats = {}

    # Update the main pipe() method to include Phase 5
    def pipe(self) -> Dict[str, Any]:
        """Enhanced pipeline execution function with entity/theme extraction."""
        try:
            # Phase 1: Setup & Initialization
            self._phase_1_setup_and_initialization()

            # Phase 2: Data Acquisition
            if self.config['execution']['mode'] in ['full_pipeline', 'data_only']:
                if 'data_acquisition' not in self.config['execution']['skip_phases']:
                    self._phase_2_data_acquisition()
                else:
                    self.logger.info("⏭️  Skipping Phase 2: Data Acquisition")

            # Phase 3: Multi-Granularity Embedding Generation
            if self.config['execution']['mode'] in ['full_pipeline', 'embedding_only']:
                if 'embedding_generation' not in self.config['execution']['skip_phases']:
                    self._phase_3_embedding_generation()
                else:
                    self.logger.info("⏭️  Skipping Phase 3: Multi-Granularity Embedding Generation")

            # Phase 4: Multi-Granularity Similarity Matrix Construction
            if self.config['execution']['mode'] in ['full_pipeline']:
                if 'similarity_matrices' not in self.config['execution']['skip_phases']:
                    self._phase_4_similarity_matrices()
                else:
                    self.logger.info("⏭️  Skipping Phase 4: Multi-Granularity Similarity Matrix Construction")

            # Phase 5: Theme Extraction (entity extraction removed for quality)
            if self.config['execution']['mode'] in ['full_pipeline']:
                if 'theme_extraction' not in self.config['execution']['skip_phases']:
                    self._phase_5_theme_extraction()
                else:
                    self.logger.info("⏭️  Skipping Phase 5: Theme Extraction")

            # Phase 6: Knowledge Graph Construction will be updated to use Phase 4 + Phase 5 data
            if self.config['execution']['mode'] in ['full_pipeline']:
                if 'knowledge_graph_construction' not in self.config['execution']['skip_phases']:
                    self._phase_6_knowledge_graph_construction()  # This will be renamed/updated
                else:
                    self.logger.info("⏭️  Skipping Phase 6: Knowledge Graph Construction")

            self.logger.info("🎉 Enhanced pipeline with entity/theme extraction completed successfully!")
            return {
                "experiment_id": self.experiment_id,
                "status": "completed",
                "execution_time": datetime.now() - self.start_time,
                "architecture": "multi_granularity_entity_theme_extraction"
            }

        except Exception as e:
            if self.logger:
                self.logger.error(f"Pipeline failed: {str(e)}")
            else:
                print(f"Pipeline failed: {str(e)}")
            raise

    def _phase_1_setup_and_initialization(self):
        """Phase 1: Setup & Initialization"""
        print("🚀 Starting Enhanced Multi-Granularity Semantic RAG Pipeline")
        print("🌟 Architecture: Three-Tier Hierarchy (Document → Chunk → Sentence)")
        print("=" * 70)

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

        self.logger.info(f"🎯 Phase 1 completed - Enhanced Experiment ID: {self.experiment_id}")
        self.logger.info(f"🏗️  Architecture: {self.config.get('experiment', {}).get('description', 'Multi-granularity system')}")

    def _phase_2_data_acquisition(self):
        """Phase 2: Data Acquisition (unchanged from original)"""
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
        """Phase 3: Enhanced Multi-Granularity Embedding Generation"""
        self.logger.info("🧠 Starting Phase 3: Enhanced Multi-Granularity Embedding Generation")

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
            
            # Step 2: Generate multi-granularity embeddings using enhanced EmbeddingEngine
            self.logger.info("✨ Generating multi-granularity embeddings for all models")
            embedding_engine = MultiGranularityEmbeddingEngine(self.config, self.logger)
            
            # Check if we should force recompute embeddings
            force_recompute = 'embeddings' in self.config['execution'].get('force_recompute', [])
            
            # Generate multi-granularity embeddings (chunks, sentences, doc_summaries)
            embeddings = embedding_engine.generate_embeddings(
                chunks, 
                self.articles,  # Pass articles for sentence and document summary embedding generation
                force_recompute=force_recompute
            )
            
            if not embeddings:
                raise RuntimeError("No multi-granularity embeddings were generated")
            
            # Get and log embedding statistics
            embedding_stats = embedding_engine.get_embedding_statistics(embeddings)
            self.logger.info("📈 Multi-Granularity Embedding Statistics:")
            for model_name, stats in embedding_stats.items():
                self.logger.info(f"   {model_name}:")
                self.logger.info(f"      Total embeddings: {stats['total_embeddings']:,}")
                for granularity_type, granularity_stats in stats['granularity_types'].items():
                    self.logger.info(f"      {granularity_type}: {granularity_stats['count']:,} embeddings, {granularity_stats['embedding_dimension']}D")
                    if 'sample_chunk_lengths' in granularity_stats:
                        self.logger.info(f"         Sample lengths: {granularity_stats['sample_chunk_lengths']}")
            
            # Store embeddings in pipeline
            self.embeddings = embeddings
            self.embedding_stats = embedding_stats
            
            self.logger.info("✅ Phase 3 Multi-Granularity Embedding Generation completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Phase 3 failed: {e}")
            raise

    def _phase_4_similarity_matrices(self):
        """Phase 4: Enhanced Multi-Granularity Similarity Matrix Construction"""
        self.logger.info("🕰 Starting Phase 4: Enhanced Multi-Granularity Similarity Matrix Construction")

        # Check if we have multi-granularity embeddings from Phase 3
        if not self.embeddings:
            self.logger.warning("No multi-granularity embeddings available from Phase 3. Loading from cache...")
            # Try to load embeddings from cache
            embedding_engine = MultiGranularityEmbeddingEngine(self.config, self.logger)
            # We'd need to reload chunks and embeddings here
            raise RuntimeError("No multi-granularity embeddings available. Please run Phase 3 first.")

        try:
            # Initialize enhanced SimilarityEngine
            self.logger.info("🔗 Initializing multi-granularity similarity engine")
            similarity_engine = SimilarityEngine(self.config, self.logger)
            
            # Check if we should force recompute similarities
            force_recompute = 'similarities' in self.config['execution'].get('force_recompute', [])
            
            # Compute multi-granularity similarity matrices
            self.logger.info(f"🎯 Computing multi-granularity similarity matrices for {len(self.embeddings)} models")
            similarities = similarity_engine.compute_similarity_matrices(
                self.embeddings, 
                force_recompute=force_recompute
            )
            
            if not similarities:
                raise RuntimeError("No multi-granularity similarity matrices were computed")
            
            # Get and log similarity statistics
            similarity_stats = similarity_engine.get_similarity_statistics(similarities)
            self.logger.info("📊 Multi-Granularity Similarity Matrix Statistics:")
            for model_name, stats in similarity_stats.items():
                self.logger.info(f"   {model_name}:")
                self.logger.info(f"      Total connections: {stats['total_connections']:,}")
                self.logger.info(f"      Memory usage: {stats['memory_usage_mb']:.1f} MB")
                self.logger.info(f"      Computation time: {stats['computation_time']:.2f}s")
                
                if 'connections_by_granularity' in stats:
                    self.logger.info(f"      Connections by granularity:")
                    for granularity_type, count in stats['connections_by_granularity'].items():
                        self.logger.info(f"         {granularity_type}: {count:,}")
                
                if 'sparsity_statistics' in stats:
                    self.logger.info(f"      Sparsity statistics:")
                    for matrix_name, sparsity_info in stats['sparsity_statistics'].items():
                        self.logger.info(f"         {matrix_name}: {sparsity_info['stored_connections']:,} connections, sparsity={sparsity_info['sparsity_ratio']:.6f}")
            
            # Store similarities in pipeline
            self.similarities = similarities
            self.similarity_stats = similarity_stats
            
            self.logger.info("✅ Phase 4 Multi-Granularity Similarity Matrix Construction completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Phase 4 failed: {e}")
            raise

    # Updated phase method for theme-only extraction
    def _phase_5_theme_extraction(self):
        """Phase 5: Theme Extraction (entity extraction removed for quality)"""
        self.logger.info("🎨 Starting Phase 5: Theme Extraction")

        # Check if we have required data from previous phases
        if not self.embeddings:
            self.logger.warning("No multi-granularity embeddings available from Phase 3. Loading from cache...")
            raise RuntimeError("No multi-granularity embeddings available. Please run Phase 3 first.")

        if not self.articles:
            self.logger.warning("No articles available from Phase 2. Loading from cache...")
            # Try to load articles from cache
            wiki_engine = WikiEngine(self.config, self.logger)
            wiki_path = Path(self.config['directories']['data']) / "wiki.json"
            if wiki_path.exists():
                self.articles = wiki_engine._load_cached_articles(wiki_path)
            else:
                raise RuntimeError("No articles available. Please run Phase 2 first.")

        try:
            # Initialize ThemeExtractionEngine (entity extraction removed)
            self.logger.info("🎨 Initializing theme extraction engine")
            extraction_engine = ThemeExtractionEngine(self.config, self.logger)

            # Check if we should force recompute
            force_recompute = 'themes' in self.config['execution'].get('force_recompute', [])

            # Extract themes at document level only (entities removed for quality)
            self.logger.info(f"⚡ Extracting themes with force_recompute={force_recompute}")
            theme_data = extraction_engine.extract_themes(
                multi_granularity_embeddings=self.embeddings,
                articles=self.articles,
                force_recompute=force_recompute
            )

            if not theme_data:
                raise RuntimeError("No theme data was extracted")

            # Get extraction statistics
            theme_stats = extraction_engine.get_extraction_statistics(theme_data)

            # Log results
            metadata = theme_data['metadata']
            self.logger.info("📊 Theme Extraction Statistics:")
            self.logger.info(f"   Total themes: {metadata.total_themes_extracted:,}")
            self.logger.info(f"   Documents processed: {metadata.document_count:,}")
            self.logger.info(f"   Processing time: {metadata.processing_time:.2f}s")
            self.logger.info(f"   Average themes per document: {metadata.total_themes_extracted / metadata.document_count:.1f}" if metadata.document_count > 0 else "   No documents processed")

            # Log extraction method used
            method_status = "✅ Ollama available" if metadata.ollama_available else "⚠️  Using fallback method"
            self.logger.info(f"   Extraction method: {method_status}")

            # Store results in pipeline
            self.theme_data = theme_data
            self.theme_stats = theme_stats

            self.logger.info("✅ Phase 5 Theme Extraction completed successfully")

        except Exception as e:
            self.logger.error(f"❌ Phase 5 failed: {e}")
            raise

    def _phase_6_knowledge_graph_construction(self):
        """Phase 6: Knowledge Graph Assembly using pre-computed data from Phases 4 & 5."""
        self.logger.info("🏗️  Starting Phase 6: Knowledge Graph Assembly")

        # Check if we have required data from previous phases
        required_data = [
            (self.chunks, "chunks", "Phase 3"),
            (self.embeddings, "multi-granularity embeddings", "Phase 3"),
            (self.similarities, "similarity matrices", "Phase 4"),
            (self.theme_data, "theme data", "Phase 5")
        ]

        for data, name, phase in required_data:
            if not data:
                self.logger.warning(f"No {name} available from {phase}. Loading from cache...")
                raise RuntimeError(f"No {name} available. Please run {phase} first.")

        try:
            # Check if we should force recompute
            force_recompute = 'knowledge_graph' in self.config['execution'].get('force_recompute', [])

            # Check cache
            kg_path = Path(self.config['directories']['data']) / "knowledge_graph.json"
            if not force_recompute and kg_path.exists():
                self.logger.info("📂 Loading cached knowledge graph")
                self.knowledge_graph = KnowledgeGraph.load(str(kg_path))
                # FIX: Load Phase 3 embeddings into cache after loading from JSON
                self.knowledge_graph.load_phase3_embeddings(self.embeddings)
                self.kg_stats = self.knowledge_graph.metadata
            else:
                # Build fresh knowledge graph using assembly approach
                self.logger.info("🔨 Assembling fresh knowledge graph from pre-computed data")
                kg_builder = KnowledgeGraphBuilder(self.config, self.logger)

                # Fixed call with all required arguments:
                self.knowledge_graph = kg_builder.build_knowledge_graph(
                    chunks=self.chunks,
                    multi_granularity_embeddings=self.embeddings,
                    similarity_data=self.similarities,  # This was missing
                    theme_data=self.theme_data
                )

                # Save knowledge graph
                self.logger.info(f"💾 Saving assembled knowledge graph to {kg_path}")
                self.knowledge_graph.save(str(kg_path))

                self.kg_stats = self.knowledge_graph.metadata

            # Log knowledge graph statistics
            self.logger.info("📊 Knowledge Graph Assembly Statistics:")
            self.logger.info(f"   Architecture: {self.kg_stats.get('architecture', 'phase6_assembly')}")

            if 'theme_bridge_stats' in self.kg_stats:
                bridge_stats = self.kg_stats['theme_bridge_stats']
                self.logger.info(f"   Theme bridges:")
                self.logger.info(f"      Unique themes: {bridge_stats['total_unique_themes']}")
                self.logger.info(f"      Themes with bridges: {bridge_stats['themes_with_bridges']}")
                self.logger.info(f"      Total bridges: {bridge_stats['total_bridges']}")

            self.logger.info(f"   Build time: {self.kg_stats.get('build_time', 0):.2f}s")

            # Initialize the new retrieval engine with knowledge graph
            self.logger.info("🎯 Initializing enhanced retrieval engine")
            self.retrieval_engine = create_retrieval_engine(
                knowledge_graph=self.knowledge_graph,
                config=self.config,
                logger=self.logger
            )
            
            # Test retrieval engine with a simple query
            try:
                test_query = "What is machine learning?"
                test_result = self.retrieval_engine.retrieve(test_query, strategy="baseline_vector")
                
                self.retrieval_stats = {
                    'status': 'initialized_successfully',
                    'test_query_results': len(test_result.retrieved_content),
                    'models_available': list(self.embeddings.keys()) if self.embeddings else [],
                    'knowledge_graph_enabled': True,
                    'available_strategies': ['baseline_vector', 'semantic_traversal']
                }
                
                self.logger.info(f"✅ Retrieval engine test successful: {len(test_result.retrieved_content)} results for test query")
                
            except Exception as e:
                self.logger.warning(f"⚠️  Retrieval engine test failed: {e}")
                self.retrieval_stats = {
                    'status': 'initialized_with_warnings',
                    'error': str(e),
                    'models_available': list(self.embeddings.keys()) if self.embeddings else [],
                    'knowledge_graph_enabled': True
                }

            self.logger.info("✅ Phase 6 Knowledge Graph Assembly completed successfully")

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

            print(f"✅ Enhanced Config loaded from {config_path}")

        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            raise

    def _initialize_experiment_tracker(self):
        """Initialize experiment tracking with unique ID."""
        # Generate experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        self.experiment_id = f"{self.config['experiment']['name']}_{timestamp}_{short_uuid}"

        print(f"🔬 Enhanced Experiment ID: {self.experiment_id}")

    def _create_output_directories(self):
        """Create all necessary output directories."""
        directories = self.config['directories']

        base_dirs = [
            directories['data'],
            directories['embeddings'],
            directories['visualizations'],
            directories['logs']
        ]

        # Create subdirectories (enhanced for multi-granularity)
        subdirs = [
            f"{directories['embeddings']}/raw",
            f"{directories['embeddings']}/similarities",
            f"{directories['embeddings']}/cross_document",
            f"{directories['visualizations']}/experiments",
            f"{directories['data']}/datasets",
            f"{directories['data']}/experiments",
            f"{directories['data']}/questions"  # Added for question generation
        ]

        all_dirs = base_dirs + subdirs

        for dir_path in all_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Create experiment-specific directories
        exp_viz_dir = Path(directories['visualizations']) / "experiments" / self.experiment_id
        exp_data_dir = Path(directories['data']) / "experiments" / self.experiment_id

        exp_viz_dir.mkdir(parents=True, exist_ok=True)
        exp_data_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Created enhanced directory structure")

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
        self.logger.info(f"Enhanced multi-granularity logging initialized for experiment: {self.experiment_id}")
        self.logger.info(f"Config: {self.config['experiment']}")

        print(f"📝 Enhanced logging initialized")

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

            # Validate multi-granularity configuration
            if 'granularity_types' in self.config['models']:
                granularity_config = self.config['models']['granularity_types']
                for granularity_type, enabled in granularity_config.items():
                    if not isinstance(enabled, dict):
                        raise ValueError(f"Invalid granularity configuration for {granularity_type}")

            print(f"✅ Enhanced configuration validated")
            if self.logger:
                self.logger.info("Enhanced configuration validation passed")

        except Exception as e:
            error_msg = f"Configuration validation failed: {e}"
            print(f"❌ {error_msg}")
            if self.logger:
                self.logger.error(error_msg)
            raise


def main():
    """Main entry point for the enhanced pipeline."""
    try:
        # Initialize and run enhanced pipeline
        pipeline = SemanticRAGPipeline()
        results = pipeline.pipe()

        print("\n" + "=" * 70)
        print("🎉 Enhanced Multi-Granularity Pipeline completed successfully!")
        print(f"🌟 Architecture: {results.get('architecture', 'multi_granularity_three_tier')}")
        print(f"📋 Experiment ID: {results['experiment_id']}")
        print(f"⏱️  Execution time: {results['execution_time']}")
        print(f"📁 Results saved in experiments directory")

    except Exception as e:
        print(f"\n❌ Enhanced pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()