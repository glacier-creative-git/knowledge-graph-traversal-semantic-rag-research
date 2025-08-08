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

            # TODO: Add remaining phases
            # self._phase_3_embedding_generation()
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