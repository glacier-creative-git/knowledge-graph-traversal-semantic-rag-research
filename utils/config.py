"""
Configuration Management for Research Pipeline
============================================

Centralized configuration for all research components including
API keys, model settings, and pipeline parameters.
"""

import os
from dataclasses import dataclass
from typing import List, Optional
import warnings

@dataclass
class ModelConfig:
    """Configuration for embedding and chat models"""
    embedding_model: str = "all-MiniLM-L6-v2"
    chat_model: str = "gpt-3.5-turbo"
    openai_api_key: Optional[str] = None

    def __post_init__(self):
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key and not self.openai_api_key.startswith("sk-"):
            warnings.warn("OpenAI API key format appears incorrect")

@dataclass
class RAGConfig:
    """Configuration for RAG traversal algorithm"""
    top_k_per_sentence: int = 20
    cross_doc_k: int = 10
    traversal_depth: int = 3
    use_sliding_window: bool = True
    num_contexts: int = 5
    retrieval_top_k: int = 10
    similarity_threshold: float = 0.5  # Drop results below this similarity to query

@dataclass
class DataConfig:
    """Configuration for dataset loading"""
    num_samples: int = 1000
    min_context_length: int = 400
    max_eval_samples: int = 20

@dataclass
class VisualizationConfig:
    """Configuration for visualizations"""
    figure_size_2d: tuple = (24, 8)
    figure_size_3d: tuple = (12, 10)
    dpi: int = 150
    max_steps_shown: int = 15
    save_plots: bool = False
    output_dir: str = "./outputs"

@dataclass
class ResearchConfig:
    """Master configuration combining all components"""
    models: ModelConfig
    rag: RAGConfig
    data: DataConfig
    viz: VisualizationConfig

    @classmethod
    def default(cls, openai_api_key: Optional[str] = None):
        """Create default configuration"""
        return cls(
            models=ModelConfig(openai_api_key=openai_api_key),
            rag=RAGConfig(),
            data=DataConfig(),
            viz=VisualizationConfig()
        )

    @classmethod
    def demo(cls, openai_api_key: Optional[str] = None):
        """Create demo configuration with smaller datasets"""
        return cls(
            models=ModelConfig(openai_api_key=openai_api_key),
            rag=RAGConfig(num_contexts=3, retrieval_top_k=5),
            data=DataConfig(num_samples=100, max_eval_samples=5),
            viz=VisualizationConfig(max_steps_shown=10)
        )

def get_config(config_type: str = "default", **kwargs) -> ResearchConfig:
    """
    Get configuration by type

    Args:
        config_type: "default" or "demo"
        **kwargs: Override any configuration parameters
    """
    # Extract openai_api_key for base config creation
    openai_api_key = kwargs.pop('openai_api_key', None)

    if config_type == "demo":
        config = ResearchConfig.demo(openai_api_key=openai_api_key)
    else:
        config = ResearchConfig.default(openai_api_key=openai_api_key)

    # Apply any overrides to nested attributes
    for key, value in kwargs.items():
        applied = False

        # Try to set nested attributes
        for attr_name in ['models', 'rag', 'data', 'viz']:
            attr_obj = getattr(config, attr_name)
            if hasattr(attr_obj, key):
                setattr(attr_obj, key, value)
                applied = True
                break

        # If not found in nested attributes, try main config
        if not applied and hasattr(config, key):
            setattr(config, key, value)

    return config