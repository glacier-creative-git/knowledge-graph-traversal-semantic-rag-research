"""
Enhanced Configuration Management for Research Pipeline
=====================================================

Updated configuration system supporting WikiEval, Natural Questions,
and future datasets with seamless dataset swapping capability.
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
    """
    Configuration for dataset loading and processing.

    Enhanced to support multiple datasets with seamless swapping.
    """
    # Dataset Selection - This is the key parameter for swapping datasets
    dataset_name: str = "wikieval"  # Options: "wikieval", "natural_questions"

    # Dataset Loading Parameters
    num_samples: int = 1000  # For datasets that support size limiting
    min_context_length: int = 400  # Minimum context length for filtering
    max_eval_samples: int = 20  # Maximum samples for evaluation

    # WikiEval Specific Parameters
    wikieval_use_all_samples: bool = True  # WikiEval only has 50 samples

    # Natural Questions Specific Parameters
    nq_split: str = "validation"  # Options: "train", "validation" (validation is much smaller)
    nq_streaming: bool = True  # Stream dataset to avoid disk usage
    nq_max_samples: int = 1000  # Maximum samples to use (prevents massive downloads)
    nq_extract_short_answers: bool = True  # Prefer short answers when available
    nq_max_context_length: int = 10000  # Limit HTML context extraction

    # General Processing Parameters
    enable_html_cleaning: bool = True  # Clean HTML content (for Natural Questions)
    enable_text_preprocessing: bool = True  # Apply text preprocessing
    min_question_length: int = 10  # Filter out very short questions

    def validate_dataset_config(self) -> bool:
        """
        Validate dataset configuration parameters.

        Returns True if configuration is valid, raises ValueError otherwise.
        """
        valid_datasets = ["wikieval", "natural_questions", "nq"]
        if self.dataset_name.lower() not in valid_datasets:
            raise ValueError(f"Invalid dataset_name: {self.dataset_name}. "
                           f"Supported datasets: {valid_datasets}")

        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got: {self.num_samples}")

        if self.max_eval_samples <= 0:
            raise ValueError(f"max_eval_samples must be positive, got: {self.max_eval_samples}")

        if self.nq_split not in ["train", "validation"]:
            raise ValueError(f"Invalid nq_split: {self.nq_split}. Options: 'train', 'validation'")

        return True

    def get_dataset_display_name(self) -> str:
        """Get human-readable dataset name for logging"""
        name_mapping = {
            "wikieval": "WikiEval",
            "natural_questions": "Natural Questions",
            "nq": "Natural Questions"
        }
        return name_mapping.get(self.dataset_name.lower(), self.dataset_name)

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

    def __post_init__(self):
        """Validate configuration after initialization"""
        self.data.validate_dataset_config()

    @classmethod
    def default(cls, openai_api_key: Optional[str] = None, dataset_name: str = "wikieval"):
        """Create default configuration with specified dataset"""
        return cls(
            models=ModelConfig(openai_api_key=openai_api_key),
            rag=RAGConfig(),
            data=DataConfig(dataset_name=dataset_name),
            viz=VisualizationConfig()
        )

    @classmethod
    def demo(cls, openai_api_key: Optional[str] = None, dataset_name: str = "wikieval"):
        """Create demo configuration with smaller datasets"""
        return cls(
            models=ModelConfig(openai_api_key=openai_api_key),
            rag=RAGConfig(num_contexts=3, retrieval_top_k=5),
            data=DataConfig(
                dataset_name=dataset_name,
                num_samples=50,  # Smaller for demo
                max_eval_samples=5
            ),
            viz=VisualizationConfig(max_steps_shown=10)
        )

    @classmethod
    def wikieval_focused(cls, openai_api_key: Optional[str] = None):
        """Create WikiEval-optimized configuration"""
        return cls(
            models=ModelConfig(openai_api_key=openai_api_key),
            rag=RAGConfig(
                num_contexts=3,  # WikiEval only has 50 samples
                similarity_threshold=0.6  # Higher threshold for quality
            ),
            data=DataConfig(
                dataset_name="wikieval",
                wikieval_use_all_samples=True,
                max_eval_samples=50  # Use all WikiEval samples
            ),
            viz=VisualizationConfig()
        )

    @classmethod
    def natural_questions_focused(cls, openai_api_key: Optional[str] = None):
        """Create Natural Questions-optimized configuration"""
        return cls(
            models=ModelConfig(openai_api_key=openai_api_key),
            rag=RAGConfig(
                num_contexts=5,
                similarity_threshold=0.5,
                top_k_per_sentence=25  # Higher for larger dataset
            ),
            data=DataConfig(
                dataset_name="natural_questions",
                num_samples=1000,
                nq_split="validation",  # Often cleaner than train
                nq_extract_short_answers=True,
                max_eval_samples=30
            ),
            viz=VisualizationConfig()
        )

def get_config(config_type: str = "default", **kwargs) -> ResearchConfig:
    """
    Get configuration by type with dataset-aware parameter handling.

    Args:
        config_type: Configuration preset ("default", "demo", "wikieval", "natural_questions")
        **kwargs: Override any configuration parameters

    Returns:
        Configured ResearchConfig instance
    """
    # Extract special parameters
    openai_api_key = kwargs.pop('openai_api_key', None)
    dataset_name = kwargs.pop('dataset_name', None)

    # Create base configuration based on type
    if config_type == "demo":
        config = ResearchConfig.demo(openai_api_key=openai_api_key,
                                   dataset_name=dataset_name or "wikieval")
    elif config_type == "wikieval":
        config = ResearchConfig.wikieval_focused(openai_api_key=openai_api_key)
    elif config_type == "natural_questions":
        config = ResearchConfig.natural_questions_focused(openai_api_key=openai_api_key)
    else:  # default
        config = ResearchConfig.default(openai_api_key=openai_api_key,
                                      dataset_name=dataset_name or "wikieval")

    # Apply any parameter overrides
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

    # Re-validate after applying overrides
    config.data.validate_dataset_config()

    return config


def get_available_datasets() -> List[str]:
    """Get list of available dataset names"""
    return ["wikieval", "natural_questions"]


def get_dataset_info(dataset_name: str) -> dict[str, str]:
    """Get information about a specific dataset"""
    dataset_info = {
        "wikieval": {
            "name": "WikiEval",
            "description": "50 Wikipedia pages with human-annotated QA pairs for RAG evaluation",
            "size": "~50 samples",
            "source": "explodinggradients/ragas team",
            "strengths": "Human-annotated, designed for RAG, high quality",
            "use_case": "RAG system validation and benchmarking"
        },
        "natural_questions": {
            "name": "Natural Questions",
            "description": "Real Google search queries with Wikipedia answers",
            "size": "~300K samples",
            "source": "Google Research",
            "strengths": "Real user queries, large scale, established benchmark",
            "use_case": "Large-scale RAG evaluation and training"
        }
    }

    return dataset_info.get(dataset_name.lower(), {
        "name": "Unknown",
        "description": "Dataset not found",
        "size": "Unknown",
        "source": "Unknown",
        "strengths": "Unknown",
        "use_case": "Unknown"
    })


def print_dataset_info():
    """Print information about all available datasets"""
    print("📊 AVAILABLE DATASETS")
    print("=" * 50)

    for dataset_name in get_available_datasets():
        info = get_dataset_info(dataset_name)
        print(f"\n🔬 {info['name']} ({dataset_name})")
        print(f"   Description: {info['description']}")
        print(f"   Size: {info['size']}")
        print(f"   Source: {info['source']}")
        print(f"   Strengths: {info['strengths']}")
        print(f"   Best for: {info['use_case']}")

    print(f"\n💡 Usage:")
    print(f"   config = get_config('default', dataset_name='wikieval')")
    print(f"   config = get_config('natural_questions')  # Preset config")
    print("=" * 50)


# Configuration validation function
def validate_research_config(config: ResearchConfig) -> bool:
    """
    Comprehensive validation of research configuration.

    Args:
        config: ResearchConfig to validate

    Returns:
        True if valid, raises appropriate exceptions otherwise
    """
    # Validate data configuration
    config.data.validate_dataset_config()

    # Validate RAG parameters
    if config.rag.top_k_per_sentence <= 0:
        raise ValueError(f"top_k_per_sentence must be positive, got: {config.rag.top_k_per_sentence}")

    if config.rag.similarity_threshold < 0 or config.rag.similarity_threshold > 1:
        raise ValueError(f"similarity_threshold must be between 0 and 1, got: {config.rag.similarity_threshold}")

    # Validate model configuration
    if config.models.openai_api_key and not config.models.openai_api_key.startswith("sk-"):
        warnings.warn("OpenAI API key format appears incorrect")

    return True