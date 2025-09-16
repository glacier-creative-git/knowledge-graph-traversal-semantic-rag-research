#!/usr/bin/env python3
"""
Phase 5: Theme Extraction Engine
===============================

Dedicated phase for extracting themes at document level.
This phase runs after Phase 4 (similarity computation) and before Phase 6 (knowledge graph assembly).

Architecture:
- Document-level themes: Ollama-based conceptual theme extraction with fallback
- Caching: Independent cache for theme data
- Entity extraction removed: was creating spurious relationships, themes work excellently
"""

import json
import hashlib
import time
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from utils.wiki import WikipediaArticle
from utils.models import ChunkEmbedding, SentenceEmbedding, DocumentSummaryEmbedding


@dataclass
class ThemeExtractionResult:
    """Container for theme extraction results at document level."""
    doc_id: str
    doc_title: str
    source_text: str  # Document summary text used for theme extraction
    themes: List[str]  # Clean theme names like ["Artificial Intelligence", "Machine Learning"]
    extraction_method: str  # 'ollama', 'fallback'
    extraction_time: float
    model_used: str  # Ollama model name

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ThemeExtractionMetadata:
    """Metadata for cached theme extraction results."""
    created_at: str
    document_count: int  # Number of documents processed
    theme_extraction_config: Dict[str, Any]
    total_themes_extracted: int
    processing_time: float
    config_hash: str
    ollama_available: bool



class OllamaThemeExtractor:
    """Ollama-based theme extraction for document summaries."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize Ollama theme extractor."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.ollama_config = config.get('ollama', {})
        self.model = self.ollama_config.get('model', 'llama3.1:8b')
        self.num_themes = config.get('theme_extraction', {}).get('num_themes', 5)
        self.available = self._test_ollama_connection()

    def _test_ollama_connection(self) -> bool:
        """Test if Ollama is available and has the required model."""
        if not OLLAMA_AVAILABLE:
            self.logger.warning("⚠️  Ollama library not available")
            return False

        try:
            models = ollama.list()
            available_models = [model.model for model in models.models]

            if self.model in available_models:
                self.logger.info(f"✅ Ollama model {self.model} available for theme extraction")
                return True
            else:
                self.logger.warning(f"⚠️  Ollama model {self.model} not found. Available: {available_models}")
                return False
        except Exception as e:
            self.logger.warning(f"⚠️  Ollama connection test failed: {e}")
            return False

    def extract_themes(self, doc_summary: str, doc_id: str, doc_title: str) -> ThemeExtractionResult:
        """Extract themes from document summary using Ollama."""
        start_time = time.time()

        if self.available:
            themes, method = self._extract_with_ollama(doc_summary, doc_title)
        else:
            themes, method = self._extract_with_fallback(doc_summary, doc_title)

        extraction_time = time.time() - start_time

        return ThemeExtractionResult(
            doc_id=doc_id,
            doc_title=doc_title,
            source_text=doc_summary,
            themes=themes,
            extraction_method=method,
            extraction_time=extraction_time,
            model_used=self.model if self.available else 'fallback'
        )

    def _extract_with_ollama(self, doc_summary: str, doc_title: str) -> Tuple[List[str], str]:
        """Extract themes using Ollama with structured JSON output."""
        prompt = self._build_theme_extraction_prompt(doc_summary, doc_title)

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.1,  # Low temperature for consistent output
                    "num_predict": 200,  # Enough tokens for JSON response
                    "stop": ["\n\nExplanation:", "\n\nNote:", "```"],  # Stop tokens
                    "timeout": 30
                }
            )

            response_text = response['response'].strip()
            themes = self._parse_ollama_response(response_text)

            if themes:
                return themes, 'ollama'
            else:
                self.logger.warning(f"Ollama returned empty themes for doc {doc_title}, using fallback")
                return self._extract_with_fallback(doc_summary, doc_title)[0], 'ollama_fallback'

        except Exception as e:
            self.logger.warning(f"Ollama theme extraction failed for {doc_title}: {e}")
            return self._extract_with_fallback(doc_summary, doc_title)[0], 'ollama_error'

    def _build_theme_extraction_prompt(self, doc_summary: str, doc_title: str) -> str:
        """Build the prompt for Ollama theme extraction."""
        prompt = f"""Extract {self.num_themes} main themes from this document summary. Return ONLY a JSON array of theme names.

Document Title: {doc_title}

Document Summary:
{doc_summary}

Requirements:
- Extract {self.num_themes} conceptual themes that capture the main topics
- Use clean, readable theme names (e.g., "Artificial Intelligence", not "artificial_intelligence")  
- Focus on substantive concepts, not specific entities
- Return ONLY the JSON array, no other text

JSON Array:"""

        return prompt

    def _parse_ollama_response(self, response_text: str) -> List[str]:
        """Parse Ollama response to extract themes from JSON."""
        try:
            # Clean response text - remove any markdown formatting
            cleaned_response = response_text.replace('```json', '').replace('```', '').strip()

            # Try to extract JSON array from response
            json_match = re.search(r'\[.*?\]', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                themes = json.loads(json_str)

                # Validate themes are strings and clean them
                cleaned_themes = []
                for theme in themes:
                    if isinstance(theme, str) and len(theme.strip()) > 0:
                        # Clean and format theme
                        clean_theme = theme.strip().strip('"').strip("'")
                        if len(clean_theme) > 0:
                            cleaned_themes.append(clean_theme)

                return cleaned_themes[:self.num_themes]  # Limit to requested number
            else:
                self.logger.warning("No JSON array found in Ollama response")
                return []

        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse Ollama JSON response: {e}")
            return []
        except Exception as e:
            self.logger.warning(f"Error parsing Ollama response: {e}")
            return []

    def _extract_with_fallback(self, doc_summary: str, doc_title: str) -> Tuple[List[str], str]:
        """Fallback theme extraction using keyword-based approach."""
        # Extract potential themes from title and summary
        text = f"{doc_title} {doc_summary}".lower()

        # Define theme patterns based on common academic/technical concepts
        theme_patterns = {
            "Artificial Intelligence": ["artificial intelligence", "ai", "machine intelligence"],
            "Machine Learning": ["machine learning", "ml", "learning algorithms"],
            "Neural Networks": ["neural networks", "neural nets", "artificial neural"],
            "Deep Learning": ["deep learning", "deep neural", "convolutional"],
            "Natural Language Processing": ["natural language", "nlp", "text processing"],
            "Computer Vision": ["computer vision", "image recognition", "visual"],
            "Data Science": ["data science", "data analysis", "big data"],
            "Robotics": ["robotics", "robots", "autonomous"],
            "Psychology": ["psychology", "psychological", "behavior", "cognitive"],
            "Neuroscience": ["neuroscience", "brain", "neural", "cognitive science"],
            "History": ["history", "historical", "past", "century"],
            "Technology": ["technology", "technological", "computing", "computer"],
            "Research": ["research", "study", "studies", "investigation"],
            "Education": ["education", "learning", "teaching", "academic"]
        }

        detected_themes = []
        for theme, patterns in theme_patterns.items():
            if any(pattern in text for pattern in patterns):
                detected_themes.append(theme)

        # If no themes detected, extract capitalized phrases as potential themes
        if not detected_themes:
            capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', doc_summary)
            # Take unique phrases and limit
            unique_phrases = list(dict.fromkeys(capitalized_phrases))
            detected_themes = unique_phrases[:self.num_themes]

        return detected_themes[:self.num_themes], 'fallback'


class ThemeExtractionEngine:
    """Main engine for Phase 5: Theme Extraction."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize the theme extraction engine."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.data_dir = Path(config['directories']['data'])

        # Initialize theme extractor only
        self.theme_extractor = OllamaThemeExtractor(config, self.logger)

        # Create theme data directory
        self.theme_dir = self.data_dir / "themes"
        self.theme_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("🎨 Phase 5: Theme Extraction Engine initialized")
        self._log_extractor_status()

    def _log_extractor_status(self):
        """Log the status of available extractors."""
        self.logger.info(
            f"   Theme extraction (Ollama): {'✅ Available' if self.theme_extractor.available else '⚠️  Fallback mode'}")

    def extract_themes(self, multi_granularity_embeddings: Dict[str, Dict[str, List[Any]]],
                      articles: List[WikipediaArticle],
                      force_recompute: bool = False) -> Dict[str, Any]:
        """
        Main extraction method for themes at document level.

        Args:
            multi_granularity_embeddings: Embedding data from Phase 3
            articles: Original Wikipedia articles
            force_recompute: Whether to force recomputation even if cached

        Returns:
            Dictionary containing extracted theme data
        """
        start_time = time.time()
        self.logger.info("🎨 Starting Phase 5: Theme Extraction")

        # Generate config hash for cache validation
        config_hash = self._generate_config_hash(multi_granularity_embeddings, articles)

        # Check cache
        cache_path = self._get_cache_path()
        if not force_recompute and self._is_cache_valid(cache_path, config_hash):
            self.logger.info("📂 Loading cached theme data")
            return self._load_cached_data(cache_path)

        self.logger.info("⚡ Extracting fresh theme data")

        # Extract document-level themes (UNCHANGED - this works excellently)
        document_themes = self._extract_document_themes(multi_granularity_embeddings, articles)
        
        # Package results (simplified structure)
        extraction_results = {
            'document_themes': document_themes
        }

        processing_time = time.time() - start_time

        # Create metadata
        metadata = self._create_metadata(extraction_results, config_hash, processing_time)

        # Package final results
        theme_data = {
            'metadata': metadata,
            'extraction_results': extraction_results
        }

        # Cache results
        self._cache_data(cache_path, theme_data)

        # Log comprehensive results
        self._log_extraction_results(metadata)

        return theme_data



    def _extract_document_themes(self, multi_granularity_embeddings: Dict[str, Dict[str, List[Any]]],
                                 articles: List[WikipediaArticle]) -> List[ThemeExtractionResult]:
        """Extract themes from document summaries."""
        # Get document summary embeddings from first model
        model_name = list(multi_granularity_embeddings.keys())[0]
        doc_summary_embeddings = multi_granularity_embeddings[model_name].get('doc_summaries', [])

        self.logger.info(f"📄 Extracting themes from {len(doc_summary_embeddings)} document summaries...")

        document_themes = []
        for doc_emb in doc_summary_embeddings:
            result = self.theme_extractor.extract_themes(
                doc_summary=doc_emb.summary_text,
                doc_id=doc_emb.doc_id,
                doc_title=doc_emb.doc_title
            )
            document_themes.append(result)

        total_themes = sum(len(result.themes) for result in document_themes)
        self.logger.info(f"   ✅ Extracted {total_themes} themes from documents")

        return document_themes

    def _create_metadata(self, extraction_results: Dict[str, List], config_hash: str,
                         processing_time: float) -> ThemeExtractionMetadata:
        """Create metadata for the extraction results."""
        document_count = len(extraction_results.get('document_themes', []))
        total_themes = sum(len(result.themes) for result in extraction_results.get('document_themes', []))

        return ThemeExtractionMetadata(
            created_at=datetime.now().isoformat(),
            document_count=document_count,
            theme_extraction_config={'num_themes': self.theme_extractor.num_themes,
                                     'model': self.theme_extractor.model},
            total_themes_extracted=total_themes,
            processing_time=processing_time,
            config_hash=config_hash,
            ollama_available=self.theme_extractor.available
        )

    def _generate_config_hash(self, multi_granularity_embeddings: Dict[str, Dict[str, List[Any]]],
                              articles: List[WikipediaArticle]) -> str:
        """Generate hash of configuration and input data for cache validation."""
        config_str = json.dumps({
            'num_themes': self.theme_extractor.num_themes,
            'ollama_model': self.theme_extractor.model,
            'doc_summary_count': len(multi_granularity_embeddings.get(list(multi_granularity_embeddings.keys())[0], {}).get('doc_summaries', [])) if multi_granularity_embeddings else 0,
            'article_count': len(articles),
            'first_article_title': articles[0].title if articles else ""
        }, sort_keys=True)

        return hashlib.md5(config_str.encode()).hexdigest()

    def _get_cache_path(self) -> Path:
        """Get cache path for theme data."""
        return self.theme_dir / "theme_extraction.json"

    def _is_cache_valid(self, cache_path: Path, expected_hash: str) -> bool:
        """Check if cached theme data is valid."""
        if not cache_path.exists():
            return False

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            metadata = cache_data.get('metadata', {})
            cached_hash = metadata.get('config_hash', '')

            return cached_hash == expected_hash

        except Exception as e:
            self.logger.warning(f"Failed to validate theme cache: {e}")
            return False

    def _cache_data(self, cache_path: Path, theme_data: Dict[str, Any]):
        """Cache theme data to disk."""
        try:
            # Convert all results to dictionaries for JSON serialization
            serializable_data = {
                'metadata': asdict(theme_data['metadata']),
                'extraction_results': {}
            }

            for result_type, results in theme_data['extraction_results'].items():
                serializable_data['extraction_results'][result_type] = [
                    result.to_dict() for result in results
                ]

            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"💾 Cached theme data to {cache_path}")

        except Exception as e:
            self.logger.error(f"Failed to cache theme data: {e}")
            raise

    def _load_cached_data(self, cache_path: Path) -> Dict[str, Any]:
        """Load cached theme data from disk."""
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # Reconstruct objects from dictionaries
            metadata = ThemeExtractionMetadata(**cache_data['metadata'])

            extraction_results = {}
            for result_type, result_data_list in cache_data['extraction_results'].items():
                if result_type == 'document_themes':
                    extraction_results[result_type] = [
                        ThemeExtractionResult(**result_data) for result_data in result_data_list
                    ]
                # No entity data to handle anymore

            return {
                'metadata': metadata,
                'extraction_results': extraction_results
            }

        except Exception as e:
            self.logger.error(f"Failed to load cached theme data: {e}")
            raise

    def _log_extraction_results(self, metadata: ThemeExtractionMetadata):
        """Log comprehensive extraction results."""
        self.logger.info("📊 Phase 5 Theme Extraction Results:")
        self.logger.info(f"   Total themes extracted: {metadata.total_themes_extracted:,}")
        self.logger.info(f"   Documents processed: {metadata.document_count:,}")
        self.logger.info(f"   Processing time: {metadata.processing_time:.2f}s")
        self.logger.info(f"   Average themes per document: {metadata.total_themes_extracted / metadata.document_count:.1f}" if metadata.document_count > 0 else "   No documents processed")

        self.logger.info(f"   Extraction method:")
        self.logger.info(
            f"      Theme extraction: {'Ollama' if metadata.ollama_available else 'Keyword-based fallback'}")

    def get_extraction_statistics(self, theme_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics about the theme extraction results."""
        metadata = theme_data['metadata']
        extraction_results = theme_data['extraction_results']

        # Calculate theme-focused statistics
        stats = {
            'total_themes': metadata.total_themes_extracted,
            'document_count': metadata.document_count,
            'processing_time': metadata.processing_time,
            'extraction_method': 'ollama' if metadata.ollama_available else 'fallback'
        }

        # Theme statistics
        if extraction_results.get('document_themes'):
            theme_lengths = [len(result.themes) for result in extraction_results['document_themes']]
            stats['theme_statistics'] = {
                'avg_themes_per_document': sum(theme_lengths) / len(theme_lengths) if theme_lengths else 0,
                'max_themes_per_document': max(theme_lengths) if theme_lengths else 0,
                'min_themes_per_document': min(theme_lengths) if theme_lengths else 0,
                'total_unique_themes': len(set(theme for result in extraction_results['document_themes'] for theme in result.themes))
            }

        return stats