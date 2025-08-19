#!/usr/bin/env python3
"""
Phase 5: Entity/Theme Extraction Engine
======================================

Dedicated phase for extracting entities and themes across all granularity levels.
This phase runs after Phase 4 (similarity computation) and before Phase 6 (knowledge graph assembly).

Architecture:
- Chunk-level entities: PERSON/ORG/GPE extraction using spaCy
- Sentence-level entities: Same entity types for fine-grained relationships
- Document-level themes: Ollama-based conceptual theme extraction
- Caching: Independent cache for entity/theme data
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

import spacy

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from wiki import WikipediaArticle
from models import ChunkEmbedding, SentenceEmbedding, DocumentSummaryEmbedding


@dataclass
class EntityExtractionResult:
    """Container for entity extraction results at any granularity level."""
    granularity_level: str  # 'chunk', 'sentence', 'document'
    source_id: str  # chunk_id, sentence_id, or doc_id
    source_text: str  # The text that entities were extracted from
    entities: Dict[str, List[str]]  # {'PERSON': [...], 'ORG': [...], 'GPE': [...]}
    extraction_method: str  # 'spacy', 'patterns'
    extraction_time: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


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
class EntityThemeExtractionMetadata:
    """Metadata for cached entity/theme extraction results."""
    created_at: str
    granularity_counts: Dict[str, int]  # chunks: 100, sentences: 300, documents: 10
    entity_extraction_config: Dict[str, Any]
    theme_extraction_config: Dict[str, Any]
    total_entities_extracted: int
    total_themes_extracted: int
    processing_time: float
    config_hash: str
    ollama_available: bool
    spacy_model_available: bool


class SpacyEntityExtractor:
    """High-quality entity extraction using spaCy NER."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize spaCy entity extractor."""
        self.logger = logger or logging.getLogger(__name__)
        self.entity_types = ['PERSON', 'ORG', 'GPE']  # High-quality entity types only
        self.nlp = None
        self.available = False

        self._load_spacy_model()

    def _load_spacy_model(self):
        """Load spaCy model with fallback handling."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.available = True
            self.logger.info("✅ spaCy model loaded successfully")
        except OSError:
            self.logger.warning("⚠️  spaCy model 'en_core_web_sm' not found, will use pattern-based fallback")
            self.available = False

    def extract_entities(self, text: str, source_id: str, granularity_level: str) -> EntityExtractionResult:
        """Extract entities from text using spaCy or pattern-based fallback."""
        start_time = time.time()

        if self.available:
            entities, method = self._extract_with_spacy(text)
        else:
            entities, method = self._extract_with_patterns(text)

        extraction_time = time.time() - start_time

        return EntityExtractionResult(
            granularity_level=granularity_level,
            source_id=source_id,
            source_text=text[:200] + "..." if len(text) > 200 else text,  # Store truncated text for reference
            entities=entities,
            extraction_method=method,
            extraction_time=extraction_time
        )

    def _extract_with_spacy(self, text: str) -> Tuple[Dict[str, List[str]], str]:
        """Extract entities using spaCy NER."""
        doc = self.nlp(text)
        entities = {entity_type: [] for entity_type in self.entity_types}

        for ent in doc.ents:
            entity_text = ent.text.strip()

            # Quality filtering
            if len(entity_text) < 2:  # Skip very short entities
                continue
            if entity_text.lower() in ['the', 'and', 'or', 'but']:  # Skip common words
                continue

            if ent.label_ in self.entity_types:
                entities[ent.label_].append(entity_text)

        # Remove duplicates while preserving order
        for entity_type in entities:
            entities[entity_type] = list(dict.fromkeys(entities[entity_type]))

        return entities, 'spacy'

    def _extract_with_patterns(self, text: str) -> Tuple[Dict[str, List[str]], str]:
        """Fallback entity extraction using basic patterns."""
        entities = {entity_type: [] for entity_type in self.entity_types}

        # Extract capitalized words/phrases (potential proper nouns)
        capitalized_patterns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

        for phrase in capitalized_patterns:
            phrase = phrase.strip()
            if len(phrase) < 3:
                continue

            # Basic heuristics for classification
            phrase_lower = phrase.lower()
            if any(word in phrase_lower for word in
                   ['university', 'company', 'corporation', 'inc', 'ltd', 'institute']):
                entities['ORG'].append(phrase)
            elif any(word in phrase_lower for word in ['dr', 'professor', 'mr', 'ms', 'mrs']):
                entities['PERSON'].append(phrase)
            elif any(word in phrase_lower for word in ['united states', 'america', 'canada', 'europe', 'asia']):
                entities['GPE'].append(phrase)
            else:
                # Default classification based on context
                if len(phrase.split()) >= 2:  # Multi-word phrases often organizations
                    entities['ORG'].append(phrase)
                else:  # Single words often places or people
                    entities['GPE'].append(phrase)

        # Remove duplicates while preserving order
        for entity_type in entities:
            entities[entity_type] = list(dict.fromkeys(entities[entity_type]))

        return entities, 'patterns'


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


class EntityThemeExtractionEngine:
    """Main engine for Phase 5: Entity/Theme Extraction."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize the entity/theme extraction engine."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.data_dir = Path(config['directories']['data'])

        # Initialize extractors
        self.entity_extractor = SpacyEntityExtractor(self.logger)
        self.theme_extractor = OllamaThemeExtractor(config, self.logger)

        # Create entity/theme data directory
        self.entity_theme_dir = self.data_dir / "entity_theme"
        self.entity_theme_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("🏷️  Phase 5: Entity/Theme Extraction Engine initialized")
        self._log_extractor_status()

    def _log_extractor_status(self):
        """Log the status of available extractors."""
        self.logger.info(
            f"   Entity extraction (spaCy): {'✅ Available' if self.entity_extractor.available else '⚠️  Fallback mode'}")
        self.logger.info(
            f"   Theme extraction (Ollama): {'✅ Available' if self.theme_extractor.available else '⚠️  Fallback mode'}")

    def extract_entities_and_themes(self, chunks: List[Dict[str, Any]],
                                    multi_granularity_embeddings: Dict[str, Dict[str, List[Any]]],
                                    articles: List[WikipediaArticle],
                                    force_recompute: bool = False) -> Dict[str, Any]:
        """
        Main extraction method for entities and themes across all granularity levels.

        Args:
            chunks: Chunk data from Phase 3
            multi_granularity_embeddings: Embedding data from Phase 3
            articles: Original Wikipedia articles
            force_recompute: Whether to force recomputation even if cached

        Returns:
            Dictionary containing all extracted entity/theme data
        """
        start_time = time.time()
        self.logger.info("🔍 Starting Phase 5: Entity/Theme Extraction")

        # Generate config hash for cache validation
        config_hash = self._generate_config_hash(chunks, multi_granularity_embeddings, articles)

        # Check cache
        cache_path = self._get_cache_path()
        if not force_recompute and self._is_cache_valid(cache_path, config_hash):
            self.logger.info("📂 Loading cached entity/theme data")
            return self._load_cached_data(cache_path)

        self.logger.info("⚡ Extracting fresh entity/theme data")

        # Extract data at each granularity level
        extraction_results = {}

        # 1. Extract chunk-level entities
        chunk_entities = self._extract_chunk_entities(chunks)
        extraction_results['chunk_entities'] = chunk_entities

        # 2. Extract sentence-level entities
        sentence_entities = self._extract_sentence_entities(multi_granularity_embeddings)
        extraction_results['sentence_entities'] = sentence_entities

        # 3. Extract document-level themes
        document_themes = self._extract_document_themes(multi_granularity_embeddings, articles)
        extraction_results['document_themes'] = document_themes

        processing_time = time.time() - start_time

        # Create metadata
        metadata = self._create_metadata(extraction_results, config_hash, processing_time)

        # Package final results
        entity_theme_data = {
            'metadata': metadata,
            'extraction_results': extraction_results
        }

        # Cache results
        self._cache_data(cache_path, entity_theme_data)

        # Log comprehensive results
        self._log_extraction_results(metadata)

        return entity_theme_data

    def _extract_chunk_entities(self, chunks: List[Dict[str, Any]]) -> List[EntityExtractionResult]:
        """Extract entities from all chunks."""
        self.logger.info(f"🔨 Extracting entities from {len(chunks)} chunks...")

        chunk_entities = []
        for chunk in chunks:
            result = self.entity_extractor.extract_entities(
                text=chunk['text'],
                source_id=chunk['chunk_id'],
                granularity_level='chunk'
            )
            chunk_entities.append(result)

        total_entities = sum(len(entities) for result in chunk_entities for entities in result.entities.values())
        self.logger.info(f"   ✅ Extracted {total_entities} entities from chunks")

        return chunk_entities

    def _extract_sentence_entities(self, multi_granularity_embeddings: Dict[str, Dict[str, List[Any]]]) -> List[
        EntityExtractionResult]:
        """Extract entities from all sentences."""
        # Get sentence embeddings from first model
        model_name = list(multi_granularity_embeddings.keys())[0]
        sentence_embeddings = multi_granularity_embeddings[model_name].get('sentences', [])

        self.logger.info(f"📝 Extracting entities from {len(sentence_embeddings)} sentences...")

        sentence_entities = []
        for sentence_emb in sentence_embeddings:
            result = self.entity_extractor.extract_entities(
                text=sentence_emb.sentence_text,
                source_id=sentence_emb.sentence_id,
                granularity_level='sentence'
            )
            sentence_entities.append(result)

        total_entities = sum(len(entities) for result in sentence_entities for entities in result.entities.values())
        self.logger.info(f"   ✅ Extracted {total_entities} entities from sentences")

        return sentence_entities

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
                         processing_time: float) -> EntityThemeExtractionMetadata:
        """Create metadata for the extraction results."""
        granularity_counts = {
            'chunks': len(extraction_results.get('chunk_entities', [])),
            'sentences': len(extraction_results.get('sentence_entities', [])),
            'documents': len(extraction_results.get('document_themes', []))
        }

        total_entities = sum(
            len(entities) for result_list in
            [extraction_results.get('chunk_entities', []), extraction_results.get('sentence_entities', [])]
            for result in result_list for entities in result.entities.values()
        )

        total_themes = sum(len(result.themes) for result in extraction_results.get('document_themes', []))

        return EntityThemeExtractionMetadata(
            created_at=datetime.now().isoformat(),
            granularity_counts=granularity_counts,
            entity_extraction_config={'entity_types': self.entity_extractor.entity_types},
            theme_extraction_config={'num_themes': self.theme_extractor.num_themes,
                                     'model': self.theme_extractor.model},
            total_entities_extracted=total_entities,
            total_themes_extracted=total_themes,
            processing_time=processing_time,
            config_hash=config_hash,
            ollama_available=self.theme_extractor.available,
            spacy_model_available=self.entity_extractor.available
        )

    def _generate_config_hash(self, chunks: List[Dict[str, Any]],
                              multi_granularity_embeddings: Dict[str, Dict[str, List[Any]]],
                              articles: List[WikipediaArticle]) -> str:
        """Generate hash of configuration and input data for cache validation."""
        config_str = json.dumps({
            'entity_types': self.entity_extractor.entity_types,
            'num_themes': self.theme_extractor.num_themes,
            'ollama_model': self.theme_extractor.model,
            'chunk_count': len(chunks),
            'granularity_counts': {
                model: {granularity: len(embeddings) for granularity, embeddings in granularity_embeddings.items()}
                for model, granularity_embeddings in multi_granularity_embeddings.items()
            },
            'article_count': len(articles),
            'first_chunk_id': chunks[0]['chunk_id'] if chunks else "",
            'first_article_title': articles[0].title if articles else ""
        }, sort_keys=True)

        return hashlib.md5(config_str.encode()).hexdigest()

    def _get_cache_path(self) -> Path:
        """Get cache path for entity/theme data."""
        return self.entity_theme_dir / "entity_theme_extraction.json"

    def _is_cache_valid(self, cache_path: Path, expected_hash: str) -> bool:
        """Check if cached entity/theme data is valid."""
        if not cache_path.exists():
            return False

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            metadata = cache_data.get('metadata', {})
            cached_hash = metadata.get('config_hash', '')

            return cached_hash == expected_hash

        except Exception as e:
            self.logger.warning(f"Failed to validate entity/theme cache: {e}")
            return False

    def _cache_data(self, cache_path: Path, entity_theme_data: Dict[str, Any]):
        """Cache entity/theme data to disk."""
        try:
            # Convert all results to dictionaries for JSON serialization
            serializable_data = {
                'metadata': asdict(entity_theme_data['metadata']),
                'extraction_results': {}
            }

            for result_type, results in entity_theme_data['extraction_results'].items():
                serializable_data['extraction_results'][result_type] = [
                    result.to_dict() for result in results
                ]

            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"💾 Cached entity/theme data to {cache_path}")

        except Exception as e:
            self.logger.error(f"Failed to cache entity/theme data: {e}")
            raise

    def _load_cached_data(self, cache_path: Path) -> Dict[str, Any]:
        """Load cached entity/theme data from disk."""
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # Reconstruct objects from dictionaries
            metadata = EntityThemeExtractionMetadata(**cache_data['metadata'])

            extraction_results = {}
            for result_type, result_data_list in cache_data['extraction_results'].items():
                if result_type == 'document_themes':
                    extraction_results[result_type] = [
                        ThemeExtractionResult(**result_data) for result_data in result_data_list
                    ]
                else:  # chunk_entities and sentence_entities
                    extraction_results[result_type] = [
                        EntityExtractionResult(**result_data) for result_data in result_data_list
                    ]

            return {
                'metadata': metadata,
                'extraction_results': extraction_results
            }

        except Exception as e:
            self.logger.error(f"Failed to load cached entity/theme data: {e}")
            raise

    def _log_extraction_results(self, metadata: EntityThemeExtractionMetadata):
        """Log comprehensive extraction results."""
        self.logger.info("📊 Phase 5 Entity/Theme Extraction Results:")
        self.logger.info(f"   Total entities extracted: {metadata.total_entities_extracted:,}")
        self.logger.info(f"   Total themes extracted: {metadata.total_themes_extracted:,}")
        self.logger.info(f"   Processing time: {metadata.processing_time:.2f}s")

        self.logger.info(f"   Granularity breakdown:")
        for granularity_level, count in metadata.granularity_counts.items():
            self.logger.info(f"      {granularity_level}: {count:,} items processed")

        self.logger.info(f"   Extraction methods:")
        self.logger.info(
            f"      Entity extraction: {'spaCy' if metadata.spacy_model_available else 'Pattern-based fallback'}")
        self.logger.info(
            f"      Theme extraction: {'Ollama' if metadata.ollama_available else 'Keyword-based fallback'}")

    def get_extraction_statistics(self, entity_theme_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics about the extraction results."""
        metadata = entity_theme_data['metadata']
        extraction_results = entity_theme_data['extraction_results']

        # Calculate detailed statistics
        stats = {
            'total_entities': metadata.total_entities_extracted,
            'total_themes': metadata.total_themes_extracted,
            'processing_time': metadata.processing_time,
            'granularity_counts': metadata.granularity_counts,
            'extraction_methods': {
                'entity': 'spacy' if metadata.spacy_model_available else 'patterns',
                'theme': 'ollama' if metadata.ollama_available else 'fallback'
            }
        }

        # Entity type breakdown
        entity_type_counts = {'PERSON': 0, 'ORG': 0, 'GPE': 0}
        for result_type in ['chunk_entities', 'sentence_entities']:
            for result in extraction_results.get(result_type, []):
                for entity_type, entity_list in result.entities.items():
                    entity_type_counts[entity_type] += len(entity_list)

        stats['entity_type_breakdown'] = entity_type_counts

        # Theme statistics
        if extraction_results.get('document_themes'):
            theme_lengths = [len(result.themes) for result in extraction_results['document_themes']]
            stats['theme_statistics'] = {
                'avg_themes_per_document': sum(theme_lengths) / len(theme_lengths) if theme_lengths else 0,
                'max_themes_per_document': max(theme_lengths) if theme_lengths else 0,
                'min_themes_per_document': min(theme_lengths) if theme_lengths else 0
            }

        return stats