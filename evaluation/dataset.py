#!/usr/bin/env python3
"""
Synthetic Dataset Generation System
==================================

Generates sophisticated evaluation datasets from knowledge graph chunks using
deepeval's evolution techniques. Creates challenging questions that test semantic
traversal capabilities through progressive complexity enhancement.

Key Features:
- Knowledge graph chunk processing and context preparation
- Evolution-based question complexity enhancement
- Multi-granularity context grouping for optimal question generation
- Comprehensive dataset statistics and quality validation
- Caching and versioning for reproducible benchmarks
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# DeepEval imports for synthetic generation
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import EvolutionConfig, Evolution
from deepeval.dataset import EvaluationDataset

# Local imports
from utils.knowledge_graph import KnowledgeGraph
from .models import ModelManager


class DatasetBuilder:
    """
    Generates synthetic evaluation datasets from knowledge graph chunks.
    
    Uses deepeval's sophisticated evolution system to create questions that require
    semantic navigation across knowledge graph connections - the perfect testbed
    for evaluating semantic traversal algorithm superiority.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize DatasetBuilder with configuration and logging.
        
        Args:
            config: Complete system configuration dictionary
            logger: Optional logger instance (creates default if None)
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.deepeval_config = config.get('deepeval', {})
        
        # Initialize model manager for question generation
        self.model_manager = ModelManager(config, logger)
        
        # Dataset generation statistics
        self.generation_stats = {
            'start_time': None,
            'end_time': None,
            'total_contexts': 0,
            'total_questions': 0,
            'evolution_distribution': {},
            'quality_scores': []
        }
        
        self.logger.info("DatasetBuilder initialized")
    
    def build(self, knowledge_graph: Optional[KnowledgeGraph] = None, 
              force_regenerate: bool = False) -> EvaluationDataset:
        """
        Main entry point for synthetic dataset generation.
        
        Creates sophisticated questions from knowledge graph chunks using evolution
        techniques specifically designed to challenge semantic traversal algorithms.
        
        Args:
            knowledge_graph: Pre-loaded KG instance (loads from file if None)
            force_regenerate: Force regeneration even if cached dataset exists
            
        Returns:
            EvaluationDataset: Ready-to-use dataset with evolved questions
            
        Raises:
            FileNotFoundError: If knowledge graph file not found
            RuntimeError: If dataset generation fails
        """
        self.generation_stats['start_time'] = time.time()
        self.logger.info("🎯 Starting synthetic dataset generation using deepeval evolution")
        
        # Load knowledge graph if not provided
        if knowledge_graph is None:
            knowledge_graph = self._load_knowledge_graph()
        
        # Check for cached dataset
        dataset_config = self.deepeval_config.get('dataset', {})
        save_path = Path(dataset_config.get('output', {}).get('save_path', 'data/synthetic_dataset.json'))
        
        if not force_regenerate and save_path.exists():
            self.logger.info(f"📂 Found cached dataset at {save_path}")
            try:
                # Try to load using add_goldens_from_json_file method
                dataset = EvaluationDataset()
                dataset.add_goldens_from_json_file(str(save_path))
                self.logger.info(f"✅ Loaded {len(dataset.goldens)} cached questions")
                return dataset
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load cached dataset: {e}. Generating new dataset.")
        
        # Generate new dataset
        self.logger.info("🧠 Generating new synthetic dataset with evolution techniques")
        
        # Prepare contexts from knowledge graph chunks
        contexts = self._prepare_contexts_from_kg(knowledge_graph)
        self.generation_stats['total_contexts'] = len(contexts)
        
        # Configure evolution strategies for semantic traversal testing
        evolution_config = self._create_evolution_config()
        
        # Initialize synthesizer with question generation model
        synthesizer = self._create_synthesizer(evolution_config)
        
        # Generate goldens with evolution applied
        goldens = self._generate_goldens_with_evolution(synthesizer, contexts)
        self.generation_stats['total_questions'] = len(goldens)
        
        # Create evaluation dataset
        dataset = EvaluationDataset(goldens=goldens)
        
        # Save dataset with metadata
        if dataset_config.get('output', {}).get('cache_enabled', True):
            self._save_dataset_with_metadata(dataset, save_path)
        
        # Log comprehensive generation statistics
        self.generation_stats['end_time'] = time.time()
        self._log_generation_statistics()
        
        return dataset
    
    def _load_knowledge_graph(self) -> KnowledgeGraph:
        """Load knowledge graph from standard data directory location."""
        kg_path = Path(self.config['directories']['data']) / "knowledge_graph.json"
        
        if not kg_path.exists():
            raise FileNotFoundError(
                f"Knowledge graph not found at {kg_path}. "
                "Please run kg_pipeline.build() first to generate the knowledge graph."
            )
        
        self.logger.info(f"📂 Loading knowledge graph from {kg_path}")
        
        # Load embeddings if available for enhanced context preparation
        try:
            kg = KnowledgeGraph.load(str(kg_path))
            self.logger.info(
                f"✅ Knowledge graph loaded: {len(kg.chunks)} chunks, "
                f"{len(kg.sentences)} sentences, {len(kg.documents)} documents"
            )
            return kg
        except Exception as e:
            raise RuntimeError(f"Failed to load knowledge graph: {e}")
    
    def _prepare_contexts_from_kg(self, kg: KnowledgeGraph) -> List[List[str]]:
        """
        Convert knowledge graph chunks into optimized context groups for synthesis.
        
        Groups chunks strategically to enable evolution techniques that require
        semantic navigation across document boundaries - crucial for testing
        triangulation algorithm superiority.
        
        Args:
            kg: Knowledge graph instance with chunks and documents
            
        Returns:
            List of context groups, each containing related chunk texts
        """
        self.logger.info("📋 Preparing contexts from knowledge graph for evolution-based generation")
        
        contexts = []
        
        # Group chunks by document for document-aware evolution
        chunks_by_doc = self._group_chunks_by_document(kg)
        
        # Create intra-document context groups (for MULTICONTEXT evolution)
        intra_doc_contexts = self._create_intra_document_contexts(chunks_by_doc)
        contexts.extend(intra_doc_contexts)
        
        # Create cross-document context groups (for COMPARATIVE and IN_BREADTH evolution)
        cross_doc_contexts = self._create_cross_document_contexts(kg, chunks_by_doc)
        contexts.extend(cross_doc_contexts)
        
        # Create theme-based context groups if themes are available
        if hasattr(kg, 'chunks') and any(hasattr(chunk, 'inherited_themes') for chunk in kg.chunks.values()):
            theme_contexts = self._create_theme_based_contexts(kg)
            contexts.extend(theme_contexts)
        
        self.logger.info(
            f"📊 Prepared {len(contexts)} context groups: "
            f"{len(intra_doc_contexts)} intra-document, "
            f"{len(cross_doc_contexts)} cross-document"
        )
        
        return contexts
    
    def _group_chunks_by_document(self, kg: KnowledgeGraph) -> Dict[str, List[str]]:
        """Group chunk texts by their source document."""
        chunks_by_doc = {}
        
        for chunk_id, chunk in kg.chunks.items():
            doc_id = chunk.source_document
            if doc_id not in chunks_by_doc:
                chunks_by_doc[doc_id] = []
            chunks_by_doc[doc_id].append(chunk.chunk_text)
        
        return chunks_by_doc
    
    def _create_intra_document_contexts(self, chunks_by_doc: Dict[str, List[str]]) -> List[List[str]]:
        """
        Create context groups within individual documents.
        
        Optimized for MULTICONTEXT and REASONING evolution types that benefit
        from related chunks within the same conceptual domain.
        """
        contexts = []
        max_chunks_per_context = 4  # Optimal for focused question generation
        
        for doc_id, chunk_texts in chunks_by_doc.items():
            # Split large documents into manageable context groups
            for i in range(0, len(chunk_texts), max_chunks_per_context):
                context_group = chunk_texts[i:i + max_chunks_per_context]
                if len(context_group) >= 2:  # Need at least 2 chunks for meaningful contexts
                    contexts.append(context_group)
        
        return contexts
    
    def _create_cross_document_contexts(self, kg: KnowledgeGraph, 
                                       chunks_by_doc: Dict[str, List[str]]) -> List[List[str]]:
        """
        Create context groups spanning multiple documents.
        
        Essential for COMPARATIVE and IN_BREADTH evolution types that create
        questions requiring semantic navigation across document boundaries.
        This is where triangulation algorithms will demonstrate superiority.
        """
        contexts = []
        doc_ids = list(chunks_by_doc.keys())
        
        # Create pairs and triplets of documents for cross-document questions
        for i in range(len(doc_ids)):
            for j in range(i + 1, min(i + 3, len(doc_ids))):  # Limit to 2-3 documents per context
                doc1_chunks = chunks_by_doc[doc_ids[i]][:2]  # Take top 2 chunks from each doc
                doc2_chunks = chunks_by_doc[doc_ids[j]][:2]
                
                # Combine chunks from different documents
                cross_doc_context = doc1_chunks + doc2_chunks
                contexts.append(cross_doc_context)
        
        return contexts
    
    def _create_theme_based_contexts(self, kg: KnowledgeGraph) -> List[List[str]]:
        """
        Create context groups based on semantic themes.
        
        Leverages knowledge graph theme information to create contexts that
        enable sophisticated thematic questions requiring semantic reasoning.
        """
        contexts = []
        theme_to_chunks = {}
        
        # Group chunks by their inherited themes
        for chunk_id, chunk in kg.chunks.items():
            if hasattr(chunk, 'inherited_themes') and chunk.inherited_themes:
                for theme in chunk.inherited_themes[:3]:  # Use top 3 themes
                    if theme not in theme_to_chunks:
                        theme_to_chunks[theme] = []
                    theme_to_chunks[theme].append(chunk.chunk_text)
        
        # Create contexts from theme-related chunks
        for theme, chunk_texts in theme_to_chunks.items():
            if len(chunk_texts) >= 2:  # Need multiple chunks for theme-based questions
                # Limit context size for focused questions
                contexts.append(chunk_texts[:4])
        
        return contexts
    
    def _create_evolution_config(self) -> EvolutionConfig:
        """
        Create evolution configuration optimized for semantic traversal testing.
        
        Configures evolution strategies to create questions that require the kind
        of semantic navigation and reasoning that triangulation algorithms excel at.
        """
        evolution_settings = self.deepeval_config.get('dataset', {}).get('evolution', {})
        
        if not evolution_settings.get('enabled', True):
            self.logger.info("Evolution disabled - using basic question generation")
            return EvolutionConfig(num_evolutions=0)
        
        # Map configuration strings to Evolution enum values
        evolution_type_mapping = {
            "REASONING": Evolution.REASONING,        # Multi-step logical reasoning
            "COMPARATIVE": Evolution.COMPARATIVE,    # Cross-document comparisons
            "IN_BREADTH": Evolution.IN_BREADTH,      # Broader scope questions
            "MULTICONTEXT": Evolution.MULTICONTEXT,  # Multi-chunk integration
            "CONCRETIZING": Evolution.CONCRETIZING,  # Specific detailed questions
            "CONSTRAINED": Evolution.CONSTRAINED,    # Constraint-based questions
            "HYPOTHETICAL": Evolution.HYPOTHETICAL   # What-if scenarios
        }
        
        # Get configured evolution types and their weights
        configured_types = evolution_settings.get('evolution_types', ['REASONING', 'COMPARATIVE'])
        evolution_distribution = evolution_settings.get('evolution_distribution', {})

        # Convert string keys to Evolution enum keys with weights
        evolutions_dict = {}

        # If distribution is provided, use it
        if evolution_distribution:
            for evo_type_str, weight in evolution_distribution.items():
                if evo_type_str in evolution_type_mapping:
                    evolutions_dict[evolution_type_mapping[evo_type_str]] = float(weight)
        else:
            # If no distribution provided, use equal weights for configured types
            weight_per_type = 1.0 / len(configured_types) if configured_types else 1.0
            for evo_type_str in configured_types:
                if evo_type_str in evolution_type_mapping:
                    evolutions_dict[evolution_type_mapping[evo_type_str]] = weight_per_type

        # Use default if no valid types found
        if not evolutions_dict:
            self.logger.warning("No valid evolution types configured - using default REASONING")
            evolutions_dict = {Evolution.REASONING: 1.0}

        config = EvolutionConfig(
            num_evolutions=evolution_settings.get('num_evolutions', 2),
            evolutions=evolutions_dict
        )
        
        self.logger.info(
            f"🧬 Evolution configured: {len(evolutions_dict)} types, "
            f"{evolution_settings.get('num_evolutions', 2)} steps per question"
        )
        
        return config
    
    def _create_synthesizer(self, evolution_config: EvolutionConfig) -> Synthesizer:
        """
        Create synthesizer instance with proper model configuration.
        
        Uses the configured question generation model with evolution settings
        optimized for creating challenging semantic traversal test cases.
        """
        # Get question generation model from model manager
        generation_model = self.model_manager.get_question_generation_model()
        
        self.logger.info(
            f"🔧 Creating synthesizer with model: {generation_model.get_model_name()}"
        )
        
        return Synthesizer(
            model=generation_model,
            evolution_config=evolution_config
        )
    
    def _generate_goldens_with_evolution(self, synthesizer: Synthesizer, 
                                        contexts: List[List[str]]) -> List:
        """
        Generate goldens using evolution techniques with comprehensive tracking.
        
        Creates sophisticated questions through iterative evolution, tracking
        the process for quality analysis and debugging.
        """
        generation_config = self.deepeval_config.get('dataset', {}).get('generation', {})
        
        num_goldens = generation_config.get('num_goldens', 100)
        max_goldens_per_context = generation_config.get('max_goldens_per_context', 3)
        include_expected_output = generation_config.get('include_expected_output', True)
        
        # Calculate how many contexts we actually need to generate the requested questions
        contexts_needed = min(len(contexts), num_goldens // max_goldens_per_context + 1)
        if contexts_needed > num_goldens:
            contexts_needed = num_goldens

        # Limit contexts to only what we need
        limited_contexts = contexts[:contexts_needed]

        # Deduplicate and truncate contexts to ensure manageable size
        deduplicated_contexts = []
        max_context_chars = 8000  # Limit each context to 8k characters max

        for context_group in limited_contexts:
            # First, extract all sentences from all chunks in this context group
            all_sentences = []
            for chunk_text in context_group:
                # Split chunk into sentences (basic sentence splitting)
                sentences = [s.strip() for s in chunk_text.split('.') if s.strip()]
                all_sentences.extend(sentences)

            # Deduplicate sentences while preserving order
            unique_sentences = list(dict.fromkeys(all_sentences))

            # Now reconstruct context with unique sentences only
            # Group sentences back into reasonable chunks for readability
            chunk_size = 3  # Group 3 sentences per "chunk"
            reconstructed_chunks = []

            for i in range(0, len(unique_sentences), chunk_size):
                sentence_group = unique_sentences[i:i + chunk_size]
                reconstructed_chunk = '. '.join(sentence_group) + '.'
                reconstructed_chunks.append(reconstructed_chunk)

            # Apply character limit to reconstructed chunks
            truncated_group = []
            total_chars = 0

            for text in reconstructed_chunks:
                if total_chars + len(text) <= max_context_chars:
                    truncated_group.append(text)
                    total_chars += len(text)
                else:
                    # Add truncated version if space allows
                    remaining_chars = max_context_chars - total_chars
                    if remaining_chars > 200:  # Only add if meaningful amount of space left
                        truncated_text = text[:remaining_chars-3] + "..."
                        truncated_group.append(truncated_text)
                    break

            deduplicated_contexts.append(truncated_group)

            # Log deduplication and truncation stats for debugging
            original_sentences = len(all_sentences)
            unique_sentence_count = len(unique_sentences)
            if unique_sentence_count < original_sentences:
                self.logger.debug(
                    f"🧹 Sentence deduplication: {original_sentences} → {unique_sentence_count} sentences, "
                    f"final context: {len(truncated_group)} chunks, {sum(len(t) for t in truncated_group)} chars"
                )

        self.logger.info(
            f"🎯 Generating {num_goldens} questions from {contexts_needed} contexts "
            f"(max {max_goldens_per_context} per context, {len(contexts)} total available)"
        )

        try:
            # Add detailed logging for generation process
            self.logger.info(f"🔍 Starting generation with {contexts_needed} selected contexts (from {len(contexts)} available)")
            self.logger.info(f"🔍 Will generate up to {num_goldens} questions")
            self.logger.info(f"🔍 Generation parameters: include_expected_output={include_expected_output}, max_per_context={max_goldens_per_context}")

            # Generate goldens using deepeval synthesizer with DEDUPLICATED contexts
            goldens = synthesizer.generate_goldens_from_contexts(
                contexts=deduplicated_contexts,
                include_expected_output=include_expected_output,
                max_goldens_per_context=max_goldens_per_context
            )

            self.logger.info(f"🎯 Generation completed, produced {len(goldens)} goldens")
            
            # Track evolution statistics from generated goldens
            self._track_evolution_statistics(goldens)
            
            # Limit to requested number of questions
            if len(goldens) > num_goldens:
                goldens = goldens[:num_goldens]
                self.logger.info(f"📏 Trimmed to requested {num_goldens} questions")
            
            return goldens
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate synthetic dataset: {e}")
    
    def _track_evolution_statistics(self, goldens: List) -> None:
        """
        Track evolution statistics from generated goldens for quality analysis.
        
        Analyzes the evolution sequences applied to understand question complexity
        and distribution for debugging and optimization purposes.
        """
        evolution_counts = {}
        quality_scores = []
        
        for golden in goldens:
            # Track evolution sequences if available in metadata
            if hasattr(golden, 'additional_metadata') and golden.additional_metadata:
                evolution_sequence = golden.additional_metadata.get('evolution_sequence', [])
                for evolution in evolution_sequence:
                    evolution_name = evolution.value if hasattr(evolution, 'value') else str(evolution)
                    evolution_counts[evolution_name] = evolution_counts.get(evolution_name, 0) + 1
                
                # Track quality scores if available
                quality_score = golden.additional_metadata.get('quality_score')
                if quality_score is not None:
                    quality_scores.append(quality_score)
        
        self.generation_stats['evolution_distribution'] = evolution_counts
        self.generation_stats['quality_scores'] = quality_scores
    
    def _save_dataset_with_metadata(self, dataset: EvaluationDataset, save_path: Path) -> None:
        """
        Save dataset with comprehensive metadata for reproducibility and analysis.
        
        Includes generation statistics, configuration snapshots, and quality metrics
        for thorough documentation of the synthetic dataset creation process.
        """
        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the dataset using save_as method (requires file_type and directory)
        # Remove .json extension since save_as will add it
        file_name = save_path.stem  # Gets filename without extension
        directory = str(save_path.parent)
        dataset.save_as(file_type='json', directory=directory, file_name=file_name)
        
        # Save metadata separately for analysis
        metadata_path = save_path.with_suffix('.metadata.json')
        metadata = {
            'generation_timestamp': datetime.now().isoformat(),
            'dataset_path': str(save_path),
            'statistics': self.generation_stats,
            'configuration_snapshot': {
                'deepeval_config': self.deepeval_config,
                'model_info': {
                    'question_generation': self.model_manager.get_model_info('question_generation'),
                }
            },
            'quality_metrics': {
                'total_questions': len(dataset.goldens),
                'average_question_length': self._calculate_average_question_length(dataset.goldens),
                'evolution_coverage': len(self.generation_stats.get('evolution_distribution', {}))
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"💾 Dataset and metadata saved: {save_path}")
    
    def _calculate_average_question_length(self, goldens: List) -> float:
        """Calculate average question length for quality metrics."""
        if not goldens:
            return 0.0
        
        total_length = sum(len(golden.input) for golden in goldens)
        return total_length / len(goldens)
    
    def _log_generation_statistics(self) -> None:
        """Log comprehensive dataset generation statistics."""
        duration = self.generation_stats['end_time'] - self.generation_stats['start_time']
        
        self.logger.info("📊 Dataset Generation Complete - Statistics:")
        self.logger.info(f"   Generation time: {duration:.2f} seconds")
        self.logger.info(f"   Total contexts processed: {self.generation_stats['total_contexts']}")
        self.logger.info(f"   Total questions generated: {self.generation_stats['total_questions']}")
        
        # Log evolution distribution
        evolution_dist = self.generation_stats.get('evolution_distribution', {})
        if evolution_dist:
            self.logger.info("   Evolution technique distribution:")
            for evolution_type, count in evolution_dist.items():
                percentage = (count / self.generation_stats['total_questions']) * 100
                self.logger.info(f"      {evolution_type}: {count} ({percentage:.1f}%)")
        
        # Log quality scores if available
        quality_scores = self.generation_stats.get('quality_scores', [])
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            self.logger.info(f"   Average quality score: {avg_quality:.3f}")
        
        self.logger.info("✅ Synthetic dataset ready for semantic traversal algorithm evaluation")
