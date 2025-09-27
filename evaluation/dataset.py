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
from deepeval.synthesizer.config import EvolutionConfig, Evolution, FiltrationConfig
from deepeval.dataset import EvaluationDataset

# Local imports
from utils.knowledge_graph import KnowledgeGraph
from .models import ModelManager
from .context_grouping import ContextGroupingOrchestrator


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
        
        # Store golden metadata separately since Golden objects are immutable
        self.golden_metadata_store = []
        
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
        
        # Check dataset loading configuration
        dataset_config = self.deepeval_config.get('dataset', {})
        output_config = dataset_config.get('output', {})
        save_path = Path(output_config.get('save_path', 'data/synthetic_dataset.json'))
        pull_from_dashboard = output_config.get('pull_from_dashboard', False)
        dataset_alias = output_config.get('dataset_alias', 'semantic-rag-benchmark')
        
        # Conditional loading: dashboard vs local file
        if pull_from_dashboard:
            self.logger.info(f"🌐 Loading dataset from DeepEval dashboard with alias '{dataset_alias}'")
            try:
                dataset = EvaluationDataset()
                dataset.pull(alias=dataset_alias)
                self.logger.info(f"✅ Loaded {len(dataset.goldens)} questions from dashboard")
                self.logger.info(f"   Dataset alias: {dataset_alias}")
                self.logger.info(f"   ✨ Evaluation runs will properly link to this uploaded dataset")
                return dataset
            except Exception as e:
                self.logger.error(f"❌ Failed to load dataset from dashboard: {e}")
                self.logger.info(f"   Verify dataset '{dataset_alias}' exists on dashboard")
                self.logger.info(f"   Falling back to local dataset generation...")
        
        elif not force_regenerate and save_path.exists():
            self.logger.info(f"📂 Found cached local dataset at {save_path}")
            try:
                # Load from local JSON file
                dataset = EvaluationDataset()
                dataset.add_goldens_from_json_file(str(save_path))
                self.logger.info(f"✅ Loaded {len(dataset.goldens)} cached questions from local file")
                self.logger.info(f"   💡 Set 'pull_from_dashboard: true' to use uploaded dataset instead")

                # Push to DeepEval dashboard if configured (even for cached datasets)
                if output_config.get('push_to_dashboard', False):
                    self._push_dataset_to_dashboard(dataset, dataset_alias)

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

        # Push to DeepEval dashboard if configured
        output_config = dataset_config.get('output', {})
        if output_config.get('push_to_dashboard', False):
            dataset_alias = output_config.get('dataset_alias', 'semantic-rag-benchmark')
            self._push_dataset_to_dashboard(dataset, dataset_alias)

        # Generate CSV for manual upload if configured
        if output_config.get('generate_csv', False):
            self._generate_csv_for_upload(save_path, output_config)

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
        Create context groups using sophisticated semantic traversal strategies.
        
        Replaces the previous basic positional grouping with advanced strategies that
        mirror retrieval algorithm capabilities, creating self-validating benchmarks.
        
        Args:
            kg: Knowledge graph instance with chunks, documents, and similarities
            
        Returns:
            List of context groups (each containing related chunk texts)
        """
        self.logger.info("📋 Preparing contexts using advanced semantic traversal strategies")
        
        # Initialize context grouping orchestrator
        context_orchestrator = ContextGroupingOrchestrator(
            kg=kg,
            config=self.config,
            logger=self.logger
        )
        
        # Determine number of context groups to generate
        generation_config = self.deepeval_config.get('dataset', {}).get('generation', {})
        num_goldens = generation_config.get('num_goldens', 100)
        max_goldens_per_context = generation_config.get('max_goldens_per_context', 3)
        
        # Calculate desired number of context groups
        # Only generate what's actually needed for efficiency
        base_contexts_needed = min(len(kg.chunks), num_goldens // max_goldens_per_context + 1)
        total_context_groups = base_contexts_needed  # Generate exactly what's needed
        
        # Generate context groups using all enabled strategies
        context_groups = context_orchestrator.generate_context_groups(total_context_groups)
        
        # Convert ContextGroup objects to List[List[str]] format expected by downstream code
        contexts = []
        strategy_distribution = {}
        context_metadata = []  # Store metadata for each context group
        
        for context_group in context_groups:
            # Add chunk texts as context group
            contexts.append(context_group.chunks)
            
            # Store metadata for later association with generated questions
            context_metadata.append({
                'strategy': context_group.strategy,
                'traversal_path': context_group.traversal_path,
                'metadata': context_group.metadata,
                'chunk_count': len(context_group.chunks),
                'sentence_count': len(context_group.sentences)
            })
            
            # Track strategy usage for logging
            strategy = context_group.strategy
            strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1
        
        # Log comprehensive statistics
        strategy_stats = context_orchestrator.get_strategy_statistics()
        
        self.logger.info(f"📊 Generated {len(contexts)} context groups using semantic traversal strategies")
        self.logger.info(f"   Strategy distribution: {strategy_distribution}")
        self.logger.info(f"   Enabled strategies: {strategy_stats['enabled_strategies']}")
        
        # Store context group metadata for analysis
        self.generation_stats['context_strategy_distribution'] = strategy_distribution
        self.generation_stats['strategy_configuration'] = strategy_stats
        self.generation_stats['context_metadata'] = context_metadata  # Store for golden association
        
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
        Optionally includes FiltrationConfig for quality filtering.
        """
        # Get question generation model from model manager
        generation_model = self.model_manager.get_question_generation_model()
        
        # Check if quality filtration is enabled
        filtration_settings = self.deepeval_config.get('dataset', {}).get('filtration', {})
        filtration_config = None
        
        if filtration_settings.get('enabled', False):
            from deepeval.synthesizer.config import FiltrationConfig
            
            critic_model = filtration_settings.get('critic_model', 'gpt-4o')
            quality_threshold = filtration_settings.get('synthetic_input_quality_threshold', 0.7)
            max_retries = filtration_settings.get('max_quality_retries', 5)
            
            filtration_config = FiltrationConfig(
                critic_model=critic_model,
                synthetic_input_quality_threshold=quality_threshold,
                max_quality_retries=max_retries
            )
            
            self.logger.info(
                f"🎯 Quality filtering enabled: threshold={quality_threshold}, "
                f"max_retries={max_retries}, critic_model={critic_model}"
            )
        
        self.logger.info(
            f"🔧 Creating synthesizer with model: {generation_model.get_model_name()}"
        )
        
        return Synthesizer(
            model=generation_model,
            evolution_config=evolution_config,
            filtration_config=filtration_config  # None if disabled
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

        # Randomly select contexts for diversity (instead of always taking first ones)
        import random
        context_index_mapping = []  # Track original indices for metadata alignment

        if contexts_needed < len(contexts):
            # Create list of (context, original_index) pairs
            indexed_contexts = [(contexts[i], i) for i in range(len(contexts))]
            # Randomly sample the pairs
            selected_pairs = random.sample(indexed_contexts, contexts_needed)
            # Split back into contexts and index mapping
            limited_contexts = [pair[0] for pair in selected_pairs]
            context_index_mapping = [pair[1] for pair in selected_pairs]
            self.logger.info(f"🎲 Randomly selected {contexts_needed} contexts from {len(contexts)} available")
        else:
            limited_contexts = contexts
            context_index_mapping = list(range(len(contexts)))

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
            
            # Associate each golden with its source context metadata
            self._associate_goldens_with_context_metadata(goldens, limited_contexts, context_index_mapping)
            
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
    
    def _associate_goldens_with_context_metadata(self, goldens: List, limited_contexts: List[List[str]], context_index_mapping: List[int]) -> None:
        """
        Associate each generated golden with metadata from its source context group.

        Stores metadata separately since Golden objects are immutable dataclasses.
        Metadata is retrieved during JSON serialization using index-based association.

        Args:
            goldens: List of generated golden objects from DeepEval
            limited_contexts: List of context groups that were actually used
            context_index_mapping: List mapping limited_contexts indices to original context indices
        """
        try:
            # Clear existing metadata store
            self.golden_metadata_store = []
            
            # Get context metadata from generation stats
            context_metadata = self.generation_stats.get('context_metadata', [])
            
            # Associate each golden with its source context metadata
            for i, golden in enumerate(goldens):
                if i < len(limited_contexts) and i < len(context_index_mapping):
                    # Map back to original context index to get correct metadata
                    original_context_index = context_index_mapping[i]
                    if original_context_index < len(context_metadata):
                        # Get metadata for this context using correct original index
                        source_context_metadata = context_metadata[original_context_index]

                        # Extract evolution strategies applied to this golden
                        applied_evolutions = self._extract_evolution_strategies(golden)

                        # Create comprehensive metadata for this golden
                        golden_metadata = {
                            'grouping_method': source_context_metadata['strategy'],
                            'evolutions': applied_evolutions,
                            'context_traversal_path': source_context_metadata['traversal_path'],
                            'context_chunk_count': source_context_metadata['chunk_count'],
                            'context_sentence_count': source_context_metadata['sentence_count'],
                            'context_strategy_metadata': source_context_metadata['metadata']
                        }

                        # Store metadata in separate store using index association
                        self.golden_metadata_store.append(golden_metadata)

                        self.logger.debug(f"📝 Stored metadata for golden {i} with strategy '{source_context_metadata['strategy']}' (original_idx: {original_context_index})")
                    else:
                        # Handle case where original context index is out of range
                        self.logger.warning(f"⚠️ Original context index {original_context_index} out of range for golden {i}")
                        fallback_metadata = {
                            'grouping_method': 'index_error',
                            'evolutions': ['unknown'],
                            'context_traversal_path': [],
                            'context_chunk_count': 0,
                            'context_sentence_count': 0,
                            'context_strategy_metadata': {}
                        }
                        self.golden_metadata_store.append(fallback_metadata)
                else:
                    # Store fallback metadata for missing associations
                    fallback_metadata = {
                        'grouping_method': 'unknown',
                        'evolutions': ['unknown'],
                        'context_traversal_path': [],
                        'context_chunk_count': 0,
                        'context_sentence_count': 0,
                        'context_strategy_metadata': {}
                    }
                    self.golden_metadata_store.append(fallback_metadata)
                    self.logger.warning(f"⚠️ Created fallback metadata for golden {i} (metadata length: {len(context_metadata)})")
            
            self.logger.info(f"💾 Stored metadata for {len(self.golden_metadata_store)} goldens")
                    
        except Exception as e:
            self.logger.error(f"❌ Error storing golden metadata: {e}")
            # Create fallback metadata for all goldens
            self.golden_metadata_store = []
            for i in range(len(goldens)):
                self.golden_metadata_store.append({
                    'grouping_method': 'error',
                    'evolutions': ['error'],
                    'context_traversal_path': [],
                    'context_chunk_count': 0,
                    'context_sentence_count': 0,
                    'context_strategy_metadata': {}
                })
    
    def _extract_evolution_strategies(self, golden) -> List[str]:
        """
        Extract evolution strategies applied to a golden from its metadata.
        
        Args:
            golden: Generated golden object from DeepEval
            
        Returns:
            List of evolution strategy names applied to this golden
        """
        try:
            applied_evolutions = []
            
            # Check if golden has evolution metadata
            if hasattr(golden, 'additional_metadata') and golden.additional_metadata:
                evolution_sequence = golden.additional_metadata.get('evolution_sequence', [])
                
                for evolution in evolution_sequence:
                    # Handle both enum and string representations
                    if hasattr(evolution, 'value'):
                        evolution_name = evolution.value
                    elif hasattr(evolution, 'name'):
                        evolution_name = evolution.name
                    else:
                        evolution_name = str(evolution)
                    
                    applied_evolutions.append(evolution_name)
            
            # If no evolution metadata found, try to infer from configured distributions
            if not applied_evolutions:
                evolution_config = self.deepeval_config.get('dataset', {}).get('evolution', {})
                if evolution_config.get('enabled', True):
                    # Return configured evolution types as fallback
                    applied_evolutions = evolution_config.get('evolution_types', ['UNKNOWN'])
            
            return applied_evolutions
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not extract evolution strategies: {e}")
            return ['UNKNOWN']
    
    def _save_dataset_with_metadata(self, dataset: EvaluationDataset, save_path: Path) -> None:
        """
        Save dataset with comprehensive metadata for reproducibility and analysis.
        
        Creates custom JSON format that includes context grouping strategy and evolution
        metadata for each generated question, enabling detailed analysis of dataset generation.
        
        Args:
            dataset: The EvaluationDataset containing generated goldens
            save_path: Path where to save the enhanced dataset JSON
        """
        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create custom dataset structure with enhanced metadata
            enhanced_dataset = []
            
            for i, golden in enumerate(dataset.goldens):
                # Create base golden structure
                golden_dict = {
                    'input': golden.input,
                    'expected_output': golden.expected_output if hasattr(golden, 'expected_output') else None,
                    'actual_output': golden.actual_output if hasattr(golden, 'actual_output') else None,
                    'context': golden.context if hasattr(golden, 'context') else None,
                    'retrieval_context': golden.retrieval_context if hasattr(golden, 'retrieval_context') else None
                }
                
                # Add enhanced metadata at the bottom using index-based lookup
                if i < len(self.golden_metadata_store):
                    stored_metadata = self.golden_metadata_store[i]
                    golden_dict['grouping_method'] = stored_metadata.get('grouping_method', 'unknown')
                    golden_dict['evolutions'] = stored_metadata.get('evolutions', ['unknown'])
                    
                    # Optional: Add detailed context metadata for debugging
                    if self.logger.level <= logging.DEBUG:
                        golden_dict['context_metadata'] = {
                            'traversal_path': stored_metadata.get('context_traversal_path', []),
                            'chunk_count': stored_metadata.get('context_chunk_count', 0),
                            'sentence_count': stored_metadata.get('context_sentence_count', 0),
                            'strategy_metadata': stored_metadata.get('context_strategy_metadata', {})
                        }
                else:
                    # Fallback metadata if no stored metadata found
                    golden_dict['grouping_method'] = 'missing'
                    golden_dict['evolutions'] = ['missing']
                    self.logger.warning(f"⚠️ No stored metadata found for golden {i}")
                
                enhanced_dataset.append(golden_dict)
            
            # Save enhanced dataset as JSON
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_dataset, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Enhanced dataset saved: {save_path}")
            self.logger.info(f"   Questions with metadata: {len(enhanced_dataset)}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save enhanced dataset: {e}")
            # Fallback to standard DeepEval saving
            self.logger.info("📄 Falling back to standard DeepEval dataset saving")
            file_name = save_path.stem
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
        
        self.logger.info(f"📊 Dataset metadata saved: {metadata_path}")
    
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

    def _push_dataset_to_dashboard(self, dataset: EvaluationDataset, alias: str) -> None:
        """
        Push dataset to DeepEval dashboard for centralized storage and sharing.

        Uploads the generated synthetic dataset to DeepEval's cloud platform,
        making it accessible for team collaboration and reproducible benchmarks.

        Args:
            dataset: The EvaluationDataset to upload
            alias: Unique name for the dataset on the dashboard
        """
        try:
            self.logger.info(f"📤 Pushing dataset to DeepEval dashboard with alias '{alias}'...")

            # Push dataset to DeepEval cloud
            dataset.push(alias=alias)

            self.logger.info(f"✅ Dataset successfully pushed to dashboard as '{alias}'")
            self.logger.info(f"   Questions uploaded: {len(dataset.goldens)}")
            self.logger.info(f"🌐 Access dataset at: https://app.confident-ai.com/datasets")

        except Exception as e:
            self.logger.error(f"❌ Failed to push dataset to dashboard: {e}")
            self.logger.warning(f"💾 Dataset still available locally - evaluation can continue")
            # Don't raise exception to avoid breaking the build process

    def _generate_csv_for_upload(self, json_path: Path, output_config: Dict[str, Any]) -> None:
        """
        Generate CSV file for manual upload to DeepEval dashboard.

        Creates a CSV version of the dataset that can be manually uploaded
        to bypass free tier API limitations while maintaining dataset linking.

        Args:
            json_path: Path to the JSON dataset file
            output_config: Output configuration dictionary
        """
        try:
            csv_path = json_path.with_suffix('.csv')
            delimiter = output_config.get('csv_context_delimiter', ' | ')
            dataset_alias = output_config.get('dataset_alias', 'semantic-rag-benchmark')

            self.logger.info(f"📄 Generating CSV for manual upload: {csv_path}")

            # Import conversion script functionality
            import subprocess
            import sys

            # Run the conversion script
            conversion_script = Path(__file__).parent / "conversion.py"
            cmd = [
                sys.executable,
                str(conversion_script),
                "--input", str(json_path),
                "--output", str(csv_path),
                "--context-delimiter", delimiter
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                self.logger.info(f"✅ CSV generated successfully: {csv_path}")
                self.logger.info(f"📋 Upload Instructions:")
                self.logger.info(f"   1. Go to DeepEval dashboard")
                self.logger.info(f"   2. Upload CSV with alias: '{dataset_alias}'")
                self.logger.info(f"   3. Set context delimiter: '{delimiter}'")
                self.logger.info(f"   4. Set config 'pull_from_dashboard: true' to use uploaded dataset")
            else:
                self.logger.error(f"❌ CSV generation failed: {result.stderr}")

        except Exception as e:
            self.logger.error(f"❌ Failed to generate CSV: {e}")
            self.logger.warning(f"💾 JSON dataset still available for local use")
