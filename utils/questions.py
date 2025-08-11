#!/usr/bin/env python3
"""
Question Generation Engine - RAGAS 0.3.0 Compatible
==================================================

RAGAS-based question generation from knowledge graphs for semantic RAG evaluation.
Uses RAGAS synthesizers with personas and proper transforms for diverse question generation.
"""

import json
import hashlib
import time
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from knowledge_graph import KnowledgeGraph


@dataclass
class EvaluationQuestion:
    """Container for an evaluation question with RAGAS compatibility."""
    question_id: str
    question: str  # RAGAS uses 'question' not 'question_text'
    reference_contexts: List[str]  # RAGAS format - chunk IDs that should be relevant
    reference: Optional[str] = None  # RAGAS format - ideal answer (optional)
    synthesizer_name: str = "unknown"  # RAGAS format
    question_type: str = "unknown"  # Our categorization
    expected_advantage: str = "neutral"  # semantic_traversal, baseline, neutral
    difficulty_level: str = "medium"  # easy, medium, hard
    source_nodes: List[str] = None  # Node IDs used to generate question
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.source_nodes is None:
            self.source_nodes = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class QuestionGenerationMetadata:
    """Metadata for generated questions."""
    created_at: str
    total_questions: int
    question_types: Dict[str, int]
    synthesizer_types: Dict[str, int]
    expected_advantages: Dict[str, int]
    generation_time: float
    config_hash: str
    knowledge_graph_stats: Dict[str, Any]
    personas_used: List[str]


class QuestionEngine:
    """Engine for generating evaluation questions using RAGAS from knowledge graphs."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize the question generation engine."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.questions_config = config.get('questions', {})
        self.questions_dir = Path(config['directories']['data']) / "questions"

        # Create questions directory
        self.questions_dir.mkdir(parents=True, exist_ok=True)

        # Initialize RAGAS components (will be set up in generate_questions)
        self.knowledge_graph = None
        self.ragas_generator = None

    def generate_questions(self, knowledge_graph: KnowledgeGraph, force_recompute: bool = False) -> List[
        EvaluationQuestion]:
        """
        Generate evaluation questions using RAGAS from the knowledge graph.

        Args:
            knowledge_graph: KnowledgeGraph instance from Phase 5
            force_recompute: Whether to force regeneration even if cached

        Returns:
            List of EvaluationQuestion objects
        """
        self.knowledge_graph = knowledge_graph

        # Generate config hash for cache validation
        config_hash = self._generate_config_hash()

        # Check cache
        cache_path = self._get_cache_path()
        if not force_recompute and self._is_cache_valid(cache_path, config_hash):
            self.logger.info("📂 Loading cached questions")
            return self._load_cached_questions(cache_path)

        self.logger.info("🎯 Generating fresh questions using RAGAS")
        start_time = time.time()

        # Step 1: Apply RAGAS transforms to knowledge graph
        ragas_kg = self._prepare_knowledge_graph_for_ragas()

        # Step 2: Create personas for diverse question generation
        personas = self._create_personas()

        # Step 3: Generate questions using RAGAS
        questions = self._generate_fresh_questions(ragas_kg, personas)

        # Step 4: Add our custom categorization and metadata
        categorized_questions = self._categorize_questions(questions)

        generation_time = time.time() - start_time

        # Step 5: Create metadata
        metadata = self._create_metadata(categorized_questions, generation_time, config_hash,
                                         [p.name for p in personas])

        # Step 6: Cache questions
        self._cache_questions(cache_path, categorized_questions, metadata)

        self.logger.info(
            f"✅ Question generation completed: {len(categorized_questions)} questions in {generation_time:.2f}s")

        return categorized_questions

    def _prepare_knowledge_graph_for_ragas(self):
        """Apply RAGAS transforms to enrich the knowledge graph."""
        try:
            from ragas.testset.graph import KnowledgeGraph as RAGASKnowledgeGraph
            from ragas.testset.graph import Node as RAGASNode, NodeType, Relationship as RAGASRelationship
            from ragas.testset.transforms import default_transforms, apply_transforms
            from ragas.testset.transforms import HeadlinesExtractor, KeyphrasesExtractor
            import uuid

            # DEBUG: Check what we're starting with
            self.logger.info(f"🔍 Starting conversion with {len(self.knowledge_graph.nodes)} nodes, {len(self.knowledge_graph.relationships)} relationships")
            
            # Sample some node IDs and relationships
            node_ids = [node.id for node in self.knowledge_graph.nodes[:3]]
            self.logger.info(f"🔍 Sample node IDs: {node_ids}")
            
            if self.knowledge_graph.relationships:
                rel_pairs = [(rel.source, rel.target, rel.type) for rel in self.knowledge_graph.relationships[:3]]
                self.logger.info(f"🔍 Sample relationship pairs: {rel_pairs}")

            self.logger.info("🔄 Converting knowledge graph to RAGAS format")

            # Create RAGAS knowledge graph
            ragas_kg = RAGASKnowledgeGraph()

            # Convert nodes to RAGAS format
            id_mapping = {}
            for node in self.knowledge_graph.nodes:
                ragas_uuid = str(uuid.uuid4())

                # Map node types
                if node.type == 'DOCUMENT':
                    node_type = NodeType.DOCUMENT
                elif node.type == 'CHUNK':
                    node_type = NodeType.CHUNK
                else:
                    node_type = NodeType.UNKNOWN

                ragas_node = RAGASNode(
                    id=ragas_uuid,
                    type=node_type,
                    properties={
                        **node.properties,
                        'original_id': node.id
                    }
                )

                id_mapping[node.id] = ragas_node
                ragas_kg.add(ragas_node)

            # DEBUG: Check after node conversion
            self.logger.info(f"🔍 After node conversion: {len(ragas_kg.nodes)} nodes, {len(ragas_kg.relationships)} relationships")
            self.logger.info(f"🔍 Node ID mapping created: {len(id_mapping)} mappings")
            
            # Convert relationships
            relationships_added = 0
            relationships_skipped = 0
            for i, rel in enumerate(self.knowledge_graph.relationships):
                if rel.source in id_mapping and rel.target in id_mapping:
                    source_node = id_mapping[rel.source]
                    target_node = id_mapping[rel.target]
                    
                    ragas_rel = RAGASRelationship(
                        source=source_node,
                        target=target_node,
                        type=rel.type,
                        properties=rel.properties
                    )
                    ragas_kg.add(ragas_rel)
                    relationships_added += 1
                    
                    # Log first few successful conversions
                    if i < 3:
                        self.logger.info(f"🔍 Converted relationship {i}: {rel.source} -> {rel.target} (type: {rel.type})")
                else:
                    relationships_skipped += 1
                    if relationships_skipped < 3:
                        self.logger.info(f"🔍 Skipped relationship {i}: {rel.source} -> {rel.target} (missing nodes)")
            
            self.logger.info(f"🔍 Relationship conversion: {relationships_added} added, {relationships_skipped} skipped")
            self.logger.info(f"🔍 After relationship conversion: {len(ragas_kg.relationships)} relationships")

            self.logger.info(f"🔧 Applying RAGAS transforms to {len(ragas_kg.nodes)} nodes")

            # Setup LLM and embeddings for transforms
            llm, embeddings = self._setup_ragas_models()

            # DEBUG: Check relationships BEFORE transforms
            self.logger.info(f"🔍 BEFORE apply_transforms: {len(ragas_kg.relationships)} relationships")
            
            # Apply RAGAS transforms to enrich the knowledge graph
            transforms = [
                HeadlinesExtractor(llm=llm, max_num=10),
                KeyphrasesExtractor(llm=llm, max_num=15)
            ]

            apply_transforms(ragas_kg, transforms=transforms)
            
            # DEBUG: Check relationships AFTER transforms
            self.logger.info(f"🔍 AFTER apply_transforms: {len(ragas_kg.relationships)} relationships")
            
            # If relationships disappeared, let's check what relationship types are left
            if len(ragas_kg.relationships) == 0:
                self.logger.error("🚨 ALL RELATIONSHIPS LOST during apply_transforms!")
            else:
                relationship_types = {}
                for rel in ragas_kg.relationships:
                    rel_type = rel.type
                    relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
                self.logger.info(f"🔍 Remaining relationship types: {relationship_types}")

            self.logger.info("✅ RAGAS transforms applied successfully")
            return ragas_kg

        except Exception as e:
            self.logger.error(f"❌ Failed to apply RAGAS transforms: {e}")
            self.logger.info("🔄 Falling back to basic conversion")
            return self._convert_to_basic_ragas_kg()

    def _convert_to_basic_ragas_kg(self):
        """Fallback conversion without transforms."""

        # At the very beginning of the method:
        self.logger.info(
            f"🔍 Starting conversion with {len(self.knowledge_graph.nodes)} nodes, {len(self.knowledge_graph.relationships)} relationships")

        # Sample some node IDs
        node_ids = [node.id for node in self.knowledge_graph.nodes[:5]]
        self.logger.info(f"🔍 Sample node IDs: {node_ids}")

        # Sample some relationship source/targets
        if self.knowledge_graph.relationships:
            rel_pairs = [(rel.source, rel.target) for rel in self.knowledge_graph.relationships[:5]]
            self.logger.info(f"🔍 Sample relationship pairs: {rel_pairs}")

        try:
            from ragas.testset.graph import KnowledgeGraph as RAGASKnowledgeGraph
            from ragas.testset.graph import Node as RAGASNode, NodeType
            import uuid

            ragas_kg = RAGASKnowledgeGraph()

            for node in self.knowledge_graph.nodes:
                ragas_uuid = str(uuid.uuid4())

                if node.type == 'DOCUMENT':
                    node_type = NodeType.DOCUMENT
                elif node.type == 'CHUNK':
                    node_type = NodeType.CHUNK
                else:
                    node_type = NodeType.UNKNOWN

                ragas_node = RAGASNode(
                    id=ragas_uuid,
                    type=node_type,
                    properties={
                        'page_content': node.properties.get('page_content', node.properties.get('text', '')),
                        **node.properties,
                        'original_id': node.id
                    }
                )

                ragas_kg.add(ragas_node)

            # Right after the node conversion loop, add this debug block:
            self.logger.info(
                f"🔍 Original KG: {len(self.knowledge_graph.nodes)} nodes, {len(self.knowledge_graph.relationships)} relationships")
            self.logger.info(f"🔍 RAGAS KG after node conversion: {len(ragas_kg.nodes)} nodes, {len(ragas_kg.relationships)} relationships")

            if self.knowledge_graph.relationships:
                sample_rel = self.knowledge_graph.relationships[0]
                self.logger.info(
                    f"🔍 Sample original relationship: {sample_rel.source} -> {sample_rel.target} (type: {sample_rel.type})")

            return ragas_kg

        except Exception as e:
            self.logger.error(f"❌ Even basic conversion failed: {e}")
            return None

    def _create_personas(self):
        """Create personas for AI/ML research domain."""
        try:
            from ragas.testset.persona import Persona

            # Define personas relevant to AI/ML research
            personas = [
                Persona(
                    name="Graduate Student",
                    role_description="A graduate student learning machine learning fundamentals. Asks clarifying questions about basic concepts, mathematical foundations, and practical applications. Seeks step-by-step explanations and examples."
                ),
                Persona(
                    name="Research Scientist",
                    role_description="An experienced researcher working on advanced AI topics. Interested in cutting-edge techniques, theoretical foundations, comparative analysis between methods, and research implications."
                ),
                Persona(
                    name="ML Engineer",
                    role_description="A machine learning engineer focused on practical implementation. Asks about implementation details, performance considerations, best practices, and real-world applications of algorithms."
                ),
                Persona(
                    name="Academic Reviewer",
                    role_description="A peer reviewer evaluating research papers. Asks critical questions about methodology, experimental design, limitations, novelty, and connections to existing work."
                ),
                Persona(
                    name="Industry Practitioner",
                    role_description="A professional applying ML in industry settings. Focused on scalability, deployment considerations, business impact, and practical constraints of different approaches."
                )
            ]

            self.logger.info(f"📋 Created {len(personas)} personas for question generation")
            return personas

        except ImportError:
            self.logger.warning("⚠️  RAGAS Persona class not available, using fallback")
            return []

    def _generate_fresh_questions(self, ragas_kg, personas) -> List[EvaluationQuestion]:
        """Generate fresh questions using RAGAS tools."""
        try:
            from ragas.testset import TestsetGenerator
            from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
            from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer
            from ragas.testset.synthesizers.multi_hop.abstract import MultiHopAbstractQuerySynthesizer

            # Setup LLM and embeddings
            llm, embeddings = self._setup_ragas_models()

            # Create TestsetGenerator with personas
            generator = TestsetGenerator(
                llm=llm,
                embedding_model=embeddings,
                knowledge_graph=ragas_kg,
                persona_list=personas if personas else None
            )

            # Define query distribution based on config
            query_distribution = self._create_query_distribution(llm)

            # Generate questions
            total_questions = self.questions_config.get('total_questions', 30)

            self.logger.info(f"🎯 Generating {total_questions} questions with {len(personas)} personas")

            # Debug: Check what relationships exist
            relationship_types = {}
            for rel in ragas_kg.relationships:
                rel_type = rel.type
                relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1

            self.logger.info(f"🔍 Knowledge graph relationships: {relationship_types}")
            self.logger.info(f"🔍 Total nodes: {len(ragas_kg.nodes)}")

            # Use RAGAS to generate the testset
            testset = generator.generate(
                testset_size=total_questions,
                query_distribution=query_distribution if query_distribution else None,
                num_personas=len(personas) if personas else 3
            )

            # Convert RAGAS testset to our format
            questions = []
            testset_df = testset.to_pandas()

            for i, row in testset_df.iterrows():
                question = EvaluationQuestion(
                    question_id=f"q_{hashlib.md5(f'{row.user_input}_{i}'.encode()).hexdigest()[:8]}",
                    question=row.user_input,
                    reference_contexts=getattr(row, 'reference_contexts', []),
                    reference=getattr(row, 'reference', None),
                    synthesizer_name=getattr(row, 'synthesizer_name', 'unknown'),
                    source_nodes=[],
                    metadata={
                        'ragas_generated': True,
                        'generation_index': i,
                        'persona_used': getattr(row, 'persona', 'unknown')
                    }
                )
                questions.append(question)

            return questions

        except Exception as e:
            self.logger.error(f"❌ RAGAS question generation failed: {e}")
            return self._generate_fallback_questions()

    def _setup_ragas_models(self) -> Tuple[Any, Any]:
        """Setup LLM and embedding models for RAGAS."""
        model_config = self.questions_config.get('model', {})

        # Get API key from environment variable
        api_key = "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"
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. Please set it before running question generation.")

        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper

            llm_model = model_config.get('llm', 'gpt-3.5-turbo')
            temperature = model_config.get('temperature', 0.1)

            chat_llm = ChatOpenAI(
                model=llm_model,
                temperature=temperature,
                api_key=api_key
            )
            llm = LangchainLLMWrapper(chat_llm)

            embedding_model = model_config.get('embeddings', 'text-embedding-ada-002')
            openai_embeddings = OpenAIEmbeddings(
                model=embedding_model,
                api_key=api_key
            )
            embeddings = LangchainEmbeddingsWrapper(openai_embeddings)

            self.logger.info(f"🤖 RAGAS setup: LLM={llm_model}, Embeddings={embedding_model}")
            return llm, embeddings

        except Exception as e:
            self.logger.error(f"❌ Failed to setup RAGAS models: {e}")
            raise

    def _create_query_distribution(self, llm):
        """Create query distribution for different question types."""
        try:
            from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
            from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer
            from ragas.testset.synthesizers.multi_hop.abstract import MultiHopAbstractQuerySynthesizer

            # DEBUG: Test which synthesizers work with our graph
            self.logger.info("🔍 Testing synthesizer compatibility...")

            distribution_config = self.questions_config.get('distribution', {})

            single_hop_weight = distribution_config.get('single_hop', 0.4)
            multi_hop_specific_weight = distribution_config.get('multi_hop_specific', 0.3)
            multi_hop_abstract_weight = distribution_config.get('multi_hop_abstract', 0.3)

            # Create synthesizers that work with extracted properties
            query_distribution = [
                (SingleHopSpecificQuerySynthesizer(llm=llm, property_name="keyphrases"), single_hop_weight),
                (MultiHopSpecificQuerySynthesizer(llm=llm), multi_hop_specific_weight),
                (MultiHopAbstractQuerySynthesizer(llm=llm), multi_hop_abstract_weight)
            ]

            return query_distribution

        except ImportError:
            self.logger.warning("⚠️  RAGAS synthesizers not available")
            return []

    def _generate_fallback_questions(self) -> List[EvaluationQuestion]:
        """Fallback question generation when RAGAS is not available."""
        self.logger.info("🔄 Using fallback question generation")

        questions = []
        total_questions = min(self.questions_config.get('total_questions', 30), 10)

        chunk_nodes = [n for n in self.knowledge_graph.nodes if n.type == 'CHUNK']

        if not chunk_nodes:
            self.logger.warning("⚠️  No chunk nodes found for question generation")
            return questions

        for i in range(total_questions):
            if i >= len(chunk_nodes):
                break

            node = chunk_nodes[i]
            page_content = node.properties.get('page_content', '')
            entities = node.properties.get('entities', {})
            themes = node.properties.get('themes', [])

            if entities.get('MISC'):
                concept = entities['MISC'][0]
                question_text = f"What is {concept}?"
                question_type = "factual"
            elif themes:
                theme = themes[0]
                question_text = f"Explain {theme.replace('_', ' ')}."
                question_type = "conceptual"
            else:
                question_text = f"What does this text discuss?"
                question_type = "general"

            question = EvaluationQuestion(
                question_id=f"fallback_{i:03d}",
                question=question_text,
                reference_contexts=[node.id],
                synthesizer_name="fallback_generator",
                question_type=question_type,
                expected_advantage="neutral",
                source_nodes=[node.id],
                metadata={
                    'fallback_generated': True,
                    'source_concept': entities.get('MISC', [''])[0] if entities.get('MISC') else '',
                    'source_theme': themes[0] if themes else ''
                }
            )
            questions.append(question)

        self.logger.info(f"Generated {len(questions)} fallback questions")
        return questions

    def _categorize_questions(self, questions: List[EvaluationQuestion]) -> List[EvaluationQuestion]:
        """Add our custom categorization to questions."""
        for question in questions:
            if 'SingleHop' in question.synthesizer_name:
                question.question_type = 'single_hop'
                question.expected_advantage = 'baseline'
                question.difficulty_level = 'easy'
            elif 'MultiHopAbstract' in question.synthesizer_name:
                question.question_type = 'multi_hop_abstract'
                question.expected_advantage = 'semantic_traversal'
                question.difficulty_level = 'hard'
            elif 'MultiHopSpecific' in question.synthesizer_name:
                question.question_type = 'multi_hop_specific'
                question.expected_advantage = 'semantic_traversal'
                question.difficulty_level = 'medium'
            elif 'fallback' in question.synthesizer_name:
                question.expected_advantage = 'neutral'
                question.difficulty_level = 'easy'
            else:
                question.question_type = 'unknown'
                question.expected_advantage = 'neutral'
                question.difficulty_level = 'medium'

        return questions

    def _generate_config_hash(self) -> str:
        """Generate hash of question generation configuration."""
        config_str = json.dumps({
            'questions_config': self.questions_config,
            'kg_node_count': len(self.knowledge_graph.nodes) if self.knowledge_graph else 0,
            'kg_relationship_count': len(self.knowledge_graph.relationships) if self.knowledge_graph else 0
        }, sort_keys=True)

        return hashlib.md5(config_str.encode()).hexdigest()

    def _get_cache_path(self) -> Path:
        """Get cache path for questions."""
        return self.questions_dir / "evaluation_questions.json"

    def _is_cache_valid(self, cache_path: Path, expected_hash: str) -> bool:
        """Check if cached questions are valid."""
        if not cache_path.exists():
            return False

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            metadata = cache_data.get('metadata', {})
            cached_hash = metadata.get('config_hash', '')

            return cached_hash == expected_hash

        except Exception as e:
            self.logger.warning(f"Failed to validate question cache: {e}")
            return False

    def _create_metadata(self, questions: List[EvaluationQuestion], generation_time: float,
                         config_hash: str, personas_used: List[str]) -> QuestionGenerationMetadata:
        """Create metadata for the generated questions."""
        question_types = {}
        synthesizer_types = {}
        expected_advantages = {}

        for question in questions:
            q_type = question.question_type
            question_types[q_type] = question_types.get(q_type, 0) + 1

            synth_type = question.synthesizer_name
            synthesizer_types[synth_type] = synthesizer_types.get(synth_type, 0) + 1

            advantage = question.expected_advantage
            expected_advantages[advantage] = expected_advantages.get(advantage, 0) + 1

        return QuestionGenerationMetadata(
            created_at=datetime.now().isoformat(),
            total_questions=len(questions),
            question_types=question_types,
            synthesizer_types=synthesizer_types,
            expected_advantages=expected_advantages,
            generation_time=generation_time,
            config_hash=config_hash,
            knowledge_graph_stats={
                'total_nodes': len(self.knowledge_graph.nodes) if self.knowledge_graph else 0,
                'total_relationships': len(self.knowledge_graph.relationships) if self.knowledge_graph else 0
            },
            personas_used=personas_used
        )

    def _cache_questions(self, cache_path: Path, questions: List[EvaluationQuestion],
                         metadata: QuestionGenerationMetadata):
        """Cache questions to disk."""
        try:
            cache_data = {
                'metadata': asdict(metadata),
                'questions': [question.to_dict() for question in questions]
            }

            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"💾 Cached questions to {cache_path}")

        except Exception as e:
            self.logger.error(f"Failed to cache questions: {e}")
            raise

    def _load_cached_questions(self, cache_path: Path) -> List[EvaluationQuestion]:
        """Load cached questions from disk."""
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            questions = []
            for q_data in cache_data['questions']:
                question = EvaluationQuestion(
                    question_id=q_data['question_id'],
                    question=q_data['question'],
                    reference_contexts=q_data.get('reference_contexts', []),
                    reference=q_data.get('reference'),
                    synthesizer_name=q_data.get('synthesizer_name', 'unknown'),
                    question_type=q_data.get('question_type', 'unknown'),
                    expected_advantage=q_data.get('expected_advantage', 'neutral'),
                    difficulty_level=q_data.get('difficulty_level', 'medium'),
                    source_nodes=q_data.get('source_nodes', []),
                    metadata=q_data.get('metadata', {})
                )
                questions.append(question)

            return questions

        except Exception as e:
            self.logger.error(f"Failed to load cached questions: {e}")
            raise

    def get_question_statistics(self, questions: List[EvaluationQuestion]) -> Dict[str, Any]:
        """Get statistics about the generated questions."""
        if not questions:
            return {}

        stats = {
            'total_questions': len(questions),
            'by_question_type': {},
            'by_synthesizer': {},
            'by_expected_advantage': {},
            'by_difficulty': {},
            'by_persona': {},
            'question_length_stats': {
                'mean': 0,
                'min': float('inf'),
                'max': 0
            }
        }

        lengths = []

        for question in questions:
            # Count by question type
            q_type = question.question_type
            stats['by_question_type'][q_type] = stats['by_question_type'].get(q_type, 0) + 1

            # Count by synthesizer
            synthesizer = question.synthesizer_name
            stats['by_synthesizer'][synthesizer] = stats['by_synthesizer'].get(synthesizer, 0) + 1

            # Count by expected advantage
            advantage = question.expected_advantage
            stats['by_expected_advantage'][advantage] = stats['by_expected_advantage'].get(advantage, 0) + 1

            # Count by difficulty
            difficulty = question.difficulty_level
            stats['by_difficulty'][difficulty] = stats['by_difficulty'].get(difficulty, 0) + 1

            # Count by persona
            persona = question.metadata.get('persona_used', 'unknown')
            stats['by_persona'][persona] = stats['by_persona'].get(persona, 0) + 1

            # Track length
            length = len(question.question)
            lengths.append(length)
            stats['question_length_stats']['min'] = min(stats['question_length_stats']['min'], length)
            stats['question_length_stats']['max'] = max(stats['question_length_stats']['max'], length)

        if lengths:
            stats['question_length_stats']['mean'] = sum(lengths) / len(lengths)

        return stats