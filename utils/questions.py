#!/usr/bin/env python3
"""
Enhanced Multi-Dimensional Question Generator
===========================================

Generates questions that exploit the full semantic lattice architecture:
- Cross-document theme bridges
- Multi-granularity hierarchical navigation
- Sequential sentence flows
- Multi-dimensional connection synthesis

Personas:
- Researcher: Complex, multi-hop questions requiring sophisticated reasoning
- Googler: Simple, direct questions in casual language
"""

import json
import hashlib
import time
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from knowledge_graph import MultiGranularityKnowledgeGraph, KGNode


class ConnectionType(str, Enum):
    """Types of connections available in the semantic lattice."""
    RAW_SIMILARITY = "raw_similarity"  # Direct cosine similarity from Phase 4
    THEME_BRIDGE = "theme_bridge"  # Cross-document theme connections
    HIERARCHICAL = "hierarchical"  # Parent-child relationships
    SEQUENTIAL = "sequential"  # Sentence-to-sentence narrative flow
    ENTITY_OVERLAP = "entity_overlap"  # Shared entity connections


class QuestionType(str, Enum):
    """Five core question types that test different aspects of the semantic lattice."""
    THEME_BRIDGE = "theme_bridge"
    GRANULARITY_CASCADE = "granularity_cascade"
    THEME_SYNTHESIS = "theme_synthesis"
    SEQUENTIAL_FLOW = "sequential_flow"
    MULTI_DIMENSIONAL = "multi_dimensional"
    RAW_SIMILARITY = "raw_similarity"  # NEW: Pure cosine similarity questions


class Persona(str, Enum):
    """Question generation personas with distinct styles."""
    RESEARCHER = "researcher"  # Complex, academic-style questions
    GOOGLER = "googler"  # Simple, casual questions


@dataclass
class QuestionBlueprint:
    """Blueprint for generating a specific question type."""
    question_type: QuestionType
    persona: Persona
    source_nodes: List[KGNode]
    connection_pathway: List[Dict[str, Any]]  # Describes the connection path required
    difficulty_level: str  # easy, medium, hard, expert
    expected_hops: int
    ground_truth_nodes: List[str]  # Node IDs that contain the answer
    theme_context: Dict[str, Any]  # Theme information driving the question


@dataclass
class GeneratedQuestion:
    """A fully generated evaluation question."""
    question_id: str
    question_text: str
    question_type: QuestionType
    persona: Persona
    difficulty_level: str
    expected_hops: int

    # Ground truth information
    ground_truth_nodes: List[str]
    reference_answer: str
    connection_pathway: List[Dict[str, Any]]

    # Theme and context
    primary_themes: List[str]
    secondary_themes: List[str]
    cross_document: bool

    # Generation metadata
    generation_time: float
    model_used: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EnhancedConnectionPathwayAnalyzer:
    """Enhanced analyzer that finds both raw similarity and theme-based pathways."""

    def __init__(self, knowledge_graph: MultiGranularityKnowledgeGraph, logger: logging.Logger):
        self.kg = knowledge_graph
        self.logger = logger

        # Cache of pathways by type
        self.pathway_cache = {
            QuestionType.THEME_BRIDGE: [],
            QuestionType.GRANULARITY_CASCADE: [],
            QuestionType.THEME_SYNTHESIS: [],
            QuestionType.SEQUENTIAL_FLOW: [],
            QuestionType.MULTI_DIMENSIONAL: [],
            QuestionType.RAW_SIMILARITY: []  # NEW
        }

        # Debug hierarchical relationships
        self._debug_knowledge_graph_structure()

        # Analyze all pathways with enhanced detection
        self._analyze_all_pathways()

    def _debug_knowledge_graph_structure(self):
        """Debug the knowledge graph structure to understand relationship patterns."""
        self.logger.info("🔍 Debugging knowledge graph structure...")

        # Count nodes by type
        node_counts = {}
        for node in self.kg.nodes:
            node_type = node.type
            node_counts[node_type] = node_counts.get(node_type, 0) + 1

        self.logger.info(f"   Node counts: {node_counts}")

        # Count relationships by type and granularity_type
        relationship_counts = {}
        granularity_type_counts = {}

        for rel in self.kg.relationships:
            rel_type = rel.type
            granularity_type = rel.granularity_type

            relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1
            granularity_type_counts[granularity_type] = granularity_type_counts.get(granularity_type, 0) + 1

        self.logger.info(f"   Relationship types: {relationship_counts}")
        self.logger.info(f"   Granularity types: {granularity_type_counts}")

        # Check for hierarchical relationships specifically
        hierarchical_rels = [r for r in self.kg.relationships if r.type in ['parent', 'child']]
        self.logger.info(f"   Hierarchical relationships found: {len(hierarchical_rels)}")

        if hierarchical_rels:
            # Show sample hierarchical relationships
            for i, rel in enumerate(hierarchical_rels[:3]):
                source_node = self.kg.get_node(rel.source)
                target_node = self.kg.get_node(rel.target)
                self.logger.info(f"      Sample {i + 1}: {source_node.type} → {target_node.type} ({rel.type})")

        # Check for raw similarity relationships
        similarity_rels = [r for r in self.kg.relationships if 'similarity' in r.type.lower()]
        self.logger.info(f"   Raw similarity relationships found: {len(similarity_rels)}")

        if similarity_rels:
            # Show sample similarity relationships
            similarity_types = {}
            for rel in similarity_rels:
                similarity_types[rel.type] = similarity_types.get(rel.type, 0) + 1
            self.logger.info(f"      Similarity types: {similarity_types}")

    def _analyze_all_pathways(self):
        """Enhanced pathway analysis with both raw similarity and theme connections."""
        self.logger.info("🔍 Enhanced pathway analysis starting...")

        # Find raw similarity pathways (NEW)
        self._find_raw_similarity_pathways()

        # Find theme bridge pathways (existing, but enhanced)
        self._find_theme_bridge_pathways()

        # Find granularity cascade pathways (FIXED)
        self._find_granularity_cascade_pathways_fixed()

        # Find theme synthesis pathways
        self._find_theme_synthesis_pathways()

        # Find sequential flow pathways
        self._find_sequential_flow_pathways()

        # Find multi-dimensional pathways
        self._find_multi_dimensional_pathways()

        total_pathways = sum(len(pathways) for pathways in self.pathway_cache.values())
        self.logger.info(f"✅ Enhanced analysis found {total_pathways} pathways across all types")

    def _find_raw_similarity_pathways(self):
        """NEW: Find pathways based on raw cosine similarity connections."""
        pathways = []

        # Look for relationships that are pure cosine similarity (not theme-based)
        similarity_relationship_types = [
            'cosine_similarity',
            'cosine_similarity_intra',
            'cosine_similarity_inter',
            'sentence_to_sentence_semantic',
            'doc_to_doc'
        ]

        for rel in self.kg.relationships:
            if rel.type in similarity_relationship_types:
                source_node = self.kg.get_node(rel.source)
                target_node = self.kg.get_node(rel.target)

                if source_node and target_node:
                    # This is a raw similarity connection
                    pathway = {
                        'source_node': source_node,
                        'target_node': target_node,
                        'similarity_score': rel.weight,
                        'connection_type': ConnectionType.RAW_SIMILARITY,
                        'relationship_type': rel.type,
                        'pathway_type': 'raw_cosine_similarity',
                        'expected_hops': 1  # Direct similarity connection
                    }
                    pathways.append(pathway)

        # Limit to prevent explosion, but sample across different similarity types
        if len(pathways) > 1000:
            # Sample proportionally from different relationship types
            type_samples = {}
            for pathway in pathways:
                rel_type = pathway['relationship_type']
                if rel_type not in type_samples:
                    type_samples[rel_type] = []
                type_samples[rel_type].append(pathway)

            sampled_pathways = []
            for rel_type, type_pathways in type_samples.items():
                sample_size = min(200, len(type_pathways))  # Max 200 per type
                sampled_pathways.extend(random.sample(type_pathways, sample_size))

            pathways = sampled_pathways

        self.pathway_cache[QuestionType.RAW_SIMILARITY] = pathways
        self.logger.info(f"   Raw similarity pathways: {len(pathways)} (sampled from available connections)")

    def _find_granularity_cascade_pathways_fixed(self):
        """FIXED: Find hierarchical pathways using proper relationship traversal."""
        pathways = []

        # Strategy: Instead of using get_children (which might be broken),
        # directly traverse the relationships to find hierarchical chains

        # Find all hierarchical relationships
        hierarchical_rels = [r for r in self.kg.relationships if r.type == 'parent']

        if not hierarchical_rels:
            self.logger.warning("   No 'parent' relationships found - checking for alternative hierarchical patterns")

            # Look for other hierarchical indicators
            contains_rels = [r for r in self.kg.relationships if 'contains' in r.type.lower()]
            hierarchical_rels.extend(contains_rels)

            child_rels = [r for r in self.kg.relationships if r.type == 'child']
            # Convert child relationships to parent relationships
            for rel in child_rels:
                # Swap source and target for parent perspective
                hierarchical_rels.append(type(rel)(
                    source=rel.target,
                    target=rel.source,
                    type='parent',
                    granularity_type=rel.granularity_type,
                    properties=rel.properties,
                    weight=rel.weight
                ))

        self.logger.info(f"   Found {len(hierarchical_rels)} hierarchical relationships to analyze")

        # Build hierarchy chains: Document → Chunk → Sentence
        doc_to_chunk = {}
        chunk_to_sentence = {}

        for rel in hierarchical_rels:
            source_node = self.kg.get_node(rel.source)
            target_node = self.kg.get_node(rel.target)

            if source_node and target_node:
                if source_node.type == 'DOCUMENT' and target_node.type == 'CHUNK':
                    if source_node.id not in doc_to_chunk:
                        doc_to_chunk[source_node.id] = []
                    doc_to_chunk[source_node.id].append(target_node)

                elif source_node.type == 'CHUNK' and target_node.type == 'SENTENCE':
                    if source_node.id not in chunk_to_sentence:
                        chunk_to_sentence[source_node.id] = []
                    chunk_to_sentence[source_node.id].append(target_node)

        self.logger.info(f"   Doc→Chunk mappings: {len(doc_to_chunk)}")
        self.logger.info(f"   Chunk→Sentence mappings: {len(chunk_to_sentence)}")

        # Create complete cascades: Document → Chunk → Sentence
        for doc_id, chunks in doc_to_chunk.items():
            doc_node = self.kg.get_node(doc_id)

            for chunk_node in chunks:
                sentences = chunk_to_sentence.get(chunk_node.id, [])

                if sentences:  # We have a complete cascade
                    # Extract themes from document node
                    doc_themes = doc_node.properties.get('direct_themes', [])

                    pathway = {
                        'document_node': doc_node,
                        'chunk_node': chunk_node,
                        'sentence_nodes': sentences[:3],  # Limit to first 3 sentences
                        'themes': doc_themes,
                        'pathway_type': 'granularity_cascade',
                        'expected_hops': 3,  # Doc → Chunk → Sentence
                        'connection_type': ConnectionType.HIERARCHICAL
                    }
                    pathways.append(pathway)

        self.pathway_cache[QuestionType.GRANULARITY_CASCADE] = pathways
        self.logger.info(f"   Granularity cascade pathways: {len(pathways)} (FIXED)")

    def _find_theme_bridge_pathways(self):
        """Enhanced theme bridge detection with better sampling."""
        pathways = []

        # Get all chunk nodes with inherited themes
        chunk_nodes = self.kg.get_nodes_by_type('CHUNK')
        chunks_with_themes = []

        for chunk_node in chunk_nodes:
            inherited_themes = chunk_node.properties.get('inherited_themes', [])
            direct_themes = chunk_node.properties.get('direct_themes', [])

            if inherited_themes or direct_themes:
                chunks_with_themes.append(chunk_node)

        self.logger.info(f"   Found {len(chunks_with_themes)} chunks with themes for bridge analysis")

        # Sample to prevent explosion
        max_source_chunks = 1000  # Limit source chunks to prevent quadratic explosion
        if len(chunks_with_themes) > max_source_chunks:
            chunks_with_themes = random.sample(chunks_with_themes, max_source_chunks)

        for source_chunk in chunks_with_themes:
            source_doc = source_chunk.properties.get('source_article')
            source_inherited = source_chunk.properties.get('inherited_themes', [])

            if source_inherited:
                # Sample target chunks for comparison (limit to prevent explosion)
                target_sample_size = min(50, len(chunks_with_themes))
                target_chunks = random.sample(chunks_with_themes, target_sample_size)

                for target_chunk in target_chunks:
                    target_doc = target_chunk.properties.get('source_article')

                    if target_doc != source_doc:  # Cross-document only
                        target_inherited = target_chunk.properties.get('inherited_themes', [])
                        target_direct = target_chunk.properties.get('direct_themes', [])

                        # Check for theme overlap
                        source_theme_names = {t['theme'] for t in source_inherited}
                        target_theme_names = set(target_direct) | {t['theme'] for t in target_inherited}

                        theme_overlap = source_theme_names.intersection(target_theme_names)

                        if theme_overlap:
                            pathway = {
                                'source_node': source_chunk,
                                'target_node': target_chunk,
                                'bridge_themes': list(theme_overlap),
                                'source_document': source_doc,
                                'target_document': target_doc,
                                'pathway_type': 'cross_document_theme_bridge',
                                'expected_hops': 2,
                                'connection_type': ConnectionType.THEME_BRIDGE
                            }
                            pathways.append(pathway)

        # Further sampling to manageable size
        if len(pathways) > 5000:
            pathways = random.sample(pathways, 5000)

        self.pathway_cache[QuestionType.THEME_BRIDGE] = pathways
        self.logger.info(f"   Theme bridge pathways: {len(pathways)} (sampled for performance)")

    def _find_theme_synthesis_pathways(self):
        """Enhanced theme synthesis with better sampling."""
        pathways = []

        chunk_nodes = self.kg.get_nodes_by_type('CHUNK')

        # Find chunks with multiple themes
        multi_theme_chunks = []
        for chunk_node in chunk_nodes:
            direct_themes = chunk_node.properties.get('direct_themes', [])
            inherited_themes = chunk_node.properties.get('inherited_themes', [])

            all_themes = set(direct_themes) | {t['theme'] for t in inherited_themes}

            if len(all_themes) >= 2:
                multi_theme_chunks.append((chunk_node, all_themes))

        self.logger.info(f"   Found {len(multi_theme_chunks)} multi-theme chunks for synthesis analysis")

        # Sample to prevent explosion
        max_chunks = 1000
        if len(multi_theme_chunks) > max_chunks:
            multi_theme_chunks = random.sample(multi_theme_chunks, max_chunks)

        for i, (source_chunk, source_themes) in enumerate(multi_theme_chunks):
            # Sample targets for comparison
            target_sample_size = min(20, len(multi_theme_chunks))
            target_sample = random.sample(multi_theme_chunks, target_sample_size)

            for target_chunk, target_themes in target_sample:
                if target_chunk.id != source_chunk.id:
                    overlap = source_themes.intersection(target_themes)
                    unique_to_source = source_themes - target_themes
                    unique_to_target = target_themes - source_themes

                    # Interesting if there's partial overlap
                    if overlap and unique_to_source and unique_to_target:
                        pathway = {
                            'source_node': source_chunk,
                            'target_node': target_chunk,
                            'shared_themes': list(overlap),
                            'source_unique_themes': list(unique_to_source),
                            'target_unique_themes': list(unique_to_target),
                            'pathway_type': 'theme_synthesis',
                            'expected_hops': 2,
                            'connection_type': ConnectionType.THEME_BRIDGE
                        }
                        pathways.append(pathway)

        # Sample final results
        if len(pathways) > 3000:
            pathways = random.sample(pathways, 3000)

        self.pathway_cache[QuestionType.THEME_SYNTHESIS] = pathways
        self.logger.info(f"   Theme synthesis pathways: {len(pathways)} (sampled for performance)")

    def _find_sequential_flow_pathways(self):
        """Find sentence-to-sentence sequential pathways."""
        pathways = []

        # Look for sequential relationships
        sequential_rels = [r for r in self.kg.relationships if 'sequential' in r.type.lower()]

        self.logger.info(f"   Found {len(sequential_rels)} sequential relationships")

        # Group sequential relationships by document
        doc_sequences = {}

        for rel in sequential_rels:
            source_node = self.kg.get_node(rel.source)
            target_node = self.kg.get_node(rel.target)

            if source_node and target_node and source_node.type == 'SENTENCE' and target_node.type == 'SENTENCE':
                doc = source_node.properties.get('source_article')
                if doc not in doc_sequences:
                    doc_sequences[doc] = []
                doc_sequences[doc].append((source_node, target_node))

        # Build sequence chains of 3+ sentences
        for doc, pairs in doc_sequences.items():
            # Sort pairs by sentence index
            sorted_pairs = sorted(pairs, key=lambda p: p[0].properties.get('sentence_index', 0))

            # Find chains of 3+ consecutive sentences
            i = 0
            while i < len(sorted_pairs) - 1:
                sequence = [sorted_pairs[i][0], sorted_pairs[i][1]]

                # Try to extend the sequence
                j = i + 1
                while j < len(sorted_pairs) and sorted_pairs[j][0].id == sequence[-1].id:
                    sequence.append(sorted_pairs[j][1])
                    j += 1

                if len(sequence) >= 3:
                    pathway = {
                        'sentence_sequence': sequence,
                        'document': doc,
                        'start_index': sequence[0].properties.get('sentence_index', 0),
                        'pathway_type': 'sequential_flow',
                        'expected_hops': len(sequence),
                        'connection_type': ConnectionType.SEQUENTIAL
                    }
                    pathways.append(pathway)

                i = j if j > i + 1 else i + 1

        self.pathway_cache[QuestionType.SEQUENTIAL_FLOW] = pathways
        self.logger.info(f"   Sequential flow pathways: {len(pathways)}")

    def _find_multi_dimensional_pathways(self):
        """Find pathways combining multiple connection types."""
        pathways = []

        # Combine theme bridges with other connection types
        theme_bridges = self.pathway_cache[QuestionType.THEME_BRIDGE]

        for bridge in theme_bridges[:100]:  # Limit to prevent explosion
            source_chunk = bridge['source_node']
            target_chunk = bridge['target_node']

            # Look for additional connections from the target chunk
            # 1. Raw similarity connections
            raw_similarities = []
            for rel in self.kg.relationships:
                if (rel.source == target_chunk.id and
                        rel.type in ['cosine_similarity', 'cosine_similarity_intra', 'cosine_similarity_inter']):
                    connected_node = self.kg.get_node(rel.target)
                    if connected_node:
                        raw_similarities.append(connected_node)

            # 2. Hierarchical connections (down to sentences)
            hierarchical_children = []
            for rel in self.kg.relationships:
                if rel.source == target_chunk.id and rel.type == 'parent':
                    child_node = self.kg.get_node(rel.target)
                    if child_node and child_node.type == 'SENTENCE':
                        hierarchical_children.append(child_node)

            if raw_similarities or hierarchical_children:
                pathway = {
                    'bridge_pathway': bridge,
                    'raw_similarity_connections': raw_similarities[:3],  # Limit
                    'hierarchical_connections': hierarchical_children[:3],  # Limit
                    'pathway_type': 'multi_dimensional',
                    'expected_hops': 4,
                    'connection_types': [ConnectionType.THEME_BRIDGE, ConnectionType.RAW_SIMILARITY,
                                         ConnectionType.HIERARCHICAL]
                }
                pathways.append(pathway)

        self.pathway_cache[QuestionType.MULTI_DIMENSIONAL] = pathways
        self.logger.info(f"   Multi-dimensional pathways: {len(pathways)}")

    def get_random_pathway(self, question_type: QuestionType) -> Optional[Dict[str, Any]]:
        """Get a random pathway of the specified type."""
        pathways = self.pathway_cache.get(question_type, [])
        return random.choice(pathways) if pathways else None


class OllamaQuestionGenerator:
    """Generates questions using Ollama with sophisticated prompts."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.model = config.get('ollama', {}).get('model', 'llama3.1:8b')
        self.available = self._test_ollama()

    def _test_ollama(self) -> bool:
        """Test if Ollama is available."""
        if not OLLAMA_AVAILABLE:
            return False

        try:
            models = ollama.list()
            available_models = [model.model for model in models.models]
            return self.model in available_models
        except Exception:
            return False

    def generate_question(self, blueprint: QuestionBlueprint) -> Optional[GeneratedQuestion]:
        """Generate a question from a blueprint."""

        print(f"🎯 Generating {blueprint.question_type.value} question (persona: {blueprint.persona.value})...")

        if not self.available:
            return self._generate_fallback_question(blueprint)

        start_time = time.time()

        try:
            prompt = self._build_prompt(blueprint)

            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "num_predict": 300,
                    "stop": ["\n\nEXPLANATION:", "\n\nCONTEXT:", "```"],
                    "timeout": 45
                }
            )

            question_text = self._parse_response(response['response'], blueprint.persona)

            if question_text:
                return self._create_generated_question(blueprint, question_text, time.time() - start_time)
            else:
                return self._generate_fallback_question(blueprint)

        except Exception as e:
            self.logger.warning(f"Ollama generation failed: {e}")
            return self._generate_fallback_question(blueprint)

    def _build_prompt(self, blueprint: QuestionBlueprint) -> str:
        """Build sophisticated prompts based on question type and persona."""

        if blueprint.persona == Persona.RESEARCHER:
            persona_instruction = "You are an academic researcher asking sophisticated questions that require multi-step reasoning and synthesis of complex information."
            style_instruction = "Generate a complex, intellectually rigorous question in ONLY one sentence. DO NOT generate more than this one sentence."
        else:  # GOOGLER
            persona_instruction = "You are a casual internet user asking simple, direct questions in lowercase, conversational style."
            style_instruction = "Generate a simple, direct question in all lowercase, in ONLY one sentence. DO NOT generate more than this one sentence. If the context given seems like it contains many broad topics, distill it down to as few words as possible."

        # Extract key context from the blueprint
        context_texts = []
        for node in blueprint.source_nodes:
            page_content = node.properties.get('page_content', node.properties.get('text', ''))
            context_texts.append(page_content[:200] + "...")

        combined_context = "\n\n".join(context_texts)

        # Type-specific instructions
        type_instructions = {
            QuestionType.THEME_BRIDGE: f"This question should require connecting information across different documents through shared themes: {blueprint.theme_context.get('bridge_themes', [])}",

            QuestionType.GRANULARITY_CASCADE: "This question should require starting with high-level themes and drilling down to specific details across multiple levels of granularity.",

            QuestionType.THEME_SYNTHESIS: f"This question should require synthesizing multiple themes: {blueprint.theme_context.get('themes_to_synthesize', [])}",

            QuestionType.SEQUENTIAL_FLOW: "This question should require following a logical sequence or argument that builds across multiple sentences.",

            QuestionType.RAW_SIMILARITY: f"This question should require understanding the direct semantic similarity between two pieces of content with a similarity score of {blueprint.theme_context.get('similarity_score', 'high')}.",

            QuestionType.MULTI_DIMENSIONAL: "This question should require complex reasoning that combines multiple types of connections and reasoning patterns."
        }

        prompt = f"""{persona_instruction}

CONTEXT INFORMATION:
{combined_context}

THEMES INVOLVED:
{blueprint.theme_context}

QUESTION REQUIREMENTS:
{type_instructions[blueprint.question_type]}

The question should require {blueprint.expected_hops} steps of reasoning.
Difficulty level: {blueprint.difficulty_level}

{style_instruction}

QUESTION:"""

        return prompt

    def _parse_response(self, response: str, persona: Persona) -> Optional[str]:
        """Parse and clean the generated response."""
        question = response.strip()

        # Remove common prefixes
        prefixes = ["question:", "here is a question:", "the question is:"]
        for prefix in prefixes:
            if question.lower().startswith(prefix):
                question = question[len(prefix):].strip()

        # Ensure it ends with a question mark
        if not question.endswith('?'):
            question += '?'

        # Apply persona-specific formatting
        if persona == Persona.GOOGLER:
            question = question.lower()
        elif persona == Persona.RESEARCHER:
            # Ensure first letter is capitalized
            if question:
                question = question[0].upper() + question[1:]

        return question if len(question) > 10 else None

    def _create_generated_question(self, blueprint: QuestionBlueprint, question_text: str,
                                   generation_time: float) -> GeneratedQuestion:
        """Create a GeneratedQuestion object."""
        question_id = f"{blueprint.question_type.value}_{hashlib.md5(question_text.encode()).hexdigest()[:8]}"

        return GeneratedQuestion(
            question_id=question_id,
            question_text=question_text,
            question_type=blueprint.question_type,
            persona=blueprint.persona,
            difficulty_level=blueprint.difficulty_level,
            expected_hops=blueprint.expected_hops,
            ground_truth_nodes=blueprint.ground_truth_nodes,
            reference_answer="",  # Could be generated separately
            connection_pathway=blueprint.connection_pathway,
            primary_themes=blueprint.theme_context.get('primary_themes', []),
            secondary_themes=blueprint.theme_context.get('secondary_themes', []),
            cross_document=blueprint.theme_context.get('cross_document', False),
            generation_time=generation_time,
            model_used=self.model
        )

    def _generate_fallback_question(self, blueprint: QuestionBlueprint) -> GeneratedQuestion:
        """Generate fallback question when Ollama is unavailable."""
        # Simple template-based fallback
        if blueprint.persona == Persona.GOOGLER:
            question_text = f"what is {blueprint.theme_context.get('primary_themes', ['this topic'])[0]}?"
        else:
            themes = blueprint.theme_context.get('primary_themes', ['these concepts'])
            question_text = f"How do {themes[0]} relate to broader theoretical frameworks?"

        return self._create_generated_question(blueprint, question_text, 0.1)


class MultiDimensionalQuestionEngine:
    """Main engine for generating multi-dimensional questions."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Initialize components
        self.ollama_generator = OllamaQuestionGenerator(config, self.logger)

        # Question distribution across types and personas
        self.question_distribution = {
            QuestionType.RAW_SIMILARITY: 0.20,  # Test the 124k+ similarity edges
            QuestionType.MULTI_DIMENSIONAL: 0.30,  # ← INCREASED! Most sophisticated
            QuestionType.THEME_BRIDGE: 0.20,  # Cross-document reasoning
            QuestionType.GRANULARITY_CASCADE: 0.15,  # Hierarchical navigation
            QuestionType.THEME_SYNTHESIS: 0.10,  # Multi-theme integration
            QuestionType.SEQUENTIAL_FLOW: 0.05,  # Narrative chains
        }

        self.persona_distribution = {
            Persona.RESEARCHER: 0.7,
            Persona.GOOGLER: 0.3
        }

        self.logger.info("🎯 Initialized multi-dimensional question generator")

    def generate_questions(self, knowledge_graph: MultiGranularityKnowledgeGraph,
                           target_count: int = 50, force_recompute: bool = False) -> List[GeneratedQuestion]:
        """Generate the full question suite."""

        # Check cache
        cache_path = self._get_cache_path()
        if not force_recompute and self._is_cache_valid(cache_path, knowledge_graph):
            self.logger.info("📂 Loading cached questions")
            return self._load_cached_questions(cache_path)

        self.logger.info(f"🎯 Generating {target_count} multi-dimensional questions")

        # Analyze pathways
        pathway_analyzer = EnhancedConnectionPathwayAnalyzer(knowledge_graph, self.logger)

        # Calculate question counts by type
        type_counts = self._calculate_question_counts(target_count)

        all_questions = []

        for question_type, count in type_counts.items():
            self.logger.info(f"Generating {count} {question_type.value} questions...")

            for _ in range(count):
                # Get pathway for this question type
                pathway = pathway_analyzer.get_random_pathway(question_type)
                if not pathway:
                    continue

                # Create blueprint
                blueprint = self._create_blueprint(question_type, pathway)
                if not blueprint:
                    continue

                # Generate question
                question = self.ollama_generator.generate_question(blueprint)
                if question:
                    all_questions.append(question)

        # Cache results
        self._cache_questions(cache_path, all_questions, knowledge_graph)

        self.logger.info(f"✅ Generated {len(all_questions)} questions")
        return all_questions

    def _calculate_question_counts(self, target_count: int) -> Dict[QuestionType, int]:
        """Calculate how many questions to generate for each type."""
        counts = {}
        for question_type, proportion in self.question_distribution.items():
            counts[question_type] = int(target_count * proportion)
        return counts

    def _create_blueprint(self, question_type: QuestionType, pathway: Dict[str, Any]) -> QuestionBlueprint:
        """Create a question blueprint from a pathway with robust error handling."""

        # Randomly select persona
        persona = random.choices(
            list(self.persona_distribution.keys()),
            weights=list(self.persona_distribution.values())
        )[0]

        # Handle each question type with defensive programming
        try:
            if question_type == QuestionType.THEME_BRIDGE:
                return self._create_theme_bridge_blueprint(pathway, persona)
            elif question_type == QuestionType.GRANULARITY_CASCADE:
                return self._create_granularity_cascade_blueprint(pathway, persona)
            elif question_type == QuestionType.THEME_SYNTHESIS:
                return self._create_theme_synthesis_blueprint(pathway, persona)
            elif question_type == QuestionType.SEQUENTIAL_FLOW:
                return self._create_sequential_flow_blueprint(pathway, persona)
            elif question_type == QuestionType.RAW_SIMILARITY:  # ← THIS IS THE MISSING CASE
                return self._create_raw_similarity_blueprint(pathway, persona)
            elif question_type == QuestionType.MULTI_DIMENSIONAL:
                return self._create_multi_dimensional_blueprint(pathway, persona)
            else:
                raise ValueError(f"Unknown question type: {question_type}")

        except KeyError as e:
            self.logger.warning(f"Pathway structure issue for {question_type}: missing key {e}")
            # Return a fallback blueprint
            return self._create_fallback_blueprint(question_type, pathway, persona)
        except Exception as e:
            self.logger.warning(f"Blueprint creation failed for {question_type}: {e}")
            return self._create_fallback_blueprint(question_type, pathway, persona)

    def _create_theme_bridge_blueprint(self, pathway: Dict[str, Any], persona: Persona) -> QuestionBlueprint:
        """Create theme bridge blueprint with validation."""
        source_nodes = [pathway['source_node'], pathway['target_node']]
        theme_context = {
            'bridge_themes': pathway.get('bridge_themes', []),
            'cross_document': True,
            'primary_themes': pathway.get('bridge_themes', [])
        }
        ground_truth_nodes = [n.id for n in source_nodes]

        return QuestionBlueprint(
            question_type=QuestionType.THEME_BRIDGE,
            persona=persona,
            source_nodes=source_nodes,
            connection_pathway=[pathway],
            difficulty_level="medium",
            expected_hops=pathway.get('expected_hops', 2),
            ground_truth_nodes=ground_truth_nodes,
            theme_context=theme_context
        )

    def _create_granularity_cascade_blueprint(self, pathway: Dict[str, Any], persona: Persona) -> QuestionBlueprint:
        """Create granularity cascade blueprint with validation."""
        source_nodes = [pathway['document_node'], pathway['chunk_node']] + pathway['sentence_nodes'][:2]
        theme_context = {
            'themes': pathway.get('themes', []),
            'cross_document': False,
            'primary_themes': pathway.get('themes', [])
        }
        ground_truth_nodes = [n.id for n in source_nodes]

        return QuestionBlueprint(
            question_type=QuestionType.GRANULARITY_CASCADE,
            persona=persona,
            source_nodes=source_nodes,
            connection_pathway=[pathway],
            difficulty_level="medium",
            expected_hops=pathway.get('expected_hops', 3),
            ground_truth_nodes=ground_truth_nodes,
            theme_context=theme_context
        )

    def _create_theme_synthesis_blueprint(self, pathway: Dict[str, Any], persona: Persona) -> QuestionBlueprint:
        """Create theme synthesis blueprint with validation."""
        source_nodes = [pathway['source_node'], pathway['target_node']]
        theme_context = {
            'themes_to_synthesize': pathway.get('shared_themes', []) + pathway.get('source_unique_themes', [])[:2],
            'primary_themes': pathway.get('shared_themes', []),
            'secondary_themes': pathway.get('source_unique_themes', [])[:2]
        }
        ground_truth_nodes = [n.id for n in source_nodes]

        return QuestionBlueprint(
            question_type=QuestionType.THEME_SYNTHESIS,
            persona=persona,
            source_nodes=source_nodes,
            connection_pathway=[pathway],
            difficulty_level="hard",
            expected_hops=pathway.get('expected_hops', 2),
            ground_truth_nodes=ground_truth_nodes,
            theme_context=theme_context
        )

    def _create_sequential_flow_blueprint(self, pathway: Dict[str, Any], persona: Persona) -> QuestionBlueprint:
        """Create sequential flow blueprint with validation."""
        source_nodes = pathway.get('sentence_sequence', [])
        theme_context = {
            'document': pathway.get('document', 'Unknown'),
            'sequential': True,
            'primary_themes': []
        }
        ground_truth_nodes = [n.id for n in source_nodes]

        return QuestionBlueprint(
            question_type=QuestionType.SEQUENTIAL_FLOW,
            persona=persona,
            source_nodes=source_nodes,
            connection_pathway=[pathway],
            difficulty_level="easy",
            expected_hops=pathway.get('expected_hops', len(source_nodes)),
            ground_truth_nodes=ground_truth_nodes,
            theme_context=theme_context
        )

    def _create_raw_similarity_blueprint(self, pathway: Dict[str, Any], persona: Persona) -> QuestionBlueprint:
        """Create raw similarity blueprint with validation."""
        source_nodes = [pathway['source_node'], pathway['target_node']]
        theme_context = {
            'similarity_score': pathway.get('similarity_score', 0.0),
            'relationship_type': pathway.get('relationship_type', 'unknown'),
            'primary_themes': []  # Raw similarity doesn't use themes
        }
        ground_truth_nodes = [n.id for n in source_nodes]

        return QuestionBlueprint(
            question_type=QuestionType.RAW_SIMILARITY,
            persona=persona,
            source_nodes=source_nodes,
            connection_pathway=[pathway],
            difficulty_level="medium",
            expected_hops=pathway.get('expected_hops', 1),
            ground_truth_nodes=ground_truth_nodes,
            theme_context=theme_context
        )

    def _create_multi_dimensional_blueprint(self, pathway: Dict[str, Any], persona: Persona) -> QuestionBlueprint:
        """Create multi-dimensional blueprint with robust pathway handling."""

        # Handle different multi-dimensional pathway structures
        if 'bridge_pathway' in pathway:
            # Original structure with bridge pathway
            bridge = pathway['bridge_pathway']
            source_nodes = [bridge['source_node'], bridge['target_node']] + pathway.get('hierarchical_connections', [])[
                                                                            :2]
            theme_context = {
                'bridge_themes': bridge.get('bridge_themes', []),
                'cross_document': True,
                'multi_dimensional': True,
                'primary_themes': bridge.get('bridge_themes', [])
            }
        else:
            # Alternative structure - use available nodes
            all_nodes = []

            # Collect nodes from various pathway components
            if 'source_node' in pathway and 'target_node' in pathway:
                all_nodes.extend([pathway['source_node'], pathway['target_node']])

            if 'raw_similarity_connections' in pathway:
                all_nodes.extend(pathway['raw_similarity_connections'][:2])

            if 'hierarchical_connections' in pathway:
                all_nodes.extend(pathway['hierarchical_connections'][:2])

            # Fallback to first available nodes
            if not all_nodes and 'sentence_sequence' in pathway:
                all_nodes = pathway['sentence_sequence'][:3]

            source_nodes = all_nodes[:4] if all_nodes else []

            theme_context = {
                'multi_dimensional': True,
                'connection_types': pathway.get('connection_types', []),
                'primary_themes': []
            }

        ground_truth_nodes = [n.id for n in source_nodes] if source_nodes else []

        return QuestionBlueprint(
            question_type=QuestionType.MULTI_DIMENSIONAL,
            persona=persona,
            source_nodes=source_nodes,
            connection_pathway=[pathway],
            difficulty_level="expert",
            expected_hops=pathway.get('expected_hops', 4),
            ground_truth_nodes=ground_truth_nodes,
            theme_context=theme_context
        )

    def _create_fallback_blueprint(self, question_type: QuestionType, pathway: Dict[str, Any],
                                   persona: Persona) -> QuestionBlueprint:
        """Create a fallback blueprint when pathway structure is unexpected."""

        # Extract any available nodes from the pathway
        source_nodes = []

        # Try common node keys
        for key in ['source_node', 'target_node', 'document_node', 'chunk_node']:
            if key in pathway and pathway[key]:
                source_nodes.append(pathway[key])

        # Try sequence keys
        for key in ['sentence_nodes', 'sentence_sequence']:
            if key in pathway and pathway[key]:
                source_nodes.extend(pathway[key][:2])

        # Minimum viable blueprint
        return QuestionBlueprint(
            question_type=question_type,
            persona=persona,
            source_nodes=source_nodes[:3],  # Limit to prevent overflow
            connection_pathway=[pathway],
            difficulty_level="medium",
            expected_hops=2,
            ground_truth_nodes=[n.id for n in source_nodes[:3]] if source_nodes else [],
            theme_context={'fallback': True, 'primary_themes': []}
        )

    def _get_cache_path(self) -> Path:
        """Get cache path for questions."""
        data_dir = Path(self.config['directories']['data'])
        return data_dir / "questions" / "multi_dimensional_questions.json"

    def _is_cache_valid(self, cache_path: Path, knowledge_graph: MultiGranularityKnowledgeGraph) -> bool:
        """Check if cached questions are valid."""
        # Implementation would check if KG structure changed
        return cache_path.exists()  # Simplified for now

    def _cache_questions(self, cache_path: Path, questions: List[GeneratedQuestion],
                         knowledge_graph: MultiGranularityKnowledgeGraph):
        """Cache questions to disk."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        cache_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_questions': len(questions),
                'ollama_available': self.ollama_generator.available,
                'model_used': self.ollama_generator.model
            },
            'questions': [q.to_dict() for q in questions]
        }

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)

    def _load_cached_questions(self, cache_path: Path) -> List[GeneratedQuestion]:
        """Load cached questions from disk."""
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        questions = []
        for q_data in cache_data['questions']:
            # Convert back to enum types
            q_data['question_type'] = QuestionType(q_data['question_type'])
            q_data['persona'] = Persona(q_data['persona'])
            questions.append(GeneratedQuestion(**q_data))

        return questions