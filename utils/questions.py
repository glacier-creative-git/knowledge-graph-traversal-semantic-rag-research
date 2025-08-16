#!/usr/bin/env python3
"""
Knowledge Graph Question Generation Engine - RAGAS-Style Implementation
======================================================================

Advanced question generation using RAGAS-style structured prompting combined with
sophisticated multi-dimensional knowledge graph traversal. Leverages rich metadata
extraction (entities, keyphrases) as themes for persona-driven question synthesis.

Key Improvements:
- RAGAS-style structured prompting with personas and themes
- Multi-hop context formatting (<1-hop>, <2-hop>)
- Strategic theme extraction from entities and keyphrases
- Structured output validation with Pydantic-style models
- Intelligent node selection based on graph relationships
- Ollama integration with fallback to structured templates

Question Generation Strategies:
1. Entity Bridge Questions (40%): Test entity-based traversal
2. Concept Similarity Questions (30%): Test cosine similarity traversal  
3. Hierarchical Questions (20%): Test multi-granularity navigation
4. Single-Hop Questions (10%): Test individual chunk/sentence retrieval
"""

import json
import hashlib
import time
import logging
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from knowledge_graph import KnowledgeGraph, KGNode


# RAGAS-style enums for structured question generation
class QueryStyle(str, Enum):
    """Question styles for diverse generation."""
    FORMAL = "Formal"
    CONVERSATIONAL = "Conversational"
    ANALYTICAL = "Analytical"
    INVESTIGATIVE = "Investigative"


class QueryLength(str, Enum):
    """Question length specifications."""
    SHORT = "Short"
    MEDIUM = "Medium"
    LONG = "Long"


@dataclass
class Persona:
    """RAGAS-style persona for question generation."""
    name: str
    role_description: str
    question_style: str
    example_questions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QuestionScenario:
    """RAGAS-style scenario for structured question generation."""
    nodes: List[KGNode]
    themes: List[str]  # Extracted from entities/keyphrases
    persona: Persona
    style: QueryStyle
    length: QueryLength
    strategy: str
    metadata: Dict[str, Any]


@dataclass
class EvaluationQuestion:
    """Container for an evaluation question with ground truth contexts."""
    question_id: str
    question: str
    reference_answer: Optional[str]  # RAGAS compatibility
    question_type: str  # "entity_bridge", "concept_similarity", "hierarchical", "single_hop"
    expected_advantage: str  # "semantic_traversal", "baseline_vector", "neutral"
    difficulty_level: str  # "easy", "medium", "hard"
    
    # Ground truth contexts (what should be retrieved)
    ground_truth_contexts: List[str]  # Node IDs that should be relevant
    reference_contexts: List[str]  # Actual text contexts for RAGAS compatibility
    primary_context_id: str  # Primary node used to generate question
    
    # Question generation metadata
    generation_strategy: str  # Specific strategy used
    source_nodes: List[str]  # Node IDs used to generate question
    relationship_types: List[str]  # Relationship types tested
    themes_used: List[str]  # Themes/concepts used in generation
    persona_used: str  # Persona name used
    
    # Ollama generation metadata
    model_used: str
    generation_prompt: str
    generation_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class QuestionGenerationMetadata:
    """Metadata for the question generation process."""
    created_at: str
    total_questions: int
    generation_strategies: Dict[str, int]
    question_types: Dict[str, int]
    expected_advantages: Dict[str, int]
    difficulty_levels: Dict[str, int]
    personas_used: Dict[str, int]
    generation_time: float
    model_used: str
    knowledge_graph_stats: Dict[str, Any]
    ollama_available: bool
    total_themes_extracted: int


class PersonaFactory:
    """Factory for creating question generation personas."""
    
    @staticmethod
    def create_personas() -> List[Persona]:
        """Create the two requested personas."""
        return [
            Persona(
                name="Research Scientist",
                role_description="An intelligent researcher interested in multi-layered concepts, complex relationships, and deep analytical questions that require connecting multiple information sources.",
                question_style="Analytical and investigative, focusing on relationships, implications, and multi-step reasoning",
                example_questions=[
                    "How do the approaches described in these documents complement each other in addressing complex challenges?",
                    "What are the underlying connections between these concepts and their broader implications?",
                    "How might the principles discussed in one context apply to the scenarios described in another?",
                    "What are the comparative advantages and limitations of these different methodologies?"
                ]
            ),
            Persona(
                name="Basic Googler",
                role_description="A general user seeking straightforward, factual information with simple, direct questions like 'what is X' or 'how does Y work'.",
                question_style="Simple, direct, and factual - seeking basic definitions and explanations",
                example_questions=[
                    "What is a transformer in AI?",
                    "How do transformers work?",
                    "What are the main features of this technology?",
                    "Who developed this approach?",
                    "When was this method introduced?"
                ]
            )
        ]


class ThemeExtractor:
    """Extract meaningful themes from node metadata for RAGAS-style question generation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_themes_per_node = config.get('question_generation', {}).get('max_themes_per_node', 6)
        
        # Define meaningful theme patterns and stopwords
        self.theme_stopwords = {
            # Generic words that aren't thematic
            'one', 'two', 'three', 'many', 'some', 'all', 'most', 'few', 'several',
            'example', 'examples', 'case', 'cases', 'instance', 'instances',
            'method', 'methods', 'approach', 'approaches', 'technique', 'techniques',
            'way', 'ways', 'means', 'manner', 'form', 'forms',
            'use', 'used', 'using', 'usage', 'application', 'applications',
            'system', 'systems', 'model', 'models', 'framework', 'frameworks',
            'type', 'types', 'kind', 'kinds', 'category', 'categories',
            'part', 'parts', 'component', 'components', 'element', 'elements',
            'feature', 'features', 'property', 'properties', 'attribute', 'attributes',
            'result', 'results', 'outcome', 'outcomes', 'output', 'outputs',
            'input', 'inputs', 'data', 'information', 'knowledge',
            'process', 'processes', 'procedure', 'procedures', 'step', 'steps',
            'function', 'functions', 'operation', 'operations', 'task', 'tasks',
            'problem', 'problems', 'issue', 'issues', 'challenge', 'challenges',
            'solution', 'solutions', 'answer', 'answers', 'response', 'responses',
            'analysis', 'evaluation', 'assessment', 'measurement', 'comparison',
            'study', 'research', 'investigation', 'examination', 'review',
            'paper', 'article', 'document', 'text', 'content', 'material',
            'value', 'values', 'number', 'numbers', 'amount', 'quantity',
            'level', 'levels', 'degree', 'degrees', 'rate', 'rates',
            'time', 'times', 'period', 'periods', 'duration', 'moment',
            'set', 'sets', 'group', 'groups', 'class', 'classes',
            'field', 'fields', 'area', 'areas', 'domain', 'domains',
            'based', 'related', 'associated', 'connected', 'linked',
            'important', 'significant', 'relevant', 'useful', 'effective',
            'different', 'various', 'multiple', 'diverse', 'alternative',
            'new', 'recent', 'current', 'modern', 'latest', 'advanced',
            'traditional', 'conventional', 'standard', 'common', 'typical',
            'general', 'specific', 'particular', 'special', 'unique',
            'main', 'primary', 'principal', 'major', 'key', 'central',
            'basic', 'fundamental', 'essential', 'core', 'critical'
        }
    
    def extract_themes_from_node(self, node: KGNode) -> List[str]:
        """Extract meaningful themes (entities + filtered keyphrases) from a node."""
        themes = []
        
        # Extract from entities (these are usually good themes)
        entities = node.properties.get('entities', {})
        priority_types = ['PERSON', 'ORG', 'PRODUCT', 'EVENT']
        other_types = ['GPE', 'MISC']
        
        # Add high-priority entities first
        for entity_type in priority_types:
            entity_list = entities.get(entity_type, [])
            for entity in entity_list[:3]:  # Max 3 per type
                if self._is_meaningful_theme(entity):
                    themes.append(entity)
        
        # Add other entities if we need more themes
        for entity_type in other_types:
            if len(themes) >= self.max_themes_per_node:
                break
            entity_list = entities.get(entity_type, [])
            for entity in entity_list[:2]:  # Max 2 per type
                if self._is_meaningful_theme(entity):
                    themes.append(entity)
        
        # Add filtered keyphrases to fill remaining slots
        keyphrases = node.properties.get('keyphrases', [])
        remaining_slots = self.max_themes_per_node - len(themes)
        
        meaningful_keyphrases = []
        for phrase in keyphrases:
            if self._is_meaningful_theme(phrase):
                meaningful_keyphrases.append(phrase)
        
        themes.extend(meaningful_keyphrases[:remaining_slots])
        
        # Clean and deduplicate themes
        cleaned_themes = []
        seen = set()
        
        for theme in themes:
            if isinstance(theme, str) and len(theme.strip()) > 2:
                theme_clean = theme.strip().lower()
                if theme_clean not in seen:
                    cleaned_themes.append(theme.strip())
                    seen.add(theme_clean)
        
        return cleaned_themes[:self.max_themes_per_node]
    
    def _is_meaningful_theme(self, theme: str) -> bool:
        """Check if a theme is meaningful (not a generic stopword)."""
        if not isinstance(theme, str) or len(theme.strip()) < 3:
            return False
        
        theme_lower = theme.strip().lower()
        
        # Reject if it's a stopword
        if theme_lower in self.theme_stopwords:
            return False
        
        # Reject if it's all stopwords
        words = theme_lower.split()
        if all(word in self.theme_stopwords for word in words):
            return False
        
        # Accept if it contains proper nouns (likely entities)
        if any(word[0].isupper() for word in theme.split()):
            return True
        
        # Accept technical terms (contains specific patterns)
        technical_patterns = [
            'learning', 'network', 'algorithm', 'neural', 'deep', 'machine',
            'regression', 'classification', 'clustering', 'optimization',
            'probability', 'statistical', 'bayesian', 'supervised', 'unsupervised',
            'reinforcement', 'training', 'validation', 'testing', 'cross-validation',
            'accuracy', 'precision', 'recall', 'f1', 'auc', 'roc',
            'gradient', 'descent', 'backpropagation', 'convolution',
            'transformer', 'attention', 'embedding', 'vector',
            'classification', 'prediction', 'inference', 'modeling'
        ]
        
        if any(pattern in theme_lower for pattern in technical_patterns):
            return True
        
        # Accept multi-word phrases (more likely to be meaningful)
        if len(words) >= 2 and len(words) <= 4:
            # At least one word should not be a stopword
            if any(word not in self.theme_stopwords for word in words):
                return True
        
        # Accept acronyms (likely meaningful)
        if theme.isupper() and len(theme) >= 2 and len(theme) <= 5:
            return True
        
        # Reject everything else
        return False
    
    def find_shared_themes(self, nodes: List[KGNode]) -> List[str]:
        """Find themes shared across multiple nodes, with fallback to best individual themes."""
        if len(nodes) < 2:
            return self.extract_themes_from_node(nodes[0]) if nodes else []
        
        # Get themes from all nodes
        all_themes = []
        node_themes = []
        
        for node in nodes:
            themes = self.extract_themes_from_node(node)
            node_themes.append(set(t.lower() for t in themes))
            all_themes.extend(themes)
        
        # Find intersection of themes (shared across nodes)
        shared_themes = set(node_themes[0])
        for themes in node_themes[1:]:
            shared_themes &= themes
        
        # Find original case versions of shared themes
        result = []
        for theme in all_themes:
            if theme.lower() in shared_themes and theme not in result:
                result.append(theme)
        
        # If no shared themes, combine best themes from all nodes
        if not result:
            # Get top themes from each node and combine
            combined_themes = []
            for node in nodes:
                node_themes_list = self.extract_themes_from_node(node)
                combined_themes.extend(node_themes_list[:3])  # Top 3 from each node
            
            # Deduplicate while preserving order
            seen = set()
            for theme in combined_themes:
                if theme.lower() not in seen:
                    result.append(theme)
                    seen.add(theme.lower())
        
        return result[:self.max_themes_per_node]


class StructuredPromptGenerator:
    """Generate RAGAS-style structured prompts for Ollama."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.theme_extractor = ThemeExtractor(config)
    
    def create_multi_hop_context(self, nodes: List[KGNode]) -> List[str]:
        """Create RAGAS-style multi-hop context formatting."""
        contexts = []
        for i, node in enumerate(nodes):
            hop_label = f"<{i+1}-hop>"
            context_text = node.properties.get('text', node.properties.get('page_content', ''))
            context = f"{hop_label}\n\n{context_text}"
            contexts.append(context)
        return contexts
    
    def generate_entity_bridge_prompt(self, scenario: QuestionScenario) -> str:
        """Generate structured prompt for entity bridge questions."""
        contexts = self.create_multi_hop_context(scenario.nodes)
        shared_themes = scenario.themes[:3]  # Limit to top 3 themes
        
        prompt = f"""You are a {scenario.persona.name}. {scenario.persona.role_description}

Generate a {scenario.length.value.lower()} question connecting these contexts through shared concepts.

CONTEXT 1: {scenario.nodes[0].properties.get('text', '')[:300]}...

CONTEXT 2: {scenario.nodes[1].properties.get('text', '')[:300]}...

SHARED CONCEPTS: {', '.join(shared_themes[:2])}

Create a question requiring both contexts that focuses on: {', '.join(shared_themes[:2])}

Question:"""
        
        return prompt
    
    def generate_concept_similarity_prompt(self, scenario: QuestionScenario) -> str:
        """Generate structured prompt for concept similarity questions."""
        contexts = self.create_multi_hop_context(scenario.nodes)
        themes = scenario.themes[:4]  # More themes for concept similarity
        
        prompt = f"""You are a {scenario.persona.name}. {scenario.persona.role_description}

Generate a question about relationships between these similar contexts.

CONTEXT 1: {scenario.nodes[0].properties.get('text', '')[:300]}...

CONTEXT 2: {scenario.nodes[1].properties.get('text', '')[:300]}...

KEY CONCEPTS: {', '.join(themes[:3])}

Create a question exploring how these concepts relate across contexts.

Question:"""
        
        return prompt
    
    def generate_hierarchical_prompt(self, scenario: QuestionScenario) -> str:
        """Generate structured prompt for hierarchical questions."""
        contexts = self.create_multi_hop_context(scenario.nodes)
        themes = scenario.themes[:3]
        
        # Identify hierarchy levels
        levels = [node.hierarchy_level for node in scenario.nodes]
        level_desc = f"document-level (level {min(levels)}) to sentence-level (level {max(levels)})"
        
        prompt = f"""You are a {scenario.persona.name}. {scenario.persona.role_description}

Generate a question connecting general and specific information.

GENERAL CONTEXT: {scenario.nodes[0].properties.get('text', '')[:300]}...

SPECIFIC DETAIL: {scenario.nodes[1].properties.get('text', '')[:200]}...

KEY CONCEPTS: {', '.join(themes[:2])}

Create a question requiring both general and specific information.

Question:"""
        
        return prompt
    
    def generate_single_hop_prompt(self, scenario: QuestionScenario) -> str:
        """Generate structured prompt for single-hop questions."""
        node = scenario.nodes[0]
        context = node.properties.get('text', node.properties.get('page_content', ''))
        themes = scenario.themes[:2]  # Fewer themes for single-hop
        
        prompt = f"""You are a {scenario.persona.name}. {scenario.persona.role_description}

Generate a simple factual question about this text.

CONTEXT: {context[:400]}...

KEY CONCEPTS: {', '.join(themes[:2])}

Create a {scenario.style.value.lower()} question about: {', '.join(themes[:2])}

Question:"""
        
        return prompt


class NodeSelectionStrategy:
    """Base class for intelligent node selection strategies."""
    
    def __init__(self, kg: KnowledgeGraph, config: Dict[str, Any], logger: logging.Logger):
        self.kg = kg
        self.config = config
        self.logger = logger
        self.theme_extractor = ThemeExtractor(config)
    
    def select_scenarios(self, num_questions: int, personas: List[Persona]) -> List[QuestionScenario]:
        """Select scenarios for question generation."""
        raise NotImplementedError


class EntityBridgeStrategy(NodeSelectionStrategy):
    """Find chunks connected via entity overlap but NOT cosine similarity."""
    
    def select_scenarios(self, num_questions: int, personas: List[Persona]) -> List[QuestionScenario]:
        """Find entity bridge candidates for multi-hop entity questions."""
        scenarios = []
        
        chunk_nodes = self.kg.get_nodes_by_type('CHUNK')
        if len(chunk_nodes) < 2:
            return scenarios
        
        # Find chunks with entity overlap but no cosine similarity
        for _ in range(num_questions):
            # Get random chunk with entities
            candidates = [n for n in chunk_nodes if n.properties.get('entities', {})]
            if len(candidates) < 2:
                break
                
            source_node = random.choice(candidates)
            
            # Find entity neighbors that are NOT cosine similar
            entity_neighbors = self.kg.get_neighbors(source_node.id, ['entity_overlap'])
            cosine_neighbors = set(n.id for n in self.kg.get_neighbors(source_node.id, ['cosine_similarity']))
            
            # Filter out cosine similar neighbors
            pure_entity_neighbors = [n for n in entity_neighbors if n.id not in cosine_neighbors]
            
            if pure_entity_neighbors:
                target_node = random.choice(pure_entity_neighbors)
                nodes = [source_node, target_node]
                
                # Extract shared themes
                themes = self.theme_extractor.find_shared_themes(nodes)
                
                # Create scenario with random persona and style
                persona = random.choice(personas)
                style = random.choice(list(QueryStyle))
                length = random.choice(list(QueryLength))
                
                scenario = QuestionScenario(
                    nodes=nodes,
                    themes=themes,
                    persona=persona,
                    style=style,
                    length=length,
                    strategy="entity_bridge",
                    metadata={
                        'shared_themes': themes,
                        'cosine_similar': False,
                        'entity_overlap': True,
                        'question_type': 'entity_bridge',
                        'expected_advantage': 'semantic_traversal',
                        'difficulty': 'medium'
                    }
                )
                scenarios.append(scenario)
        
        return scenarios


class ConceptSimilarityStrategy(NodeSelectionStrategy):
    """Find chunks with high cosine similarity for concept relationship questions."""
    
    def select_scenarios(self, num_questions: int, personas: List[Persona]) -> List[QuestionScenario]:
        """Find concept similarity candidates for semantic traversal questions."""
        scenarios = []
        
        chunk_nodes = self.kg.get_nodes_by_type('CHUNK')
        if len(chunk_nodes) < 2:
            return scenarios
        
        # Find chunks with high cosine similarity
        for _ in range(num_questions):
            source_node = random.choice(chunk_nodes)
            
            # Get cosine similar neighbors with high similarity
            best_neighbor = None
            best_similarity = 0.0
            
            for rel in self.kg.relationships:
                if (rel.source == source_node.id and rel.type == 'cosine_similarity' and 
                    rel.weight > best_similarity and rel.weight > 0.5):  # High similarity threshold
                    target_node = self.kg.get_node(rel.target)
                    if target_node:
                        best_neighbor = target_node
                        best_similarity = rel.weight
            
            if best_neighbor:
                nodes = [source_node, best_neighbor]
                
                # Extract combined themes from both nodes
                themes = self.theme_extractor.find_shared_themes(nodes)
                
                # Create scenario
                persona = random.choice(personas)
                style = random.choice(list(QueryStyle))
                length = random.choice(list(QueryLength))
                
                scenario = QuestionScenario(
                    nodes=nodes,
                    themes=themes,
                    persona=persona,
                    style=style,
                    length=length,
                    strategy="concept_similarity",
                    metadata={
                        'cosine_similarity': best_similarity,
                        'question_type': 'concept_similarity',
                        'expected_advantage': 'semantic_traversal',
                        'difficulty': 'easy'
                    }
                )
                scenarios.append(scenario)
        
        return scenarios


class HierarchicalStrategy(NodeSelectionStrategy):
    """Use parent-child relationships for multi-granularity questions."""
    
    def select_scenarios(self, num_questions: int, personas: List[Persona]) -> List[QuestionScenario]:
        """Find hierarchical candidates for granularity navigation questions."""
        scenarios = []
        
        # Strategy: Document + one of its sentences (skipping chunk level)
        document_nodes = self.kg.get_nodes_by_type('DOCUMENT')
        
        for _ in range(num_questions):
            if not document_nodes:
                break
                
            doc_node = random.choice(document_nodes)
            
            # Find chunks belonging to this document
            doc_chunks = self.kg.get_children(doc_node.id)
            
            if doc_chunks:
                # Get sentences from one of the chunks
                chunk_node = random.choice(doc_chunks)
                chunk_sentences = self.kg.get_children(chunk_node.id)
                
                if chunk_sentences:
                    sentence_node = random.choice(chunk_sentences)
                    nodes = [doc_node, sentence_node]
                    
                    # Extract themes across hierarchy levels
                    themes = self.theme_extractor.find_shared_themes(nodes)
                    
                    # Create scenario
                    persona = random.choice(personas)
                    style = random.choice(list(QueryStyle))
                    length = random.choice(list(QueryLength))
                    
                    scenario = QuestionScenario(
                        nodes=nodes,
                        themes=themes,
                        persona=persona,
                        style=style,
                        length=length,
                        strategy="hierarchical",
                        metadata={
                            'hierarchy_levels': [0, 2],  # Document to sentence
                            'intermediate_chunk': chunk_node.id,
                            'question_type': 'hierarchical',
                            'expected_advantage': 'semantic_traversal',
                            'difficulty': 'hard'
                        }
                    )
                    scenarios.append(scenario)
        
        return scenarios


class SingleHopStrategy(NodeSelectionStrategy):
    """Select individual chunks or sentences for simple factual questions."""
    
    def select_scenarios(self, num_questions: int, personas: List[Persona]) -> List[QuestionScenario]:
        """Find single node candidates for baseline comparison questions."""
        scenarios = []
        
        # Mix of chunks and sentences
        chunk_nodes = self.kg.get_nodes_by_type('CHUNK')
        sentence_nodes = self.kg.get_nodes_by_type('SENTENCE')
        
        all_candidates = chunk_nodes + sentence_nodes
        
        for _ in range(min(num_questions, len(all_candidates))):
            node = random.choice(all_candidates)
            
            # Prefer nodes with good entity/keyphrase content
            entities = node.properties.get('entities', {})
            keyphrases = node.properties.get('keyphrases', [])
            
            if entities or keyphrases:
                # Extract themes from single node
                themes = self.theme_extractor.extract_themes_from_node(node)
                
                # Prefer "Basic Googler" persona for single-hop questions
                persona = random.choice([p for p in personas if "Basic" in p.name] or personas)
                style = random.choice([QueryStyle.CONVERSATIONAL, QueryStyle.FORMAL])
                length = random.choice([QueryLength.SHORT, QueryLength.MEDIUM])
                
                scenario = QuestionScenario(
                    nodes=[node],
                    themes=themes,
                    persona=persona,
                    style=style,
                    length=length,
                    strategy="single_hop",
                    metadata={
                        'node_type': node.type,
                        'has_entities': bool(entities),
                        'has_keyphrases': bool(keyphrases),
                        'question_type': 'single_hop',
                        'expected_advantage': 'baseline_vector',
                        'difficulty': 'easy'
                    }
                )
                scenarios.append(scenario)
        
        return scenarios


class OllamaQuestionGenerator:
    """Generates questions using Ollama with RAGAS-style structured prompts."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.ollama_config = config.get('ollama', {})
        self.model = self.ollama_config.get('model', 'llama3.1:8b')
        self.available = self._test_ollama_connection()
        self.prompt_generator = StructuredPromptGenerator(config)
    
    def _test_ollama_connection(self) -> bool:
        """Test if Ollama is available and model is ready."""
        if not OLLAMA_AVAILABLE:
            return False
        
        try:
            models = ollama.list()
            available_models = [model.model for model in models.models]
            
            if self.model in available_models:
                self.logger.info(f"✅ Ollama model {self.model} available for question generation")
                return True
            else:
                self.logger.warning(f"⚠️  Ollama model {self.model} not found. Available: {available_models}")
                return False
        except Exception as e:
            self.logger.warning(f"⚠️  Ollama not available: {e}")
            return False
    
    def generate_question_and_answer(self, scenario: QuestionScenario) -> Tuple[Optional[str], Optional[str]]:
        """Generate question and answer based on scenario."""
        if not self.available:
            return self._generate_fallback_question(scenario), None
        
        try:
            # Generate structured prompt based on strategy
            if scenario.strategy == "entity_bridge":
                prompt = self.prompt_generator.generate_entity_bridge_prompt(scenario)
            elif scenario.strategy == "concept_similarity":
                prompt = self.prompt_generator.generate_concept_similarity_prompt(scenario)
            elif scenario.strategy == "hierarchical":
                prompt = self.prompt_generator.generate_hierarchical_prompt(scenario)
            elif scenario.strategy == "single_hop":
                prompt = self.prompt_generator.generate_single_hop_prompt(scenario)
            else:
                prompt = self._generate_generic_prompt(scenario)
            
            # Generate question using Ollama
            # Add timeout and better error handling
            self.logger.info(f"🤖 Generating question with {self.model} (prompt length: {len(prompt)} chars)")
            
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.3,  # Slightly creative but consistent
                    "num_predict": 150,  # Reduced for faster generation
                    "stop": ["\n\n", "Answer:", "Explanation:", "Context:"],
                    "timeout": 30  # 30 second timeout
                }
            )
            
            self.logger.info(f"✅ Generated response: {response['response'][:100]}...")
            
            question = response['response'].strip()
            
            # Clean up common artifacts
            question = self._clean_question(question)
            
            # Validate question
            if self._is_valid_question(question):
                # Try to generate answer if this is multi-hop
                answer = None
                if len(scenario.nodes) > 1:
                    answer = self._generate_reference_answer(question, scenario)
                return question, answer
            else:
                self.logger.warning(f"Generated invalid question: {question}")
                return self._generate_fallback_question(scenario), None
        
        except Exception as e:
            self.logger.warning(f"Ollama question generation failed: {e}")
            self.logger.warning(f"Prompt that failed (first 500 chars): {prompt[:500]}...")
            return self._generate_fallback_question(scenario), None
    
    def _generate_reference_answer(self, question: str, scenario: QuestionScenario) -> Optional[str]:
        """Generate reference answer for multi-hop questions."""
        try:
            contexts = self.prompt_generator.create_multi_hop_context(scenario.nodes)
            combined_context = "\n\n".join(contexts)
            
            # Simplified answer prompt to avoid timeouts
            answer_prompt = f"""Answer this question using the provided context.

CONTEXT: {combined_context[:800]}...

QUESTION: {question}

Answer:"""  # Shortened context to avoid timeouts
            
            response = ollama.generate(
                model=self.model,
                prompt=answer_prompt,
                options={
                    "temperature": 0.1,  # Very consistent for answers
                    "num_predict": 150,  # Reduced for faster generation
                    "stop": ["\n\nQuestion:", "Context:"],
                    "timeout": 20  # Shorter timeout for answers
                }
            )
            
            answer = response['response'].strip()
            return answer if len(answer) > 10 else None
            
        except Exception as e:
            self.logger.warning(f"Failed to generate reference answer: {e}")
            return None
    
    def _clean_question(self, question: str) -> str:
        """Clean and normalize generated question."""
        # Remove common prefixes
        prefixes_to_remove = [
            "here is a question", "here's a question", "question:", "the question is",
            "a good question would be", "i would ask", "one could ask", "generate only the question"
        ]
        
        question_lower = question.lower().strip()
        for prefix in prefixes_to_remove:
            if question_lower.startswith(prefix):
                prefix_end = len(prefix)
                question = question[prefix_end:].strip()
                if question.startswith(':'):
                    question = question[1:].strip()
                break
        
        # Remove common suffixes
        suffixes_to_remove = ["(no explanations)", "(no explanation)", "no explanations needed"]
        for suffix in suffixes_to_remove:
            if question.lower().endswith(suffix):
                question = question[:-len(suffix)].strip()
        
        # Ensure question ends with question mark
        question = question.strip()
        if question and not question.endswith('?'):
            question += '?'
        
        # Capitalize first letter
        if question:
            question = question[0].upper() + question[1:]
        
        return question
    
    def _is_valid_question(self, question: str) -> bool:
        """Validate generated question quality."""
        if not question or len(question) < 10:
            return False
        
        # Must end with question mark
        if not question.endswith('?'):
            return False
        
        # Should contain question words or be imperative
        question_indicators = ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'explain', 'describe', 'compare', 'analyze']
        question_lower = question.lower()
        
        has_question_indicator = any(indicator in question_lower for indicator in question_indicators)
        
        return has_question_indicator
    
    def _generate_fallback_question(self, scenario: QuestionScenario) -> str:
        """Generate fallback question when Ollama unavailable."""
        node = scenario.nodes[0]
        themes = scenario.themes[:2]
        
        # Use persona style for fallback
        if "Basic" in scenario.persona.name:
            # Simple factual questions
            if themes:
                return f"What is {themes[0]}?"
            else:
                return "What are the main concepts discussed in this text?"
        else:
            # Research-style questions
            if len(scenario.nodes) > 1 and themes:
                return f"How are {themes[0]} and {themes[1] if len(themes) > 1 else 'related concepts'} connected across different contexts?"
            elif themes:
                return f"What are the implications of {themes[0]} in this context?"
            else:
                return "What are the key relationships and concepts discussed?"
    
    def _generate_generic_prompt(self, scenario: QuestionScenario) -> str:
        """Generate generic prompt for unknown strategies."""
        contexts = self.prompt_generator.create_multi_hop_context(scenario.nodes)
        combined_context = '\n\n'.join(contexts)
        
        prompt = f"""You are a {scenario.persona.name}. {scenario.persona.role_description}

Generate a {scenario.length.value.lower()} {scenario.style.value.lower()} question based on this content.

CONTENT:
{combined_context}

Generate only the question (no explanations):"""
        
        return prompt


class KnowledgeGraphQuestionGenerator:
    """Main engine for generating evaluation questions from knowledge graphs using RAGAS-style approach."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize the knowledge graph question generator."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.qg_config = config.get('question_generation', {})
        
        # Initialize components
        self.personas = PersonaFactory.create_personas()
        self.ollama_generator = OllamaQuestionGenerator(config, self.logger)
        self.theme_extractor = ThemeExtractor(config)
        
        # Question type distribution (from config or defaults)
        self.question_distribution = self.qg_config.get('question_types', {
            'entity_bridge': 0.4,      # 40%
            'concept_similarity': 0.3,  # 30%
            'hierarchical': 0.2,       # 20%
            'single_hop': 0.1          # 10%
        })
        
        self.logger.info(f"🎯 Initialized Knowledge Graph Question Generator with {len(self.personas)} personas")
        for persona in self.personas:
            self.logger.info(f"   👤 {persona.name}: {persona.role_description}")
    
    def generate_questions(self, knowledge_graph: KnowledgeGraph, force_recompute: bool = False) -> List[EvaluationQuestion]:
        """Generate evaluation questions from knowledge graph using RAGAS-style approach."""
        # Check cache
        cache_path = self._get_cache_path()
        if not force_recompute and self._is_cache_valid(cache_path, knowledge_graph):
            self.logger.info("📂 Loading cached questions")
            return self._load_cached_questions(cache_path)
        
        self.logger.info("🎯 Generating fresh questions using RAGAS-style structured approach")
        start_time = time.time()
        
        # Calculate question counts by type
        target_questions = self.qg_config.get('target_questions', 50)
        question_counts = self._calculate_question_counts(target_questions)
        
        self.logger.info(f"📊 Question distribution: {question_counts}")
        
        # Initialize selection strategies
        strategies = {
            'entity_bridge': EntityBridgeStrategy(knowledge_graph, self.config, self.logger),
            'concept_similarity': ConceptSimilarityStrategy(knowledge_graph, self.config, self.logger),
            'hierarchical': HierarchicalStrategy(knowledge_graph, self.config, self.logger),
            'single_hop': SingleHopStrategy(knowledge_graph, self.config, self.logger)
        }
        
        # Generate scenarios for each strategy
        all_scenarios = []
        generation_stats = {}
        total_themes_extracted = 0
        
        for strategy_name, count in question_counts.items():
            if count == 0:
                continue
                
            self.logger.info(f"🔄 Generating {count} {strategy_name} scenarios...")
            
            strategy = strategies[strategy_name]
            scenarios = strategy.select_scenarios(count, self.personas)
            
            all_scenarios.extend(scenarios)
            generation_stats[strategy_name] = len(scenarios)
            
            # Count themes extracted
            for scenario in scenarios:
                total_themes_extracted += len(scenario.themes)
            
            self.logger.info(f"✅ Generated {len(scenarios)} {strategy_name} scenarios")
        
        # Generate questions from scenarios with better progress tracking
        self.logger.info(f"🤖 Generating questions from {len(all_scenarios)} scenarios using Ollama...")
        
        all_questions = []
        failed_scenarios = 0
        
        for i, scenario in enumerate(all_scenarios):
            self.logger.info(f"   Processing scenario {i+1}/{len(all_scenarios)} ({scenario.strategy})...")
            
            try:
                question = self._generate_question_from_scenario(scenario)
                if question:
                    all_questions.append(question)
                    self.logger.info(f"   ✅ Generated question {len(all_questions)}: {question.question[:60]}...")
                else:
                    failed_scenarios += 1
                    self.logger.warning(f"   ❌ Failed to generate question for scenario {i+1}")
            except Exception as e:
                failed_scenarios += 1
                self.logger.error(f"   ❌ Error generating question for scenario {i+1}: {e}")
                continue
        
        self.logger.info(f"📊 Question generation summary: {len(all_questions)} successful, {failed_scenarios} failed")
        
        generation_time = time.time() - start_time
        
        # Create metadata
        metadata = QuestionGenerationMetadata(
            created_at=datetime.now().isoformat(),
            total_questions=len(all_questions),
            generation_strategies=generation_stats,
            question_types={},
            expected_advantages={},
            difficulty_levels={},
            personas_used={},
            generation_time=generation_time,
            model_used=self.ollama_generator.model,
            knowledge_graph_stats={
                'total_nodes': len(knowledge_graph.nodes),
                'total_relationships': len(knowledge_graph.relationships)
            },
            ollama_available=self.ollama_generator.available,
            total_themes_extracted=total_themes_extracted
        )
        
        # Count actual distributions
        for question in all_questions:
            metadata.question_types[question.question_type] = metadata.question_types.get(question.question_type, 0) + 1
            metadata.expected_advantages[question.expected_advantage] = metadata.expected_advantages.get(question.expected_advantage, 0) + 1
            metadata.difficulty_levels[question.difficulty_level] = metadata.difficulty_levels.get(question.difficulty_level, 0) + 1
            metadata.personas_used[question.persona_used] = metadata.personas_used.get(question.persona_used, 0) + 1
        
        # Cache questions
        self._cache_questions(cache_path, all_questions, metadata)
        
        self.logger.info(f"✅ Question generation completed: {len(all_questions)} questions in {generation_time:.2f}s")
        self.logger.info(f"🎭 Personas used: {metadata.personas_used}")
        self.logger.info(f"🏷️  Themes extracted: {total_themes_extracted} total themes")
        
        return all_questions
    
    def _calculate_question_counts(self, target_questions: int) -> Dict[str, int]:
        """Calculate how many questions to generate for each type."""
        question_counts = {}
        
        for strategy, proportion in self.question_distribution.items():
            count = int(target_questions * proportion)
            question_counts[strategy] = count
        
        return question_counts
    
    def _generate_question_from_scenario(self, scenario: QuestionScenario) -> Optional[EvaluationQuestion]:
        """Generate an evaluation question from a scenario."""
        start_time = time.time()
        
        # Generate question and answer using Ollama with error handling
        try:
            question_text, reference_answer = self.ollama_generator.generate_question_and_answer(scenario)
            if not question_text:
                self.logger.warning(f"Failed to generate question for {scenario.strategy} scenario")
                return None
        except Exception as e:
            self.logger.error(f"Error in question generation for {scenario.strategy}: {e}")
            return None
        
        generation_time = time.time() - start_time
        
        # Create ground truth contexts (node IDs that should be retrieved)
        ground_truth_contexts = [node.id for node in scenario.nodes]
        primary_context_id = scenario.nodes[0].id
        
        # Create reference contexts (actual text for RAGAS compatibility)
        reference_contexts = []
        for node in scenario.nodes:
            context_text = node.properties.get('text', node.properties.get('page_content', ''))
            reference_contexts.append(context_text)
        
        # Determine relationship types being tested
        relationship_types = []
        if scenario.strategy == "entity_bridge":
            relationship_types = ["entity_overlap"]
        elif scenario.strategy == "concept_similarity":
            relationship_types = ["cosine_similarity"]
        elif scenario.strategy == "hierarchical":
            relationship_types = ["parent", "child"]
        elif scenario.strategy == "single_hop":
            relationship_types = []
        
        # Create question ID
        question_id = f"kg_{scenario.strategy}_{hashlib.md5(question_text.encode()).hexdigest()[:8]}"
        
        # Create evaluation question
        question = EvaluationQuestion(
            question_id=question_id,
            question=question_text,
            reference_answer=reference_answer,
            question_type=scenario.metadata.get('question_type', scenario.strategy),
            expected_advantage=scenario.metadata.get('expected_advantage', 'neutral'),
            difficulty_level=scenario.metadata.get('difficulty', 'medium'),
            ground_truth_contexts=ground_truth_contexts,
            reference_contexts=reference_contexts,
            primary_context_id=primary_context_id,
            generation_strategy=scenario.strategy,
            source_nodes=[node.id for node in scenario.nodes],
            relationship_types=relationship_types,
            themes_used=scenario.themes,
            persona_used=scenario.persona.name,
            model_used=self.ollama_generator.model,
            generation_prompt="",  # Don't store full prompt to save space
            generation_time=generation_time
        )
        
        return question
    
    def _get_cache_path(self) -> Path:
        """Get cache path for questions."""
        data_dir = Path(self.config['directories']['data'])
        return data_dir / "knowledge_graph_questions.json"
    
    def _is_cache_valid(self, cache_path: Path, knowledge_graph: KnowledgeGraph) -> bool:
        """Check if cached questions are valid for current knowledge graph."""
        if not cache_path.exists():
            return False
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            cached_kg_stats = cache_data.get('metadata', {}).get('knowledge_graph_stats', {})
            current_kg_stats = {
                'total_nodes': len(knowledge_graph.nodes),
                'total_relationships': len(knowledge_graph.relationships)
            }
            
            return cached_kg_stats == current_kg_stats
            
        except Exception as e:
            self.logger.warning(f"Failed to validate question cache: {e}")
            return False
    
    def _cache_questions(self, cache_path: Path, questions: List[EvaluationQuestion], metadata: QuestionGenerationMetadata):
        """Cache questions to disk."""
        try:
            cache_data = {
                'metadata': asdict(metadata),
                'questions': [question.to_dict() for question in questions]
            }
            
            cache_path.parent.mkdir(parents=True, exist_ok=True)
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
                question = EvaluationQuestion(**q_data)
                questions.append(question)
            
            return questions
            
        except Exception as e:
            self.logger.error(f"Failed to load cached questions: {e}")
            raise
    
    def get_question_statistics(self, questions: List[EvaluationQuestion]) -> Dict[str, Any]:
        """Get statistics about generated questions."""
        if not questions:
            return {}
        
        stats = {
            'total_questions': len(questions),
            'by_question_type': {},
            'by_expected_advantage': {},
            'by_difficulty_level': {},
            'by_generation_strategy': {},
            'by_persona_used': {},
            'average_generation_time': sum(q.generation_time for q in questions) / len(questions),
            'themes_coverage': {
                'total_themes_used': sum(len(q.themes_used) for q in questions),
                'average_themes_per_question': sum(len(q.themes_used) for q in questions) / len(questions),
                'unique_themes': len(set(theme for q in questions for theme in q.themes_used))
            },
            'ground_truth_coverage': {
                'mean_contexts_per_question': sum(len(q.ground_truth_contexts) for q in questions) / len(questions),
                'total_unique_contexts': len(set(ctx for q in questions for ctx in q.ground_truth_contexts))
            }
        }
        
        # Count distributions
        for question in questions:
            # Question type distribution
            q_type = question.question_type
            stats['by_question_type'][q_type] = stats['by_question_type'].get(q_type, 0) + 1
            
            # Expected advantage distribution
            advantage = question.expected_advantage
            stats['by_expected_advantage'][advantage] = stats['by_expected_advantage'].get(advantage, 0) + 1
            
            # Difficulty distribution
            difficulty = question.difficulty_level
            stats['by_difficulty_level'][difficulty] = stats['by_difficulty_level'].get(difficulty, 0) + 1
            
            # Generation strategy distribution
            strategy = question.generation_strategy
            stats['by_generation_strategy'][strategy] = stats['by_generation_strategy'].get(strategy, 0) + 1
            
            # Persona distribution
            persona = question.persona_used
            stats['by_persona_used'][persona] = stats['by_persona_used'].get(persona, 0) + 1
        
        return stats


# Factory function for backward compatibility
def QuestionEngine(config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> KnowledgeGraphQuestionGenerator:
    """Factory function for creating question generator."""
    return KnowledgeGraphQuestionGenerator(config, logger)
