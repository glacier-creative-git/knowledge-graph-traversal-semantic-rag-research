#!/usr/bin/env python3
"""
Simplified Question Generation Engine
===================================

Minimalist question generation focusing on three core relationship types:
1. Entity overlap (PERSON/ORG/GPE only)
2. Cosine similarity 
3. Hierarchical relationships

Removes themes, keyphrases, and complex metadata extraction for cognitive simplicity.
Lets Ollama discover connections autonomously from high-quality contexts.
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

from knowledge_graph import KnowledgeGraph, KGNode


class QueryStyle(str, Enum):
    """Question styles for diverse generation."""
    FORMAL = "Formal"
    CONVERSATIONAL = "Conversational"
    ANALYTICAL = "Analytical"


@dataclass
class Persona:
    """Simplified persona for question generation."""
    name: str
    role_description: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QuestionScenario:
    """Simplified scenario for question generation."""
    nodes: List[KGNode]
    shared_entities: List[str]  # Only entities, no themes
    persona: Persona
    style: QueryStyle
    strategy: str
    metadata: Dict[str, Any]


@dataclass
class EvaluationQuestion:
    """Container for an evaluation question."""
    question_id: str
    question: str
    reference_answer: Optional[str]
    question_type: str
    expected_advantage: str
    difficulty_level: str
    
    # Ground truth contexts
    ground_truth_contexts: List[str]
    reference_contexts: List[str]
    primary_context_id: str
    
    # Generation metadata
    generation_strategy: str
    source_nodes: List[str]
    relationship_types: List[str]
    entities_used: List[str]  # Changed from themes_used
    persona_used: str
    
    # Ollama metadata
    model_used: str
    generation_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class PersonaFactory:
    """Factory for creating simplified personas."""
    
    @staticmethod
    def create_personas() -> List[Persona]:
        """Create two basic personas."""
        return [
            Persona(
                name="Research Scientist",
                role_description="An intelligent researcher interested in complex relationships and multi-step reasoning that requires connecting multiple information sources."
            ),
            Persona(
                name="Basic Googler", 
                role_description="A general user seeking straightforward, factual information with simple, direct questions."
            )
        ]


class SimplifiedPromptGenerator:
    """Generate simplified prompts that let Ollama discover connections autonomously."""
    
    def generate_entity_bridge_prompt(self, scenario: QuestionScenario) -> str:
        """Generate prompt for entity bridge questions."""
        shared_entities = scenario.shared_entities[:3]
        
        prompt = f"""You are a {scenario.persona.name}. {scenario.persona.role_description}

Generate a question that connects these two contexts through their shared factual elements.

CONTEXT 1: {scenario.nodes[0].properties.get('text', '')[:300]}...

CONTEXT 2: {scenario.nodes[1].properties.get('text', '')[:300]}...

These contexts share: {', '.join(shared_entities)}

Create a question that requires understanding both contexts and their connection.

Question:"""
        
        return prompt
    
    def generate_concept_similarity_prompt(self, scenario: QuestionScenario) -> str:
        """Generate prompt for concept similarity questions."""
        prompt = f"""You are a {scenario.persona.name}. {scenario.persona.role_description}

Generate a question about the relationship between these similar contexts.

CONTEXT 1: {scenario.nodes[0].properties.get('text', '')[:300]}...

CONTEXT 2: {scenario.nodes[1].properties.get('text', '')[:300]}...

These contexts discuss related concepts. Create a question exploring how they relate.

Question:"""
        
        return prompt
    
    def generate_hierarchical_prompt(self, scenario: QuestionScenario) -> str:
        """Generate prompt for hierarchical questions."""
        prompt = f"""You are a {scenario.persona.name}. {scenario.persona.role_description}

Generate a question connecting general and specific information.

GENERAL CONTEXT: {scenario.nodes[0].properties.get('text', '')[:300]}...

SPECIFIC DETAIL: {scenario.nodes[1].properties.get('text', '')[:200]}...

Create a question that requires both general and specific information.

Question:"""
        
        return prompt
    
    def generate_single_hop_prompt(self, scenario: QuestionScenario) -> str:
        """Generate prompt for single-hop questions."""
        node = scenario.nodes[0]
        context = node.properties.get('text', node.properties.get('page_content', ''))
        entities = scenario.shared_entities[:2]
        
        if entities:
            entity_focus = f"Create a {scenario.style.value.lower()} question about: {', '.join(entities)}"
        else:
            entity_focus = "Create a simple factual question about this text."
        
        prompt = f"""You are a {scenario.persona.name}. {scenario.persona.role_description}

Generate a simple factual question about this text.

CONTEXT: {context[:400]}...

{entity_focus}

Question:"""
        
        return prompt


class EntityExtractor:
    """Extract shared entities from nodes (PERSON/ORG/GPE only)."""
    
    def __init__(self):
        self.entity_types = ['PERSON', 'ORG', 'GPE']
    
    def get_shared_entities(self, nodes: List[KGNode]) -> List[str]:
        """Get entities shared across all nodes."""
        if not nodes:
            return []
        
        # Get entity sets for each node
        entity_sets = []
        for node in nodes:
            entities = node.properties.get('entities', {})
            all_entities = set()
            for entity_type in self.entity_types:
                all_entities.update(entities.get(entity_type, []))
            # Normalize to lowercase for comparison
            entity_sets.append({e.lower() for e in all_entities})
        
        # Find intersection
        if len(entity_sets) == 1:
            shared = entity_sets[0]
        else:
            shared = entity_sets[0]
            for entity_set in entity_sets[1:]:
                shared &= entity_set
        
        # Return original case versions (from first node)
        result = []
        if nodes:
            first_node_entities = nodes[0].properties.get('entities', {})
            for entity_type in self.entity_types:
                for entity in first_node_entities.get(entity_type, []):
                    if entity.lower() in shared and entity not in result:
                        result.append(entity)
        
        return result[:5]  # Limit to top 5
    
    def get_node_entities(self, node: KGNode) -> List[str]:
        """Get all entities from a single node."""
        entities = node.properties.get('entities', {})
        all_entities = []
        for entity_type in self.entity_types:
            all_entities.extend(entities.get(entity_type, []))
        return all_entities[:5]  # Limit to top 5


class NodeSelectionStrategy:
    """Base class for node selection strategies."""
    
    def __init__(self, kg: KnowledgeGraph, logger: logging.Logger):
        self.kg = kg
        self.logger = logger
        self.entity_extractor = EntityExtractor()
    
    def select_scenarios(self, num_questions: int, personas: List[Persona]) -> List[QuestionScenario]:
        """Select scenarios for question generation."""
        raise NotImplementedError


class EntityBridgeStrategy(NodeSelectionStrategy):
    """Find chunks connected via entity overlap but NOT cosine similarity."""
    
    def select_scenarios(self, num_questions: int, personas: List[Persona]) -> List[QuestionScenario]:
        """Find entity bridge candidates."""
        scenarios = []
        
        chunk_nodes = [n for n in self.kg.nodes if n.type == 'CHUNK']
        if len(chunk_nodes) < 2:
            return scenarios
        
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
                
                # Extract shared entities
                shared_entities = self.entity_extractor.get_shared_entities(nodes)
                
                # Create scenario
                persona = random.choice(personas)
                style = random.choice(list(QueryStyle))
                
                scenario = QuestionScenario(
                    nodes=nodes,
                    shared_entities=shared_entities,
                    persona=persona,
                    style=style,
                    strategy="entity_bridge",
                    metadata={
                        'question_type': 'entity_bridge',
                        'expected_advantage': 'semantic_traversal',
                        'difficulty': 'medium'
                    }
                )
                scenarios.append(scenario)
        
        return scenarios


class ConceptSimilarityStrategy(NodeSelectionStrategy):
    """Find chunks with high cosine similarity."""
    
    def select_scenarios(self, num_questions: int, personas: List[Persona]) -> List[QuestionScenario]:
        """Find concept similarity candidates."""
        scenarios = []
        
        chunk_nodes = [n for n in self.kg.nodes if n.type == 'CHUNK']
        if len(chunk_nodes) < 2:
            return scenarios
        
        for _ in range(num_questions):
            source_node = random.choice(chunk_nodes)
            
            # Get cosine similar neighbors with high similarity
            best_neighbor = None
            best_similarity = 0.0
            
            for rel in self.kg.relationships:
                if (rel.source == source_node.id and rel.type == 'cosine_similarity' and 
                    rel.weight > best_similarity and rel.weight > 0.5):
                    target_node = self.kg.get_node(rel.target)
                    if target_node:
                        best_neighbor = target_node
                        best_similarity = rel.weight
            
            if best_neighbor:
                nodes = [source_node, best_neighbor]
                
                # Extract entities from both nodes
                shared_entities = self.entity_extractor.get_shared_entities(nodes)
                
                # Create scenario
                persona = random.choice(personas)
                style = random.choice(list(QueryStyle))
                
                scenario = QuestionScenario(
                    nodes=nodes,
                    shared_entities=shared_entities,
                    persona=persona,
                    style=style,
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
    """Use parent-child relationships."""
    
    def select_scenarios(self, num_questions: int, personas: List[Persona]) -> List[QuestionScenario]:
        """Find hierarchical candidates."""
        scenarios = []
        
        document_nodes = [n for n in self.kg.nodes if n.type == 'DOCUMENT']
        
        for _ in range(num_questions):
            if not document_nodes:
                break
                
            doc_node = random.choice(document_nodes)
            
            # Find chunks belonging to this document
            doc_chunks = self.kg.get_neighbors(doc_node.id, ['contains'])
            
            if doc_chunks:
                chunk_node = random.choice(doc_chunks)
                nodes = [doc_node, chunk_node]
                
                # Extract entities across hierarchy levels
                shared_entities = self.entity_extractor.get_shared_entities(nodes)
                
                # Create scenario
                persona = random.choice(personas)
                style = random.choice(list(QueryStyle))
                
                scenario = QuestionScenario(
                    nodes=nodes,
                    shared_entities=shared_entities,
                    persona=persona,
                    style=style,
                    strategy="hierarchical",
                    metadata={
                        'question_type': 'hierarchical',
                        'expected_advantage': 'semantic_traversal',
                        'difficulty': 'hard'
                    }
                )
                scenarios.append(scenario)
        
        return scenarios


class SingleHopStrategy(NodeSelectionStrategy):
    """Select individual chunks for simple questions."""
    
    def select_scenarios(self, num_questions: int, personas: List[Persona]) -> List[QuestionScenario]:
        """Find single node candidates."""
        scenarios = []
        
        chunk_nodes = [n for n in self.kg.nodes if n.type == 'CHUNK']
        
        for _ in range(min(num_questions, len(chunk_nodes))):
            node = random.choice(chunk_nodes)
            
            # Extract entities from single node
            entities = self.entity_extractor.get_node_entities(node)
            
            # Prefer "Basic Googler" persona for single-hop questions
            persona = random.choice([p for p in personas if "Basic" in p.name] or personas)
            style = random.choice([QueryStyle.CONVERSATIONAL, QueryStyle.FORMAL])
            
            scenario = QuestionScenario(
                nodes=[node],
                shared_entities=entities,
                persona=persona,
                style=style,
                strategy="single_hop",
                metadata={
                    'question_type': 'single_hop',
                    'expected_advantage': 'baseline_vector',
                    'difficulty': 'easy'
                }
            )
            scenarios.append(scenario)
        
        return scenarios


class OllamaQuestionGenerator:
    """Generates questions using Ollama with simplified prompts."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.model = config.get('ollama', {}).get('model', 'llama3.1:8b')
        self.available = self._test_ollama_connection()
        self.prompt_generator = SimplifiedPromptGenerator()
    
    def _test_ollama_connection(self) -> bool:
        """Test if Ollama is available."""
        if not OLLAMA_AVAILABLE:
            return False
        
        try:
            models = ollama.list()
            available_models = [model.model for model in models.models]
            
            if self.model in available_models:
                self.logger.info(f"✅ Ollama model {self.model} available")
                return True
            else:
                self.logger.warning(f"⚠️  Ollama model {self.model} not found. Available: {available_models}")
                return False
        except Exception as e:
            self.logger.warning(f"⚠️  Ollama not available: {e}")
            return False
    
    def generate_question_and_answer(self, scenario: QuestionScenario) -> Tuple[Optional[str], Optional[str]]:
        """Generate question based on scenario."""
        if not self.available:
            return self._generate_fallback_question(scenario), None
        
        try:
            # Generate prompt based on strategy
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
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "num_predict": 150,
                    "stop": ["\n\n", "Answer:", "Explanation:", "Context:"],
                    "timeout": 30
                }
            )
            
            question = response['response'].strip()
            question = self._clean_question(question)
            
            if self._is_valid_question(question):
                return question, None
            else:
                return self._generate_fallback_question(scenario), None
        
        except Exception as e:
            self.logger.warning(f"Ollama question generation failed: {e}")
            return self._generate_fallback_question(scenario), None
    
    def _clean_question(self, question: str) -> str:
        """Clean and normalize generated question."""
        # Remove common prefixes
        prefixes_to_remove = [
            "here is a question", "here's a question", "question:", "the question is",
            "a good question would be", "i would ask", "one could ask"
        ]
        
        question_lower = question.lower().strip()
        for prefix in prefixes_to_remove:
            if question_lower.startswith(prefix):
                question = question[len(prefix):].strip()
                if question.startswith(':'):
                    question = question[1:].strip()
                break
        
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
        
        if not question.endswith('?'):
            return False
        
        # Should contain question words or be imperative
        question_indicators = ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'explain', 'describe', 'compare']
        question_lower = question.lower()
        
        return any(indicator in question_lower for indicator in question_indicators)
    
    def _generate_fallback_question(self, scenario: QuestionScenario) -> str:
        """Generate fallback question when Ollama unavailable."""
        entities = scenario.shared_entities[:2]
        
        if "Basic" in scenario.persona.name:
            if entities:
                return f"What is {entities[0]}?"
            else:
                return "What are the main concepts discussed in this text?"
        else:
            if len(scenario.nodes) > 1 and entities:
                return f"How are {entities[0]} and {entities[1] if len(entities) > 1 else 'related concepts'} connected?"
            elif entities:
                return f"What are the implications of {entities[0]} in this context?"
            else:
                return "What are the key relationships discussed?"
    
    def _generate_generic_prompt(self, scenario: QuestionScenario) -> str:
        """Generate generic prompt for unknown strategies."""
        combined_context = '\n\n'.join([node.properties.get('text', '') for node in scenario.nodes])
        
        prompt = f"""You are a {scenario.persona.name}. {scenario.persona.role_description}

Generate a {scenario.style.value.lower()} question based on this content.

CONTENT:
{combined_context[:800]}

Generate only the question:"""
        
        return prompt


class QuestionEngine:
    """Main engine for generating evaluation questions from knowledge graphs."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize the question engine."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components
        self.personas = PersonaFactory.create_personas()
        self.ollama_generator = OllamaQuestionGenerator(config, self.logger)
        
        # Question type distribution
        self.question_distribution = {
            'entity_bridge': 0.4,
            'concept_similarity': 0.3,
            'hierarchical': 0.2,
            'single_hop': 0.1
        }
        
        self.logger.info(f"🎯 Initialized simplified question generator with {len(self.personas)} personas")
    
    def generate_questions(self, knowledge_graph: KnowledgeGraph, force_recompute: bool = False) -> List[EvaluationQuestion]:
        """Generate evaluation questions from knowledge graph."""
        # Check cache
        cache_path = self._get_cache_path()
        if not force_recompute and self._is_cache_valid(cache_path, knowledge_graph):
            self.logger.info("📂 Loading cached questions")
            return self._load_cached_questions(cache_path)
        
        self.logger.info("🎯 Generating fresh questions using simplified approach")
        start_time = time.time()
        
        # Calculate question counts by type
        target_questions = 50  # Default
        question_counts = self._calculate_question_counts(target_questions)
        
        self.logger.info(f"📊 Question distribution: {question_counts}")
        
        # Initialize selection strategies
        strategies = {
            'entity_bridge': EntityBridgeStrategy(knowledge_graph, self.logger),
            'concept_similarity': ConceptSimilarityStrategy(knowledge_graph, self.logger),
            'hierarchical': HierarchicalStrategy(knowledge_graph, self.logger),
            'single_hop': SingleHopStrategy(knowledge_graph, self.logger)
        }
        
        # Generate scenarios for each strategy
        all_scenarios = []
        
        for strategy_name, count in question_counts.items():
            if count == 0:
                continue
                
            self.logger.info(f"🔄 Generating {count} {strategy_name} scenarios...")
            
            strategy = strategies[strategy_name]
            scenarios = strategy.select_scenarios(count, self.personas)
            all_scenarios.extend(scenarios)
            
            self.logger.info(f"✅ Generated {len(scenarios)} {strategy_name} scenarios")
        
        # Generate questions from scenarios
        self.logger.info(f"🤖 Generating questions from {len(all_scenarios)} scenarios...")
        
        all_questions = []
        
        for i, scenario in enumerate(all_scenarios):
            question = self._generate_question_from_scenario(scenario)
            if question:
                all_questions.append(question)
        
        generation_time = time.time() - start_time
        
        # Cache questions
        metadata = {
            'created_at': datetime.now().isoformat(),
            'total_questions': len(all_questions),
            'generation_time': generation_time,
            'model_used': self.ollama_generator.model,
            'knowledge_graph_stats': {
                'total_nodes': len(knowledge_graph.nodes),
                'total_relationships': len(knowledge_graph.relationships)
            }
        }
        
        self._cache_questions(cache_path, all_questions, metadata)
        
        self.logger.info(f"✅ Question generation completed: {len(all_questions)} questions in {generation_time:.2f}s")
        
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
        
        question_text, reference_answer = self.ollama_generator.generate_question_and_answer(scenario)
        if not question_text:
            return None
        
        generation_time = time.time() - start_time
        
        # Create ground truth contexts
        ground_truth_contexts = [node.id for node in scenario.nodes]
        primary_context_id = scenario.nodes[0].id
        
        # Create reference contexts
        reference_contexts = []
        for node in scenario.nodes:
            context_text = node.properties.get('text', node.properties.get('page_content', ''))
            reference_contexts.append(context_text)
        
        # Determine relationship types
        relationship_types = []
        if scenario.strategy == "entity_bridge":
            relationship_types = ["entity_overlap"]
        elif scenario.strategy == "concept_similarity":
            relationship_types = ["cosine_similarity"]
        elif scenario.strategy == "hierarchical":
            relationship_types = ["contains"]
        
        # Create question ID
        question_id = f"simple_{scenario.strategy}_{hashlib.md5(question_text.encode()).hexdigest()[:8]}"
        
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
            entities_used=scenario.shared_entities,
            persona_used=scenario.persona.name,
            model_used=self.ollama_generator.model,
            generation_time=generation_time
        )
        
        return question
    
    def _get_cache_path(self) -> Path:
        """Get cache path for questions."""
        data_dir = Path(self.config['directories']['data'])
        return data_dir / "questions" / "evaluation_questions.json"
    
    def _is_cache_valid(self, cache_path: Path, knowledge_graph: KnowledgeGraph) -> bool:
        """Check if cached questions are valid."""
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
    
    def _cache_questions(self, cache_path: Path, questions: List[EvaluationQuestion], metadata: Dict[str, Any]):
        """Cache questions to disk."""
        try:
            cache_data = {
                'metadata': metadata,
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
            
            # Count by expected advantage
            advantage = question.expected_advantage
            stats['by_expected_advantage'][advantage] = stats['by_expected_advantage'].get(advantage, 0) + 1
            
            # Count by difficulty
            difficulty = question.difficulty_level
            stats['by_difficulty'][difficulty] = stats['by_difficulty'].get(difficulty, 0) + 1
            
            # Count by persona
            persona = question.persona_used
            stats['by_persona'][persona] = stats['by_persona'].get(persona, 0) + 1
            
            # Track length
            length = len(question.question)
            lengths.append(length)
            stats['question_length_stats']['min'] = min(stats['question_length_stats']['min'], length)
            stats['question_length_stats']['max'] = max(stats['question_length_stats']['max'], length)
        
        if lengths:
            stats['question_length_stats']['mean'] = sum(lengths) / len(lengths)
        
        return stats
