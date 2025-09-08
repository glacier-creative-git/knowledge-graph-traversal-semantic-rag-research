#!/usr/bin/env python3
"""
Question Generation Engine
=========================

Generates evaluation questions using traversal.py rules to ensure question-retrieval coherence.
Implements the core principle: questions are generated to match retrieval capabilities.
"""

import hashlib
import random
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from traversal import (
    TraversalValidator, TraversalPath, ConnectionType, GranularityLevel,
    NavigationLogic, TraversalConstraints
)
from knowledge_graph import KnowledgeGraph


@dataclass
class GeneratedQuestion:
    """Container for a generated question with ground truth."""
    question_id: str
    question_text: str
    ground_truth_path: TraversalPath  # Uses shared structure from traversal.py
    expected_answer: str
    difficulty_level: str
    question_type: str  # "raw_similarity", "hierarchical", etc.
    generation_metadata: Dict[str, Any]


@dataclass
class EvaluationDataset:
    """Container for a complete evaluation dataset."""
    questions: List[GeneratedQuestion]
    dataset_metadata: Dict[str, Any]
    generation_config: Dict[str, Any]


class QuestionDifficultyAssessment:
    """
    Question difficulty and viability assessment moved from traversal.py.
    This is computational work, not pure rules.
    """
    
    @staticmethod
    def calculate_plan_viability(template: Dict[str, Any], target_complexity: str) -> float:
        """Calculate viability score for a traversal plan."""
        base_score = 0.5
        
        # Complexity matching
        complexity_scores = {
            "simple": {"raw_similarity": 1.0, "hierarchical": 0.8, "sequential_flow": 0.6, "theme_bridge": 0.4, "multi_dimensional": 0.2},
            "medium": {"raw_similarity": 0.8, "hierarchical": 1.0, "sequential_flow": 0.9, "theme_bridge": 0.8, "multi_dimensional": 0.6},
            "hard": {"raw_similarity": 0.6, "hierarchical": 0.8, "sequential_flow": 1.0, "theme_bridge": 1.0, "multi_dimensional": 0.9},
            "expert": {"raw_similarity": 0.4, "hierarchical": 0.6, "sequential_flow": 0.8, "theme_bridge": 0.9, "multi_dimensional": 1.0}
        }
        
        pattern_score = complexity_scores.get(target_complexity, {}).get(template["pattern"], 0.5)
        
        # Hop count appropriateness (more hops = more complexity)
        hop_bonus = min(0.3, template["hops"] * 0.05)
        
        # Cross-document capability bonus
        cross_doc_bonus = 0.1 if template.get("cross_document", False) else 0.0
        
        total_score = min(1.0, base_score + pattern_score * 0.4 + hop_bonus + cross_doc_bonus)
        return total_score
    
    @staticmethod
    def estimate_context_sufficiency(template: Dict[str, Any], target_complexity: str) -> float:
        """Estimate context sufficiency for a traversal template."""
        # Base sufficiency depends on number of nodes (more nodes = more context)
        node_count = template["hops"] + 1
        base_sufficiency = min(1.0, node_count * 0.15)
        
        # Granularity bonus (sentence level provides most detailed context)
        granularity_bonus = 0.0
        for granularity in template["granularities"]:
            if granularity == GranularityLevel.SENTENCE:
                granularity_bonus += 0.1
            elif granularity == GranularityLevel.CHUNK:
                granularity_bonus += 0.05
        
        # Complexity adjustment
        complexity_multipliers = {
            "simple": 1.2,
            "medium": 1.0, 
            "hard": 0.8,
            "expert": 0.6
        }
        
        multiplier = complexity_multipliers.get(target_complexity, 1.0)
        
        total_sufficiency = min(1.0, (base_sufficiency + granularity_bonus) * multiplier)
        return total_sufficiency


class TraversalPlanner:
    """
    High-level planner for creating traversal strategies for question generation.
    Moved from traversal.py as it does computational work (scoring/planning).
    """
    
    def __init__(self, validator: TraversalValidator, logger: Optional[logging.Logger] = None):
        """Initialize planner with validator and optional logger."""
        self.validator = validator
        self.logger = logger or logging.getLogger(__name__)
    
    def plan_traversal(self, 
                      start_node_granularity: GranularityLevel,
                      target_complexity: str,
                      traversal_pattern: str,
                      cross_document_required: bool = False) -> List[Dict[str, Any]]:
        """
        Plan a traversal strategy based on requirements.
        Returns list of possible traversal plans ranked by viability.
        """
        plans = []
        
        # Get valid templates for the pattern
        templates = self.validator.generate_valid_path_templates(traversal_pattern)
        
        for template in templates:
            # Check if template matches requirements
            if cross_document_required and not template.get("cross_document", False):
                continue
            
            # Check if starting granularity is compatible
            if template["granularities"][0] != start_node_granularity:
                continue
            
            # Calculate plan viability score
            viability_score = QuestionDifficultyAssessment.calculate_plan_viability(template, target_complexity)
            
            plan = {
                "template": template,
                "viability_score": viability_score,
                "estimated_context_sufficiency": QuestionDifficultyAssessment.estimate_context_sufficiency(template, target_complexity),
                "navigation_complexity": len(template["connection_types"]),
                "recommended": viability_score > 0.7
            }
            
            plans.append(plan)
        
        # Sort by viability score (highest first)
        plans.sort(key=lambda x: x["viability_score"], reverse=True)
        
        if self.logger:
            self.logger.debug(f"Generated {len(plans)} traversal plans for {traversal_pattern} pattern")
            if plans:
                best_plan = plans[0]
                self.logger.debug(f"Best plan: {best_plan['template']['description']} (score: {best_plan['viability_score']:.2f})")
        
        return plans


class QuestionTemplate:
    """Question templates for different traversal patterns."""
    
    @staticmethod
    def raw_similarity_template(source_chunk_text: str, target_chunk_text: str) -> str:
        """Generate question for raw similarity traversal."""
        templates = [
            f"What concept connects the idea of '{source_chunk_text[:50]}...' to '{target_chunk_text[:50]}...'?",
            f"How are the topics discussed in '{source_chunk_text[:30]}...' and '{target_chunk_text[:30]}...' related?",
            f"What is the semantic relationship between these two concepts: '{source_chunk_text[:40]}...' and '{target_chunk_text[:40]}...'?"
        ]
        return random.choice(templates)
    
    @staticmethod
    def hierarchical_template(doc_title: str, chunk_text: str, sentence_text: str) -> str:
        """Generate question for hierarchical traversal."""
        templates = [
            f"In the document about '{doc_title}', what specific detail is mentioned regarding '{chunk_text[:30]}...'?",
            f"According to the article on '{doc_title}', what is stated about '{chunk_text[:40]}...'?",
            f"What specific information does '{doc_title}' provide about '{chunk_text[:35]}...'?"
        ]
        return random.choice(templates)
    
    @staticmethod
    def theme_bridge_template(source_theme: str, target_theme: str, source_doc: str, target_doc: str) -> str:
        """Generate question for theme bridge traversal."""
        templates = [
            f"How does the concept of '{source_theme}' in '{source_doc}' relate to '{target_theme}' discussed in '{target_doc}'?",
            f"What connections exist between '{source_theme}' and '{target_theme}' across different documents?",
            f"Compare the treatment of '{source_theme}' in '{source_doc}' with '{target_theme}' in '{target_doc}'."
        ]
        return random.choice(templates)
    
    @staticmethod
    def sequential_flow_template(sentence_sequence: List[str]) -> str:
        """Generate question for sequential flow traversal."""
        if len(sentence_sequence) >= 2:
            first_concept = sentence_sequence[0][:30]
            last_concept = sentence_sequence[-1][:30]
            
            templates = [
                f"What is the logical progression from '{first_concept}...' to '{last_concept}...'?",
                f"How does the discussion evolve from '{first_concept}...' to eventually cover '{last_concept}...'?",
                f"What are the intermediate steps in the reasoning that connects '{first_concept}...' to '{last_concept}...'?"
            ]
            return random.choice(templates)
        else:
            return f"What is the main point of this sequence: '{sentence_sequence[0][:50]}...'?"
    
    @staticmethod
    def multi_dimensional_template(path_description: str) -> str:
        """Generate question for multi-dimensional traversal."""
        templates = [
            f"Explain the complex relationship described by this path: {path_description}",
            f"What insights can be gained by following this multi-step reasoning: {path_description}?",
            f"How do the different concepts in this path connect: {path_description}?"
        ]
        return random.choice(templates)


class QuestionGenerator:
    """Main question generator using traversal.py rules."""
    
    def __init__(self, knowledge_graph: KnowledgeGraph, config: Dict[str, Any], 
                 logger: Optional[logging.Logger] = None):
        """Initialize question generator."""
        self.kg = knowledge_graph
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components
        self.validator = TraversalValidator(logger)
        self.planner = TraversalPlanner(self.validator, logger)
        
        # Question generation settings
        self.question_config = config.get('question_generation', {
            'difficulty_distribution': {'simple': 0.3, 'medium': 0.4, 'hard': 0.2, 'expert': 0.1},
            'pattern_distribution': {'raw_similarity': 0.4, 'hierarchical': 0.3, 'theme_bridge': 0.2, 'multi_dimensional': 0.1},
            'min_questions_per_pattern': 5
        })
    
    def generate_single_hop_questions(self, num_questions: int) -> List[GeneratedQuestion]:
        """Generate single-hop questions (raw similarity only)."""
        questions = []
        target_pattern = "raw_similarity"
        
        self.logger.info(f"Generating {num_questions} single-hop questions")
        
        # Get chunk pairs with high similarity
        chunk_pairs = self._get_chunk_similarity_pairs(num_questions * 2)  # Generate extra to filter
        
        for i, (source_chunk_id, target_chunk_id, similarity_score) in enumerate(chunk_pairs[:num_questions]):
            try:
                # Create ground truth path
                path_nodes = [source_chunk_id, target_chunk_id]
                connection_types = [ConnectionType.RAW_SIMILARITY]
                granularities = [GranularityLevel.CHUNK, GranularityLevel.CHUNK]
                documents = [self._get_chunk_document(source_chunk_id), self._get_chunk_document(target_chunk_id)]
                
                # Validate path using traversal rules
                traversal_path = self.validator.validate_path(
                    path_nodes, connection_types, granularities, documents, target_pattern
                )
                
                if not traversal_path.is_valid:
                    self.logger.warning(f"Invalid path generated for question {i}: {traversal_path.validation_errors}")
                    continue
                
                # Generate question text
                source_text = self._get_chunk_text(source_chunk_id)
                target_text = self._get_chunk_text(target_chunk_id)
                question_text = QuestionTemplate.raw_similarity_template(source_text, target_text)
                
                # Generate expected answer
                expected_answer = f"Both chunks discuss related concepts with a similarity score of {similarity_score:.3f}. " + \
                                f"The connection is: {source_text[:100]}... relates to {target_text[:100]}..."
                
                # Create question
                question = GeneratedQuestion(
                    question_id=f"single_hop_{i}_{hashlib.md5(question_text.encode()).hexdigest()[:8]}",
                    question_text=question_text,
                    ground_truth_path=traversal_path,
                    expected_answer=expected_answer,
                    difficulty_level="simple",
                    question_type=target_pattern,
                    generation_metadata={
                        'source_chunk': source_chunk_id,
                        'target_chunk': target_chunk_id,
                        'similarity_score': similarity_score,
                        'generation_method': 'chunk_similarity_pairs'
                    }
                )
                
                questions.append(question)
                self.logger.debug(f"Generated single-hop question {i + 1}: {question.question_id}")
                
            except Exception as e:
                self.logger.warning(f"Failed to generate question {i}: {e}")
                continue
        
        self.logger.info(f"Successfully generated {len(questions)} single-hop questions")
        return questions
    
    def generate_multi_hop_questions(self, num_questions: int, 
                                   difficulty_levels: List[str] = ["medium", "hard"]) -> List[GeneratedQuestion]:
        """Generate multi-hop questions using various traversal patterns."""
        questions = []
        patterns = ["hierarchical", "theme_bridge", "multi_dimensional"]
        
        self.logger.info(f"Generating {num_questions} multi-hop questions")
        
        questions_per_pattern = max(1, num_questions // len(patterns))
        
        for pattern in patterns:
            pattern_questions = self._generate_pattern_questions(
                pattern, questions_per_pattern, difficulty_levels
            )
            questions.extend(pattern_questions)
        
        # Shuffle and trim to exact number requested
        random.shuffle(questions)
        questions = questions[:num_questions]
        
        self.logger.info(f"Successfully generated {len(questions)} multi-hop questions")
        return questions
    
    def _generate_pattern_questions(self, pattern: str, num_questions: int, 
                                  difficulty_levels: List[str]) -> List[GeneratedQuestion]:
        """Generate questions for a specific traversal pattern."""
        questions = []
        
        for i in range(num_questions):
            try:
                difficulty = random.choice(difficulty_levels)
                
                # Plan traversal for this pattern and difficulty
                plans = self.planner.plan_traversal(
                    start_node_granularity=GranularityLevel.CHUNK,
                    target_complexity=difficulty,
                    traversal_pattern=pattern,
                    cross_document_required=(pattern in ["theme_bridge", "multi_dimensional"])
                )
                
                if not plans:
                    self.logger.warning(f"No valid plans for pattern {pattern}, difficulty {difficulty}")
                    continue
                
                best_plan = plans[0]
                template = best_plan["template"]
                
                # Generate concrete path using template
                concrete_path = self._generate_concrete_path(template)
                
                if not concrete_path:
                    self.logger.warning(f"Failed to generate concrete path for pattern {pattern}")
                    continue
                
                # Generate question text based on pattern
                question_text = self._generate_question_text(pattern, concrete_path, template)
                expected_answer = self._generate_expected_answer(concrete_path)
                
                # Create question
                question = GeneratedQuestion(
                    question_id=f"{pattern}_{i}_{hashlib.md5(question_text.encode()).hexdigest()[:8]}",
                    question_text=question_text,
                    ground_truth_path=concrete_path,
                    expected_answer=expected_answer,
                    difficulty_level=difficulty,
                    question_type=pattern,
                    generation_metadata={
                        'template': template,
                        'plan_viability': best_plan["viability_score"],
                        'context_sufficiency': best_plan["estimated_context_sufficiency"]
                    }
                )
                
                questions.append(question)
                self.logger.debug(f"Generated {pattern} question {i + 1}: {question.question_id}")
                
            except Exception as e:
                self.logger.warning(f"Failed to generate {pattern} question {i}: {e}")
                continue
        
        return questions
    
    def _generate_concrete_path(self, template: Dict[str, Any]) -> Optional[TraversalPath]:
        """Generate a concrete traversal path from a template."""
        try:
            # Select random starting chunk
            chunk_ids = list(self.kg.chunks.keys())
            if not chunk_ids:
                return None
            
            start_chunk = random.choice(chunk_ids)
            
            # Build path according to template
            path_nodes = [start_chunk]
            granularities = [template["granularities"][0]]
            documents = [self._get_chunk_document(start_chunk)]
            connection_types = []
            
            current_node = start_chunk
            current_granularity = template["granularities"][0]
            
            for i, (connection_type, target_granularity) in enumerate(
                zip(template["connection_types"], template["granularities"][1:])
            ):
                next_node = self._find_next_node(current_node, current_granularity, 
                                               connection_type, target_granularity)
                
                if not next_node:
                    break
                
                path_nodes.append(next_node)
                connection_types.append(connection_type)
                granularities.append(target_granularity)
                documents.append(self._get_node_document(next_node, target_granularity))
                
                current_node = next_node
                current_granularity = target_granularity
            
            # Validate the generated path
            traversal_path = self.validator.validate_path(
                path_nodes, connection_types, granularities, documents, template["pattern"]
            )
            
            return traversal_path if traversal_path.is_valid else None
            
        except Exception as e:
            self.logger.warning(f"Failed to generate concrete path: {e}")
            return None
    
    def _find_next_node(self, current_node: str, current_granularity: GranularityLevel,
                       connection_type: ConnectionType, target_granularity: GranularityLevel) -> Optional[str]:
        """Find the next node in a traversal path."""
        if current_granularity == GranularityLevel.CHUNK:
            chunk = self.kg.chunks.get(current_node)
            if not chunk:
                return None
            
            if connection_type == ConnectionType.RAW_SIMILARITY:
                # Find a similar chunk
                candidates = chunk.intra_doc_connections + chunk.inter_doc_connections
                return random.choice(candidates) if candidates else None
            
            elif connection_type == ConnectionType.HIERARCHICAL and target_granularity == GranularityLevel.SENTENCE:
                # Go to a sentence in this chunk
                sentences = self.kg.get_chunk_sentences(current_node)
                return sentences[0].sentence_id if sentences else None
            
            elif connection_type == ConnectionType.THEME_BRIDGE:
                # Find cross-document chunk
                candidates = chunk.inter_doc_connections
                return random.choice(candidates) if candidates else None
        
        return None
    
    def _generate_question_text(self, pattern: str, path: TraversalPath, template: Dict[str, Any]) -> str:
        """Generate question text based on pattern and path."""
        if pattern == "raw_similarity":
            source_text = self._get_chunk_text(path.nodes[0])
            target_text = self._get_chunk_text(path.nodes[1])
            return QuestionTemplate.raw_similarity_template(source_text, target_text)
        
        elif pattern == "hierarchical":
            # Assuming Document → Chunk → Sentence
            if len(path.nodes) >= 2:
                chunk_text = self._get_chunk_text(path.nodes[0])
                doc_title = self._get_chunk_document(path.nodes[0])
                sentence_text = ""
                if len(path.nodes) > 1 and path.granularity_levels[1] == GranularityLevel.SENTENCE:
                    sentence = self.kg.sentences.get(path.nodes[1])
                    sentence_text = sentence.sentence_text if sentence else ""
                
                return QuestionTemplate.hierarchical_template(doc_title, chunk_text, sentence_text)
        
        elif pattern == "theme_bridge":
            source_doc = self._get_chunk_document(path.nodes[0])
            target_doc = self._get_chunk_document(path.nodes[1]) if len(path.nodes) > 1 else source_doc
            
            # Get themes from documents
            source_themes = self._get_document_themes(source_doc)
            target_themes = self._get_document_themes(target_doc)
            
            source_theme = source_themes[0] if source_themes else "general topic"
            target_theme = target_themes[0] if target_themes else "related topic"
            
            return QuestionTemplate.theme_bridge_template(source_theme, target_theme, source_doc, target_doc)
        
        else:
            # Default question for complex patterns
            path_description = f"Path with {len(path.nodes)} nodes using {len(set(path.connection_types))} different connection types"
            return QuestionTemplate.multi_dimensional_template(path_description)
    
    def _generate_expected_answer(self, path: TraversalPath) -> str:
        """Generate expected answer based on path content."""
        content_pieces = []
        
        for node_id, granularity in zip(path.nodes, path.granularity_levels):
            if granularity == GranularityLevel.CHUNK:
                text = self._get_chunk_text(node_id)
                content_pieces.append(text[:100] + "...")
            elif granularity == GranularityLevel.SENTENCE:
                sentence = self.kg.sentences.get(node_id)
                if sentence:
                    content_pieces.append(sentence.sentence_text)
        
        return " The answer involves connecting these concepts: " + " → ".join(content_pieces)
    
    # Helper methods
    def _get_chunk_similarity_pairs(self, num_pairs: int) -> List[Tuple[str, str, float]]:
        """Get chunk pairs sorted by similarity using pre-computed connections."""
        pairs = []
        
        # Iterate through all chunks and their pre-computed connections
        for chunk_id, chunk_obj in self.kg.chunks.items():
            # Get all connections (both intra and inter-document)
            all_connections = chunk_obj.intra_doc_connections + chunk_obj.inter_doc_connections
            
            for connected_chunk_id in all_connections:
                if connected_chunk_id in chunk_obj.connection_scores:
                    similarity = chunk_obj.connection_scores[connected_chunk_id]
                    pairs.append((chunk_id, connected_chunk_id, similarity))
        
        # Sort by similarity (descending) and return top pairs
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        self.logger.debug(f"Found {len(pairs)} total chunk pairs, returning top {num_pairs}")
        return pairs[:num_pairs]
    
    def _get_chunk_text(self, chunk_id: str) -> str:
        """Get text content of a chunk."""
        chunk = self.kg.chunks.get(chunk_id)
        return chunk.chunk_text if chunk else ""
    
    def _get_chunk_document(self, chunk_id: str) -> str:
        """Get document name for a chunk."""
        chunk = self.kg.chunks.get(chunk_id)
        return chunk.source_document if chunk else "unknown"
    
    def _get_node_document(self, node_id: str, granularity: GranularityLevel) -> str:
        """Get document name for any node type."""
        if granularity == GranularityLevel.CHUNK:
            return self._get_chunk_document(node_id)
        elif granularity == GranularityLevel.SENTENCE:
            sentence = self.kg.sentences.get(node_id)
            return sentence.source_document if sentence else "unknown"
        elif granularity == GranularityLevel.DOCUMENT:
            document = self.kg.documents.get(node_id)
            return document.title if document else "unknown"
        return "unknown"
    
    def _get_document_themes(self, doc_title: str) -> List[str]:
        """Get themes for a document."""
        for doc in self.kg.documents.values():
            if doc.title == doc_title:
                return doc.doc_themes
        return []


class DatasetGenerator:
    """Creates complete evaluation datasets with question-answer pairs."""
    
    def __init__(self, knowledge_graph: KnowledgeGraph, config: Dict[str, Any], 
                 logger: Optional[logging.Logger] = None):
        """Initialize dataset generator."""
        self.kg = knowledge_graph
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.question_generator = QuestionGenerator(knowledge_graph, config, logger)
    
    def create_dataset(self, num_questions: int, difficulty_levels: List[str] = None) -> EvaluationDataset:
        """Create a complete evaluation dataset."""
        if difficulty_levels is None:
            difficulty_levels = ["simple", "medium", "hard"]
        
        self.logger.info(f"Creating evaluation dataset with {num_questions} questions")
        
        all_questions = []
        
        # Generate mix of single-hop and multi-hop questions
        single_hop_count = int(num_questions * 0.4)  # 40% single-hop
        multi_hop_count = num_questions - single_hop_count
        
        # Generate single-hop questions
        single_hop_questions = self.question_generator.generate_single_hop_questions(single_hop_count)
        all_questions.extend(single_hop_questions)
        
        # Generate multi-hop questions
        multi_hop_questions = self.question_generator.generate_multi_hop_questions(multi_hop_count, difficulty_levels)
        all_questions.extend(multi_hop_questions)
        
        # Shuffle final dataset
        random.shuffle(all_questions)
        
        # Create dataset metadata
        metadata = {
            'total_questions': len(all_questions),
            'single_hop_questions': len(single_hop_questions),
            'multi_hop_questions': len(multi_hop_questions),
            'difficulty_distribution': self._analyze_difficulty_distribution(all_questions),
            'pattern_distribution': self._analyze_pattern_distribution(all_questions),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'knowledge_graph_stats': {
                'total_chunks': len(self.kg.chunks),
                'total_sentences': len(self.kg.sentences),
                'total_documents': len(self.kg.documents)
            }
        }
        
        dataset = EvaluationDataset(
            questions=all_questions,
            dataset_metadata=metadata,
            generation_config=self.config.get('question_generation', {})
        )
        
        self.logger.info(f"Created evaluation dataset with {len(all_questions)} questions")
        self.logger.info(f"Difficulty distribution: {metadata['difficulty_distribution']}")
        self.logger.info(f"Pattern distribution: {metadata['pattern_distribution']}")
        
        return dataset
    
    def validate_question_answerability(self, dataset: EvaluationDataset) -> Dict[str, Any]:
        """Validate that generated questions are answerable using their ground truth paths."""
        validation_results = {
            'total_questions': len(dataset.questions),
            'valid_questions': 0,
            'invalid_questions': 0,
            'validation_errors': []
        }
        
        for question in dataset.questions:
            if question.ground_truth_path.is_valid:
                validation_results['valid_questions'] += 1
            else:
                validation_results['invalid_questions'] += 1
                validation_results['validation_errors'].append({
                    'question_id': question.question_id,
                    'errors': question.ground_truth_path.validation_errors
                })
        
        validation_results['validity_rate'] = validation_results['valid_questions'] / validation_results['total_questions']
        
        self.logger.info(f"Question validation: {validation_results['valid_questions']}/{validation_results['total_questions']} valid ({validation_results['validity_rate']:.2%})")
        
        return validation_results
    
    def _analyze_difficulty_distribution(self, questions: List[GeneratedQuestion]) -> Dict[str, int]:
        """Analyze difficulty distribution in questions."""
        distribution = {}
        for question in questions:
            difficulty = question.difficulty_level
            distribution[difficulty] = distribution.get(difficulty, 0) + 1
        return distribution
    
    def _analyze_pattern_distribution(self, questions: List[GeneratedQuestion]) -> Dict[str, int]:
        """Analyze pattern distribution in questions."""
        distribution = {}
        for question in questions:
            pattern = question.question_type
            distribution[pattern] = distribution.get(pattern, 0) + 1
        return distribution


# Factory functions for easy initialization
def create_question_generator(knowledge_graph: KnowledgeGraph, config: Dict[str, Any], 
                            logger: Optional[logging.Logger] = None) -> QuestionGenerator:
    """Factory function to create a question generator."""
    return QuestionGenerator(knowledge_graph, config, logger)


def create_dataset_generator(knowledge_graph: KnowledgeGraph, config: Dict[str, Any], 
                           logger: Optional[logging.Logger] = None) -> DatasetGenerator:
    """Factory function to create a dataset generator."""
    return DatasetGenerator(knowledge_graph, config, logger)
