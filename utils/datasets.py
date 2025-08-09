#!/usr/bin/env python3
"""
Dataset Generation Engine
========================

Handles dataset generation for evaluating semantic RAG systems.
Integrates RAGAS for standard evaluation and generates custom questions
that showcase semantic traversal advantages.
"""

import json
import hashlib
import time
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from models import ChunkEmbedding
from retrieval import RetrievalEngine, RetrievalResult


@dataclass
class EvaluationQuestion:
    """Container for an evaluation question with metadata."""
    question_id: str
    question_text: str
    question_type: str  # factual_lookup, conceptual_relationships, etc.
    expected_advantage: str  # baseline, semantic_traversal, neutral
    difficulty_level: str  # easy, medium, hard
    source_articles: List[str]
    related_chunks: List[str]  # Chunk IDs that should be relevant
    human_answer: Optional[str] = None
    generation_method: str = "unknown"
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class DatasetMetadata:
    """Metadata for generated datasets."""
    created_at: str
    total_questions: int
    generation_methods: Dict[str, int]
    question_types: Dict[str, int]
    difficulty_levels: Dict[str, int]
    expected_advantages: Dict[str, int]
    generation_time: float
    config_hash: str


class CustomQuestionGenerator:
    """Generates custom questions that favor semantic traversal."""
    
    # Question templates for different types
    QUESTION_TEMPLATES = {
        "conceptual_relationships": [
            "How does {concept_a} relate to {concept_b}?",
            "What is the connection between {concept_a} and {concept_b}?", 
            "Explain the relationship between {concept_a} and {concept_b}.",
            "How are {concept_a} and {concept_b} connected?",
            "What links {concept_a} with {concept_b}?"
        ],
        "multi_hop_reasoning": [
            "How does {concept_a} lead to {concept_b} through {concept_c}?",
            "Trace the development from {concept_a} to {concept_b}.",
            "Explain the progression from {concept_a} to {concept_b}.",
            "How does understanding {concept_a} help with {concept_b}?",
            "What steps connect {concept_a} to {concept_b}?"
        ],
        "contextual_understanding": [
            "What background knowledge is needed to understand {concept}?",
            "What context is required to grasp {concept}?",
            "What foundational concepts support {concept}?",
            "How do different aspects of {concept} work together?",
            "What broader framework includes {concept}?"
        ],
        "reading_flow": [
            "How do these concepts build upon each other: {concept_list}?",
            "Explain the logical progression through {concept_list}.",
            "How do {concept_list} form a coherent narrative?",
            "What is the natural flow from {concept_a} through {concept_b} to {concept_c}?",
            "How do these ideas connect in sequence: {concept_list}?"
        ]
    }
    
    def __init__(self, config: Dict[str, Any], embeddings: Dict[str, List[ChunkEmbedding]], 
                 logger: Optional[logging.Logger] = None):
        """Initialize custom question generator."""
        self.config = config
        self.embeddings = embeddings  
        self.logger = logger or logging.getLogger(__name__)
        self.custom_config = config['datasets']['custom']
        
        # Extract concepts from chunks for question generation
        self._extract_concepts()
    
    def _extract_concepts(self):
        """Extract key concepts from chunks for question generation."""
        self.concepts = []
        self.concept_chunks = {}  # concept -> list of relevant chunks
        
        # Simple concept extraction - look for noun phrases and technical terms
        for model_name, chunks in self.embeddings.items():
            for chunk in chunks:
                # Extract potential concepts (simplified approach)
                text = chunk.chunk_text.lower()
                
                # Look for common AI/ML concepts
                ai_concepts = [
                    "machine learning", "artificial intelligence", "neural networks", 
                    "deep learning", "natural language processing", "computer vision",
                    "algorithms", "data science", "statistics", "supervised learning",
                    "unsupervised learning", "reinforcement learning", "training data",
                    "feature extraction", "classification", "regression", "clustering"
                ]
                
                for concept in ai_concepts:
                    if concept in text:
                        if concept not in self.concept_chunks:
                            self.concept_chunks[concept] = []
                        self.concept_chunks[concept].append(chunk)
                        if concept not in self.concepts:
                            self.concepts.append(concept)
        
        self.logger.info(f"Extracted {len(self.concepts)} concepts from corpus")
        self.logger.debug(f"Concepts: {self.concepts[:10]}...")  # Log first 10
    
    def generate_questions(self, num_questions: int) -> List[EvaluationQuestion]:
        """Generate custom questions focused on semantic traversal advantages."""
        questions = []
        focus_areas = self.custom_config['focus_areas']
        
        questions_per_type = num_questions // len(focus_areas)
        
        for question_type in focus_areas:
            type_questions = self._generate_questions_for_type(question_type, questions_per_type)
            questions.extend(type_questions)
        
        # Fill remaining slots
        remaining = num_questions - len(questions)
        if remaining > 0:
            extra_questions = self._generate_questions_for_type(
                random.choice(focus_areas), remaining
            )
            questions.extend(extra_questions)
        
        self.logger.info(f"Generated {len(questions)} custom questions")
        return questions[:num_questions]
    
    def _generate_questions_for_type(self, question_type: str, num_questions: int) -> List[EvaluationQuestion]:
        """Generate questions for a specific type."""
        questions = []
        templates = self.QUESTION_TEMPLATES[question_type]
        
        for i in range(num_questions):
            try:
                question = self._create_question(question_type, templates)
                if question:
                    questions.append(question)
            except Exception as e:
                self.logger.warning(f"Failed to generate question of type {question_type}: {e}")
                continue
        
        return questions
    
    def _create_question(self, question_type: str, templates: List[str]) -> Optional[EvaluationQuestion]:
        """Create a single question of the specified type."""
        template = random.choice(templates)
        
        if question_type == "conceptual_relationships":
            return self._create_relationship_question(template)
        elif question_type == "multi_hop_reasoning":
            return self._create_multi_hop_question(template)
        elif question_type == "contextual_understanding":
            return self._create_contextual_question(template)
        elif question_type == "reading_flow":
            return self._create_reading_flow_question(template)
        
        return None
    
    def _create_relationship_question(self, template: str) -> Optional[EvaluationQuestion]:
        """Create a question about relationships between concepts."""
        if len(self.concepts) < 2:
            return None
        
        # Find two related concepts
        concept_a, concept_b = random.sample(self.concepts, 2)
        
        question_text = template.format(concept_a=concept_a, concept_b=concept_b)
        question_id = f"rel_{hashlib.md5(question_text.encode()).hexdigest()[:8]}"
        
        # Get related chunks
        related_chunks = []
        if concept_a in self.concept_chunks:
            related_chunks.extend([c.chunk_id for c in self.concept_chunks[concept_a][:3]])
        if concept_b in self.concept_chunks:
            related_chunks.extend([c.chunk_id for c in self.concept_chunks[concept_b][:3]])
        
        # Get source articles
        source_articles = []
        for concept in [concept_a, concept_b]:
            if concept in self.concept_chunks:
                articles = [c.source_article for c in self.concept_chunks[concept][:2]]
                source_articles.extend(articles)
        
        return EvaluationQuestion(
            question_id=question_id,
            question_text=question_text,
            question_type="conceptual_relationships",
            expected_advantage="semantic_traversal",
            difficulty_level=random.choice(["medium", "hard"]),
            source_articles=list(set(source_articles)),
            related_chunks=related_chunks,
            generation_method="custom",
            metadata={
                "concept_a": concept_a,
                "concept_b": concept_b,
                "template": template
            }
        )
    
    def _create_multi_hop_question(self, template: str) -> Optional[EvaluationQuestion]:
        """Create a question requiring multi-hop reasoning."""
        if len(self.concepts) < 3:
            return None
        
        # Find three related concepts for multi-hop
        concept_a, concept_b, concept_c = random.sample(self.concepts, 3)
        
        if "{concept_c}" in template:
            question_text = template.format(
                concept_a=concept_a, concept_b=concept_b, concept_c=concept_c
            )
        else:
            question_text = template.format(concept_a=concept_a, concept_b=concept_b)
        
        question_id = f"hop_{hashlib.md5(question_text.encode()).hexdigest()[:8]}"
        
        # Get related chunks from all concepts
        related_chunks = []
        source_articles = []
        for concept in [concept_a, concept_b, concept_c]:
            if concept in self.concept_chunks:
                chunks = self.concept_chunks[concept][:2]
                related_chunks.extend([c.chunk_id for c in chunks])
                source_articles.extend([c.source_article for c in chunks])
        
        return EvaluationQuestion(
            question_id=question_id,
            question_text=question_text,
            question_type="multi_hop_reasoning",
            expected_advantage="semantic_traversal",
            difficulty_level="hard",
            source_articles=list(set(source_articles)),
            related_chunks=related_chunks,
            generation_method="custom",
            metadata={
                "concepts": [concept_a, concept_b, concept_c],
                "template": template
            }
        )
    
    def _create_contextual_question(self, template: str) -> Optional[EvaluationQuestion]:
        """Create a question about contextual understanding."""
        if not self.concepts:
            return None
        
        concept = random.choice(self.concepts)
        question_text = template.format(concept=concept)
        question_id = f"ctx_{hashlib.md5(question_text.encode()).hexdigest()[:8]}"
        
        # Get related chunks and articles
        related_chunks = []
        source_articles = []
        if concept in self.concept_chunks:
            chunks = self.concept_chunks[concept][:4]
            related_chunks = [c.chunk_id for c in chunks]
            source_articles = [c.source_article for c in chunks]
        
        return EvaluationQuestion(
            question_id=question_id,
            question_text=question_text,
            question_type="contextual_understanding", 
            expected_advantage="semantic_traversal",
            difficulty_level=random.choice(["medium", "hard"]),
            source_articles=list(set(source_articles)),
            related_chunks=related_chunks,
            generation_method="custom",
            metadata={
                "concept": concept,
                "template": template
            }
        )
    
    def _create_reading_flow_question(self, template: str) -> Optional[EvaluationQuestion]:
        """Create a question about reading flow and narrative coherence."""
        if len(self.concepts) < 3:
            return None
        
        # Select 3-4 concepts for reading flow
        num_concepts = random.choice([3, 4])
        concept_list = random.sample(self.concepts, num_concepts)
        
        if "{concept_list}" in template:
            question_text = template.format(concept_list=", ".join(concept_list))
        else:
            # For templates with specific concept slots
            question_text = template.format(
                concept_a=concept_list[0],
                concept_b=concept_list[1], 
                concept_c=concept_list[2] if len(concept_list) > 2 else concept_list[1]
            )
        
        question_id = f"flow_{hashlib.md5(question_text.encode()).hexdigest()[:8]}"
        
        # Get related chunks from all concepts
        related_chunks = []
        source_articles = []
        for concept in concept_list:
            if concept in self.concept_chunks:
                chunks = self.concept_chunks[concept][:2]
                related_chunks.extend([c.chunk_id for c in chunks])
                source_articles.extend([c.source_article for c in chunks])
        
        return EvaluationQuestion(
            question_id=question_id,
            question_text=question_text,
            question_type="reading_flow",
            expected_advantage="semantic_traversal",
            difficulty_level="hard",
            source_articles=list(set(source_articles)),
            related_chunks=related_chunks,
            generation_method="custom",
            metadata={
                "concepts": concept_list,
                "template": template
            }
        )


class RAGASGenerator:
    """Generates questions using RAGAS methodology."""
    
    def __init__(self, config: Dict[str, Any], embeddings: Dict[str, List[ChunkEmbedding]], 
                 logger: Optional[logging.Logger] = None):
        """Initialize RAGAS generator."""
        self.config = config
        self.embeddings = embeddings
        self.logger = logger or logging.getLogger(__name__)
        self.ragas_config = config['datasets']['ragas']
    
    def generate_questions(self, num_questions: int) -> List[EvaluationQuestion]:
        """Generate questions using RAGAS-style methodology."""
        questions = []
        
        # For now, implement a simplified RAGAS-style approach
        # In a full implementation, you'd integrate with the actual RAGAS library
        
        question_types = self.ragas_config['question_types']
        difficulty_levels = self.ragas_config['difficulty_levels']
        
        questions_per_type = num_questions // len(question_types)
        
        for q_type in question_types:
            for difficulty in difficulty_levels:
                type_questions = self._generate_ragas_questions(
                    q_type, difficulty, questions_per_type // len(difficulty_levels)
                )
                questions.extend(type_questions)
        
        self.logger.info(f"Generated {len(questions)} RAGAS-style questions")
        return questions[:num_questions]
    
    def _generate_ragas_questions(self, question_type: str, difficulty: str, count: int) -> List[EvaluationQuestion]:
        """Generate RAGAS-style questions for a specific type and difficulty."""
        questions = []
        
        # Simple templates for RAGAS-style questions
        templates = {
            "factual": [
                "What is {concept}?",
                "Define {concept}.",
                "Explain what {concept} means.",
                "What does {concept} refer to?"
            ],
            "reasoning": [
                "Why is {concept} important?",
                "How does {concept} work?",
                "What are the benefits of {concept}?",
                "What are the applications of {concept}?"
            ],
            "multi_context": [
                "Compare {concept_a} and {concept_b}.",
                "What are the differences between {concept_a} and {concept_b}?",
                "How do {concept_a} and {concept_b} complement each other?",
                "Which is better: {concept_a} or {concept_b}?"
            ]
        }
        
        if question_type not in templates:
            return questions
        
        # Get available concepts
        concepts = list(self._get_available_concepts().keys())
        if not concepts:
            return questions
        
        for i in range(count):
            try:
                template = random.choice(templates[question_type])
                
                if "{concept_b}" in template:
                    # Multi-concept question
                    if len(concepts) >= 2:
                        concept_a, concept_b = random.sample(concepts, 2)
                        question_text = template.format(concept_a=concept_a, concept_b=concept_b)
                    else:
                        continue
                else:
                    # Single concept question  
                    concept = random.choice(concepts)
                    question_text = template.format(concept=concept)
                
                question_id = f"ragas_{hashlib.md5(question_text.encode()).hexdigest()[:8]}"
                
                # Determine expected advantage
                if question_type == "factual":
                    expected_advantage = "baseline"  # Factual questions favor baseline
                elif question_type == "multi_context":
                    expected_advantage = "semantic_traversal"  # Multi-context favors traversal
                else:
                    expected_advantage = "neutral"
                
                questions.append(EvaluationQuestion(
                    question_id=question_id,
                    question_text=question_text,
                    question_type=f"ragas_{question_type}",
                    expected_advantage=expected_advantage,
                    difficulty_level=difficulty,
                    source_articles=[],  # Will be populated during validation
                    related_chunks=[],   # Will be populated during validation
                    generation_method="ragas",
                    metadata={
                        "ragas_type": question_type,
                        "template": template
                    }
                ))
                
            except Exception as e:
                self.logger.warning(f"Failed to generate RAGAS question: {e}")
                continue
        
        return questions
    
    def _get_available_concepts(self) -> Dict[str, List[ChunkEmbedding]]:
        """Get available concepts from embeddings."""
        concepts = {}
        
        # Simple concept extraction
        for model_name, chunks in self.embeddings.items():
            for chunk in chunks:
                text = chunk.chunk_text.lower()
                
                # Extract common AI/ML terms
                ai_terms = [
                    "machine learning", "artificial intelligence", "neural networks",
                    "deep learning", "data science", "algorithm", "model"
                ]
                
                for term in ai_terms:
                    if term in text:
                        if term not in concepts:
                            concepts[term] = []
                        concepts[term].append(chunk)
        
        return concepts


class DatasetValidator:
    """Validates and improves generated datasets."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize dataset validator."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.validation_config = config['datasets']['validation']
    
    def validate_questions(self, questions: List[EvaluationQuestion]) -> List[EvaluationQuestion]:
        """Validate and filter questions based on quality criteria."""
        validated = []
        
        for question in questions:
            if self._is_valid_question(question):
                validated.append(question)
            else:
                self.logger.debug(f"Filtered out question: {question.question_text[:50]}...")
        
        self.logger.info(f"Validated {len(validated)}/{len(questions)} questions")
        return validated
    
    def _is_valid_question(self, question: EvaluationQuestion) -> bool:
        """Check if a question meets quality criteria."""
        # Basic length check
        if len(question.question_text) < 10:
            return False
        
        # Check for question marks or question words
        text = question.question_text.lower()
        question_indicators = ['?', 'what', 'how', 'why', 'when', 'where', 'which', 'explain', 'describe']
        if not any(indicator in text for indicator in question_indicators):
            return False
        
        # Additional quality checks could be added here
        return True


class DatasetEngine:
    """Main engine for dataset generation and management."""
    
    def __init__(self, config: Dict[str, Any], embeddings: Dict[str, List[ChunkEmbedding]], 
                 logger: Optional[logging.Logger] = None):
        """Initialize the dataset engine."""
        self.config = config
        self.embeddings = embeddings  
        self.logger = logger or logging.getLogger(__name__)
        self.dataset_config = config['datasets']
        self.datasets_dir = Path(config['directories']['data']) / "datasets"
        
        # Create datasets directory
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize generators
        self.custom_generator = CustomQuestionGenerator(config, embeddings, logger)
        self.ragas_generator = RAGASGenerator(config, embeddings, logger)
        self.validator = DatasetValidator(config, logger)
    
    def generate_dataset(self, force_recompute: bool = False) -> List[EvaluationQuestion]:
        """
        Generate evaluation dataset using configured methods.
        
        Args:
            force_recompute: Whether to force regeneration even if cached
            
        Returns:
            List of EvaluationQuestion objects
        """
        # Generate config hash for cache validation
        config_hash = self._generate_config_hash()
        
        # Check cache
        cache_path = self._get_cache_path()
        if not force_recompute and self._is_cache_valid(cache_path, config_hash):
            self.logger.info("Loading cached dataset")
            return self._load_cached_dataset(cache_path)
        
        self.logger.info("Generating fresh dataset")
        start_time = time.time()
        
        # Generate questions based on method
        all_questions = []
        generation_method = self.dataset_config['generation_method']
        
        if generation_method in ['ragas', 'mixed']:
            ragas_count = self.dataset_config['ragas']['num_questions']
            ragas_questions = self.ragas_generator.generate_questions(ragas_count)
            all_questions.extend(ragas_questions)
            self.logger.info(f"Generated {len(ragas_questions)} RAGAS questions")
        
        if generation_method in ['custom', 'mixed']:
            custom_count = self.dataset_config['custom']['num_questions']
            custom_questions = self.custom_generator.generate_questions(custom_count)
            all_questions.extend(custom_questions)
            self.logger.info(f"Generated {len(custom_questions)} custom questions")
        
        # Validate questions
        validated_questions = self.validator.validate_questions(all_questions)
        
        # Shuffle for good distribution
        random.shuffle(validated_questions)
        
        generation_time = time.time() - start_time
        
        # Create metadata
        metadata = self._create_dataset_metadata(validated_questions, generation_time, config_hash)
        
        # Cache dataset
        self._cache_dataset(cache_path, validated_questions, metadata)
        
        self.logger.info(f"Dataset generation completed: {len(validated_questions)} questions in {generation_time:.2f}s")
        
        return validated_questions
    
    def _generate_config_hash(self) -> str:
        """Generate hash of dataset configuration for cache validation."""
        config_str = json.dumps({
            'datasets': self.dataset_config,
            'embedding_count': {model: len(chunks) for model, chunks in self.embeddings.items()}
        }, sort_keys=True)
        
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _get_cache_path(self) -> Path:
        """Get cache path for dataset."""
        return self.datasets_dir / "evaluation_dataset.json"
    
    def _is_cache_valid(self, cache_path: Path, expected_hash: str) -> bool:
        """Check if cached dataset is valid."""
        if not cache_path.exists():
            return False
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            metadata = cache_data.get('metadata', {})
            cached_hash = metadata.get('config_hash', '')
            
            return cached_hash == expected_hash
            
        except Exception as e:
            self.logger.warning(f"Failed to validate dataset cache: {e}")
            return False
    
    def _create_dataset_metadata(self, questions: List[EvaluationQuestion], 
                               generation_time: float, config_hash: str) -> DatasetMetadata:
        """Create metadata for the dataset."""
        generation_methods = {}
        question_types = {}
        difficulty_levels = {}
        expected_advantages = {}
        
        for question in questions:
            # Count generation methods
            method = question.generation_method
            generation_methods[method] = generation_methods.get(method, 0) + 1
            
            # Count question types
            q_type = question.question_type
            question_types[q_type] = question_types.get(q_type, 0) + 1
            
            # Count difficulty levels
            difficulty = question.difficulty_level
            difficulty_levels[difficulty] = difficulty_levels.get(difficulty, 0) + 1
            
            # Count expected advantages
            advantage = question.expected_advantage
            expected_advantages[advantage] = expected_advantages.get(advantage, 0) + 1
        
        return DatasetMetadata(
            created_at=datetime.now().isoformat(),
            total_questions=len(questions),
            generation_methods=generation_methods,
            question_types=question_types,
            difficulty_levels=difficulty_levels,
            expected_advantages=expected_advantages,
            generation_time=generation_time,
            config_hash=config_hash
        )
    
    def _cache_dataset(self, cache_path: Path, questions: List[EvaluationQuestion], 
                      metadata: DatasetMetadata):
        """Cache dataset to disk."""
        try:
            cache_data = {
                'metadata': asdict(metadata),
                'questions': [question.to_dict() for question in questions]
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Cached dataset to {cache_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to cache dataset: {e}")
            raise
    
    def _load_cached_dataset(self, cache_path: Path) -> List[EvaluationQuestion]:
        """Load cached dataset from disk."""
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            questions = []
            for q_data in cache_data['questions']:
                # Handle missing fields gracefully
                question = EvaluationQuestion(
                    question_id=q_data['question_id'],
                    question_text=q_data['question_text'],
                    question_type=q_data['question_type'],
                    expected_advantage=q_data['expected_advantage'],
                    difficulty_level=q_data['difficulty_level'],
                    source_articles=q_data.get('source_articles', []),
                    related_chunks=q_data.get('related_chunks', []),
                    human_answer=q_data.get('human_answer'),
                    generation_method=q_data.get('generation_method', 'unknown'),
                    metadata=q_data.get('metadata', {})
                )
                questions.append(question)
            
            return questions
            
        except Exception as e:
            self.logger.error(f"Failed to load cached dataset: {e}")
            raise
    
    def get_dataset_statistics(self, questions: List[EvaluationQuestion]) -> Dict[str, Any]:
        """Get statistics about the dataset."""
        if not questions:
            return {}
        
        # Count by various categories
        stats = {
            'total_questions': len(questions),
            'by_generation_method': {},
            'by_question_type': {},
            'by_difficulty': {},
            'by_expected_advantage': {},
            'question_length_stats': {
                'mean': 0,
                'min': float('inf'),
                'max': 0
            }
        }
        
        lengths = []
        
        for question in questions:
            # Generation method
            method = question.generation_method
            stats['by_generation_method'][method] = stats['by_generation_method'].get(method, 0) + 1
            
            # Question type
            q_type = question.question_type
            stats['by_question_type'][q_type] = stats['by_question_type'].get(q_type, 0) + 1
            
            # Difficulty
            difficulty = question.difficulty_level
            stats['by_difficulty'][difficulty] = stats['by_difficulty'].get(difficulty, 0) + 1
            
            # Expected advantage
            advantage = question.expected_advantage
            stats['by_expected_advantage'][advantage] = stats['by_expected_advantage'].get(advantage, 0) + 1
            
            # Length
            length = len(question.question_text)
            lengths.append(length)
            stats['question_length_stats']['min'] = min(stats['question_length_stats']['min'], length)
            stats['question_length_stats']['max'] = max(stats['question_length_stats']['max'], length)
        
        if lengths:
            stats['question_length_stats']['mean'] = sum(lengths) / len(lengths)
        
        return stats
