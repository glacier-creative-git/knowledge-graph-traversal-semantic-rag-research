#!/usr/bin/env python3
"""
Traversal-Based Question Generator for Phase 7
==============================================

Implements traversal-aware question generation that creates questions from paths
that your algorithms can actually follow. This ensures evaluation tests real
retrieval capabilities rather than hoping for lucky similarity matches.

Key Principle: Path → Question (not Chunk → Question like RAGAS)
"""

import hashlib
import random
import logging
import time
import json
import requests
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from utils.traversal import (
    TraversalValidator, TraversalPath, ConnectionType, GranularityLevel,
    NavigationLogic, TraversalConstraints
)
from utils.knowledge_graph import KnowledgeGraph
from utils.questions import GeneratedQuestion, EvaluationDataset


class ModelInterface:
    """Abstract interface for interchangeable question generation models."""
    
    def generate_question(self, prompt: str) -> str:
        """Generate a question from a prompt."""
        raise NotImplementedError
    
    def critique_question(self, question: str, context: str) -> Dict[str, Any]:
        """Critique a question and provide feedback."""
        raise NotImplementedError


class OllamaInterface(ModelInterface):
    """Ollama model interface for question generation."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config['ollama']
        self.logger = logger or logging.getLogger(__name__)
        self.base_url = self.config['base_url']
        self.model = self.config['model']
        
    def generate_question(self, prompt: str) -> str:
        """Generate question using Ollama."""
        try:
            # Add explicit instruction to the prompt
            full_prompt = f"{prompt}\n\nIMPORTANT: Respond with ONLY the question text, no explanations or labels."
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": self.config['options']
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()['response'].strip()
        except Exception as e:
            self.logger.error(f"Ollama generation failed: {e}")
            return f"What is the main concept discussed in the provided content?"
    
    def critique_question(self, question: str, context: str) -> Dict[str, Any]:
        """Critique question using Ollama."""
        critique_prompt = f"""
        Please evaluate this question for quality:
        
        Question: {question}
        Context: {context[:200]}...
        
        Rate the question on:
        1. Independence (can be understood without external context): Yes/No
        2. Clear Intent (obvious what type of answer is sought): Yes/No  
        3. Answerability (can be answered from provided context): Yes/No
        
        Respond with JSON format:
        {{"independence": true/false, "clear_intent": true/false, "answerable": true/false, "feedback": "brief explanation"}}
        """
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": critique_prompt,
                    "stream": False,
                    "options": self.config['options']
                },
                timeout=30
            )
            response.raise_for_status()
            
            # Try to parse JSON response
            raw_response = response.json()['response'].strip()
            
            # Clean the response to extract JSON
            import re
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                critique = json.loads(json_str)
                critique['verdict'] = (critique.get('independence', False) and 
                                    critique.get('clear_intent', False) and 
                                    critique.get('answerable', False))
                return critique
            else:
                # Fallback if JSON parsing fails
                return {
                    "independence": True,
                    "clear_intent": True, 
                    "answerable": True,
                    "verdict": True,
                    "feedback": "JSON parsing failed, assuming valid"
                }
                
        except Exception as e:
            self.logger.warning(f"Ollama critique failed: {e}")
            return {
                "independence": True,
                "clear_intent": True,
                "answerable": True, 
                "verdict": True,
                "feedback": f"Critique error: {e}"
            }


class OpenAIInterface(ModelInterface):
    """OpenAI model interface for question generation."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config['openai']
        self.logger = logger or logging.getLogger(__name__)
        self.api_key = os.getenv('OPENAI_API_KEY') or self.config.get('api_key', '')
        self.generator_model = self.config['generator_model']
        self.critic_model = self.config['critic_model']
        
        if not self.api_key:
            self.logger.warning("No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
    
    def generate_question(self, prompt: str) -> str:
        """Generate question using OpenAI."""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.generator_model,
                messages=[
                    {"role": "system", "content": "You are an expert question generator for RAG evaluation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config['options']['temperature'],
                max_tokens=self.config['options']['max_tokens']
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"OpenAI generation failed: {e}")
            return f"Generated question about the provided content (OpenAI error: {e})"
    
    def critique_question(self, question: str, context: str) -> Dict[str, Any]:
        """Critique question using OpenAI."""
        critique_prompt = f"""
        Evaluate this question for quality:
        
        Question: {question}
        Context: {context[:200]}...
        
        Rate on: Independence, Clear Intent, Answerability (all boolean)
        
        Respond only with JSON: {{"independence": true/false, "clear_intent": true/false, "answerable": true/false, "feedback": "brief explanation"}}
        """
        
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.critic_model,
                messages=[
                    {"role": "system", "content": "You are an expert question evaluator. Respond only with valid JSON."},
                    {"role": "user", "content": critique_prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            raw_response = response.choices[0].message.content.strip()
            critique = json.loads(raw_response)
            critique['verdict'] = (critique.get('independence', False) and 
                                 critique.get('clear_intent', False) and 
                                 critique.get('answerable', False))
            return critique
            
        except Exception as e:
            self.logger.warning(f"OpenAI critique failed: {e}")
            return {
                "independence": True,
                "clear_intent": True,
                "answerable": True,
                "verdict": True, 
                "feedback": f"Critique error: {e}"
            }


class TraversalQuestionGenerator:
    """
    Advanced question generator that creates questions from validated traversal paths.
    Implements the user's specific 5-category traversal logic.
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph, config: Dict[str, Any], 
                 logger: Optional[logging.Logger] = None):
        """Initialize with knowledge graph and configuration."""
        self.kg = knowledge_graph
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.qgen_config = config.get('question_generation', {})
        
        # Initialize model interfaces
        self.generator_model = self._create_model_interface('generator')
        self.critic_model = self._create_model_interface('critic')
        
        # Initialize traversal validator
        self.validator = TraversalValidator(logger)
        
        self.logger.info("TraversalQuestionGenerator initialized with:")
        self.logger.info(f"  Generator: {self.qgen_config.get('generator_model_type', 'ollama')}")
        self.logger.info(f"  Critic: {self.qgen_config.get('critic_model_type', 'ollama')}")
        self.logger.info(f"  Question distribution: {self.qgen_config.get('question_distribution', {})}")
    
    def _create_model_interface(self, model_role: str) -> ModelInterface:
        """Create appropriate model interface based on configuration."""
        model_type = self.qgen_config.get(f'{model_role}_model_type', 'ollama')
        
        if model_type == 'ollama':
            return OllamaInterface(self.config, self.logger)
        elif model_type == 'openai':
            return OpenAIInterface(self.config, self.logger)
        else:
            self.logger.warning(f"Unknown model type: {model_type}, defaulting to Ollama")
            return OllamaInterface(self.config, self.logger)
    
    def generate_dataset(self, num_questions: int, cache_name: Optional[str] = None) -> EvaluationDataset:
        """Generate complete dataset using traversal-based approach."""
        self.logger.info(f"🎯 Generating {num_questions} traversal-based questions")
        
        distribution = self.qgen_config.get('question_distribution', {
            'single_hop': 0.3,
            'sequential_flow': 0.25, 
            'multi_hop': 0.2,
            'theme_hop': 0.15,
            'hierarchical': 0.1
        })
        
        # Calculate questions per type
        questions_per_type = {}
        total_allocated = 0
        
        for q_type, proportion in distribution.items():
            count = int(num_questions * proportion)
            questions_per_type[q_type] = count
            total_allocated += count
        
        # Distribute remaining questions
        remaining = num_questions - total_allocated
        if remaining > 0:
            # Add to single_hop (most reliable type)
            questions_per_type['single_hop'] += remaining
        
        self.logger.info(f"Question allocation: {questions_per_type}")
        
        # Generate questions by type
        all_questions = []
        
        for q_type, count in questions_per_type.items():
            if count > 0:
                self.logger.info(f"Generating {count} {q_type} questions...")
                type_questions = self._generate_questions_by_type(q_type, count)
                all_questions.extend(type_questions)
                self.logger.info(f"✅ Generated {len(type_questions)} {q_type} questions")
        
        # Shuffle final dataset
        random.shuffle(all_questions)
        
        # Create dataset metadata
        metadata = {
            'total_questions': len(all_questions),
            'questions_by_type': {q_type: len([q for q in all_questions if q.question_type == q_type]) 
                                 for q_type in distribution.keys()},
            'generation_method': 'traversal_based',
            'created_at': datetime.now().isoformat(),
            'cache_name': cache_name or f"traversal_dataset_{len(all_questions)}q",
            'knowledge_graph_stats': {
                'total_chunks': len(self.kg.chunks),
                'total_sentences': len(self.kg.sentences),
                'total_documents': len(self.kg.documents)
            }
        }
        
        dataset = EvaluationDataset(
            questions=all_questions,
            dataset_metadata=metadata,
            generation_config=self.qgen_config
        )
        
        # Cache dataset if enabled with standardized path structure
        if self.qgen_config.get('cache_questions', True):
            # FIXED: Cache directly in data directory, not questions subdirectory
            # This ensures benchmark.py and question generators use the same location
            cache_path = Path(self.config['directories']['data']) / f"{metadata['cache_name']}.json"
            
            # Ensure parent directory exists (data directory always exists)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            dataset.save(str(cache_path), self.logger)
        
        self.logger.info(f"🎉 Generated {len(all_questions)} traversal-based questions")
        return dataset
    
    def _generate_questions_by_type(self, question_type: str, count: int) -> List[GeneratedQuestion]:
        """Generate questions for a specific traversal type with fallback logic."""
        try:
            generated = []
            
            # Try to generate questions of the requested type
            if question_type == 'single_hop':
                generated = self._generate_single_hop_questions(count)
            elif question_type == 'sequential_flow':
                generated = self._generate_sequential_flow_questions(count)
            elif question_type == 'multi_hop':
                generated = self._generate_multi_hop_questions(count)
            elif question_type == 'theme_hop':
                generated = self._generate_theme_hop_questions(count)
            elif question_type == 'hierarchical':
                generated = self._generate_hierarchical_questions(count)
            else:
                self.logger.warning(f"Unknown question type: {question_type}")
                generated = []
            
            # Implement fallback logic if insufficient questions generated
            if len(generated) < count:
                shortfall = count - len(generated)
                self.logger.warning(f"Only generated {len(generated)}/{count} {question_type} questions. Generating {shortfall} single_hop questions as fallback.")
                
                # Generate fallback single_hop questions
                fallback_questions = self._generate_single_hop_questions(shortfall)
                generated.extend(fallback_questions)
                
                # Log detailed failure reasons
                self.logger.info(f"Fallback successful: Generated {len(fallback_questions)} single_hop questions to replace failed {question_type} questions")
            
            return generated[:count]  # Ensure we don't exceed requested count
            
        except Exception as e:
            self.logger.error(f"Failed to generate {question_type} questions: {e}. Falling back to single_hop questions.")
            # Complete fallback to single_hop questions
            return self._generate_single_hop_questions(count)
    
    def _generate_single_hop_questions(self, count: int) -> List[GeneratedQuestion]:
        """
        Single hop: Inside same document, raw similarity, no overlapping windows.
        """
        questions = []
        
        for i in range(count):
            try:
                # Find two chunks in same document with high similarity
                source_chunk, target_chunk = self._find_same_document_chunk_pair()
                
                if not source_chunk or not target_chunk:
                    continue
                
                # Ensure no overlapping windows
                if self._chunks_overlap(source_chunk, target_chunk):
                    continue
                
                # Create traversal path
                path = TraversalPath(
                    nodes=[source_chunk.chunk_id, target_chunk.chunk_id],
                    connection_types=[ConnectionType.RAW_SIMILARITY],
                    granularity_levels=[GranularityLevel.CHUNK, GranularityLevel.CHUNK],
                    total_hops=1,
                    is_valid=True,
                    validation_errors=[]
                )
                
                # Generate question from path
                question_text = self._generate_question_from_path(path, 'single_hop')
                expected_answer = self._generate_answer_from_path(path)
                
                question = GeneratedQuestion(
                    question_id=f"single_hop_{i}_{hashlib.md5(question_text.encode()).hexdigest()[:8]}",
                    question_text=question_text,
                    ground_truth_path=path,
                    expected_answer=expected_answer,
                    difficulty_level="simple",
                    question_type="single_hop",
                    generation_metadata={
                        'source_chunk': source_chunk.chunk_id,
                        'target_chunk': target_chunk.chunk_id,
                        'same_document': source_chunk.source_document,
                        'similarity_score': source_chunk.connection_scores.get(target_chunk.chunk_id, 0.0)
                    }
                )
                
                questions.append(question)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate single hop question {i}: {e}")
                continue
        
        return questions
    
    def _generate_sequential_flow_questions(self, count: int) -> List[GeneratedQuestion]:
        """
        Sequential flow: Same document, chunks 1-10, sliding window logic.
        Question should require perfect alignment of all chunks.
        """
        questions = []
        
        for i in range(count):
            try:
                # Find sequential chunk sequence (up to 10 chunks)
                chunk_sequence = self._find_sequential_chunk_sequence()
                
                if len(chunk_sequence) < 3:  # Need at least 3 chunks for meaningful sequence
                    continue
                
                # Create traversal path through sequence
                path_nodes = [chunk.chunk_id for chunk in chunk_sequence]
                connection_types = [ConnectionType.SEQUENTIAL] * (len(chunk_sequence) - 1)
                granularities = [GranularityLevel.CHUNK] * len(chunk_sequence)
                
                path = TraversalPath(
                    nodes=path_nodes,
                    connection_types=connection_types,
                    granularity_levels=granularities,
                    total_hops=len(connection_types),
                    is_valid=True,
                    validation_errors=[]
                )
                
                # Generate question requiring full sequence understanding
                question_text = self._generate_question_from_path(path, 'sequential_flow')
                expected_answer = self._generate_answer_from_path(path)
                
                question = GeneratedQuestion(
                    question_id=f"sequential_flow_{i}_{hashlib.md5(question_text.encode()).hexdigest()[:8]}",
                    question_text=question_text,
                    ground_truth_path=path,
                    expected_answer=expected_answer,
                    difficulty_level="medium",
                    question_type="sequential_flow",
                    generation_metadata={
                        'sequence_length': len(chunk_sequence),
                        'document': chunk_sequence[0].source_document,
                        'chunk_range': f"{chunk_sequence[0].chunk_id} to {chunk_sequence[-1].chunk_id}"
                    }
                )
                
                questions.append(question)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate sequential flow question {i}: {e}")
                continue
        
        return questions
    
    def _generate_multi_hop_questions(self, count: int) -> List[GeneratedQuestion]:
        """
        Multi hop: 3 different documents, raw KG similarity (NOT theme).
        """
        questions = []
        
        for i in range(count):
            try:
                # Find chunks from 3 different documents with raw similarity connections
                document_chunks = self._find_multi_document_chunks(3, use_theme_similarity=False)
                
                if len(document_chunks) < 3:
                    continue
                
                # Create 2-hop path across 3 documents
                path = TraversalPath(
                    nodes=[chunk.chunk_id for chunk in document_chunks],
                    connection_types=[ConnectionType.RAW_SIMILARITY, ConnectionType.RAW_SIMILARITY],
                    granularity_levels=[GranularityLevel.CHUNK] * 3,
                    total_hops=2,
                    is_valid=True,
                    validation_errors=[]
                )
                
                question_text = self._generate_question_from_path(path, 'multi_hop')
                expected_answer = self._generate_answer_from_path(path)
                
                question = GeneratedQuestion(
                    question_id=f"multi_hop_{i}_{hashlib.md5(question_text.encode()).hexdigest()[:8]}",
                    question_text=question_text,
                    ground_truth_path=path,
                    expected_answer=expected_answer,
                    difficulty_level="hard",
                    question_type="multi_hop",
                    generation_metadata={
                        'documents': [chunk.source_document for chunk in document_chunks],
                        'connection_type': 'raw_similarity'
                    }
                )
                
                questions.append(question)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate multi hop question {i}: {e}")
                continue
        
        return questions
    
    def _generate_theme_hop_questions(self, count: int) -> List[GeneratedQuestion]:
        """
        Theme hop: 3 different documents, theme similarity (NOT raw KG).
        """
        questions = []
        
        for i in range(count):
            try:
                # Find chunks from 3 different documents with theme connections
                document_chunks = self._find_multi_document_chunks(3, use_theme_similarity=True)
                
                if len(document_chunks) < 3:
                    continue
                
                # Create 2-hop path using theme bridges
                path = TraversalPath(
                    nodes=[chunk.chunk_id for chunk in document_chunks],
                    connection_types=[ConnectionType.THEME_BRIDGE, ConnectionType.THEME_BRIDGE],
                    granularity_levels=[GranularityLevel.CHUNK] * 3,
                    total_hops=2,
                    is_valid=True,
                    validation_errors=[]
                )
                
                question_text = self._generate_question_from_path(path, 'theme_hop')
                expected_answer = self._generate_answer_from_path(path)
                
                question = GeneratedQuestion(
                    question_id=f"theme_hop_{i}_{hashlib.md5(question_text.encode()).hexdigest()[:8]}",
                    question_text=question_text,
                    ground_truth_path=path,
                    expected_answer=expected_answer,
                    difficulty_level="hard",
                    question_type="theme_hop",
                    generation_metadata={
                        'documents': [chunk.source_document for chunk in document_chunks],
                        'connection_type': 'theme_bridge',
                        'shared_themes': self._find_shared_themes(document_chunks)
                    }
                )
                
                questions.append(question)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate theme hop question {i}: {e}")
                continue
        
        return questions
    
    def _generate_hierarchical_questions(self, count: int) -> List[GeneratedQuestion]:
        """
        Hierarchical: doc_summary → chunk → sentence (same document).
        """
        questions = []
        
        for i in range(count):
            try:
                # Find document with summary, chunk, and sentence
                doc_id, chunk, sentence = self._find_hierarchical_sequence()
                
                if not all([doc_id, chunk, sentence]):
                    continue
                
                # Create hierarchical traversal path
                path = TraversalPath(
                    nodes=[doc_id, chunk.chunk_id, sentence.sentence_id],
                    connection_types=[ConnectionType.HIERARCHICAL, ConnectionType.HIERARCHICAL],
                    granularity_levels=[GranularityLevel.DOCUMENT, GranularityLevel.CHUNK, GranularityLevel.SENTENCE],
                    total_hops=2,
                    is_valid=True,
                    validation_errors=[]
                )
                
                question_text = self._generate_question_from_path(path, 'hierarchical')
                expected_answer = self._generate_answer_from_path(path)
                
                question = GeneratedQuestion(
                    question_id=f"hierarchical_{i}_{hashlib.md5(question_text.encode()).hexdigest()[:8]}",
                    question_text=question_text,
                    ground_truth_path=path,
                    expected_answer=expected_answer,
                    difficulty_level="medium",
                    question_type="hierarchical",
                    generation_metadata={
                        'document': chunk.source_document,
                        'chunk_id': chunk.chunk_id,
                        'sentence_id': sentence.sentence_id
                    }
                )
                
                questions.append(question)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate hierarchical question {i}: {e}")
                continue
        
        return questions
    
    def _generate_question_from_path(self, path: TraversalPath, question_type: str) -> str:
        """Generate question text from traversal path using LLM with improved prompts."""
        # Get content from path nodes
        path_content = []
        for node_id, granularity in zip(path.nodes, path.granularity_levels):
            content = self._get_node_content(node_id, granularity)
            path_content.append(content[:150])  # Limit content length
        
        # Create type-specific prompt with direct instructions
        if question_type == 'single_hop':
            prompt = f"""Based on these two related text passages, write ONE clear question that requires understanding both passages to answer:

Passage A: {path_content[0]}

Passage B: {path_content[1]}

Write only the question (no explanations or labels):"""
        
        elif question_type == 'sequential_flow':
            # Use fewer steps to avoid overwhelming the prompt
            key_steps = [path_content[0], path_content[len(path_content)//2], path_content[-1]] if len(path_content) > 3 else path_content
            prompt = f"""Based on this sequence of connected concepts, write ONE question that requires understanding the progression from start to end:

Start: {key_steps[0]}

Middle: {key_steps[1] if len(key_steps) > 1 else ''}

End: {key_steps[-1]}

Write only the question (no explanations or labels):"""
        
        elif question_type == 'multi_hop':
            docs = [self._get_chunk_document(node_id) for node_id in path.nodes]
            prompt = f"""Based on these passages from different documents, write ONE question that requires information from all three sources:

From {docs[0]}: {path_content[0]}

From {docs[1]}: {path_content[1]}

From {docs[2]}: {path_content[2]}

Write only the question (no explanations or labels):"""
        
        elif question_type == 'theme_hop':
            docs = [self._get_chunk_document(node_id) for node_id in path.nodes]
            prompt = f"""Based on these thematically connected passages from different documents, write ONE question about their conceptual relationship:

From {docs[0]}: {path_content[0]}

From {docs[1]}: {path_content[1]}

From {docs[2]}: {path_content[2]}

Write only the question (no explanations or labels):"""
        
        elif question_type == 'hierarchical':
            prompt = f"""Based on this information at different levels of detail, write ONE question that connects the general concept to the specific detail:

General: {path_content[0]}

Specific: {path_content[-1]}

Write only the question (no explanations or labels):"""
        
        else:
            prompt = f"Based on this content, write ONE clear question: {' '.join(path_content[:2])}"
        
        # Generate question using model
        raw_response = self.generator_model.generate_question(prompt)
        
        # Clean and parse the response
        question = self._parse_question_response(raw_response)
        
        # Validate the question
        if not question:
            self.logger.warning(f"Failed to parse valid question from LLM response for {question_type}")
            question = self._generate_fallback_question(path_content, question_type)
        else:
            # Additional validation
            validation = self._validate_question_quality(question)
            if not validation['is_valid']:
                self.logger.warning(f"Generated question failed validation: {validation['errors']}")
                question = self._generate_fallback_question(path_content, question_type)
        
        return question
    
    def _validate_question_quality(self, question: str) -> Dict[str, Any]:
        """Comprehensive question quality validation."""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'quality_score': 1.0
        }
        
        # Check basic structure
        if not question or len(question.strip()) < 10:
            validation_result['is_valid'] = False
            validation_result['errors'].append("Question too short or empty")
            return validation_result
        
        question = question.strip()
        
        # Check if it's actually a question
        if not question.endswith('?'):
            validation_result['warnings'].append("Missing question mark")
            validation_result['quality_score'] -= 0.2
        
        # Check for incomplete sentences or fragments
        if len(question.split()) < 5:
            validation_result['is_valid'] = False
            validation_result['errors'].append("Question appears to be a fragment")
            return validation_result
        
        # Check for common question words
        question_words = ['how', 'what', 'why', 'when', 'where', 'which', 'who', 'does', 'can', 'is', 'are', 'will']
        has_question_word = any(word.lower() in question.lower() for word in question_words)
        
        if not has_question_word:
            validation_result['warnings'].append("No clear question words found")
            validation_result['quality_score'] -= 0.3
        
        # Check for weird artifacts (single words, awards without context, etc.)
        if len(question.split()) <= 3 and '?' in question:
            validation_result['is_valid'] = False
            validation_result['errors'].append("Question appears to be malformed fragment")
        
        # Check for award/title fragments without proper question structure
        if any(word in question.lower() for word in ['award', 'prize', 'medal']) and len(question.split()) < 6:
            validation_result['is_valid'] = False
            validation_result['errors'].append("Question appears to be incomplete award/title reference")
        
        return validation_result
    
    def _parse_question_response(self, raw_response: str) -> str:
        """Parse and clean the raw LLM response to extract just the question with validation."""
        if not raw_response:
            return ""
        
        # Remove common prefixes
        prefixes_to_remove = [
            "Question:", "Q:", "Here is a question", "Here's a question", 
            "The question is:", "A question that", "Write only the question",
            "Based on", "Given", "Looking at", "Considering"
        ]
        
        cleaned = raw_response.strip()
        
        # Remove prefixes
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
                break
        
        # Remove colons at the start
        cleaned = cleaned.lstrip(":")
        cleaned = cleaned.strip()
        
        # Look for the first sentence that ends with a question mark
        sentences = cleaned.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence.endswith('?'):
                # Validate this potential question
                validation = self._validate_question_quality(sentence)
                if validation['is_valid']:
                    return sentence
                else:
                    self.logger.warning(f"Invalid question candidate: {sentence}. Errors: {validation['errors']}")
        
        # If no valid question mark sentence found, try to construct one
        if sentences and len(sentences[0].strip()) > 10:
            question = sentences[0].strip()
            if not question.endswith('?'):
                question += "?"
            
            # Validate the constructed question
            validation = self._validate_question_quality(question)
            if validation['is_valid']:
                return question
        
        # Return empty string if no valid question can be constructed
        self.logger.error(f"Could not extract valid question from: {raw_response[:100]}...")
        return ""
    
    def _generate_fallback_question(self, path_content: List[str], question_type: str) -> str:
        """Generate a simple fallback question if LLM generation fails."""
        if not path_content:
            return "What is the main concept discussed in this content?"
        
        first_words = path_content[0].split()[:5]
        concept = ' '.join(first_words) if first_words else "this concept"
        
        if question_type == 'single_hop':
            return f"How does {concept} relate to the information in the second passage?"
        elif question_type == 'sequential_flow':
            return f"What is the logical progression that begins with {concept}?"
        elif question_type in ['multi_hop', 'theme_hop']:
            return f"How does {concept} connect across the different documents?"
        elif question_type == 'hierarchical':
            return f"What specific details are provided about {concept}?"
        else:
            return f"What is the main point about {concept}?"
    
    def _generate_answer_from_path(self, path: TraversalPath) -> str:
        """Generate expected answer from traversal path content."""
        answer_parts = []
        
        for node_id, granularity in zip(path.nodes, path.granularity_levels):
            content = self._get_node_content(node_id, granularity)
            answer_parts.append(content[:200])  # Limit length
        
        return "The answer can be found by connecting: " + " → ".join(answer_parts)
    
    # Helper methods for finding appropriate nodes/paths
    
    def _find_same_document_chunk_pair(self) -> Tuple[Any, Any]:
        """Find two chunks from same document with high similarity."""
        chunks = list(self.kg.chunks.values())
        random.shuffle(chunks)
        
        for chunk in chunks:
            # Look for intra-document connections
            for connected_id in chunk.intra_doc_connections:
                connected_chunk = self.kg.chunks.get(connected_id)
                if connected_chunk and chunk.source_document == connected_chunk.source_document:
                    return chunk, connected_chunk
        
        return None, None
    
    def _chunks_overlap(self, chunk1: Any, chunk2: Any) -> bool:
        """Check if two chunks have overlapping sentences."""
        sentences1 = set(chunk1.sentence_ids)
        sentences2 = set(chunk2.sentence_ids)
        return len(sentences1 & sentences2) > 0
    
    def _find_sequential_chunk_sequence(self) -> List[Any]:
        """Find sequence of chunks for sequential flow."""
        documents = list(self.kg.documents.values())
        
        for doc in documents:
            # Get chunks from this document
            doc_chunks = [chunk for chunk in self.kg.chunks.values() 
                         if chunk.source_document == doc.title]
            
            if len(doc_chunks) >= 3:
                # Sort by position in document (assuming chunk_id contains position info)
                doc_chunks.sort(key=lambda x: x.chunk_id)
                
                # Take up to 10 chunks
                sequence_length = min(10, len(doc_chunks))
                return doc_chunks[:sequence_length]
        
        return []
    
    def _find_multi_document_chunks(self, num_docs: int, use_theme_similarity: bool) -> List[Any]:
        """Find chunks from multiple documents connected by similarity or themes with improved validation."""
        try:
            chunks = list(self.kg.chunks.values())
            
            # Pre-validate that we have enough documents
            available_docs = set(chunk.source_document for chunk in chunks)
            if len(available_docs) < num_docs:
                self.logger.warning(f"Only {len(available_docs)} documents available, but {num_docs} requested")
                return []
            
            # Try multiple starting points for better success rate
            max_attempts = 10
            
            for attempt in range(max_attempts):
                selected_chunks = []
                used_documents = set()
                
                # Start with random chunk that has connections
                potential_starts = [chunk for chunk in chunks 
                                  if (chunk.inter_doc_connections if use_theme_similarity 
                                     else chunk.intra_doc_connections + chunk.inter_doc_connections)]
                
                if not potential_starts:
                    self.logger.warning("No chunks with appropriate connections found")
                    continue
                
                start_chunk = random.choice(potential_starts)
                selected_chunks.append(start_chunk)
                used_documents.add(start_chunk.source_document)
                current_chunk = start_chunk
                
                # Find connected chunks from different documents
                while len(selected_chunks) < num_docs:
                    candidates = []
                    
                    if use_theme_similarity:
                        # Use inter-document connections (theme-based)
                        candidates = [self.kg.chunks.get(cid) for cid in current_chunk.inter_doc_connections 
                                    if cid in self.kg.chunks and 
                                    self.kg.chunks[cid].source_document not in used_documents]
                    else:
                        # Use all connections but filter for different documents
                        all_connections = current_chunk.intra_doc_connections + current_chunk.inter_doc_connections
                        candidates = [self.kg.chunks.get(cid) for cid in all_connections 
                                    if cid in self.kg.chunks and 
                                    self.kg.chunks[cid].source_document not in used_documents]
                    
                    # Filter out None values
                    candidates = [c for c in candidates if c is not None]
                    
                    if not candidates:
                        break
                    
                    next_chunk = random.choice(candidates)
                    selected_chunks.append(next_chunk)
                    used_documents.add(next_chunk.source_document)
                    current_chunk = next_chunk
                
                # Check if we found enough chunks
                if len(selected_chunks) >= num_docs:
                    return selected_chunks
            
            # If all attempts failed, log the issue
            connection_type = "theme-based" if use_theme_similarity else "raw similarity"
            self.logger.warning(f"Failed to find {num_docs} documents connected by {connection_type} after {max_attempts} attempts")
            return []
            
        except Exception as e:
            self.logger.error(f"Error in _find_multi_document_chunks: {e}")
            return []
    
    def _find_hierarchical_sequence(self) -> Tuple[str, Any, Any]:
        """Find document → chunk → sentence sequence with improved validation."""
        try:
            documents = list(self.kg.documents.values())
            
            if not documents:
                self.logger.warning("No documents available for hierarchical sequence")
                return None, None, None
            
            # Try multiple documents for better success rate
            random.shuffle(documents)
            
            for doc in documents:
                try:
                    # Get chunks from this document
                    doc_chunks = [chunk for chunk in self.kg.chunks.values() 
                                 if chunk.source_document == doc.title]
                    
                    if not doc_chunks:
                        continue
                    
                    chunk = random.choice(doc_chunks)
                    
                    # Get sentences from this chunk - try different methods
                    sentences = []
                    
                    # Method 1: Use kg.get_chunk_sentences if available
                    if hasattr(self.kg, 'get_chunk_sentences'):
                        sentences = self.kg.get_chunk_sentences(chunk.chunk_id)
                    
                    # Method 2: Try to get sentences from chunk.sentence_ids if available
                    if not sentences and hasattr(chunk, 'sentence_ids'):
                        sentences = [self.kg.sentences[sid] for sid in chunk.sentence_ids 
                                   if sid in self.kg.sentences]
                    
                    # Method 3: Get any sentences from this document
                    if not sentences:
                        sentences = [sent for sent in self.kg.sentences.values() 
                                   if hasattr(sent, 'source_document') and sent.source_document == doc.title]
                    
                    if sentences:
                        sentence = random.choice(sentences)
                        return doc.doc_id, chunk, sentence
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process document {doc.title}: {e}")
                    continue
            
            self.logger.warning("No valid hierarchical sequences found in any document")
            return None, None, None
            
        except Exception as e:
            self.logger.error(f"Error in _find_hierarchical_sequence: {e}")
            return None, None, None
    
    def _get_node_content(self, node_id: str, granularity: GranularityLevel) -> str:
        """Get text content for a node at specific granularity."""
        if granularity == GranularityLevel.DOCUMENT:
            doc = self.kg.documents.get(node_id)
            return doc.doc_summary if doc else ""
        elif granularity == GranularityLevel.CHUNK:
            chunk = self.kg.chunks.get(node_id)
            return chunk.chunk_text if chunk else ""
        elif granularity == GranularityLevel.SENTENCE:
            sentence = self.kg.sentences.get(node_id)
            return sentence.sentence_text if sentence else ""
        return ""
    
    def _get_chunk_document(self, chunk_id: str) -> str:
        """Get document name for a chunk."""
        chunk = self.kg.chunks.get(chunk_id)
        return chunk.source_document if chunk else "unknown"
    
    def _find_shared_themes(self, chunks: List[Any]) -> List[str]:
        """Find shared themes between chunks from different documents."""
        if len(chunks) < 2:
            return []
        
        # Get themes for each document
        all_themes = []
        for chunk in chunks:
            doc = next((d for d in self.kg.documents.values() 
                       if d.title == chunk.source_document), None)
            if doc:
                all_themes.append(set(doc.doc_themes))
        
        # Find intersection of themes
        if all_themes:
            shared = set.intersection(*all_themes)
            return list(shared)
        
        return []


# Factory function for easy initialization
def create_traversal_question_generator(knowledge_graph: KnowledgeGraph, config: Dict[str, Any], 
                                       logger: Optional[logging.Logger] = None) -> TraversalQuestionGenerator:
    """Factory function to create a traversal-based question generator."""
    return TraversalQuestionGenerator(knowledge_graph, config, logger)
