#!/usr/bin/env python3
"""
Knowledge Graph Construction
===========================

RAGAS-compatible knowledge graph construction for the semantic RAG pipeline.
Builds unified knowledge graphs with entities, relationships, and hierarchical structure
for both retrieval and question generation.
"""

import json
import hashlib
import time
import logging
import re
import heapq
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

from models import ChunkEmbedding


@dataclass
class KGNode:
    """RAGAS-compatible knowledge graph node."""
    id: str  # RAGAS format
    type: str  # RAGAS format - "DOCUMENT", "CHUNK", etc.
    properties: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'type': self.type,
            'properties': self.properties
        }


@dataclass  
class KGRelationship:
    """RAGAS-compatible knowledge graph relationship."""
    source: str  # RAGAS format
    target: str  # RAGAS format
    type: str  # RAGAS format
    properties: Dict[str, Any]
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'source': self.source,
            'target': self.target,
            'type': self.type,
            'properties': self.properties,
            'weight': self.weight
        }


class BaseExtractor(ABC):
    """Base class for information extractors."""
    
    @abstractmethod
    def extract(self, text: str) -> Dict[str, Any]:
        """Extract information from text."""
        pass


class NERExtractor(BaseExtractor):
    """Named Entity Recognition extractor using spaCy."""
    
    def __init__(self):
        """Initialize the NER extractor."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback to simple pattern matching if spaCy model not available
            self.nlp = None
    
    def extract(self, text: str) -> Dict[str, Any]:
        """Extract named entities from text."""
        if self.nlp:
            return self._extract_with_spacy(text)
        else:
            return self._extract_with_patterns(text)
    
    def _extract_with_spacy(self, text: str) -> Dict[str, Any]:
        """Extract entities using spaCy."""
        doc = self.nlp(text)
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],  # Geopolitical entities (countries, cities, states)
            'MISC': []
        }
        
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE']:
                entities[ent.label_].append(ent.text)
            else:
                entities['MISC'].append(ent.text)
        
        # Remove duplicates and clean up
        for key in entities:
            entities[key] = list(set(entities[key]))
            
        return {'entities': entities}
    
    def _extract_with_patterns(self, text: str) -> Dict[str, Any]:
        """Fallback entity extraction using patterns."""
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],
            'MISC': []
        }
        
        # Simple pattern-based extraction for common terms
        tech_terms = [
            'machine learning', 'artificial intelligence', 'neural networks',
            'deep learning', 'natural language processing', 'computer vision',
            'algorithms', 'data science', 'statistics', 'supervised learning',
            'unsupervised learning', 'reinforcement learning', 'training data',
            'feature extraction', 'classification', 'regression', 'clustering'
        ]
        
        text_lower = text.lower()
        for term in tech_terms:
            if term in text_lower:
                entities['MISC'].append(term)
        
        # Look for capitalized words that might be names/organizations
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', text)
        for word in capitalized_words:
            if len(word) > 2 and word not in ['The', 'This', 'That', 'These', 'Those']:
                entities['PERSON'].append(word)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
            
        return {'entities': entities}


class KeyphraseExtractor(BaseExtractor):
    """Extract key phrases using TF-IDF."""
    
    def __init__(self, max_features: int = 20):
        """Initialize the keyphrase extractor."""
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1
        )
    
    def extract(self, text: str) -> Dict[str, Any]:
        """Extract key phrases from text."""
        try:
            # Fit and transform the text
            tfidf_matrix = self.vectorizer.fit_transform([text])
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get TF-IDF scores
            scores = tfidf_matrix.toarray()[0]
            
            # Get top phrases
            top_indices = scores.argsort()[-10:][::-1]  # Top 10 phrases
            keyphrases = [feature_names[i] for i in top_indices if scores[i] > 0]
            
            return {'keyphrases': keyphrases}
            
        except Exception as e:
            # Fallback to simple term extraction
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            word_freq = defaultdict(int)
            for word in words:
                if word not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'man', 'end', 'few', 'got', 'own', 'say', 'she', 'too', 'use']:
                    word_freq[word] += 1
            
            # Get top words
            keyphrases = sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:10]
            return {'keyphrases': keyphrases}


class ThemeExtractor(BaseExtractor):
    """Extract themes/topics from text."""
    
    def __init__(self):
        """Initialize the theme extractor."""
        self.theme_keywords = {
            'machine_learning': ['machine learning', 'ml', 'algorithm', 'model', 'training', 'prediction'],
            'artificial_intelligence': ['artificial intelligence', 'ai', 'intelligent', 'automation', 'cognitive'],
            'neural_networks': ['neural network', 'neuron', 'layer', 'activation', 'backpropagation'],
            'data_science': ['data science', 'data analysis', 'statistics', 'visualization', 'analytics'],
            'deep_learning': ['deep learning', 'cnn', 'rnn', 'transformer', 'attention'],
            'nlp': ['natural language processing', 'nlp', 'text', 'language', 'linguistic'],
            'computer_vision': ['computer vision', 'image', 'visual', 'recognition', 'detection'],
            'mathematics': ['mathematics', 'mathematical', 'equation', 'formula', 'theorem'],
            'programming': ['programming', 'code', 'software', 'development', 'implementation'],
            'research': ['research', 'study', 'experiment', 'analysis', 'investigation']
        }
    
    def extract(self, text: str) -> Dict[str, Any]:
        """Extract themes from text."""
        text_lower = text.lower()
        themes = []
        theme_scores = {}
        
        for theme, keywords in self.theme_keywords.items():
            score = 0
            for keyword in keywords:
                score += text_lower.count(keyword)
            
            if score > 0:
                themes.append(theme)
                theme_scores[theme] = score
        
        # Sort themes by relevance
        themes.sort(key=lambda x: theme_scores[x], reverse=True)
        
        return {'themes': themes[:5], 'theme_scores': theme_scores}


class SummaryExtractor(BaseExtractor):
    """Extract/generate summaries from text."""
    
    def __init__(self, max_sentences: int = 2):
        """Initialize the summary extractor."""
        self.max_sentences = max_sentences
    
    def extract(self, text: str) -> Dict[str, Any]:
        """Extract summary from text."""
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) <= self.max_sentences:
            summary = text.strip()
        else:
            # Simple extractive summarization - take first and a middle sentence
            summary_sentences = [sentences[0]]
            if len(sentences) > 2:
                middle_idx = len(sentences) // 2
                summary_sentences.append(sentences[middle_idx])
            
            summary = ' '.join(summary_sentences)
        
        return {'summary': summary}


class BaseRelationshipBuilder(ABC):
    """Base class for relationship builders."""
    
    @abstractmethod
    def build_relationships(self, nodes: List[KGNode]) -> List[KGRelationship]:
        """Build relationships between nodes."""
        pass


class EntitySimilarityBuilder(BaseRelationshipBuilder):
    """Build sparse relationships based on entity overlap using top-k selection."""
    
    def __init__(self, config: Dict[str, Any], entity_types: List[str] = None):
        """Initialize the sparse entity similarity builder."""
        self.entity_types = entity_types or ['PERSON', 'ORG', 'GPE', 'MISC']
        
        # Get sparsity settings from config
        sparsity_config = config.get('knowledge_graph', {}).get('sparsity', {})
        self.top_k = sparsity_config.get('relationship_limits', {}).get('entity_similarity', 10)
        self.min_similarity = sparsity_config.get('min_thresholds', {}).get('entity_similarity', 0.3)
        
        print(f"EntitySimilarityBuilder: top_k={self.top_k}, min_similarity={self.min_similarity}")
    
    def build_relationships(self, nodes: List[KGNode]) -> List[KGRelationship]:
        """Build sparse entity-based relationships using top-k selection."""
        relationships = []
        total_computed = 0
        total_added = 0
        
        print(f"Building entity relationships for {len(nodes)} nodes...")
        
        # For each node, find its top-k most similar neighbors by entity overlap
        for i, source_node in enumerate(nodes):
            if i % 500 == 0:
                print(f"Processing entity similarities for node {i+1}/{len(nodes)}...")
            
            # Compute similarities to all other nodes
            similarities = []
            
            for j, target_node in enumerate(nodes):
                if i != j:  # Skip self
                    similarity = self._compute_entity_similarity(source_node, target_node)
                    total_computed += 1
                    
                    if similarity >= self.min_similarity:
                        similarities.append((similarity, target_node, j))
            
            # Keep only top-k most similar
            top_k_similar = heapq.nlargest(self.top_k, similarities, key=lambda x: x[0])
            
            # Create relationships for top-k
            for similarity, target_node, target_idx in top_k_similar:
                relationships.append(KGRelationship(
                    source=source_node.id,
                    target=target_node.id,
                    type="entity_similarity",
                    properties={
                        'similarity_score': similarity, 
                        'entity_types': self.entity_types,
                        'rank': len([r for r in relationships if r.source == source_node.id]) + 1
                    },
                    weight=similarity
                ))
                total_added += 1
        
        print(f"EntitySimilarityBuilder: computed {total_computed:,} pairs, added {total_added:,} relationships (reduction: {100*(1-total_added/max(total_computed,1)):.1f}%)")
        return relationships
    
    def _compute_entity_similarity(self, node_a: KGNode, node_b: KGNode) -> float:
        """Compute Jaccard similarity between entities."""
        entities_a = node_a.properties.get('entities', {})
        entities_b = node_b.properties.get('entities', {})
        
        all_entities_a = set()
        all_entities_b = set()
        
        for entity_type in self.entity_types:
            all_entities_a.update(entities_a.get(entity_type, []))
            all_entities_b.update(entities_b.get(entity_type, []))
        
        # Convert to lowercase for comparison
        all_entities_a = {e.lower() for e in all_entities_a}
        all_entities_b = {e.lower() for e in all_entities_b}
        
        if not all_entities_a and not all_entities_b:
            return 0.0
        
        intersection = len(all_entities_a.intersection(all_entities_b))
        union = len(all_entities_a.union(all_entities_b))
        
        return intersection / union if union > 0 else 0.0


class ThematicSimilarityBuilder(BaseRelationshipBuilder):
    """Build sparse relationships based on theme overlap using top-k selection."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the sparse thematic similarity builder."""
        # Get sparsity settings from config
        sparsity_config = config.get('knowledge_graph', {}).get('sparsity', {})
        self.top_k = sparsity_config.get('relationship_limits', {}).get('thematic_similarity', 5)
        self.min_similarity = sparsity_config.get('min_thresholds', {}).get('thematic_similarity', 0.6)
        
        print(f"ThematicSimilarityBuilder: top_k={self.top_k}, min_similarity={self.min_similarity}")
    
    def build_relationships(self, nodes: List[KGNode]) -> List[KGRelationship]:
        """Build sparse theme-based relationships using top-k selection."""
        relationships = []
        total_computed = 0
        total_added = 0
        
        # Only consider chunk nodes for efficiency
        chunk_nodes = [n for n in nodes if n.type == 'CHUNK']
        print(f"Building thematic relationships for {len(chunk_nodes)} chunk nodes...")
        
        # For each node, find its top-k most similar neighbors by theme overlap
        for i, source_node in enumerate(chunk_nodes):
            if i % 500 == 0:
                print(f"Processing thematic similarities for node {i+1}/{len(chunk_nodes)}...")
            
            # Compute similarities to all other nodes
            similarities = []
            
            for j, target_node in enumerate(chunk_nodes):
                if i != j:  # Skip self
                    similarity = self._compute_theme_similarity(source_node, target_node)
                    total_computed += 1
                    
                    if similarity >= self.min_similarity:
                        similarities.append((similarity, target_node, j))
            
            # Keep only top-k most similar
            top_k_similar = heapq.nlargest(self.top_k, similarities, key=lambda x: x[0])
            
            # Create relationships for top-k
            for similarity, target_node, target_idx in top_k_similar:
                relationships.append(KGRelationship(
                    source=source_node.id,
                    target=target_node.id,
                    type="thematic_similarity",
                    properties={
                        'similarity_score': similarity,
                        'rank': len([r for r in relationships if r.source == source_node.id]) + 1
                    },
                    weight=similarity
                ))
                total_added += 1
        
        print(f"ThematicSimilarityBuilder: computed {total_computed:,} pairs, added {total_added:,} relationships (reduction: {100*(1-total_added/max(total_computed,1)):.1f}%)")
        return relationships
    
    def _compute_theme_similarity(self, node_a: KGNode, node_b: KGNode) -> float:
        """Compute theme similarity between nodes."""
        themes_a = set(node_a.properties.get('themes', []))
        themes_b = set(node_b.properties.get('themes', []))
        
        if not themes_a and not themes_b:
            return 0.0
        
        intersection = len(themes_a.intersection(themes_b))
        union = len(themes_a.union(themes_b))
        
        return intersection / union if union > 0 else 0.0


class EmbeddingSimilarityBuilder(BaseRelationshipBuilder):
    """Build relationships based on embedding similarity."""
    
    def __init__(self, similarity_data: Dict[str, Any], min_similarity: float = 0.5):
        """Initialize the embedding similarity builder."""
        self.similarity_data = similarity_data
        self.min_similarity = min_similarity
    
    def build_relationships(self, nodes: List[KGNode]) -> List[KGRelationship]:
        """Build embedding-based relationships."""
        relationships = []
        
        # Use existing similarity matrices from Phase 4
        for model_name, model_similarity_data in self.similarity_data.items():
            chunk_index_map = model_similarity_data['chunk_index_map']
            similarity_matrix = model_similarity_data['matrices']['combined']
            
            # Create chunk_id to node mapping
            chunk_to_node = {}
            for node in nodes:
                if node.type == 'CHUNK':
                    chunk_id = node.properties.get('chunk_id')
                    if chunk_id:
                        chunk_to_node[chunk_id] = node
            
            # Extract relationships from similarity matrix
            for chunk_id, chunk_idx in chunk_index_map.items():
                if chunk_id not in chunk_to_node:
                    continue
                
                source_node = chunk_to_node[chunk_id]
                similarity_row = similarity_matrix[chunk_idx].toarray().flatten()
                
                for target_idx, similarity_score in enumerate(similarity_row):
                    if similarity_score >= self.min_similarity and target_idx != chunk_idx:
                        # Find the target chunk
                        target_chunk_id = None
                        for cid, cidx in chunk_index_map.items():
                            if cidx == target_idx:
                                target_chunk_id = cid
                                break
                        
                        if target_chunk_id and target_chunk_id in chunk_to_node:
                            target_node = chunk_to_node[target_chunk_id]
                            
                            relationships.append(KGRelationship(
                                source=source_node.id,
                                target=target_node.id,
                                type="embedding_similarity",
                                properties={
                                    'similarity_score': float(similarity_score),
                                    'model_name': model_name
                                },
                                weight=float(similarity_score)
                            ))
        
        return relationships


class HierarchicalRelationshipBuilder(BaseRelationshipBuilder):
    """Build hierarchical relationships (document -> chunk)."""
    
    def build_relationships(self, nodes: List[KGNode]) -> List[KGRelationship]:
        """Build hierarchical relationships."""
        relationships = []
        
        # Group nodes by type
        document_nodes = [n for n in nodes if n.type == 'DOCUMENT']
        chunk_nodes = [n for n in nodes if n.type == 'CHUNK']
        
        # Create document -> chunk relationships
        for chunk_node in chunk_nodes:
            source_article = chunk_node.properties.get('source_article')
            if source_article:
                # Find matching document node
                for doc_node in document_nodes:
                    if doc_node.properties.get('title') == source_article:
                        relationships.append(KGRelationship(
                            source=doc_node.id,
                            target=chunk_node.id,
                            type="contains",
                            properties={'relationship': 'document_contains_chunk'},
                            weight=1.0
                        ))
                        break
        
        return relationships


class KnowledgeGraph:
    """RAGAS-compatible knowledge graph."""
    
    def __init__(self):
        """Initialize empty knowledge graph."""
        self.nodes: List[KGNode] = []
        self.relationships: List[KGRelationship] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_node(self, node: KGNode):
        """Add a node to the graph."""
        self.nodes.append(node)
    
    def add_relationship(self, relationship: KGRelationship):
        """Add a relationship to the graph."""
        self.relationships.append(relationship)
    
    def get_node(self, node_id: str) -> Optional[KGNode]:
        """Get a node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_neighbors(self, node_id: str, relationship_types: List[str] = None) -> List[KGNode]:
        """Get neighboring nodes."""
        neighbors = []
        
        for rel in self.relationships:
            if relationship_types and rel.type not in relationship_types:
                continue
                
            if rel.source == node_id:
                neighbor = self.get_node(rel.target)
                if neighbor:
                    neighbors.append(neighbor)
            elif rel.target == node_id:
                neighbor = self.get_node(rel.source)
                if neighbor:
                    neighbors.append(neighbor)
        
        return neighbors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'metadata': self.metadata,
            'nodes': [node.to_dict() for node in self.nodes],
            'relationships': [rel.to_dict() for rel in self.relationships]
        }
    
    def save(self, file_path: str):
        """Save knowledge graph to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, file_path: str) -> 'KnowledgeGraph':
        """Load knowledge graph from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        kg = cls()
        kg.metadata = data.get('metadata', {})
        
        # Load nodes
        for node_data in data.get('nodes', []):
            # Handle both old and new formats for backwards compatibility
            node_id = node_data.get('id') or node_data.get('node_id')
            node_type = node_data.get('type') or node_data.get('node_type')
            
            node = KGNode(
                id=node_id,
                type=node_type,
                properties=node_data['properties']
            )
            kg.add_node(node)
        
        # Load relationships
        for rel_data in data.get('relationships', []):
            # Handle both old and new formats for backwards compatibility
            source = rel_data.get('source') or rel_data.get('source_id')
            target = rel_data.get('target') or rel_data.get('target_id')
            rel_type = rel_data.get('type') or rel_data.get('relationship_type')
            
            rel = KGRelationship(
                source=source,
                target=target,
                type=rel_type,
                properties=rel_data['properties'],
                weight=rel_data.get('weight', 1.0)
            )
            kg.add_relationship(rel)
        
        return kg


class KnowledgeGraphBuilder:
    """Builder for constructing RAGAS-compatible knowledge graphs."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize the knowledge graph builder."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.kg_config = config.get('knowledge_graph', {})
        
        # Initialize extractors
        self.extractors = {
            'ner': NERExtractor(),
            'keyphrases': KeyphraseExtractor(),
            'themes': ThemeExtractor(),
            'summary': SummaryExtractor()
        }
        
        # Initialize relationship builders
        self.relationship_builders = {}
    
    def build_knowledge_graph(self, chunks: List[Dict[str, Any]], embeddings: Dict[str, List[ChunkEmbedding]], 
                            similarities: Dict[str, Any]) -> KnowledgeGraph:
        """Build complete knowledge graph from chunks."""
        start_time = time.time()
        
        self.logger.info("🏗️  Building RAGAS-compatible knowledge graph")
        
        # Create knowledge graph
        kg = KnowledgeGraph()
        
        # Step 1: Create document nodes
        document_nodes = self._create_document_nodes(chunks)
        for node in document_nodes:
            kg.add_node(node)
        
        # Step 2: Create chunk nodes with extracted information
        chunk_nodes = self._create_chunk_nodes(chunks, embeddings)
        for node in chunk_nodes:
            kg.add_node(node)
        
        # Step 3: Build relationships
        relationships = self._build_relationships(kg.nodes, similarities)
        for rel in relationships:
            kg.add_relationship(rel)
        
        # Step 4: Add metadata
        build_time = time.time() - start_time
        kg.metadata = {
            'created_at': datetime.now().isoformat(),
            'total_nodes': len(kg.nodes),
            'total_relationships': len(kg.relationships),
            'build_time': build_time,
            'config': self.kg_config,
            'node_types': {
                'DOCUMENT': len([n for n in kg.nodes if n.type == 'DOCUMENT']),
                'CHUNK': len([n for n in kg.nodes if n.type == 'CHUNK'])
            },
            'relationship_types': {
                rel_type: len([r for r in kg.relationships if r.type == rel_type])
                for rel_type in set(r.type for r in kg.relationships)
            }
        }
        
        self.logger.info(f"✅ Knowledge graph built: {len(kg.nodes)} nodes, {len(kg.relationships)} relationships in {build_time:.2f}s")
        
        return kg
    
    def _create_document_nodes(self, chunks: List[Dict[str, Any]]) -> List[KGNode]:
        """Create document-level nodes."""
        documents = {}
        
        # Group chunks by source article
        for chunk in chunks:
            article = chunk['source_article']
            if article not in documents:
                documents[article] = {
                    'title': article,
                    'chunks': []
                }
            documents[article]['chunks'].append(chunk)
        
        # Create document nodes
        document_nodes = []
        for article, doc_data in documents.items():
            # Combine all chunk text for document-level extraction
            full_text = ' '.join([chunk['text'] for chunk in doc_data['chunks']])
            
            # Extract information at document level
            extracted_info = {}
            for extractor_name, extractor in self.extractors.items():
                try:
                    info = extractor.extract(full_text)
                    extracted_info.update(info)
                except Exception as e:
                    self.logger.warning(f"Failed to extract {extractor_name} for document {article}: {e}")
            
            # Create document node
            node_id = f"doc_{hashlib.md5(article.encode()).hexdigest()[:8]}"
            node = KGNode(
                id=node_id,
                type="DOCUMENT",  # RAGAS format
                properties={
                    'title': article,
                    'page_content': full_text[:1000],  # RAGAS expects 'page_content'
                    'full_text': full_text[:1000],  # Keep for backwards compatibility
                    'chunk_count': len(doc_data['chunks']),
                    **extracted_info
                }
            )
            document_nodes.append(node)
        
        return document_nodes
    
    def _create_chunk_nodes(self, chunks: List[Dict[str, Any]], embeddings: Dict[str, List[ChunkEmbedding]]) -> List[KGNode]:
        """Create chunk-level nodes with extracted information."""
        chunk_nodes = []
        
        # Create embedding lookup
        embedding_lookup = {}
        for model_name, chunk_embeddings in embeddings.items():
            for chunk_emb in chunk_embeddings:
                embedding_lookup[chunk_emb.chunk_id] = chunk_emb
        
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            
            # Extract information from chunk
            extracted_info = {}
            for extractor_name, extractor in self.extractors.items():
                try:
                    info = extractor.extract(chunk['text'])
                    extracted_info.update(info)
                except Exception as e:
                    self.logger.warning(f"Failed to extract {extractor_name} for chunk {chunk_id}: {e}")
            
            # Create chunk node
            node_id = f"chunk_{hashlib.md5(chunk_id.encode()).hexdigest()[:8]}"
            properties = {
                # RAGAS required fields
                'page_content': chunk['text'],  # RAGAS expects 'page_content'
                
                # Your algorithm required fields (keep all of these)
                'chunk_id': chunk_id,
                'text': chunk['text'],  # Keep for backwards compatibility
                'source_article': chunk['source_article'],
                'source_sentences': chunk['source_sentences'],
                'anchor_sentence_idx': chunk['anchor_sentence_idx'],
                'window_position': chunk['window_position'],
                
                # RAGAS extracted information
                **extracted_info
            }
            
            # Add embedding info if available
            if chunk_id in embedding_lookup:
                chunk_emb = embedding_lookup[chunk_id]
                properties['embedding_available'] = True
                properties['embedding_model'] = list(embeddings.keys())[0]  # Use first model as reference
            
            node = KGNode(
                id=node_id,
                type="CHUNK",  # RAGAS format
                properties=properties
            )
            chunk_nodes.append(node)
        
        return chunk_nodes
    
    def _build_relationships(self, nodes: List[KGNode], similarities: Dict[str, Any]) -> List[KGRelationship]:
        """Build all relationships between nodes using sparse builders."""
        all_relationships = []
        
        self.logger.info("Building sparse relationships...")
        
        # Initialize sparse relationship builders with config
        builders = [
            EntitySimilarityBuilder(self.config),
            ThematicSimilarityBuilder(self.config),
            EmbeddingSimilarityBuilder(similarities, min_similarity=0.5),  # Keep existing
            HierarchicalRelationshipBuilder()
        ]
        
        # Build relationships with each builder
        for builder in builders:
            try:
                builder_start = time.time()
                relationships = builder.build_relationships(nodes)
                builder_time = time.time() - builder_start
                
                all_relationships.extend(relationships)
                self.logger.info(f"Built {len(relationships):,} {builder.__class__.__name__} relationships in {builder_time:.2f}s")
            except Exception as e:
                self.logger.warning(f"Failed to build relationships with {builder.__class__.__name__}: {e}")
        
        return all_relationships
