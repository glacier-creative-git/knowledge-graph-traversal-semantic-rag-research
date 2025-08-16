#!/usr/bin/env python3
"""
Multi-Dimensional Knowledge Graph Construction
=============================================

Revolutionary three-tier hierarchical architecture (Document → Chunk → Sentence)
with multi-dimensional relationship system for domain-agnostic semantic RAG.

Architecture:
- Hierarchical Structure Layer: Parent/child navigation across granularity levels
- Cosine Similarity Layer: Mathematical semantic relationships (from Phase 4)  
- Entity Overlap Layer: Factual bridges via shared entities (domain-agnostic NER)
- Domain Agnostic: Works with ANY content type without theme bias
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
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

from models import ChunkEmbedding


@dataclass
class KGNode:
    """Multi-dimensional knowledge graph node with hierarchical structure."""
    id: str
    type: str  # "DOCUMENT", "CHUNK", "SENTENCE"
    properties: Dict[str, Any]
    
    # Hierarchical relationships
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    hierarchy_level: int = 0  # 0=document, 1=chunk, 2=sentence
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'type': self.type,
            'properties': self.properties,
            'parent_id': self.parent_id,
            'children_ids': self.children_ids,
            'hierarchy_level': self.hierarchy_level
        }


@dataclass  
class KGRelationship:
    """Multi-dimensional knowledge graph relationship."""
    source: str
    target: str
    type: str  # "parent", "child", "cosine_similarity", "entity_overlap"
    properties: Dict[str, Any]
    weight: float = 1.0
    
    # Traversal metadata
    traversal_cost: float = 1.0
    bidirectional: bool = True
    context_dependent: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'source': self.source,
            'target': self.target,
            'type': self.type,
            'properties': self.properties,
            'weight': self.weight,
            'traversal_cost': self.traversal_cost,
            'bidirectional': self.bidirectional,
            'context_dependent': self.context_dependent
        }


class BaseExtractor(ABC):
    """Base class for domain-agnostic information extractors."""
    
    @abstractmethod
    def extract(self, text: str, granularity_level: str = "chunk") -> Dict[str, Any]:
        """Extract information from text at specified granularity level."""
        pass


class DomainAgnosticNERExtractor(BaseExtractor):
    """Completely domain-agnostic Named Entity Recognition."""
    
    def __init__(self):
        """Initialize domain-agnostic NER extractor."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.available = True
        except OSError:
            self.nlp = None
            self.available = False
    
    def extract(self, text: str, granularity_level: str = "chunk") -> Dict[str, Any]:
        """Extract entities with zero domain bias."""
        if self.available:
            return self._extract_with_spacy(text)
        else:
            return self._extract_with_patterns(text)
    
    def _extract_with_spacy(self, text: str) -> Dict[str, Any]:
        """Extract entities using spaCy with domain-agnostic categories."""
        doc = self.nlp(text)
        entities = {
            'PERSON': [],      # People (any domain)
            'ORG': [],         # Organizations (any domain)  
            'GPE': [],         # Geopolitical entities (countries, cities, states)
            'PRODUCT': [],     # Products, objects, things
            'EVENT': [],       # Named events
            'MISC': []         # Everything else
        }
        
        for ent in doc.ents:
            entity_text = ent.text.strip()
            if len(entity_text) < 2:  # Skip very short entities
                continue
                
            if ent.label_ in ['PERSON']:
                entities['PERSON'].append(entity_text)
            elif ent.label_ in ['ORG']:
                entities['ORG'].append(entity_text)
            elif ent.label_ in ['GPE']:
                entities['GPE'].append(entity_text)
            elif ent.label_ in ['PRODUCT']:
                entities['PRODUCT'].append(entity_text)
            elif ent.label_ in ['EVENT']:
                entities['EVENT'].append(entity_text)
            else:
                entities['MISC'].append(entity_text)
        
        # Remove duplicates and clean up
        for key in entities:
            entities[key] = list(set(entities[key]))
            
        return {'entities': entities}
    
    def _extract_with_patterns(self, text: str) -> Dict[str, Any]:
        """Fallback extraction using completely domain-agnostic patterns."""
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],
            'PRODUCT': [],
            'EVENT': [],
            'MISC': []
        }
        
        # Extract capitalized words/phrases (potential proper nouns)
        # This is domain-agnostic - works for any field
        capitalized_patterns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        for phrase in capitalized_patterns:
            phrase = phrase.strip()
            if len(phrase) < 3:
                continue
                
            # Very basic heuristics (domain-agnostic)
            if any(word in phrase.lower() for word in ['university', 'company', 'corporation', 'inc', 'ltd']):
                entities['ORG'].append(phrase)
            elif any(word in phrase.lower() for word in ['dr', 'professor', 'mr', 'ms', 'mrs']):
                entities['PERSON'].append(phrase)
            else:
                entities['MISC'].append(phrase)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
            
        return {'entities': entities}


class DomainAgnosticKeyphraseExtractor(BaseExtractor):
    """Extract key phrases without domain bias using TF-IDF."""
    
    def __init__(self, max_features: int = 15):
        """Initialize keyphrase extractor."""
        self.max_features = max_features
    
    def extract(self, text: str, granularity_level: str = "chunk") -> Dict[str, Any]:
        """Extract domain-agnostic key phrases."""
        try:
            # Adjust max features based on granularity
            if granularity_level == "sentence":
                max_features = min(5, self.max_features)
            elif granularity_level == "chunk":
                max_features = self.max_features
            else:  # document
                max_features = max(20, self.max_features)
            
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=1,
                token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # Only alphabetic tokens
            )
            
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top phrases with scores
            scored_phrases = [(feature_names[i], scores[i]) for i in range(len(scores)) if scores[i] > 0]
            scored_phrases.sort(key=lambda x: x[1], reverse=True)
            
            keyphrases = [phrase for phrase, score in scored_phrases[:max_features]]
            
            return {'keyphrases': keyphrases}
            
        except Exception as e:
            # Fallback to simple word frequency
            return self._fallback_keyphrases(text, granularity_level)
    
    def _fallback_keyphrases(self, text: str, granularity_level: str) -> Dict[str, Any]:
        """Fallback keyphrase extraction using word frequency."""
        # Common stop words to avoid
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 
            'boy', 'did', 'man', 'end', 'few', 'got', 'own', 'say', 'she', 'too', 
            'use', 'this', 'that', 'with', 'have', 'from', 'they', 'been', 'said',
            'each', 'which', 'their', 'time', 'will', 'about', 'would', 'there',
            'could', 'other', 'after', 'first', 'well', 'many', 'some', 'what'
        }
        
        # Extract words and calculate frequency
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = defaultdict(int)
        
        for word in words:
            if word not in stop_words:
                word_freq[word] += 1
        
        # Adjust number based on granularity
        if granularity_level == "sentence":
            max_phrases = 3
        elif granularity_level == "chunk":
            max_phrases = 8
        else:
            max_phrases = 15
        
        keyphrases = sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:max_phrases]
        return {'keyphrases': keyphrases}


class DomainAgnosticSummaryExtractor(BaseExtractor):
    """Generate extractive summaries without domain bias."""
    
    def __init__(self):
        """Initialize summary extractor."""
        pass
    
    def extract(self, text: str, granularity_level: str = "chunk") -> Dict[str, Any]:
        """Extract summary appropriate for granularity level."""
        sentences = nltk.sent_tokenize(text)
        
        if granularity_level == "sentence":
            # For sentences, just return the sentence itself
            summary = text.strip()
        elif granularity_level == "chunk":
            # For chunks, take first sentence + key middle sentence
            if len(sentences) <= 2:
                summary = text.strip()
            else:
                # First sentence + middle sentence with highest word overlap with first
                first_sentence = sentences[0]
                best_middle = self._find_best_supporting_sentence(first_sentence, sentences[1:])
                summary = f"{first_sentence} {best_middle}" if best_middle else first_sentence
        else:  # document
            # For documents, take first + middle + last sentences
            if len(sentences) <= 3:
                summary = text.strip()
            else:
                key_sentences = [
                    sentences[0],  # First
                    sentences[len(sentences) // 2],  # Middle
                    sentences[-1]  # Last
                ]
                summary = ' '.join(key_sentences)
        
        return {'summary': summary[:500]}  # Limit length
    
    def _find_best_supporting_sentence(self, first_sentence: str, other_sentences: List[str]) -> str:
        """Find sentence with highest word overlap with first sentence."""
        if not other_sentences:
            return ""
        
        first_words = set(re.findall(r'\b[a-zA-Z]+\b', first_sentence.lower()))
        
        best_sentence = ""
        best_overlap = 0
        
        for sentence in other_sentences:
            sentence_words = set(re.findall(r'\b[a-zA-Z]+\b', sentence.lower()))
            overlap = len(first_words & sentence_words)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence
        
        return best_sentence


class BaseRelationshipBuilder(ABC):
    """Base class for multi-dimensional relationship builders."""
    
    @abstractmethod
    def build_relationships(self, nodes: List[KGNode]) -> List[KGRelationship]:
        """Build relationships between nodes."""
        pass


class HierarchicalRelationshipBuilder(BaseRelationshipBuilder):
    """Build parent-child hierarchical relationships across three tiers."""
    
    def build_relationships(self, nodes: List[KGNode]) -> List[KGRelationship]:
        """Build bidirectional parent-child relationships."""
        relationships = []
        
        # Create lookup dictionaries for efficient access
        nodes_by_type = defaultdict(list)
        nodes_by_id = {}
        
        for node in nodes:
            nodes_by_type[node.type].append(node)
            nodes_by_id[node.id] = node
        
        # Build Document → Chunk relationships
        for chunk_node in nodes_by_type['CHUNK']:
            source_article = chunk_node.properties.get('source_article')
            if source_article:
                # Find matching document node
                for doc_node in nodes_by_type['DOCUMENT']:
                    if doc_node.properties.get('title') == source_article:
                        # Parent relationship: Document → Chunk
                        relationships.append(KGRelationship(
                            source=doc_node.id,
                            target=chunk_node.id,
                            type="parent",
                            properties={'relationship_level': 'document_to_chunk'},
                            weight=1.0,
                            traversal_cost=0.5,  # Low cost for hierarchical navigation
                            bidirectional=True
                        ))
                        
                        # Child relationship: Chunk → Document  
                        relationships.append(KGRelationship(
                            source=chunk_node.id,
                            target=doc_node.id,
                            type="child",
                            properties={'relationship_level': 'chunk_to_document'},
                            weight=1.0,
                            traversal_cost=0.5,
                            bidirectional=True
                        ))
                        
                        # Update node hierarchical metadata
                        chunk_node.parent_id = doc_node.id
                        doc_node.children_ids.append(chunk_node.id)
                        break
        
        # Build Chunk → Sentence relationships
        for sentence_node in nodes_by_type['SENTENCE']:
            parent_chunk_id = sentence_node.properties.get('parent_chunk_id')
            if parent_chunk_id and parent_chunk_id in nodes_by_id:
                chunk_node = nodes_by_id[parent_chunk_id]
                
                # Parent relationship: Chunk → Sentence
                relationships.append(KGRelationship(
                    source=chunk_node.id,
                    target=sentence_node.id,
                    type="parent",
                    properties={'relationship_level': 'chunk_to_sentence'},
                    weight=1.0,
                    traversal_cost=0.3,  # Very low cost for fine-grained navigation
                    bidirectional=True
                ))
                
                # Child relationship: Sentence → Chunk
                relationships.append(KGRelationship(
                    source=sentence_node.id,
                    target=chunk_node.id,
                    type="child",
                    properties={'relationship_level': 'sentence_to_chunk'},
                    weight=1.0,
                    traversal_cost=0.3,
                    bidirectional=True
                ))
                
                # Update node hierarchical metadata
                sentence_node.parent_id = chunk_node.id
                chunk_node.children_ids.append(sentence_node.id)
        
        return relationships


class CosineSimilarityRelationshipBuilder(BaseRelationshipBuilder):
    """Build relationships based on Phase 4 cosine similarity matrices."""
    
    def __init__(self, similarities: Dict[str, Any], config: Dict[str, Any]):
        """Initialize cosine similarity relationship builder."""
        self.similarities = similarities
        self.config = config
        
        # Get configuration for cosine similarity relationships
        kg_config = config.get('knowledge_graph', {})
        sparsity_config = kg_config.get('sparsity', {})
        self.top_k = sparsity_config.get('relationship_limits', {}).get('cosine_similarity', 15)
        self.min_similarity = sparsity_config.get('min_thresholds', {}).get('cosine_similarity', 0.4)
    
    def build_relationships(self, nodes: List[KGNode]) -> List[KGRelationship]:
        """Build cosine similarity relationships using Phase 4 matrices."""
        relationships = []
        
        # Use existing similarity matrices from Phase 4
        for model_name, model_similarity_data in self.similarities.items():
            chunk_index_map = model_similarity_data['chunk_index_map']
            similarity_matrix = model_similarity_data['matrices']['combined']
            
            # Create chunk_id to node mapping
            chunk_to_node = {}
            for node in nodes:
                if node.type == 'CHUNK':
                    chunk_id = node.properties.get('chunk_id')
                    if chunk_id:
                        chunk_to_node[chunk_id] = node
            
            # Extract top-k relationships for each chunk
            for chunk_id, chunk_idx in chunk_index_map.items():
                if chunk_id not in chunk_to_node:
                    continue
                
                source_node = chunk_to_node[chunk_id]
                similarity_row = similarity_matrix[chunk_idx].toarray().flatten()
                
                # Get indices and scores of similar chunks
                similar_indices = []
                for target_idx, similarity_score in enumerate(similarity_row):
                    if target_idx != chunk_idx and similarity_score >= self.min_similarity:
                        similar_indices.append((target_idx, similarity_score))
                
                # Keep only top-k most similar
                similar_indices.sort(key=lambda x: x[1], reverse=True)
                top_k_indices = similar_indices[:self.top_k]
                
                # Create relationships for top-k
                for target_idx, similarity_score in top_k_indices:
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
                            type="cosine_similarity",
                            properties={
                                'similarity_score': float(similarity_score),
                                'model_name': model_name,
                                'rank': len([r for r in relationships 
                                           if r.source == source_node.id and r.type == "cosine_similarity"]) + 1
                            },
                            weight=float(similarity_score),
                            traversal_cost=1.0 - similarity_score,  # Lower cost for higher similarity
                            bidirectional=True,
                            context_dependent=False
                        ))
        
        return relationships


class EntityOverlapRelationshipBuilder(BaseRelationshipBuilder):
    """Build relationships based on shared entities (domain-agnostic)."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize entity overlap relationship builder."""
        self.config = config
        
        # Get configuration
        kg_config = config.get('knowledge_graph', {})
        sparsity_config = kg_config.get('sparsity', {})
        self.top_k = sparsity_config.get('relationship_limits', {}).get('entity_overlap', 8)
        self.min_similarity = sparsity_config.get('min_thresholds', {}).get('entity_overlap', 0.2)
        
        # Entity types to consider (domain-agnostic)
        self.entity_types = ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'MISC']
    
    def build_relationships(self, nodes: List[KGNode]) -> List[KGRelationship]:
        """Build sparse entity overlap relationships using Jaccard similarity."""
        relationships = []
        total_computed = 0
        total_added = 0
        
        # Consider all node types for entity relationships
        entity_nodes = [n for n in nodes if 'entities' in n.properties]
        
        print(f"Building entity overlap relationships for {len(entity_nodes)} nodes with entities...")
        
        # For each node, find its top-k most similar neighbors by entity overlap
        for i, source_node in enumerate(entity_nodes):
            if i % 200 == 0:
                print(f"Processing entity overlaps for node {i+1}/{len(entity_nodes)}...")
            
            # Compute similarities to all other nodes
            similarities = []
            
            for j, target_node in enumerate(entity_nodes):
                if i != j:  # Skip self
                    similarity = self._compute_entity_jaccard_similarity(source_node, target_node)
                    total_computed += 1
                    
                    if similarity >= self.min_similarity:
                        similarities.append((similarity, target_node, j))
            
            # Keep only top-k most similar
            top_k_similar = heapq.nlargest(self.top_k, similarities, key=lambda x: x[0])
            
            # Create relationships for top-k
            for similarity, target_node, target_idx in top_k_similar:
                shared_entities = self._get_shared_entities(source_node, target_node)
                
                relationships.append(KGRelationship(
                    source=source_node.id,
                    target=target_node.id,
                    type="entity_overlap",
                    properties={
                        'jaccard_similarity': similarity,
                        'shared_entities': shared_entities,
                        'entity_types': self.entity_types,
                        'rank': len([r for r in relationships 
                                   if r.source == source_node.id and r.type == "entity_overlap"]) + 1
                    },
                    weight=similarity,
                    traversal_cost=1.0 - similarity,
                    bidirectional=True,
                    context_dependent=False
                ))
                total_added += 1
        
        print(f"EntityOverlapBuilder: computed {total_computed:,} pairs, added {total_added:,} relationships (reduction: {100*(1-total_added/max(total_computed,1)):.1f}%)")
        return relationships
    
    def _compute_entity_jaccard_similarity(self, node_a: KGNode, node_b: KGNode) -> float:
        """Compute Jaccard similarity between entity sets."""
        entities_a = node_a.properties.get('entities', {})
        entities_b = node_b.properties.get('entities', {})
        
        all_entities_a = set()
        all_entities_b = set()
        
        # Collect all entities across types
        for entity_type in self.entity_types:
            all_entities_a.update(entities_a.get(entity_type, []))
            all_entities_b.update(entities_b.get(entity_type, []))
        
        # Normalize to lowercase for comparison
        all_entities_a = {e.lower().strip() for e in all_entities_a if len(e.strip()) > 1}
        all_entities_b = {e.lower().strip() for e in all_entities_b if len(e.strip()) > 1}
        
        if not all_entities_a and not all_entities_b:
            return 0.0
        
        intersection = len(all_entities_a & all_entities_b)
        union = len(all_entities_a | all_entities_b)
        
        return intersection / union if union > 0 else 0.0
    
    def _get_shared_entities(self, node_a: KGNode, node_b: KGNode) -> List[str]:
        """Get list of shared entities between two nodes."""
        entities_a = node_a.properties.get('entities', {})
        entities_b = node_b.properties.get('entities', {})
        
        all_entities_a = set()
        all_entities_b = set()
        
        for entity_type in self.entity_types:
            all_entities_a.update(entities_a.get(entity_type, []))
            all_entities_b.update(entities_b.get(entity_type, []))
        
        # Normalize and find intersection
        all_entities_a = {e.lower().strip() for e in all_entities_a if len(e.strip()) > 1}
        all_entities_b = {e.lower().strip() for e in all_entities_b if len(e.strip()) > 1}
        
        shared = list(all_entities_a & all_entities_b)
        return shared[:10]  # Limit to top 10 shared entities


class KnowledgeGraph:
    """Multi-dimensional knowledge graph with three-tier hierarchy."""
    
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
        """Get neighboring nodes with optional relationship type filtering."""
        neighbors = []
        
        for rel in self.relationships:
            if relationship_types and rel.type not in relationship_types:
                continue
                
            if rel.source == node_id:
                neighbor = self.get_node(rel.target)
                if neighbor:
                    neighbors.append(neighbor)
            elif rel.bidirectional and rel.target == node_id:
                neighbor = self.get_node(rel.source)
                if neighbor:
                    neighbors.append(neighbor)
        
        return neighbors
    
    def get_children(self, node_id: str) -> List[KGNode]:
        """Get direct children of a node."""
        return self.get_neighbors(node_id, relationship_types=["parent"])
    
    def get_parent(self, node_id: str) -> Optional[KGNode]:
        """Get direct parent of a node."""
        parents = self.get_neighbors(node_id, relationship_types=["child"])
        return parents[0] if parents else None
    
    def get_nodes_by_type(self, node_type: str) -> List[KGNode]:
        """Get all nodes of a specific type."""
        return [node for node in self.nodes if node.type == node_type]
    
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
            node = KGNode(
                id=node_data['id'],
                type=node_data['type'],
                properties=node_data['properties'],
                parent_id=node_data.get('parent_id'),
                children_ids=node_data.get('children_ids', []),
                hierarchy_level=node_data.get('hierarchy_level', 0)
            )
            kg.add_node(node)
        
        # Load relationships
        for rel_data in data.get('relationships', []):
            rel = KGRelationship(
                source=rel_data['source'],
                target=rel_data['target'],
                type=rel_data['type'],
                properties=rel_data['properties'],
                weight=rel_data.get('weight', 1.0),
                traversal_cost=rel_data.get('traversal_cost', 1.0),
                bidirectional=rel_data.get('bidirectional', True),
                context_dependent=rel_data.get('context_dependent', False)
            )
            kg.add_relationship(rel)
        
        return kg


class MultiDimensionalKnowledgeGraphBuilder:
    """Builder for multi-dimensional three-tier knowledge graphs."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize the multi-dimensional knowledge graph builder."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.kg_config = config.get('knowledge_graph', {})
        
        # Initialize domain-agnostic extractors
        self.extractors = {
            'ner': DomainAgnosticNERExtractor(),
            'keyphrases': DomainAgnosticKeyphraseExtractor(),
            'summary': DomainAgnosticSummaryExtractor()
        }
        
        self.logger.info("🔧 Initialized domain-agnostic multi-dimensional knowledge graph builder")
    
    def build_knowledge_graph(self, chunks: List[Dict[str, Any]], embeddings: Dict[str, List[ChunkEmbedding]], 
                            similarities: Dict[str, Any]) -> KnowledgeGraph:
        """Build complete three-tier multi-dimensional knowledge graph."""
        start_time = time.time()
        
        self.logger.info("🏗️  Building multi-dimensional knowledge graph with three-tier hierarchy")
        
        # Create knowledge graph
        kg = KnowledgeGraph()
        
        # Step 1: Create document nodes (Level 0)
        self.logger.info("📄 Creating document-level nodes...")
        document_nodes = self._create_document_nodes(chunks)
        for node in document_nodes:
            kg.add_node(node)
        
        # Step 2: Create chunk nodes (Level 1)  
        self.logger.info("📝 Creating chunk-level nodes...")
        chunk_nodes = self._create_chunk_nodes(chunks, embeddings)
        for node in chunk_nodes:
            kg.add_node(node)
        
        # Step 3: Create sentence nodes (Level 2)
        self.logger.info("📖 Creating sentence-level nodes...")
        sentence_nodes = self._create_sentence_nodes(chunks)
        for node in sentence_nodes:
            kg.add_node(node)
        
        # Step 4: Build multi-dimensional relationships
        self.logger.info("🔗 Building multi-dimensional relationships...")
        relationships = self._build_multi_dimensional_relationships(kg.nodes, similarities)
        for rel in relationships:
            kg.add_relationship(rel)
        
        # Step 5: Add metadata
        build_time = time.time() - start_time
        kg.metadata = {
            'created_at': datetime.now().isoformat(),
            'total_nodes': len(kg.nodes),
            'total_relationships': len(kg.relationships),
            'build_time': build_time,
            'architecture': 'multi_dimensional_three_tier',
            'config': self.kg_config,
            'node_types': {
                'DOCUMENT': len([n for n in kg.nodes if n.type == 'DOCUMENT']),
                'CHUNK': len([n for n in kg.nodes if n.type == 'CHUNK']),
                'SENTENCE': len([n for n in kg.nodes if n.type == 'SENTENCE'])
            },
            'relationship_types': {
                rel_type: len([r for r in kg.relationships if r.type == rel_type])
                for rel_type in set(r.type for r in kg.relationships)
            },
            'hierarchy_levels': {
                'document_level': 0,
                'chunk_level': 1, 
                'sentence_level': 2
            }
        }
        
        self.logger.info(f"✅ Multi-dimensional knowledge graph built:")
        self.logger.info(f"   📊 {len(kg.nodes)} total nodes ({kg.metadata['node_types']})")
        self.logger.info(f"   🔗 {len(kg.relationships)} total relationships")
        self.logger.info(f"   ⏱️  Build time: {build_time:.2f}s")
        
        return kg
    
    def _create_document_nodes(self, chunks: List[Dict[str, Any]]) -> List[KGNode]:
        """Create document-level nodes (Level 0)."""
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
                    info = extractor.extract(full_text, granularity_level="document")
                    extracted_info.update(info)
                except Exception as e:
                    self.logger.warning(f"Failed to extract {extractor_name} for document {article}: {e}")
            
            # Create document node
            node_id = f"doc_{hashlib.md5(article.encode()).hexdigest()[:8]}"
            node = KGNode(
                id=node_id,
                type="DOCUMENT",
                properties={
                    'title': article,
                    'page_content': full_text[:500000],  # RAGAS compatibility
                    'text': full_text[:500000],
                    'chunk_count': len(doc_data['chunks']),
                    'char_count': len(full_text),
                    'word_count': len(full_text.split()),
                    **extracted_info
                },
                parent_id=None,
                children_ids=[],
                hierarchy_level=0
            )
            document_nodes.append(node)
        
        return document_nodes
    
    def _create_chunk_nodes(self, chunks: List[Dict[str, Any]], embeddings: Dict[str, List[ChunkEmbedding]]) -> List[KGNode]:
        """Create chunk-level nodes (Level 1)."""
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
                    info = extractor.extract(chunk['text'], granularity_level="chunk")
                    extracted_info.update(info)
                except Exception as e:
                    self.logger.warning(f"Failed to extract {extractor_name} for chunk {chunk_id}: {e}")
            
            # Create chunk node
            node_id = f"chunk_{hashlib.md5(chunk_id.encode()).hexdigest()[:8]}"
            properties = {
                # Core properties
                'chunk_id': chunk_id,
                'text': chunk['text'],
                'page_content': chunk['text'],  # RAGAS compatibility
                'source_article': chunk['source_article'],
                'source_sentences': chunk['source_sentences'],
                'anchor_sentence_idx': chunk['anchor_sentence_idx'],
                'window_position': chunk['window_position'],
                'char_count': len(chunk['text']),
                'word_count': len(chunk['text'].split()),
                'sentence_count': len(chunk['source_sentences']),
                
                # Extracted information
                **extracted_info
            }
            
            # Add embedding availability info
            if chunk_id in embedding_lookup:
                properties['embedding_available'] = True
                properties['embedding_models'] = list(embeddings.keys())
            
            node = KGNode(
                id=node_id,
                type="CHUNK",
                properties=properties,
                parent_id=None,  # Will be set by hierarchical relationship builder
                children_ids=[],
                hierarchy_level=1
            )
            chunk_nodes.append(node)
        
        return chunk_nodes
    
    def _create_sentence_nodes(self, chunks: List[Dict[str, Any]]) -> List[KGNode]:
        """Create sentence-level nodes (Level 2)."""
        sentence_nodes = []
        
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            chunk_node_id = f"chunk_{hashlib.md5(chunk_id.encode()).hexdigest()[:8]}"
            
            # Split chunk into individual sentences
            sentences = nltk.sent_tokenize(chunk['text'])
            
            for i, sentence_text in enumerate(sentences):
                sentence_text = sentence_text.strip()
                if len(sentence_text) < 10:  # Skip very short sentences
                    continue
                
                # Extract information from sentence
                extracted_info = {}
                for extractor_name, extractor in self.extractors.items():
                    try:
                        info = extractor.extract(sentence_text, granularity_level="sentence")
                        extracted_info.update(info)
                    except Exception as e:
                        self.logger.warning(f"Failed to extract {extractor_name} for sentence: {e}")
                
                # Create sentence node
                sentence_id = f"{chunk_id}_sent_{i}"
                node_id = f"sent_{hashlib.md5(sentence_id.encode()).hexdigest()[:8]}"
                
                node = KGNode(
                    id=node_id,
                    type="SENTENCE",
                    properties={
                        'sentence_id': sentence_id,
                        'text': sentence_text,
                        'page_content': sentence_text,  # RAGAS compatibility
                        'parent_chunk_id': chunk_node_id,
                        'source_article': chunk['source_article'],
                        'sentence_index': i,
                        'char_count': len(sentence_text),
                        'word_count': len(sentence_text.split()),
                        
                        # Extracted information
                        **extracted_info
                    },
                    parent_id=None,  # Will be set by hierarchical relationship builder
                    children_ids=[],
                    hierarchy_level=2
                )
                sentence_nodes.append(node)
        
        return sentence_nodes
    
    def _build_multi_dimensional_relationships(self, nodes: List[KGNode], similarities: Dict[str, Any]) -> List[KGRelationship]:
        """Build relationships across all dimensions."""
        all_relationships = []
        
        # Initialize multi-dimensional relationship builders
        builders = [
            # Dimension 1: Hierarchical Structure Layer
            HierarchicalRelationshipBuilder(),
            
            # Dimension 2: Cosine Similarity Layer  
            CosineSimilarityRelationshipBuilder(similarities, self.config),
            
            # Dimension 3: Entity Overlap Layer
            EntityOverlapRelationshipBuilder(self.config)
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


# Factory function for backward compatibility
def KnowledgeGraphBuilder(config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> MultiDimensionalKnowledgeGraphBuilder:
    """Factory function for creating knowledge graph builder."""
    return MultiDimensionalKnowledgeGraphBuilder(config, logger)
