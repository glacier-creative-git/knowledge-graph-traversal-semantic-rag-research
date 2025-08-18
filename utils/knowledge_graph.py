#!/usr/bin/env python3
"""
Enhanced Multi-Granularity Knowledge Graph Construction
=====================================================

Pure translation layer that builds knowledge graphs from pre-computed multi-granularity similarity matrices.
All heavy numerical computation is handled in Phase 4; this phase focuses on graph structure translation.

Architecture:
- Document → Chunk → Sentence three-tier hierarchy
- Pre-computed similarity relationships from Phase 4
- Entity-based factual connections
- Pure O(n) translation performance
"""

import json
import hashlib
import time
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

import numpy as np
import spacy

from models import ChunkEmbedding, SentenceEmbedding, DocumentSummaryEmbedding


@dataclass
class KGNode:
    """Multi-granularity knowledge graph node."""
    id: str
    type: str  # "DOCUMENT", "CHUNK", "SENTENCE"
    granularity_level: int  # 0=Document, 1=Chunk, 2=Sentence
    properties: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'type': self.type,
            'granularity_level': self.granularity_level,
            'properties': self.properties
        }


@dataclass  
class KGRelationship:
    """Multi-granularity knowledge graph relationship."""
    source: str
    target: str
    type: str  # "parent", "child", "cosine_similarity", "entity_overlap", etc.
    granularity_type: str  # "hierarchical", "chunk_to_chunk", "doc_to_doc", etc.
    properties: Dict[str, Any]
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'source': self.source,
            'target': self.target,
            'type': self.type,
            'granularity_type': self.granularity_type,
            'properties': self.properties,
            'weight': self.weight
        }


class SimplifiedNERExtractor:
    """Extract only PERSON, ORG, and GPE entities - no domain bias."""
    
    def __init__(self):
        """Initialize simplified NER extractor."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.available = True
        except OSError:
            self.nlp = None
            self.available = False
    
    def extract(self, text: str) -> Dict[str, Any]:
        """Extract only high-quality entity types."""
        if self.available:
            return self._extract_with_spacy(text)
        else:
            return self._extract_with_patterns(text)
    
    def _extract_with_spacy(self, text: str) -> Dict[str, Any]:
        """Extract entities using spaCy - PERSON/ORG/GPE only."""
        doc = self.nlp(text)
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': []
        }
        
        for ent in doc.ents:
            entity_text = ent.text.strip()
            if len(entity_text) < 2:  # Skip very short entities
                continue
                
            if ent.label_ == 'PERSON':
                entities['PERSON'].append(entity_text)
            elif ent.label_ == 'ORG':
                entities['ORG'].append(entity_text)
            elif ent.label_ == 'GPE':
                entities['GPE'].append(entity_text)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
            
        return {'entities': entities}
    
    def _extract_with_patterns(self, text: str) -> Dict[str, Any]:
        """Fallback extraction using basic patterns."""
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': []
        }
        
        # Extract capitalized words/phrases (potential proper nouns)
        capitalized_patterns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        for phrase in capitalized_patterns:
            phrase = phrase.strip()
            if len(phrase) < 3:
                continue
                
            # Basic heuristics
            if any(word in phrase.lower() for word in ['university', 'company', 'corporation', 'inc', 'ltd']):
                entities['ORG'].append(phrase)
            elif any(word in phrase.lower() for word in ['dr', 'professor', 'mr', 'ms', 'mrs']):
                entities['PERSON'].append(phrase)
            else:
                # Default to ORG for capitalized phrases
                entities['ORG'].append(phrase)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
            
        return {'entities': entities}


class BaseRelationshipBuilder(ABC):
    """Base class for relationship builders."""
    
    @abstractmethod
    def build_relationships(self, nodes_by_type: Dict[str, List[KGNode]], 
                          similarity_data: Dict[str, Any]) -> List[KGRelationship]:
        """Build relationships between nodes."""
        pass


class HierarchicalRelationshipBuilder(BaseRelationshipBuilder):
    """Build hierarchical relationships across three tiers."""
    
    def build_relationships(self, nodes_by_type: Dict[str, List[KGNode]], 
                          similarity_data: Dict[str, Any]) -> List[KGRelationship]:
        """Build document→chunk→sentence hierarchical relationships."""
        relationships = []
        
        # Document → Chunk relationships
        for chunk_node in nodes_by_type.get('CHUNK', []):
            source_article = chunk_node.properties.get('source_article')
            if source_article:
                # Find matching document node
                for doc_node in nodes_by_type.get('DOCUMENT', []):
                    if doc_node.properties.get('title') == source_article:
                        # Parent relationship: Document → Chunk
                        relationships.append(KGRelationship(
                            source=doc_node.id,
                            target=chunk_node.id,
                            type="parent",
                            granularity_type="hierarchical",
                            properties={'relationship': 'document_contains_chunk'},
                            weight=1.0
                        ))
                        # Child relationship: Chunk → Document
                        relationships.append(KGRelationship(
                            source=chunk_node.id,
                            target=doc_node.id,
                            type="child",
                            granularity_type="hierarchical", 
                            properties={'relationship': 'chunk_belongs_to_document'},
                            weight=1.0
                        ))
                        break
        
        # Chunk → Sentence relationships
        for sentence_node in nodes_by_type.get('SENTENCE', []):
            containing_chunks = sentence_node.properties.get('containing_chunks', [])
            
            for chunk_id in containing_chunks:
                # Find matching chunk node by chunk_id
                for chunk_node in nodes_by_type.get('CHUNK', []):
                    if chunk_node.properties.get('chunk_id') == chunk_id:
                        # Parent relationship: Chunk → Sentence
                        relationships.append(KGRelationship(
                            source=chunk_node.id,
                            target=sentence_node.id,
                            type="parent",
                            granularity_type="hierarchical",
                            properties={'relationship': 'chunk_contains_sentence'},
                            weight=1.0
                        ))
                        # Child relationship: Sentence → Chunk
                        relationships.append(KGRelationship(
                            source=sentence_node.id,
                            target=chunk_node.id,
                            type="child",
                            granularity_type="hierarchical",
                            properties={'relationship': 'sentence_belongs_to_chunk'},
                            weight=1.0
                        ))
                        break
        
        return relationships


class PreComputedSimilarityRelationshipBuilder(BaseRelationshipBuilder):
    """Build relationships from pre-computed Phase 4 similarity matrices (pure translation)."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize pre-computed similarity relationship builder."""
        self.logger = logger or logging.getLogger(__name__)
    
    def build_relationships(self, nodes_by_type: Dict[str, List[KGNode]], 
                          similarity_data: Dict[str, Any]) -> List[KGRelationship]:
        """Build similarity relationships by reading pre-computed matrices (O(n) translation)."""
        relationships = []
        
        # Extract similarity matrices and index maps from Phase 4
        matrices = similarity_data.get('matrices', {})
        index_maps = similarity_data.get('index_maps', {})
        
        self.logger.info(f"🔄 Translating pre-computed similarities: {list(matrices.keys())}")
        
        # Translate chunk-to-chunk similarities
        relationships.extend(self._translate_chunk_to_chunk_similarities(
            nodes_by_type, matrices, index_maps
        ))
        
        # Translate document-to-document similarities  
        relationships.extend(self._translate_doc_to_doc_similarities(
            nodes_by_type, matrices, index_maps
        ))
        
        # Translate sentence-to-sentence similarities
        relationships.extend(self._translate_sentence_to_sentence_similarities(
            nodes_by_type, matrices, index_maps
        ))
        
        # Translate cross-granularity similarities
        relationships.extend(self._translate_cross_granularity_similarities(
            nodes_by_type, matrices, index_maps
        ))
        
        self.logger.info(f"✅ Translated {len(relationships):,} pre-computed similarity relationships")
        return relationships
    
    def _translate_chunk_to_chunk_similarities(self, nodes_by_type: Dict[str, List[KGNode]], 
                                             matrices: Dict[str, Any], 
                                             index_maps: Dict[str, Any]) -> List[KGRelationship]:
        """Translate chunk-to-chunk similarity matrices to relationships."""
        relationships = []
        
        # Get chunk nodes and index mapping
        chunk_nodes = nodes_by_type.get('CHUNK', [])
        chunk_index_map = index_maps.get('chunks', {})
        
        if not chunk_nodes or not chunk_index_map:
            return relationships
        
        # Create reverse mapping: index → node
        index_to_node = {}
        for node in chunk_nodes:
            chunk_id = node.properties.get('chunk_id')
            if chunk_id in chunk_index_map:
                chunk_idx = chunk_index_map[chunk_id]
                index_to_node[chunk_idx] = node
        
        # Translate intra-document similarities
        if 'chunk_to_chunk_intra' in matrices:
            matrix = matrices['chunk_to_chunk_intra']
            relationships.extend(self._extract_relationships_from_matrix(
                matrix, index_to_node, "cosine_similarity_intra", "chunk_to_chunk"
            ))
        
        # Translate inter-document similarities
        if 'chunk_to_chunk_inter' in matrices:
            matrix = matrices['chunk_to_chunk_inter']
            relationships.extend(self._extract_relationships_from_matrix(
                matrix, index_to_node, "cosine_similarity_inter", "chunk_to_chunk"
            ))
        
        # Translate combined similarities
        if 'chunk_to_chunk_combined' in matrices:
            matrix = matrices['chunk_to_chunk_combined']
            relationships.extend(self._extract_relationships_from_matrix(
                matrix, index_to_node, "cosine_similarity", "chunk_to_chunk"
            ))
        
        return relationships
    
    def _translate_doc_to_doc_similarities(self, nodes_by_type: Dict[str, List[KGNode]], 
                                         matrices: Dict[str, Any], 
                                         index_maps: Dict[str, Any]) -> List[KGRelationship]:
        """Translate document-to-document similarity matrices to relationships."""
        relationships = []
        
        # Get document nodes and index mapping
        doc_nodes = nodes_by_type.get('DOCUMENT', [])
        doc_index_map = index_maps.get('doc_summaries', {})
        
        if not doc_nodes or not doc_index_map:
            return relationships
        
        # Create reverse mapping: index → node (match by title/source_article)
        index_to_node = {}
        for node in doc_nodes:
            title = node.properties.get('title')
            # Find matching doc_id in index map
            for doc_id, doc_idx in doc_index_map.items():
                # Match by title (this is a simple heuristic, could be improved)
                if title in doc_id or any(word in doc_id for word in title.split()[:3]):
                    index_to_node[doc_idx] = node
                    break
        
        # Translate document similarities
        if 'doc_to_doc' in matrices:
            matrix = matrices['doc_to_doc']
            relationships.extend(self._extract_relationships_from_matrix(
                matrix, index_to_node, "document_similarity", "doc_to_doc"
            ))
        
        return relationships
    
    def _translate_sentence_to_sentence_similarities(self, nodes_by_type: Dict[str, List[KGNode]], 
                                                   matrices: Dict[str, Any], 
                                                   index_maps: Dict[str, Any]) -> List[KGRelationship]:
        """Translate sentence-to-sentence similarity matrices to relationships."""
        relationships = []
        
        # Get sentence nodes and index mapping
        sentence_nodes = nodes_by_type.get('SENTENCE', [])
        sentence_index_map = index_maps.get('sentences', {})
        
        if not sentence_nodes or not sentence_index_map:
            return relationships
        
        # Create reverse mapping: index → node
        index_to_node = {}
        for node in sentence_nodes:
            sentence_id = node.properties.get('sentence_id')
            if sentence_id in sentence_index_map:
                sentence_idx = sentence_index_map[sentence_id]
                index_to_node[sentence_idx] = node
        
        # Translate semantic similarities
        if 'sentence_to_sentence_semantic' in matrices:
            matrix = matrices['sentence_to_sentence_semantic']
            relationships.extend(self._extract_relationships_from_matrix(
                matrix, index_to_node, "sentence_similarity_semantic", "sentence_to_sentence"
            ))
        
        # Translate sequential similarities
        if 'sentence_to_sentence_sequential' in matrices:
            matrix = matrices['sentence_to_sentence_sequential']
            relationships.extend(self._extract_relationships_from_matrix(
                matrix, index_to_node, "sentence_similarity_sequential", "sentence_to_sentence"
            ))
        
        return relationships
    
    def _translate_cross_granularity_similarities(self, nodes_by_type: Dict[str, List[KGNode]], 
                                                matrices: Dict[str, Any], 
                                                index_maps: Dict[str, Any]) -> List[KGRelationship]:
        """Translate cross-granularity similarity matrices to relationships."""
        relationships = []
        
        # Translate sentence-to-chunk similarities
        if 'sentence_to_chunk' in matrices:
            sentence_nodes = nodes_by_type.get('SENTENCE', [])
            chunk_nodes = nodes_by_type.get('CHUNK', [])
            sentence_index_map = index_maps.get('sentences', {})
            chunk_index_map = index_maps.get('chunks', {})
            
            if sentence_nodes and chunk_nodes and sentence_index_map and chunk_index_map:
                # Create index mappings
                sentence_index_to_node = {}
                for node in sentence_nodes:
                    sentence_id = node.properties.get('sentence_id')
                    if sentence_id in sentence_index_map:
                        sentence_idx = sentence_index_map[sentence_id]
                        sentence_index_to_node[sentence_idx] = node
                
                chunk_index_to_node = {}
                for node in chunk_nodes:
                    chunk_id = node.properties.get('chunk_id')
                    if chunk_id in chunk_index_map:
                        chunk_idx = chunk_index_map[chunk_id]
                        chunk_index_to_node[chunk_idx] = node
                
                # Extract relationships from cross-granularity matrix
                matrix = matrices['sentence_to_chunk']
                relationships.extend(self._extract_cross_granularity_relationships(
                    matrix, sentence_index_to_node, chunk_index_to_node, 
                    "sentence_to_chunk_similarity", "cross_granularity"
                ))
        
        # Translate chunk-to-doc similarities
        if 'chunk_to_doc' in matrices:
            chunk_nodes = nodes_by_type.get('CHUNK', [])
            doc_nodes = nodes_by_type.get('DOCUMENT', [])
            chunk_index_map = index_maps.get('chunks', {})
            doc_index_map = index_maps.get('doc_summaries', {})
            
            if chunk_nodes and doc_nodes and chunk_index_map and doc_index_map:
                # Create index mappings (similar to above)
                chunk_index_to_node = {}
                for node in chunk_nodes:
                    chunk_id = node.properties.get('chunk_id')
                    if chunk_id in chunk_index_map:
                        chunk_idx = chunk_index_map[chunk_id]
                        chunk_index_to_node[chunk_idx] = node
                
                doc_index_to_node = {}
                for node in doc_nodes:
                    title = node.properties.get('title')
                    for doc_id, doc_idx in doc_index_map.items():
                        if title in doc_id or any(word in doc_id for word in title.split()[:3]):
                            doc_index_to_node[doc_idx] = node
                            break
                
                # Extract relationships from cross-granularity matrix
                matrix = matrices['chunk_to_doc']
                relationships.extend(self._extract_cross_granularity_relationships(
                    matrix, chunk_index_to_node, doc_index_to_node,
                    "chunk_to_doc_similarity", "cross_granularity"
                ))
        
        return relationships
    
    def _extract_relationships_from_matrix(self, matrix, index_to_node: Dict[int, KGNode], 
                                         relationship_type: str, granularity_type: str) -> List[KGRelationship]:
        """Extract relationships from a sparse similarity matrix."""
        relationships = []
        
        # Get non-zero entries from sparse matrix
        coo_matrix = matrix.tocoo()
        
        for i, j, similarity_score in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
            if i in index_to_node and j in index_to_node:
                source_node = index_to_node[i]
                target_node = index_to_node[j]
                
                relationship = KGRelationship(
                    source=source_node.id,
                    target=target_node.id,
                    type=relationship_type,
                    granularity_type=granularity_type,
                    properties={
                        'similarity_score': float(similarity_score),
                        'source_type': source_node.type,
                        'target_type': target_node.type
                    },
                    weight=float(similarity_score)
                )
                relationships.append(relationship)
        
        return relationships
    
    def _extract_cross_granularity_relationships(self, matrix, source_index_to_node: Dict[int, KGNode], 
                                               target_index_to_node: Dict[int, KGNode],
                                               relationship_type: str, granularity_type: str) -> List[KGRelationship]:
        """Extract relationships from a cross-granularity similarity matrix."""
        relationships = []
        
        # Get non-zero entries from sparse matrix
        coo_matrix = matrix.tocoo()
        
        for i, j, similarity_score in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
            if i in source_index_to_node and j in target_index_to_node:
                source_node = source_index_to_node[i]
                target_node = target_index_to_node[j]
                
                relationship = KGRelationship(
                    source=source_node.id,
                    target=target_node.id,
                    type=relationship_type,
                    granularity_type=granularity_type,
                    properties={
                        'similarity_score': float(similarity_score),
                        'source_type': source_node.type,
                        'target_type': target_node.type
                    },
                    weight=float(similarity_score)
                )
                relationships.append(relationship)
        
        return relationships


class EntityOverlapRelationshipBuilder(BaseRelationshipBuilder):
    """Build relationships based on PERSON/ORG/GPE entity overlap only."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize entity overlap relationship builder."""
        self.config = config
        
        # Get sparsity settings from knowledge graph config (simplified)
        kg_config = config.get('knowledge_graph', {})
        extractors_config = kg_config.get('extractors', {})
        
        # Use simplified top-k approach
        self.top_k = 8  # Reduced for performance
        self.min_similarity = 0.15  # Lower threshold for more connections
        
        # Only high-quality entity types
        self.entity_types = extractors_config.get('ner', {}).get('entity_types', ['PERSON', 'ORG', 'GPE'])[:3]  # Limit to first 3
    
    def build_relationships(self, nodes_by_type: Dict[str, List[KGNode]], 
                          similarity_data: Dict[str, Any]) -> List[KGRelationship]:
        """Build entity overlap relationships using simplified top-k selection."""
        relationships = []
        
        # Get all nodes for entity overlap computation
        all_nodes = []
        for node_list in nodes_by_type.values():
            all_nodes.extend(node_list)
        
        if len(all_nodes) < 2:
            return relationships
        
        self.logger.info(f"🏷️  Building entity overlap relationships for {len(all_nodes)} nodes...")
        
        # For each node, find its top-k most similar neighbors by entity overlap
        for i, source_node in enumerate(all_nodes):
            if i % 1000 == 0 and i > 0:
                self.logger.debug(f"Processing entity similarities for node {i+1}/{len(all_nodes)}...")
            
            # Compute similarities to all other nodes (vectorized for performance)
            similarities = []
            
            for j, target_node in enumerate(all_nodes):
                if i != j:  # Skip self
                    similarity = self._compute_entity_similarity(source_node, target_node)
                    
                    if similarity >= self.min_similarity:
                        similarities.append((similarity, target_node))
            
            # Keep only top-k most similar
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_k_similar = similarities[:self.top_k]
            
            # Create relationships for top-k
            for similarity, target_node in top_k_similar:
                relationships.append(KGRelationship(
                    source=source_node.id,
                    target=target_node.id,
                    type="entity_overlap",
                    granularity_type="entity_based",
                    properties={
                        'similarity_score': similarity, 
                        'entity_types': self.entity_types,
                        'source_type': source_node.type,
                        'target_type': target_node.type
                    },
                    weight=similarity
                ))
        
        self.logger.info(f"✅ Built {len(relationships):,} entity overlap relationships")
        return relationships
    
    def _compute_entity_similarity(self, node_a: KGNode, node_b: KGNode) -> float:
        """Compute Jaccard similarity between PERSON/ORG/GPE entities only."""
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


class MultiGranularityKnowledgeGraph:
    """Enhanced multi-granularity knowledge graph."""
    
    def __init__(self):
        """Initialize empty multi-granularity knowledge graph."""
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
    
    def get_nodes_by_type(self, node_type: str) -> List[KGNode]:
        """Get all nodes of a specific type."""
        return [node for node in self.nodes if node.type == node_type]
    
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
    
    def get_parent(self, node_id: str) -> Optional[KGNode]:
        """Get parent node in hierarchy."""
        for rel in self.relationships:
            if rel.target == node_id and rel.type == "parent":
                return self.get_node(rel.source)
        return None
    
    def get_children(self, node_id: str) -> List[KGNode]:
        """Get child nodes in hierarchy."""
        children = []
        for rel in self.relationships:
            if rel.source == node_id and rel.type == "parent":
                child = self.get_node(rel.target)
                if child:
                    children.append(child)
        return children
    
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
    def load(cls, file_path: str) -> 'MultiGranularityKnowledgeGraph':
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
            granularity_level = node_data.get('granularity_level', 0)  # Default to 0 for old format
            
            node = KGNode(
                id=node_id,
                type=node_type,
                granularity_level=granularity_level,
                properties=node_data['properties']
            )
            kg.add_node(node)
        
        # Load relationships
        for rel_data in data.get('relationships', []):
            # Handle both old and new formats for backwards compatibility
            source = rel_data.get('source') or rel_data.get('source_id')
            target = rel_data.get('target') or rel_data.get('target_id')
            rel_type = rel_data.get('type') or rel_data.get('relationship_type')
            granularity_type = rel_data.get('granularity_type', 'unknown')  # Default for old format
            
            rel = KGRelationship(
                source=source,
                target=target,
                type=rel_type,
                granularity_type=granularity_type,
                properties=rel_data['properties'],
                weight=rel_data.get('weight', 1.0)
            )
            kg.add_relationship(rel)
        
        return kg


class MultiGranularityKnowledgeGraphBuilder:
    """Builder for enhanced multi-granularity knowledge graphs."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize the multi-granularity knowledge graph builder."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.kg_config = config.get('knowledge_graph', {})
        
        # Initialize extractors
        self.ner_extractor = SimplifiedNERExtractor()
        
        self.logger.info("🏗️  Initialized multi-granularity knowledge graph builder")
    
    def build_knowledge_graph(self, chunks: List[Dict[str, Any]], 
                            multi_granularity_embeddings: Dict[str, Dict[str, List[Any]]], 
                            multi_granularity_similarities: Dict[str, Dict[str, Any]]) -> MultiGranularityKnowledgeGraph:
        """Build multi-granularity knowledge graph from embeddings and pre-computed similarities."""
        start_time = time.time()
        
        self.logger.info("🌟 Building enhanced multi-granularity knowledge graph")
        
        # Create knowledge graph
        kg = MultiGranularityKnowledgeGraph()
        
        # Extract embeddings for the first model (assume single model for now)
        model_name = list(multi_granularity_embeddings.keys())[0]
        granularity_embeddings = multi_granularity_embeddings[model_name]
        similarity_data = multi_granularity_similarities[model_name]
        
        # Step 1: Create document nodes (level 0)
        document_nodes = self._create_document_nodes(chunks, granularity_embeddings.get('doc_summaries', []))
        for node in document_nodes:
            kg.add_node(node)
        
        # Step 2: Create chunk nodes (level 1)
        chunk_nodes = self._create_chunk_nodes(chunks, granularity_embeddings.get('chunks', []))
        for node in chunk_nodes:
            kg.add_node(node)
        
        # Step 3: Create sentence nodes (level 2)
        sentence_nodes = self._create_sentence_nodes(granularity_embeddings.get('sentences', []))
        for node in sentence_nodes:
            kg.add_node(node)
        
        # Step 4: Group nodes by type for relationship building
        nodes_by_type = {
            'DOCUMENT': document_nodes,
            'CHUNK': chunk_nodes,
            'SENTENCE': sentence_nodes
        }
        
        # Step 5: Build relationships using multiple builders
        relationships = self._build_relationships(nodes_by_type, similarity_data)
        for rel in relationships:
            kg.add_relationship(rel)
        
        # Step 6: Add metadata
        build_time = time.time() - start_time
        kg.metadata = {
            'created_at': datetime.now().isoformat(),
            'architecture': 'multi_granularity_three_tier',
            'total_nodes': len(kg.nodes),
            'total_relationships': len(kg.relationships),
            'build_time': build_time,
            'config': self.kg_config,
            'granularity_counts': {
                'documents': len(document_nodes),
                'chunks': len(chunk_nodes),
                'sentences': len(sentence_nodes)
            },
            'relationship_types': {
                rel_type: len([r for r in kg.relationships if r.type == rel_type])
                for rel_type in set(r.type for r in kg.relationships)
            },
            'granularity_relationship_types': {
                granularity_type: len([r for r in kg.relationships if r.granularity_type == granularity_type])
                for granularity_type in set(r.granularity_type for r in kg.relationships)
            }
        }
        
        self.logger.info(f"🎉 Multi-granularity knowledge graph built: {len(kg.nodes)} nodes, {len(kg.relationships)} relationships in {build_time:.2f}s")
        self._log_detailed_statistics(kg.metadata)
        
        return kg
    
    def _create_document_nodes(self, chunks: List[Dict[str, Any]], 
                             doc_embeddings: List[DocumentSummaryEmbedding]) -> List[KGNode]:
        """Create document-level nodes (level 0)."""
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
            
            # Extract entities at document level
            extracted_info = self.ner_extractor.extract(full_text)
            
            # Find matching document summary embedding
            doc_summary_text = ""
            for doc_emb in doc_embeddings:
                if doc_emb.source_article == article:
                    doc_summary_text = doc_emb.summary_text
                    break
            
            # Create document node
            node_id = f"doc_{hashlib.md5(article.encode()).hexdigest()[:8]}"
            node = KGNode(
                id=node_id,
                type="DOCUMENT",
                granularity_level=0,
                properties={
                    'title': article,
                    'page_content': doc_summary_text or full_text[:1000],  # RAGAS expects 'page_content'
                    'full_text': full_text[:1000],  # Keep for backwards compatibility
                    'summary_text': doc_summary_text,
                    'chunk_count': len(doc_data['chunks']),
                    **extracted_info
                }
            )
            document_nodes.append(node)
        
        return document_nodes
    
    def _create_chunk_nodes(self, chunks: List[Dict[str, Any]], 
                          chunk_embeddings: List[ChunkEmbedding]) -> List[KGNode]:
        """Create chunk-level nodes (level 1)."""
        chunk_nodes = []
        
        # Create embedding lookup
        embedding_lookup = {emb.chunk_id: emb for emb in chunk_embeddings}
        
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            
            # Extract entities from chunk
            extracted_info = self.ner_extractor.extract(chunk['text'])
            
            # Create chunk node
            node_id = f"chunk_{hashlib.md5(chunk_id.encode()).hexdigest()[:8]}"
            properties = {
                # RAGAS required fields
                'page_content': chunk['text'],  # RAGAS expects 'page_content'
                
                # Core chunk properties
                'chunk_id': chunk_id,
                'text': chunk['text'],  # Keep for backwards compatibility
                'source_article': chunk['source_article'],
                'source_sentences': chunk['source_sentences'],
                'anchor_sentence_idx': chunk['anchor_sentence_idx'],
                'window_position': chunk['window_position'],
                
                # Entity information
                **extracted_info
            }
            
            # Add embedding info if available
            if chunk_id in embedding_lookup:
                properties['embedding_available'] = True
                properties['embedding_dimension'] = len(embedding_lookup[chunk_id].embedding)
            
            node = KGNode(
                id=node_id,
                type="CHUNK",
                granularity_level=1,
                properties=properties
            )
            chunk_nodes.append(node)
        
        return chunk_nodes
    
    def _create_sentence_nodes(self, sentence_embeddings: List[SentenceEmbedding]) -> List[KGNode]:
        """Create sentence-level nodes (level 2)."""
        sentence_nodes = []
        
        for sentence_emb in sentence_embeddings:
            # Extract entities from sentence
            extracted_info = self.ner_extractor.extract(sentence_emb.sentence_text)
            
            # Create sentence node
            node_id = f"sent_{hashlib.md5(sentence_emb.sentence_id.encode()).hexdigest()[:8]}"
            properties = {
                # RAGAS required fields
                'page_content': sentence_emb.sentence_text,  # RAGAS expects 'page_content'
                
                # Core sentence properties
                'sentence_id': sentence_emb.sentence_id,
                'text': sentence_emb.sentence_text,  # Keep for backwards compatibility
                'source_article': sentence_emb.source_article,
                'sentence_index': sentence_emb.sentence_index,
                'containing_chunks': sentence_emb.containing_chunks,
                
                # Embedding info
                'embedding_available': True,
                'embedding_dimension': len(sentence_emb.embedding),
                
                # Entity information
                **extracted_info
            }
            
            node = KGNode(
                id=node_id,
                type="SENTENCE",
                granularity_level=2,
                properties=properties
            )
            sentence_nodes.append(node)
        
        return sentence_nodes
    
    def _build_relationships(self, nodes_by_type: Dict[str, List[KGNode]], 
                           similarity_data: Dict[str, Any]) -> List[KGRelationship]:
        """Build relationships using multiple builders."""
        all_relationships = []
        
        self.logger.info("Building multi-granularity relationships...")
        
        # Initialize relationship builders
        builders = [
            HierarchicalRelationshipBuilder(),
            PreComputedSimilarityRelationshipBuilder(self.logger),
            EntityOverlapRelationshipBuilder(self.config)
        ]
        
        # Build relationships with each builder
        for builder in builders:
            try:
                builder_start = time.time()
                relationships = builder.build_relationships(nodes_by_type, similarity_data)
                builder_time = time.time() - builder_start
                
                all_relationships.extend(relationships)
                self.logger.info(f"✅ Built {len(relationships):,} {builder.__class__.__name__} relationships in {builder_time:.2f}s")
            except Exception as e:
                self.logger.warning(f"⚠️  Failed to build relationships with {builder.__class__.__name__}: {e}")
        
        return all_relationships
    
    def _log_detailed_statistics(self, metadata: Dict[str, Any]):
        """Log detailed statistics about the knowledge graph."""
        self.logger.info("📊 Multi-Granularity Knowledge Graph Statistics:")
        self.logger.info(f"   Architecture: {metadata['architecture']}")
        self.logger.info(f"   Total nodes: {metadata['total_nodes']:,}")
        self.logger.info(f"   Total relationships: {metadata['total_relationships']:,}")
        
        # Granularity breakdown
        granularity_counts = metadata['granularity_counts']
        self.logger.info(f"   Granularity breakdown:")
        self.logger.info(f"      Documents (L0): {granularity_counts['documents']:,}")
        self.logger.info(f"      Chunks (L1): {granularity_counts['chunks']:,}")
        self.logger.info(f"      Sentences (L2): {granularity_counts['sentences']:,}")
        
        # Relationship type breakdown
        rel_types = metadata['relationship_types']
        self.logger.info(f"   Relationship types:")
        for rel_type, count in rel_types.items():
            self.logger.info(f"      {rel_type}: {count:,}")
        
        # Granularity relationship breakdown
        granularity_rel_types = metadata['granularity_relationship_types']
        self.logger.info(f"   Granularity relationship types:")
        for granularity_type, count in granularity_rel_types.items():
            self.logger.info(f"      {granularity_type}: {count:,}")


# Backwards compatibility aliases
KnowledgeGraph = MultiGranularityKnowledgeGraph
KnowledgeGraphBuilder = MultiGranularityKnowledgeGraphBuilder