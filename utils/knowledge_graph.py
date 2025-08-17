#!/usr/bin/env python3
"""
Simplified Knowledge Graph Construction
=====================================

Minimalist architecture focusing on three core relationship types:
1. Cosine Similarity Layer: Mathematical semantic relationships (from Phase 4)
2. Entity Overlap Layer: Factual connections via PERSON/ORG/GPE entities only
3. Hierarchical Layer: Document → Chunk structural navigation

Removes themes, keyphrases, and summaries for cognitive simplicity.
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

from models import ChunkEmbedding


@dataclass
class KGNode:
    """Simplified knowledge graph node."""
    id: str
    type: str  # "DOCUMENT", "CHUNK"
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
    """Simplified knowledge graph relationship."""
    source: str
    target: str
    type: str  # "contains", "cosine_similarity", "entity_overlap"
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
    def build_relationships(self, nodes: List[KGNode]) -> List[KGRelationship]:
        """Build relationships between nodes."""
        pass


class HierarchicalRelationshipBuilder(BaseRelationshipBuilder):
    """Build document→chunk hierarchical relationships."""
    
    def build_relationships(self, nodes: List[KGNode]) -> List[KGRelationship]:
        """Build hierarchical relationships."""
        relationships = []
        
        # Group nodes by type
        document_nodes = [n for n in nodes if n.type == 'DOCUMENT']
        chunk_nodes = [n for n in nodes if n.type == 'CHUNK']
        
        # Create document → chunk relationships
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


class CosineSimilarityRelationshipBuilder(BaseRelationshipBuilder):
    """Build relationships using Phase 4 cosine similarity matrices."""
    
    def __init__(self, similarities: Dict[str, Any], config: Dict[str, Any]):
        """Initialize cosine similarity relationship builder."""
        self.similarities = similarities
        self.config = config
        
        # Get sparsity settings
        sparsity_config = config.get('knowledge_graph', {}).get('sparsity', {})
        self.top_k = sparsity_config.get('relationship_limits', {}).get('embedding_similarity', 15)
        self.min_similarity = sparsity_config.get('min_thresholds', {}).get('embedding_similarity', 0.5)
    
    def build_relationships(self, nodes: List[KGNode]) -> List[KGRelationship]:
        """Build cosine similarity relationships using existing matrices."""
        relationships = []
        
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
                
                # Get similar chunks above threshold
                similar_indices = []
                for target_idx, similarity_score in enumerate(similarity_row):
                    if target_idx != chunk_idx and similarity_score >= self.min_similarity:
                        similar_indices.append((target_idx, similarity_score))
                
                # Keep only top-k most similar
                similar_indices.sort(key=lambda x: x[1], reverse=True)
                top_k_indices = similar_indices[:self.top_k]
                
                # Create relationships
                for target_idx, similarity_score in top_k_indices:
                    # Find target chunk
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
                                'model_name': model_name
                            },
                            weight=float(similarity_score)
                        ))
        
        return relationships


class EntityOverlapRelationshipBuilder(BaseRelationshipBuilder):
    """Build relationships based on PERSON/ORG/GPE entity overlap only."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize entity overlap relationship builder."""
        self.config = config
        
        # Get sparsity settings
        sparsity_config = config.get('knowledge_graph', {}).get('sparsity', {})
        self.top_k = sparsity_config.get('relationship_limits', {}).get('entity_similarity', 10)
        self.min_similarity = sparsity_config.get('min_thresholds', {}).get('entity_similarity', 0.3)
        
        # Only high-quality entity types
        self.entity_types = ['PERSON', 'ORG', 'GPE']
    
    def build_relationships(self, nodes: List[KGNode]) -> List[KGRelationship]:
        """Build entity overlap relationships using top-k selection."""
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
                    type="entity_overlap",
                    properties={
                        'similarity_score': similarity, 
                        'entity_types': self.entity_types,
                        'rank': len([r for r in relationships if r.source == source_node.id]) + 1
                    },
                    weight=similarity
                ))
                total_added += 1
        
        print(f"EntityOverlapBuilder: computed {total_computed:,} pairs, added {total_added:,} relationships (reduction: {100*(1-total_added/max(total_computed,1)):.1f}%)")
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


class KnowledgeGraph:
    """Simplified knowledge graph."""
    
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
    """Builder for simplified knowledge graphs."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize the simplified knowledge graph builder."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.kg_config = config.get('knowledge_graph', {})
        
        # Initialize single extractor
        self.ner_extractor = SimplifiedNERExtractor()
        
        self.logger.info("🔧 Initialized simplified knowledge graph builder (PERSON/ORG/GPE only)")
    
    def build_knowledge_graph(self, chunks: List[Dict[str, Any]], embeddings: Dict[str, List[ChunkEmbedding]], 
                            similarities: Dict[str, Any]) -> KnowledgeGraph:
        """Build simplified knowledge graph from chunks."""
        start_time = time.time()
        
        self.logger.info("🏗️  Building simplified knowledge graph")
        
        # Create knowledge graph
        kg = KnowledgeGraph()
        
        # Step 1: Create document nodes
        document_nodes = self._create_document_nodes(chunks)
        for node in document_nodes:
            kg.add_node(node)
        
        # Step 2: Create chunk nodes with entity extraction only
        chunk_nodes = self._create_chunk_nodes(chunks, embeddings)
        for node in chunk_nodes:
            kg.add_node(node)
        
        # Step 3: Build three core relationship types
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
        
        self.logger.info(f"✅ Simplified knowledge graph built: {len(kg.nodes)} nodes, {len(kg.relationships)} relationships in {build_time:.2f}s")
        
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
            
            # Extract entities at document level
            extracted_info = self.ner_extractor.extract(full_text)
            
            # Create document node
            node_id = f"doc_{hashlib.md5(article.encode()).hexdigest()[:8]}"
            node = KGNode(
                id=node_id,
                type="DOCUMENT",
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
        """Create chunk-level nodes with entity extraction only."""
        chunk_nodes = []
        
        # Create embedding lookup
        embedding_lookup = {}
        for model_name, chunk_embeddings in embeddings.items():
            for chunk_emb in chunk_embeddings:
                embedding_lookup[chunk_emb.chunk_id] = chunk_emb
        
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
                
                # Entity information (simplified)
                **extracted_info
            }
            
            # Add embedding info if available
            if chunk_id in embedding_lookup:
                properties['embedding_available'] = True
                properties['embedding_model'] = list(embeddings.keys())[0]  # Use first model as reference
            
            node = KGNode(
                id=node_id,
                type="CHUNK",
                properties=properties
            )
            chunk_nodes.append(node)
        
        return chunk_nodes
    
    def _build_relationships(self, nodes: List[KGNode], similarities: Dict[str, Any]) -> List[KGRelationship]:
        """Build three core relationship types."""
        all_relationships = []
        
        self.logger.info("Building three core relationship types...")
        
        # Initialize simplified relationship builders
        builders = [
            HierarchicalRelationshipBuilder(),
            CosineSimilarityRelationshipBuilder(similarities, self.config),
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
