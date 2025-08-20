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

from models import ChunkEmbedding, SentenceEmbedding, DocumentSummaryEmbedding, EmbeddingModel


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


class ThemeBridgeBuilder:
    """Builds cross-document theme similarity bridges for semantic highways."""

    def __init__(self, entity_theme_data: Dict[str, Any], embedding_model,
                 config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize theme bridge builder."""
        self.entity_theme_data = entity_theme_data
        self.embedding_model = embedding_model
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.theme_config = config.get('theme_bridging', {})

        # Extract all themes and organize by document
        self.themes_by_document = self._organize_themes_by_document()
        self.all_unique_themes = self._extract_unique_themes()

        self.logger.info(
            f"🌉 ThemeBridgeBuilder initialized: {len(self.all_unique_themes)} unique themes across {len(self.themes_by_document)} documents")

    def _organize_themes_by_document(self) -> Dict[str, List[str]]:
        """Organize themes by their source documents (universal data access)."""
        themes_by_doc = {}

        document_themes = self.entity_theme_data['extraction_results'].get('document_themes', [])

        for theme_result in document_themes:
            try:
                doc_title = self._get_field(theme_result, 'doc_title')
                themes = self._get_field(theme_result, 'themes')

                if doc_title and themes:
                    themes_by_doc[doc_title] = themes

            except ValueError as e:
                self.logger.warning(f"Failed to extract theme data: {e}")
                continue

        return themes_by_doc

    def _get_field(self, obj, field_name: str):
        """Universal field accessor for both objects and dictionaries."""
        if hasattr(obj, field_name):
            return getattr(obj, field_name)
        elif isinstance(obj, dict):
            return obj.get(field_name)
        else:
            raise ValueError(f"Cannot access field '{field_name}' from {type(obj)}")

    def _extract_unique_themes(self) -> List[str]:
        """Extract all unique themes across all documents."""
        all_themes = set()
        for themes in self.themes_by_document.values():
            all_themes.update(themes)
        return list(all_themes)

    def compute_cross_document_theme_bridges(self) -> dict[Any, Any] | dict[str, list[tuple[float, str]]]:
        """
        Compute cross-document theme similarity bridges.

        Returns:
            Dictionary mapping each theme to its top-k cross-document similar themes
        """
        self.logger.info(f"🔗 Computing cross-document theme bridges for {len(self.all_unique_themes)} themes")

        if len(self.all_unique_themes) < 2:
            self.logger.warning("Less than 2 themes found, no bridges to compute")
            return {}

        # Embed all themes
        theme_embeddings = self.embedding_model.encode_batch(
            self.all_unique_themes,
            batch_size=self.config['models']['embedding_batch_size'],
            show_progress=True
        )

        # Compute similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(theme_embeddings)

        # Build cross-document bridges
        theme_bridges = {}
        top_k = self.theme_config.get('top_k_bridges', 3)
        min_similarity = self.theme_config.get('min_bridge_similarity', 0.2)

        for i, source_theme in enumerate(self.all_unique_themes):
            source_doc = self._get_document_for_theme(source_theme)
            bridges = []

            # Get similarities for this theme
            similarities = similarity_matrix[i]

            # Create (similarity, target_theme, target_doc) tuples for filtering
            similarity_candidates = []
            for j, target_theme in enumerate(self.all_unique_themes):
                if i != j:  # Skip self
                    target_doc = self._get_document_for_theme(target_theme)
                    similarity_score = float(similarities[j])

                    # Only include cross-document connections
                    if target_doc != source_doc and similarity_score >= min_similarity:
                        similarity_candidates.append((similarity_score, target_theme))

            # Sort by similarity and take top-k
            similarity_candidates.sort(key=lambda x: x[0], reverse=True)
            bridges = similarity_candidates[:top_k]

            if bridges:
                theme_bridges[source_theme] = bridges
                self.logger.debug(f"Theme '{source_theme}' bridges to {len(bridges)} cross-document themes")

        total_bridges = sum(len(bridges) for bridges in theme_bridges.values())
        self.logger.info(f"✅ Built {total_bridges} cross-document theme bridges")

        return theme_bridges

    def _get_document_for_theme(self, theme: str) -> Optional[str]:
        """Find which document a theme belongs to (returns document title)."""

        for doc_title, themes in self.themes_by_document.items():
            if theme in themes:
                return doc_title

        # Theme not found - show all available themes for debugging
        all_available_themes = []
        for doc_title, themes in self.themes_by_document.items():
            all_available_themes.extend(themes)

        return None

    def get_inherited_themes_for_node(self, source_document: str, theme_bridges: Dict[str, List[Tuple[str, float]]]) -> \
    Dict[str, Any]:
        """Get both direct and inherited themes for a node based on its source document."""
        direct_themes = self.themes_by_document.get(source_document, [])

        # Collect cross-document inherited themes
        inherited_themes = []
        theme_inheritance_map = {}

        for direct_theme in direct_themes:
            if direct_theme in theme_bridges:
                bridges = theme_bridges[direct_theme]

                # Fix: The bridges are actually (similarity_float, theme_string) tuples
                # So we need to unpack as (similarity, bridge_theme) not (bridge_theme, similarity)
                for similarity, bridge_theme in bridges:  # This should work now
                    inherited_themes.append({
                        'theme': bridge_theme,  # ✅ Now string
                        'similarity': float(similarity),  # ✅ Now float
                        'inherited_from': direct_theme,
                        'source_document': self._get_document_for_theme(bridge_theme)
                    })

                theme_inheritance_map[direct_theme] = [(similarity, theme) for similarity, theme in bridges]

        return {
            'direct_themes': direct_themes,
            'inherited_themes': inherited_themes,
            'theme_inheritance_map': theme_inheritance_map,
            'total_themes': len(direct_themes) + len(inherited_themes)
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
        """Get neighboring nodes with updated relationship type mapping."""
        neighbors = []

        # If no specific types requested, get all neighbors
        if relationship_types is None:
            search_types = None
        else:
            # Map old relationship type names to new multi-granularity names
            type_mapping = {
                'cosine_similarity': [
                    'sentence_to_sentence_semantic',
                    'doc_to_doc',
                    'cosine_similarity_intra',
                    'cosine_similarity_inter',
                    'cosine_similarity'
                ],
                'entity_overlap': [
                    'high_confidence_entity_overlap',
                    'entity_overlap'
                ],
                'similarity': [
                    'sentence_to_sentence_semantic',
                    'sentence_to_sentence_sequential',
                    'doc_to_doc',
                    'cosine_similarity_intra',
                    'cosine_similarity_inter',
                    'cosine_similarity'
                ]
            }

            # Expand requested types to include all variants
            search_types = set()
            for req_type in relationship_types:
                if req_type in type_mapping:
                    search_types.update(type_mapping[req_type])
                else:
                    search_types.add(req_type)  # Include exact matches too

            search_types = list(search_types)

        for rel in self.relationships:
            # If search_types is None, include all relationships
            if search_types is None or rel.type in search_types:
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

    def _create_enhanced_document_nodes(self, chunks: List[Dict[str, Any]],
                                        doc_embeddings: List[DocumentSummaryEmbedding],
                                        entity_theme_data: Dict[str, Any],
                                        theme_bridge_builder: ThemeBridgeBuilder,
                                        theme_bridges: Dict[str, List[Tuple[str, float]]]) -> List[KGNode]:
        """Create document-level nodes with themes and inherited theme bridges."""
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

        # Get document themes from entity_theme_data
        document_themes_lookup = {}
        document_themes = entity_theme_data['extraction_results'].get('document_themes', [])
        for theme_result in document_themes:
            document_themes_lookup[theme_result.doc_title] = {
                'themes': theme_result.themes,
                'doc_id': theme_result.doc_id,
                'summary_text': theme_result.source_text
            }

        # Create document nodes
        document_nodes = []
        for article, doc_data in documents.items():
            # Get themes for this document
            theme_info = document_themes_lookup.get(article, {
                'themes': [],
                'doc_id': f"doc_{hashlib.md5(article.encode()).hexdigest()[:8]}",
                'summary_text': ''
            })

            # Get theme bridges for this document
            theme_data = theme_bridge_builder.get_inherited_themes_for_node(
                theme_info['doc_id'], theme_bridges
            )

            # Combine all chunk text for document-level extraction (fallback)
            full_text = ' '.join([chunk['text'] for chunk in doc_data['chunks']])

            # Find matching document summary embedding
            doc_summary_text = theme_info['summary_text']
            for doc_emb in doc_embeddings:
                if doc_emb.source_article == article:
                    doc_summary_text = doc_emb.summary_text
                    break

            # Create document node with enhanced theme information
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

                    # Enhanced theme information
                    'direct_themes': theme_data['direct_themes'],
                    'inherited_themes': theme_data['inherited_themes'],
                    'theme_inheritance_map': theme_data['theme_inheritance_map'],
                    'total_semantic_themes': theme_data['total_themes'],

                    # Document-level metadata
                    'doc_id': theme_info['doc_id']
                }
            )
            document_nodes.append(node)

        return document_nodes

    def _create_enhanced_chunk_nodes(self, chunks: List[Dict[str, Any]],
                                     chunk_embeddings: List[ChunkEmbedding],
                                     entity_theme_data: Dict[str, Any],
                                     theme_bridge_builder: ThemeBridgeBuilder,
                                     theme_bridges: Dict[str, List[Tuple[str, float]]]) -> List[KGNode]:
        """Create chunk-level nodes with entities and inherited themes."""
        chunk_nodes = []

        # Create embedding lookup
        embedding_lookup = {emb.chunk_id: emb for emb in chunk_embeddings}

        # Create entity lookup
        entity_lookup = {}
        chunk_entities = entity_theme_data['extraction_results'].get('chunk_entities', [])
        for entity_result in chunk_entities:
            entity_lookup[entity_result.source_id] = entity_result.entities

        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            source_document = chunk['source_article']

            # Get themes (direct + inherited)
            theme_data = theme_bridge_builder.get_inherited_themes_for_node(source_document, theme_bridges)

            # Get entities
            entities = entity_lookup.get(chunk_id, {'PERSON': [], 'ORG': [], 'GPE': []})

            # Create enhanced chunk node
            node_id = f"chunk_{hashlib.md5(chunk_id.encode()).hexdigest()[:8]}"
            properties = {
                # RAGAS required fields
                'page_content': chunk['text'],

                # Core chunk properties
                'chunk_id': chunk_id,
                'text': chunk['text'],
                'source_article': source_document,
                'source_sentences': chunk['source_sentences'],
                'anchor_sentence_idx': chunk['anchor_sentence_idx'],

                # Entities
                'entities': entities,

                # Theme inheritance (the new semantic highways)
                'direct_themes': theme_data['direct_themes'],
                'inherited_themes': theme_data['inherited_themes'],
                'theme_inheritance_map': theme_data['theme_inheritance_map'],
                'total_semantic_themes': theme_data['total_themes']
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

    def _create_enhanced_sentence_nodes(self, sentence_embeddings: List[SentenceEmbedding],
                                        entity_theme_data: Dict[str, Any],
                                        theme_bridge_builder: ThemeBridgeBuilder,
                                        theme_bridges: Dict[str, List[Tuple[str, float]]]) -> List[KGNode]:
        """Create sentence-level nodes with entities and inherited themes."""
        sentence_nodes = []

        # Create entity lookup
        entity_lookup = {}
        sentence_entities = entity_theme_data['extraction_results'].get('sentence_entities', [])
        for entity_result in sentence_entities:
            entity_lookup[entity_result.source_id] = entity_result.entities

        for sentence_emb in sentence_embeddings:
            source_document = sentence_emb.source_article

            # Get themes (inherited from document)
            theme_data = theme_bridge_builder.get_inherited_themes_for_node(source_document, theme_bridges)

            # Get entities
            entities = entity_lookup.get(sentence_emb.sentence_id, {'PERSON': [], 'ORG': [], 'GPE': []})

            # Create enhanced sentence node
            node_id = f"sent_{hashlib.md5(sentence_emb.sentence_id.encode()).hexdigest()[:8]}"
            properties = {
                # RAGAS required fields
                'page_content': sentence_emb.sentence_text,

                # Core sentence properties
                'sentence_id': sentence_emb.sentence_id,
                'text': sentence_emb.sentence_text,
                'source_article': source_document,
                'sentence_index': sentence_emb.sentence_index,
                'containing_chunks': sentence_emb.containing_chunks,

                # Entities
                'entities': entities,

                # Theme inheritance (the new semantic highways)
                'direct_themes': theme_data['direct_themes'],
                'inherited_themes': theme_data['inherited_themes'],
                'theme_inheritance_map': theme_data['theme_inheritance_map'],
                'total_semantic_themes': theme_data['total_themes']
            }

            node = KGNode(
                id=node_id,
                type="SENTENCE",
                granularity_level=2,
                properties=properties
            )
            sentence_nodes.append(node)

        return sentence_nodes

    def _build_high_confidence_entity_relationships(self, nodes_by_type: Dict[str, List[KGNode]]) -> List[
        KGRelationship]:
        """Build only high-confidence entity overlap relationships."""
        relationships = []

        # Get configuration
        entity_config = self.config.get('knowledge_graph_assembly', {}).get('entity_relationships', {})
        if not entity_config.get('enabled', True):
            return relationships

        min_entity_overlap = entity_config.get('min_entity_overlap', 2)
        min_jaccard_similarity = entity_config.get('min_jaccard_similarity', 0.3)

        # Get all nodes for entity overlap computation (chunks and sentences only for efficiency)
        processable_nodes = []
        for node_type in ['CHUNK', 'SENTENCE']:
            processable_nodes.extend(nodes_by_type.get(node_type, []))

        if len(processable_nodes) < 2:
            return relationships

        self.logger.info(f"🏷️  Building high-confidence entity relationships for {len(processable_nodes)} nodes...")

        entity_types = ['PERSON', 'ORG', 'GPE']
        relationship_count = 0

        # Compare each pair of nodes for entity overlap
        for i, source_node in enumerate(processable_nodes):
            source_entities = self._extract_node_entities(source_node, entity_types)

            if len(source_entities) == 0:
                continue

            # Only check against subsequent nodes to avoid duplicates
            for target_node in processable_nodes[i + 1:]:
                target_entities = self._extract_node_entities(target_node, entity_types)

                if len(target_entities) == 0:
                    continue

                # Calculate entity overlap
                overlap = source_entities.intersection(target_entities)
                union = source_entities.union(target_entities)

                # Apply high-confidence filters
                if len(overlap) >= min_entity_overlap and len(union) > 0:
                    jaccard_similarity = len(overlap) / len(union)

                    if jaccard_similarity >= min_jaccard_similarity:
                        # Create bidirectional relationships
                        relationships.extend([
                            KGRelationship(
                                source=source_node.id,
                                target=target_node.id,
                                type="high_confidence_entity_overlap",
                                granularity_type="entity_based",
                                properties={
                                    'jaccard_similarity': jaccard_similarity,
                                    'shared_entities': list(overlap),
                                    'entity_overlap_count': len(overlap),
                                    'source_type': source_node.type,
                                    'target_type': target_node.type
                                },
                                weight=jaccard_similarity
                            ),
                            KGRelationship(
                                source=target_node.id,
                                target=source_node.id,
                                type="high_confidence_entity_overlap",
                                granularity_type="entity_based",
                                properties={
                                    'jaccard_similarity': jaccard_similarity,
                                    'shared_entities': list(overlap),
                                    'entity_overlap_count': len(overlap),
                                    'source_type': target_node.type,
                                    'target_type': source_node.type
                                },
                                weight=jaccard_similarity
                            )
                        ])
                        relationship_count += 2

        self.logger.info(f"✅ Built {relationship_count} high-confidence entity relationships")
        return relationships

    def _extract_node_entities(self, node: KGNode, entity_types: List[str]) -> set:
        """Extract entities from a node for overlap calculation."""
        entities = node.properties.get('entities', {})
        all_entities = set()

        for entity_type in entity_types:
            node_entities = entities.get(entity_type, [])
            # Normalize to lowercase for comparison
            all_entities.update({e.lower() for e in node_entities})

        return all_entities

    def _build_enhanced_relationships(self, nodes_by_type: Dict[str, List[KGNode]],
                                      similarity_data: Dict[str, Any],
                                      theme_bridges: Dict[str, List[Tuple[str, float]]]) -> List[KGRelationship]:
        """Build relationships using pre-computed connections from Phase 4."""
        all_relationships = []

        self.logger.info("🔗 Building relationships using pre-computed similarity connections")

        # Build hierarchical relationships
        hierarchical_builder = HierarchicalRelationshipBuilder()
        hierarchical_relationships = hierarchical_builder.build_relationships(nodes_by_type, similarity_data)
        all_relationships.extend(hierarchical_relationships)
        self.logger.info(f"✅ Built {len(hierarchical_relationships)} hierarchical relationships")

        # Build similarity relationships using pre-computed connections
        similarity_relationships = self._build_similarity_relationships_from_connections(
            nodes_by_type, similarity_data
        )
        all_relationships.extend(similarity_relationships)
        self.logger.info(
            f"✅ Built {len(similarity_relationships)} similarity relationships from pre-computed connections")

        # Build entity overlap relationships (high-confidence only)
        entity_relationships = self._build_high_confidence_entity_relationships(nodes_by_type)
        all_relationships.extend(entity_relationships)
        self.logger.info(f"✅ Built {len(entity_relationships)} high-confidence entity relationships")

        return all_relationships

    def _build_similarity_relationships_from_connections(self, nodes_by_type: Dict[str, List[KGNode]],
                                                         similarity_data: Dict[str, Any]) -> List[KGRelationship]:
        """Build relationships directly from Phase 4 pre-computed connections."""
        relationships = []

        # Get pre-computed connections from Phase 4
        connections = similarity_data.get('connections', [])
        if not connections:
            self.logger.warning("No pre-computed connections found in similarity data")
            return relationships

        # Create node lookup by source IDs
        node_lookup = {}
        for node_list in nodes_by_type.values():
            for node in node_list:
                # Map various ID types to nodes
                props = node.properties
                if 'chunk_id' in props:
                    node_lookup[props['chunk_id']] = node
                if 'sentence_id' in props:
                    node_lookup[props['sentence_id']] = node
                if 'doc_id' in props:
                    node_lookup[props['doc_id']] = node

        # Convert connections to relationships
        for connection in connections:
            source_node = node_lookup.get(connection.source_id)
            target_node = node_lookup.get(connection.target_id)

            if source_node and target_node:
                relationship = KGRelationship(
                    source=source_node.id,
                    target=target_node.id,
                    type=connection.connection_type,
                    granularity_type=connection.granularity_type,
                    properties={
                        'similarity_score': connection.similarity_score,
                        'source_type': source_node.type,
                        'target_type': target_node.type,
                        'computed_in_phase': '4_similarity_matrices'
                    },
                    weight=connection.similarity_score
                )
                relationships.append(relationship)

        return relationships

    def build_knowledge_graph(self, chunks: List[Dict[str, Any]],
                              multi_granularity_embeddings: Dict[str, Dict[str, List[Any]]],
                              multi_granularity_similarities: Dict[str, Dict[str, Any]],
                              entity_theme_data: Dict[str, Any]) -> MultiGranularityKnowledgeGraph:
        """Build multi-granularity knowledge graph using pre-computed similarities and extracted entities/themes."""
        start_time = time.time()

        self.logger.info("🌟 Building Phase 6: Knowledge Graph Assembly using pre-computed data")

        # Create knowledge graph
        kg = MultiGranularityKnowledgeGraph()

        # Extract embeddings and similarities for the first model
        model_name = list(multi_granularity_embeddings.keys())[0]
        granularity_embeddings = multi_granularity_embeddings[model_name]
        similarity_data = multi_granularity_similarities[model_name]

        # Initialize theme bridge builder
        embedding_model = EmbeddingModel(model_name, self.config['system']['device'], self.logger)
        theme_bridge_builder = ThemeBridgeBuilder(entity_theme_data, embedding_model, self.config, self.logger)

        # Compute cross-document theme bridges
        theme_bridges = theme_bridge_builder.compute_cross_document_theme_bridges()

        # Step 1: Create document nodes (level 0) with themes
        document_nodes = self._create_enhanced_document_nodes(
            chunks, granularity_embeddings.get('doc_summaries', []),
            entity_theme_data, theme_bridge_builder, theme_bridges
        )
        for node in document_nodes:
            kg.add_node(node)

        # Step 2: Create chunk nodes (level 1) with inherited themes and entities
        chunk_nodes = self._create_enhanced_chunk_nodes(
            chunks, granularity_embeddings.get('chunks', []),
            entity_theme_data, theme_bridge_builder, theme_bridges
        )
        for node in chunk_nodes:
            kg.add_node(node)

        # Step 3: Create sentence nodes (level 2) with inherited themes and entities
        sentence_nodes = self._create_enhanced_sentence_nodes(
            granularity_embeddings.get('sentences', []),
            entity_theme_data, theme_bridge_builder, theme_bridges
        )
        for node in sentence_nodes:
            kg.add_node(node)

        # Step 4: Group nodes by type for relationship building
        nodes_by_type = {
            'DOCUMENT': document_nodes,
            'CHUNK': chunk_nodes,
            'SENTENCE': sentence_nodes
        }

        # Step 5: Build relationships using pre-computed similarity connections
        relationships = self._build_enhanced_relationships(nodes_by_type, similarity_data, theme_bridges)
        for rel in relationships:
            kg.add_relationship(rel)

        # Step 6: Add metadata
        build_time = time.time() - start_time
        kg.metadata = {
            'created_at': datetime.now().isoformat(),
            'architecture': 'phase6_assembly_with_theme_bridges',
            'total_nodes': len(kg.nodes),
            'total_relationships': len(kg.relationships),
            'build_time': build_time,
            'config': self.kg_config,
            'granularity_counts': {
                'documents': len(document_nodes),
                'chunks': len(chunk_nodes),
                'sentences': len(sentence_nodes)
            },
            'theme_bridge_stats': {
                'total_unique_themes': len(theme_bridge_builder.all_unique_themes),
                'themes_with_bridges': len(theme_bridges),
                'total_bridges': sum(len(bridges) for bridges in theme_bridges.values())
            },
            'relationship_types': {
                rel_type: len([r for r in kg.relationships if r.type == rel_type])
                for rel_type in set(r.type for r in kg.relationships)
            }
        }

        self.logger.info(
            f"🎉 Knowledge graph assembled: {len(kg.nodes)} nodes, {len(kg.relationships)} relationships in {build_time:.2f}s")
        self.logger.info(f"🌉 Theme bridges: {len(theme_bridges)} themes with cross-document connections")

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