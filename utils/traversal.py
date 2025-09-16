#!/usr/bin/env python3
"""
Semantic RAG Traversal Rules Engine
==================================

Unified traversal rulebook for both question generation and retrieval algorithms.
Ensures systemic coherence by defining exactly where and how semantic navigation can occur.

This module implements the core principle: Question generation methodology must directly 
reflect retrieval algorithm capabilities.

PURE RULES ONLY - No computational logic or scoring.
"""

import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass


class ConnectionType(Enum):
    """
    Defines all possible connection types in the semantic knowledge graph.
    Each type has specific granularity constraints and navigation rules.
    """
    RAW_SIMILARITY = "raw_similarity"           # Direct cosine similarity between nodes
    THEME_BRIDGE = "theme_bridge"               # Cross-document theme connections  
    HIERARCHICAL = "hierarchical"               # Parent-child relationships in Doc→Chunk→Sentence
    SEQUENTIAL = "sequential"                   # Narrative flow connections (document-scoped)
    ENTITY_OVERLAP = "entity_overlap"           # Shared entity connections (if enabled)


class GranularityLevel(Enum):
    """Node granularity levels in the three-tier hierarchy."""
    DOCUMENT = 0
    CHUNK = 1  
    SENTENCE = 2


@dataclass
class TraversalPath:
    """
    Represents a complete traversal path through the knowledge graph.
    Used for both question generation and retrieval validation.
    """
    nodes: List[str]                    # Node IDs in traversal order
    connection_types: List[ConnectionType]  # Connection type used for each hop
    granularity_levels: List[GranularityLevel]  # Granularity of each node
    total_hops: int
    is_valid: bool
    validation_errors: List[str]
    
    def __post_init__(self):
        """Validate path consistency after initialization."""
        self.total_hops = len(self.connection_types)
        if len(self.nodes) != self.total_hops + 1:
            self.is_valid = False
            if not hasattr(self, 'validation_errors'):
                self.validation_errors = []
            self.validation_errors.append("Node count must be connection count + 1")


class GranularityRules:
    """
    Defines which granularity levels are allowed for each connection type.
    Implements the core constraint that different connection types operate at different scales.
    """
    
    # Raw similarity: Chunk-level only (no sentence-to-sentence semantic jumping)
    RAW_SIMILARITY_GRANULARITIES = {GranularityLevel.CHUNK}
    
    # Theme bridges: Document and chunk levels (provide cross-document navigation)
    THEME_BRIDGE_GRANULARITIES = {GranularityLevel.DOCUMENT, GranularityLevel.CHUNK}
    
    # Hierarchical: All levels (but only in downward progression)
    HIERARCHICAL_GRANULARITIES = {GranularityLevel.DOCUMENT, GranularityLevel.CHUNK, GranularityLevel.SENTENCE}
    
    # Sequential flow: Document-scoped sentences, inter-document chunks/docs
    SEQUENTIAL_GRANULARITIES = {GranularityLevel.DOCUMENT, GranularityLevel.CHUNK, GranularityLevel.SENTENCE}
    
    # Entity overlap: All levels (if enabled, but removed for quality in current system)
    ENTITY_OVERLAP_GRANULARITIES = {GranularityLevel.DOCUMENT, GranularityLevel.CHUNK, GranularityLevel.SENTENCE}
    
    @classmethod
    def get_allowed_granularities(cls, connection_type: ConnectionType) -> Set[GranularityLevel]:
        """Get allowed granularities for a connection type."""
        mapping = {
            ConnectionType.RAW_SIMILARITY: cls.RAW_SIMILARITY_GRANULARITIES,
            ConnectionType.THEME_BRIDGE: cls.THEME_BRIDGE_GRANULARITIES,
            ConnectionType.HIERARCHICAL: cls.HIERARCHICAL_GRANULARITIES,
            ConnectionType.SEQUENTIAL: cls.SEQUENTIAL_GRANULARITIES,
            ConnectionType.ENTITY_OVERLAP: cls.ENTITY_OVERLAP_GRANULARITIES,
        }
        return mapping.get(connection_type, set())
    
    @classmethod
    def is_granularity_allowed(cls, connection_type: ConnectionType, granularity: GranularityLevel) -> bool:
        """Check if a granularity level is allowed for a connection type."""
        allowed = cls.get_allowed_granularities(connection_type)
        return granularity in allowed


class ScopeRules:
    """
    Defines scope constraints for different connection types and granularities.
    Critical rule: Sentence-level sequential connections must remain document-scoped.
    """
    
    # Sequential scope rules: Sentences never cross documents
    SEQUENTIAL_SCOPE_RULES = {
        GranularityLevel.SENTENCE: "document_only",      # Never cross documents at sentence level
        GranularityLevel.CHUNK: "inter_document_allowed", # Can cross documents
        GranularityLevel.DOCUMENT: "always_inter_document" # Always crosses to different document
    }
    
    # Raw similarity scope: Chunks can be intra or inter-document
    RAW_SIMILARITY_SCOPE_RULES = {
        GranularityLevel.CHUNK: "intra_and_inter_document"
    }
    
    # Theme bridge scope: Always cross-document by definition
    THEME_BRIDGE_SCOPE_RULES = {
        GranularityLevel.DOCUMENT: "always_inter_document",
        GranularityLevel.CHUNK: "always_inter_document"
    }
    
    @classmethod
    def get_scope_constraint(cls, connection_type: ConnectionType, granularity: GranularityLevel) -> str:
        """Get scope constraint for a connection type and granularity."""
        if connection_type == ConnectionType.SEQUENTIAL:
            return cls.SEQUENTIAL_SCOPE_RULES.get(granularity, "unknown")
        elif connection_type == ConnectionType.RAW_SIMILARITY:
            return cls.RAW_SIMILARITY_SCOPE_RULES.get(granularity, "unknown")
        elif connection_type == ConnectionType.THEME_BRIDGE:
            return cls.THEME_BRIDGE_SCOPE_RULES.get(granularity, "unknown")
        elif connection_type == ConnectionType.HIERARCHICAL:
            return "intra_document_only"  # Hierarchical relationships are always within document
        else:
            return "unknown"


class TraversalConstraints:
    """
    Defines constraints for different types of traversal patterns.
    Implements the progressive complexity approach from simple to expert-level navigation.
    """
    
    # Phase 1: Raw similarity (chunk-level only)
    RAW_SIMILARITY_CONSTRAINTS = {
        "max_hops": 1,
        "allowed_granularities": [GranularityLevel.CHUNK],
        "unique_connections_required": False,
        "min_context_sentences": 3
    }
    
    # Phase 2: Hierarchical navigation (3-node consistency)
    HIERARCHICAL_CONSTRAINTS = {
        "max_hops": 2,
        "required_progression": [GranularityLevel.DOCUMENT, GranularityLevel.CHUNK, GranularityLevel.SENTENCE],
        "unique_connections_required": False,
        "context_inheritance_required": True
    }
    
    # Phase 3: Sequential flow (minimum 5 hops for meaningful narrative)
    SEQUENTIAL_FLOW_CONSTRAINTS = {
        "min_hops": 5,
        "max_hops": 10,
        "document_scope_required": True,  # For sentence-level
        "unique_connections_required": False,
        "narrative_coherence_required": True
    }
    
    # Phase 4: Theme bridge (intelligent cross-document navigation)
    THEME_BRIDGE_CONSTRAINTS = {
        "max_hops": 3,
        "cross_document_required": True,
        "landing_strategy_required": True,
        "context_preservation_required": True
    }
    
    # Phase 5: Multi-dimensional integration (3-hop non-repeating)
    MULTI_DIMENSIONAL_CONSTRAINTS = {
        "required_hops": 3,
        "unique_connections_required": True,
        "max_connection_types": 3,
        "complexity_level": "expert"
    }
    
    @classmethod
    def get_constraints_for_pattern(cls, pattern_name: str) -> Dict[str, Any]:
        """Get constraints for a specific traversal pattern."""
        mapping = {
            "raw_similarity": cls.RAW_SIMILARITY_CONSTRAINTS,
            "hierarchical": cls.HIERARCHICAL_CONSTRAINTS,
            "sequential_flow": cls.SEQUENTIAL_FLOW_CONSTRAINTS,
            "theme_bridge": cls.THEME_BRIDGE_CONSTRAINTS,
            "multi_dimensional": cls.MULTI_DIMENSIONAL_CONSTRAINTS,
        }
        return mapping.get(pattern_name, {})


class NavigationLogic:
    """
    Core navigation logic for semantic traversal.
    PURE RULES ONLY - No computational work.
    """
    
    @staticmethod
    def choose_landing_strategy(source_granularity: GranularityLevel, 
                              target_document: str, 
                              connection_type: ConnectionType) -> str:
        """
        Determine how to land in target document based on connection type and source.
        Critical for theme bridge navigation - use document-aware theme analysis.
        """
        if connection_type == ConnectionType.THEME_BRIDGE:
            if source_granularity == GranularityLevel.DOCUMENT:
                return "document_to_document_theme_matching"
            elif source_granularity == GranularityLevel.CHUNK:
                return "chunk_to_most_relevant_chunk_in_target_document"
        
        elif connection_type == ConnectionType.RAW_SIMILARITY:
            return "direct_similarity_matching"
        
        elif connection_type == ConnectionType.HIERARCHICAL:
            if source_granularity == GranularityLevel.DOCUMENT:
                return "document_to_child_chunk"
            elif source_granularity == GranularityLevel.CHUNK:
                return "chunk_to_child_sentence"
        
        elif connection_type == ConnectionType.SEQUENTIAL:
            return "narrative_flow_continuation"
        
        return "default_similarity_based"
    
    @staticmethod
    def validate_path_coherence(path_nodes: List[str], 
                              connection_types: List[ConnectionType],
                              node_granularities: List[GranularityLevel],
                              node_documents: List[str]) -> Tuple[bool, List[str]]:
        """
        Ensure traversal path maintains semantic coherence.
        Validates all granularity, scope, and progression rules.
        PURE VALIDATION - No computational work.
        """
        errors = []
        
        if len(path_nodes) != len(connection_types) + 1:
            errors.append("Path node count must equal connection count + 1")
            return False, errors
        
        if len(path_nodes) != len(node_granularities):
            errors.append("Path node count must equal granularity count")
            return False, errors
        
        if len(path_nodes) != len(node_documents):
            errors.append("Path node count must equal document count")
            return False, errors
        
        # Validate each connection
        for i, (connection_type, source_granularity, target_granularity, source_doc, target_doc) in enumerate(
            zip(connection_types, node_granularities[:-1], node_granularities[1:], 
                node_documents[:-1], node_documents[1:])
        ):
            # Check granularity rules
            if not GranularityRules.is_granularity_allowed(connection_type, source_granularity):
                errors.append(f"Hop {i}: Source granularity {source_granularity} not allowed for {connection_type}")
            
            if not GranularityRules.is_granularity_allowed(connection_type, target_granularity):
                errors.append(f"Hop {i}: Target granularity {target_granularity} not allowed for {connection_type}")
            
            # Check scope rules
            scope_constraint = ScopeRules.get_scope_constraint(connection_type, source_granularity)
            
            if scope_constraint == "document_only" and source_doc != target_doc:
                errors.append(f"Hop {i}: {connection_type} at {source_granularity} level cannot cross documents")
            
            if scope_constraint == "always_inter_document" and source_doc == target_doc:
                errors.append(f"Hop {i}: {connection_type} at {source_granularity} level must cross documents")
            
            # Hierarchical progression validation
            if connection_type == ConnectionType.HIERARCHICAL:
                if target_granularity.value != source_granularity.value + 1:
                    errors.append(f"Hop {i}: Hierarchical connection must progress down one level")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_multi_dimensional_constraints(connection_types: List[ConnectionType]) -> Tuple[bool, List[str]]:
        """
        Validate multi-dimensional traversal constraints.
        Ensures no repeating connection types and proper complexity.
        """
        errors = []
        
        # Check unique connections requirement
        if len(set(connection_types)) != len(connection_types):
            errors.append("Multi-dimensional traversal requires unique connection types")
        
        # Check required hop count
        constraints = TraversalConstraints.MULTI_DIMENSIONAL_CONSTRAINTS
        required_hops = constraints["required_hops"]
        
        if len(connection_types) != required_hops:
            errors.append(f"Multi-dimensional traversal requires exactly {required_hops} hops")
        
        return len(errors) == 0, errors


class TraversalValidator:
    """
    High-level validator that combines all rules to validate complete traversal paths.
    Used by both question generation and retrieval algorithms.
    PURE VALIDATION ONLY - No computational scoring.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize validator with optional logger."""
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_path(self, 
                     path_nodes: List[str],
                     connection_types: List[ConnectionType], 
                     node_granularities: List[GranularityLevel],
                     node_documents: List[str],
                     traversal_pattern: str) -> TraversalPath:
        """
        Comprehensive path validation using all traversal rules.
        Returns a TraversalPath object with validation results.
        NO CONTEXT SCORING - Pure structural validation only.
        """
        validation_errors = []
        
        # Basic path coherence validation
        is_coherent, coherence_errors = NavigationLogic.validate_path_coherence(
            path_nodes, connection_types, node_granularities, node_documents
        )
        validation_errors.extend(coherence_errors)
        
        # Pattern-specific constraint validation
        pattern_constraints = TraversalConstraints.get_constraints_for_pattern(traversal_pattern)
        
        if pattern_constraints:
            # Check hop count constraints
            if "max_hops" in pattern_constraints and len(connection_types) > pattern_constraints["max_hops"]:
                validation_errors.append(f"Path exceeds max hops for {traversal_pattern}: {len(connection_types)} > {pattern_constraints['max_hops']}")
            
            if "min_hops" in pattern_constraints and len(connection_types) < pattern_constraints["min_hops"]:
                validation_errors.append(f"Path below min hops for {traversal_pattern}: {len(connection_types)} < {pattern_constraints['min_hops']}")
            
            if "required_hops" in pattern_constraints and len(connection_types) != pattern_constraints["required_hops"]:
                validation_errors.append(f"Path must have exactly {pattern_constraints['required_hops']} hops for {traversal_pattern}")
            
            # Multi-dimensional specific validation
            if traversal_pattern == "multi_dimensional":
                is_multi_valid, multi_errors = NavigationLogic.validate_multi_dimensional_constraints(connection_types)
                validation_errors.extend(multi_errors)
        
        # Create traversal path object (no context scoring)
        traversal_path = TraversalPath(
            nodes=path_nodes,
            connection_types=connection_types,
            granularity_levels=node_granularities,
            total_hops=len(connection_types),
            is_valid=len(validation_errors) == 0,
            validation_errors=validation_errors
        )
        
        # Log validation results
        if self.logger:
            if traversal_path.is_valid:
                self.logger.debug(f"✅ Path validation successful for {traversal_pattern}: {len(path_nodes)} nodes, {len(connection_types)} hops")
            else:
                self.logger.warning(f"❌ Path validation failed for {traversal_pattern}: {'; '.join(validation_errors)}")
        
        return traversal_path
    
    def generate_valid_path_templates(self, traversal_pattern: str) -> List[Dict[str, Any]]:
        """
        Generate templates for valid paths based on traversal pattern.
        Used by question generation to create coherent questions.
        STRUCTURAL TEMPLATES ONLY - No viability scoring.
        """
        templates = []
        
        if traversal_pattern == "raw_similarity":
            templates.append({
                "pattern": "raw_similarity",
                "hops": 1,
                "granularities": [GranularityLevel.CHUNK, GranularityLevel.CHUNK],
                "connection_types": [ConnectionType.RAW_SIMILARITY],
                "cross_document": True,  # Can be intra or inter
                "description": "Direct chunk-to-chunk similarity"
            })
        
        elif traversal_pattern == "hierarchical":
            templates.append({
                "pattern": "hierarchical", 
                "hops": 2,
                "granularities": [GranularityLevel.DOCUMENT, GranularityLevel.CHUNK, GranularityLevel.SENTENCE],
                "connection_types": [ConnectionType.HIERARCHICAL, ConnectionType.HIERARCHICAL],
                "cross_document": False,
                "description": "Document → Chunk → Sentence progression"
            })
        
        elif traversal_pattern == "theme_bridge":
            templates.append({
                "pattern": "theme_bridge",
                "hops": 2,
                "granularities": [GranularityLevel.CHUNK, GranularityLevel.CHUNK],
                "connection_types": [ConnectionType.THEME_BRIDGE, ConnectionType.HIERARCHICAL],
                "cross_document": True,
                "description": "Cross-document theme bridge with hierarchical refinement"
            })
        
        elif traversal_pattern == "multi_dimensional":
            templates.append({
                "pattern": "multi_dimensional",
                "hops": 3,
                "granularities": [GranularityLevel.DOCUMENT, GranularityLevel.CHUNK, GranularityLevel.CHUNK, GranularityLevel.SENTENCE],
                "connection_types": [ConnectionType.HIERARCHICAL, ConnectionType.THEME_BRIDGE, ConnectionType.SEQUENTIAL],
                "cross_document": True,
                "description": "Complex multi-hop with unique connection types"
            })
        
        elif traversal_pattern == "sequential_flow":
            # Multiple templates for different sequential patterns
            templates.extend([
                {
                    "pattern": "sequential_flow",
                    "hops": 5,
                    "granularities": [GranularityLevel.SENTENCE] * 6,
                    "connection_types": [ConnectionType.SEQUENTIAL] * 5,
                    "cross_document": False,
                    "description": "Document-scoped sentence flow (minimum 5 hops)"
                },
                {
                    "pattern": "sequential_flow", 
                    "hops": 3,
                    "granularities": [GranularityLevel.CHUNK, GranularityLevel.CHUNK, GranularityLevel.CHUNK, GranularityLevel.CHUNK],
                    "connection_types": [ConnectionType.SEQUENTIAL] * 3,
                    "cross_document": True,
                    "description": "Inter-document chunk flow"
                }
            ])
        
        return templates


# Public API for easy import and usage
__all__ = [
    "ConnectionType",
    "GranularityLevel", 
    "TraversalPath",
    "GranularityRules",
    "ScopeRules", 
    "TraversalConstraints",
    "NavigationLogic",
    "TraversalValidator"
]
