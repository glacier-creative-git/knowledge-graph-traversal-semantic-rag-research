#!/usr/bin/env python3
"""
Traversal Rules Engine Test Suite
================================

Comprehensive test suite for the semantic RAG traversal rules engine.
Tests all validation logic, constraints, and path planning capabilities.

This script can run standalone tests or validate against a real knowledge_graph.json file.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict

# Add utils to Python path for imports
sys.path.append(str(Path(__file__).parent / "utils"))

from traversal import (
    ConnectionType, GranularityLevel, TraversalPath,
    GranularityRules, ScopeRules, TraversalConstraints,
    NavigationLogic, TraversalValidator, TraversalPlanner
)


class TraversalRulesTestSuite:
    """Comprehensive test suite for traversal rules validation."""
    
    def __init__(self):
        """Initialize test suite."""
        self.setup_logging()
        self.validator = TraversalValidator(self.logger)
        self.planner = TraversalPlanner(self.validator, self.logger)
        self.test_results = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "failures": []
        }
        
        # Sample knowledge graph data for testing
        self.sample_kg_data = None
        self.load_knowledge_graph()
    
    def setup_logging(self):
        """Setup logging for test output."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("TraversalTests")
    
    def load_knowledge_graph(self):
        """Load knowledge graph data if available."""
        kg_paths = [
            Path("data/knowledge_graph.json"),
            Path("knowledge_graph.json")
        ]
        
        for kg_path in kg_paths:
            if kg_path.exists():
                try:
                    with open(kg_path, 'r', encoding='utf-8') as f:
                        self.sample_kg_data = json.load(f)
                    self.logger.info(f"✅ Loaded knowledge graph from {kg_path}")
                    return
                except Exception as e:
                    self.logger.warning(f"⚠️  Failed to load KG from {kg_path}: {e}")
        
        self.logger.info("📝 No knowledge graph found, using synthetic test data")
    
    def assert_test(self, condition: bool, test_name: str, error_msg: str = ""):
        """Assert a test condition and track results."""
        self.test_results["tests_run"] += 1
        
        if condition:
            self.test_results["tests_passed"] += 1
            self.logger.info(f"✅ {test_name}")
        else:
            self.test_results["tests_failed"] += 1
            failure_msg = f"{test_name}: {error_msg}" if error_msg else test_name
            self.test_results["failures"].append(failure_msg)
            self.logger.error(f"❌ {failure_msg}")
    
    def test_granularity_rules(self):
        """Test granularity rules for different connection types."""
        self.logger.info("🔍 Testing Granularity Rules...")
        
        # Test raw similarity constraints (chunk-level only)
        self.assert_test(
            GranularityRules.is_granularity_allowed(ConnectionType.RAW_SIMILARITY, GranularityLevel.CHUNK),
            "Raw similarity allows chunks",
        )
        
        self.assert_test(
            not GranularityRules.is_granularity_allowed(ConnectionType.RAW_SIMILARITY, GranularityLevel.SENTENCE),
            "Raw similarity disallows sentences",
            "Should not allow sentence-to-sentence semantic jumping"
        )
        
        # Test hierarchical constraints (all levels allowed)
        for level in [GranularityLevel.DOCUMENT, GranularityLevel.CHUNK, GranularityLevel.SENTENCE]:
            self.assert_test(
                GranularityRules.is_granularity_allowed(ConnectionType.HIERARCHICAL, level),
                f"Hierarchical allows {level.name}",
            )
        
        # Test theme bridge constraints (document and chunk only)
        self.assert_test(
            GranularityRules.is_granularity_allowed(ConnectionType.THEME_BRIDGE, GranularityLevel.DOCUMENT),
            "Theme bridge allows documents",
        )
        
        self.assert_test(
            GranularityRules.is_granularity_allowed(ConnectionType.THEME_BRIDGE, GranularityLevel.CHUNK),
            "Theme bridge allows chunks",
        )
        
        self.assert_test(
            not GranularityRules.is_granularity_allowed(ConnectionType.THEME_BRIDGE, GranularityLevel.SENTENCE),
            "Theme bridge disallows sentences",
            "Theme bridges should not operate at sentence level"
        )
    
    def test_scope_rules(self):
        """Test scope rules for cross-document navigation."""
        self.logger.info("🌐 Testing Scope Rules...")
        
        # Test sequential scope rules (sentences document-only)
        sentence_scope = ScopeRules.get_scope_constraint(ConnectionType.SEQUENTIAL, GranularityLevel.SENTENCE)
        self.assert_test(
            sentence_scope == "document_only",
            "Sequential sentences are document-scoped",
            f"Got {sentence_scope}, expected 'document_only'"
        )
        
        # Test theme bridge scope (always inter-document)
        doc_theme_scope = ScopeRules.get_scope_constraint(ConnectionType.THEME_BRIDGE, GranularityLevel.DOCUMENT)
        self.assert_test(
            doc_theme_scope == "always_inter_document",
            "Theme bridges always cross documents",
            f"Got {doc_theme_scope}, expected 'always_inter_document'"
        )
        
        # Test hierarchical scope (intra-document only)
        hier_scope = ScopeRules.get_scope_constraint(ConnectionType.HIERARCHICAL, GranularityLevel.CHUNK)
        self.assert_test(
            hier_scope == "intra_document_only",
            "Hierarchical connections are intra-document",
            f"Got {hier_scope}, expected 'intra_document_only'"
        )
    
    def test_valid_path_validation(self):
        """Test validation of valid traversal paths."""
        self.logger.info("✅ Testing Valid Path Validation...")
        
        # Test valid raw similarity path (chunk to chunk)
        valid_raw_similarity = self.validator.validate_path(
            path_nodes=["chunk_1", "chunk_2"],
            connection_types=[ConnectionType.RAW_SIMILARITY],
            node_granularities=[GranularityLevel.CHUNK, GranularityLevel.CHUNK],
            node_documents=["doc_a", "doc_b"],  # Can cross documents
            traversal_pattern="raw_similarity"
        )
        
        self.assert_test(
            valid_raw_similarity.is_valid,
            "Valid raw similarity path passes validation",
            f"Errors: {valid_raw_similarity.validation_errors}"
        )
        
        # Test valid hierarchical path (doc → chunk → sentence)
        valid_hierarchical = self.validator.validate_path(
            path_nodes=["doc_1", "chunk_1", "sentence_1"],
            connection_types=[ConnectionType.HIERARCHICAL, ConnectionType.HIERARCHICAL],
            node_granularities=[GranularityLevel.DOCUMENT, GranularityLevel.CHUNK, GranularityLevel.SENTENCE],
            node_documents=["doc_a", "doc_a", "doc_a"],  # Must stay within document
            traversal_pattern="hierarchical"
        )
        
        self.assert_test(
            valid_hierarchical.is_valid,
            "Valid hierarchical path passes validation",
            f"Errors: {valid_hierarchical.validation_errors}"
        )
        
        # Test valid theme bridge path
        valid_theme_bridge = self.validator.validate_path(
            path_nodes=["chunk_1", "chunk_2", "sentence_2"],
            connection_types=[ConnectionType.THEME_BRIDGE, ConnectionType.HIERARCHICAL],
            node_granularities=[GranularityLevel.CHUNK, GranularityLevel.CHUNK, GranularityLevel.SENTENCE],
            node_documents=["doc_a", "doc_b", "doc_b"],  # Cross documents then hierarchical
            traversal_pattern="theme_bridge"
        )
        
        self.assert_test(
            valid_theme_bridge.is_valid,
            "Valid theme bridge path passes validation",
            f"Errors: {valid_theme_bridge.validation_errors}"
        )
    
    def test_invalid_path_validation(self):
        """Test validation of invalid traversal paths."""
        self.logger.info("❌ Testing Invalid Path Validation...")
        
        # Test invalid: sentence-to-sentence via raw similarity
        invalid_sentence_similarity = self.validator.validate_path(
            path_nodes=["sentence_1", "sentence_2"],
            connection_types=[ConnectionType.RAW_SIMILARITY],
            node_granularities=[GranularityLevel.SENTENCE, GranularityLevel.SENTENCE],
            node_documents=["doc_a", "doc_b"],
            traversal_pattern="raw_similarity"
        )
        
        self.assert_test(
            not invalid_sentence_similarity.is_valid,
            "Invalid sentence-to-sentence similarity rejected",
            "Should reject sentence-level raw similarity"
        )
        
        # Test invalid: cross-document sequential at sentence level
        invalid_cross_doc_sequential = self.validator.validate_path(
            path_nodes=["sentence_1", "sentence_2"],
            connection_types=[ConnectionType.SEQUENTIAL],
            node_granularities=[GranularityLevel.SENTENCE, GranularityLevel.SENTENCE],
            node_documents=["doc_a", "doc_b"],  # Cross documents - invalid for sentences
            traversal_pattern="sequential_flow"
        )
        
        self.assert_test(
            not invalid_cross_doc_sequential.is_valid,
            "Invalid cross-document sentence sequential rejected",
            "Should reject cross-document sentence flow"
        )
        
        # Test invalid: wrong hierarchical progression
        invalid_hierarchical = self.validator.validate_path(
            path_nodes=["chunk_1", "doc_1"],  # Wrong direction
            connection_types=[ConnectionType.HIERARCHICAL],
            node_granularities=[GranularityLevel.CHUNK, GranularityLevel.DOCUMENT],
            node_documents=["doc_a", "doc_a"],
            traversal_pattern="hierarchical"
        )
        
        self.assert_test(
            not invalid_hierarchical.is_valid,
            "Invalid hierarchical progression rejected",
            "Should reject upward hierarchical progression"
        )
        
        # Test invalid: theme bridge within same document
        invalid_intra_theme_bridge = self.validator.validate_path(
            path_nodes=["chunk_1", "chunk_2"],
            connection_types=[ConnectionType.THEME_BRIDGE],
            node_granularities=[GranularityLevel.CHUNK, GranularityLevel.CHUNK],
            node_documents=["doc_a", "doc_a"],  # Same document - invalid for theme bridge
            traversal_pattern="theme_bridge"
        )
        
        self.assert_test(
            not invalid_intra_theme_bridge.is_valid,
            "Invalid intra-document theme bridge rejected",
            "Should reject theme bridges within same document"
        )
    
    def test_constraint_validation(self):
        """Test pattern-specific constraint validation."""
        self.logger.info("📏 Testing Pattern Constraints...")
        
        # Test raw similarity constraints (max 1 hop)
        invalid_multi_hop_raw = self.validator.validate_path(
            path_nodes=["chunk_1", "chunk_2", "chunk_3"],
            connection_types=[ConnectionType.RAW_SIMILARITY, ConnectionType.RAW_SIMILARITY],
            node_granularities=[GranularityLevel.CHUNK, GranularityLevel.CHUNK, GranularityLevel.CHUNK],
            node_documents=["doc_a", "doc_b", "doc_c"],
            traversal_pattern="raw_similarity"
        )
        
        self.assert_test(
            not invalid_multi_hop_raw.is_valid,
            "Multi-hop raw similarity rejected",
            "Raw similarity should be limited to 1 hop"
        )
        
        # Test sequential flow constraints (min 5 hops)
        invalid_short_sequential = self.validator.validate_path(
            path_nodes=["sentence_1", "sentence_2"],
            connection_types=[ConnectionType.SEQUENTIAL],
            node_granularities=[GranularityLevel.SENTENCE, GranularityLevel.SENTENCE],
            node_documents=["doc_a", "doc_a"],
            traversal_pattern="sequential_flow"
        )
        
        self.assert_test(
            not invalid_short_sequential.is_valid,
            "Short sequential flow rejected",
            "Sequential flow should require minimum 5 hops"
        )
        
        # Test multi-dimensional constraints (unique connection types)
        invalid_repeated_connections = self.validator.validate_path(
            path_nodes=["doc_1", "chunk_1", "chunk_2", "sentence_2"],
            connection_types=[ConnectionType.HIERARCHICAL, ConnectionType.HIERARCHICAL, ConnectionType.HIERARCHICAL],
            node_granularities=[GranularityLevel.DOCUMENT, GranularityLevel.CHUNK, GranularityLevel.CHUNK, GranularityLevel.SENTENCE],
            node_documents=["doc_a", "doc_a", "doc_b", "doc_b"],
            traversal_pattern="multi_dimensional"
        )
        
        self.assert_test(
            not invalid_repeated_connections.is_valid,
            "Repeated connection types in multi-dimensional rejected",
            "Multi-dimensional should require unique connection types"
        )
    
    def test_context_sufficiency_assessment(self):
        """Test context sufficiency scoring."""
        self.logger.info("📊 Testing Context Sufficiency Assessment...")
        
        # Create sample node properties
        node_properties = {
            "rich_node": {
                "page_content": "This is a comprehensive chunk of text with multiple sentences. It contains detailed information about the topic. The content is rich and provides substantial context for understanding complex relationships. Additional sentences provide even more depth and nuance to the discussion."
            },
            "poor_node": {
                "page_content": "Short text."
            },
            "empty_node": {
                "page_content": ""
            }
        }
        
        # Test context assessment for different complexity levels
        nodes = ["rich_node", "poor_node", "empty_node"]
        
        for complexity in ["simple", "medium", "hard", "expert"]:
            context_scores = NavigationLogic.assess_context_sufficiency(
                nodes, complexity, node_properties
            )
            
            # Rich node should have good scores
            self.assert_test(
                context_scores[0] > 0.5,
                f"Rich node has good context for {complexity} complexity",
                f"Score: {context_scores[0]:.2f}"
            )
            
            # Poor node should have lower scores for higher complexity
            if complexity in ["hard", "expert"]:
                self.assert_test(
                    context_scores[1] < 0.5,
                    f"Poor node has low context for {complexity} complexity",
                    f"Score: {context_scores[1]:.2f}"
                )
            
            # Empty node should have very low scores
            self.assert_test(
                context_scores[2] < 0.2,
                f"Empty node has very low context for {complexity} complexity",
                f"Score: {context_scores[2]:.2f}"
            )
    
    def test_path_template_generation(self):
        """Test generation of valid path templates."""
        self.logger.info("📋 Testing Path Template Generation...")
        
        # Test raw similarity templates
        raw_templates = self.validator.generate_valid_path_templates("raw_similarity")
        self.assert_test(
            len(raw_templates) > 0,
            "Raw similarity templates generated",
            "Should generate at least one template"
        )
        
        template = raw_templates[0]
        self.assert_test(
            template["hops"] == 1,
            "Raw similarity template has 1 hop",
            f"Got {template['hops']} hops"
        )
        
        # Test hierarchical templates
        hier_templates = self.validator.generate_valid_path_templates("hierarchical")
        self.assert_test(
            len(hier_templates) > 0,
            "Hierarchical templates generated"
        )
        
        template = hier_templates[0]
        self.assert_test(
            template["hops"] == 2,
            "Hierarchical template has 2 hops",
            f"Got {template['hops']} hops"
        )
        
        self.assert_test(
            template["granularities"] == [GranularityLevel.DOCUMENT, GranularityLevel.CHUNK, GranularityLevel.SENTENCE],
            "Hierarchical template follows Doc→Chunk→Sentence progression"
        )
        
        # Test sequential flow templates
        seq_templates = self.validator.generate_valid_path_templates("sequential_flow")
        self.assert_test(
            len(seq_templates) >= 2,
            "Sequential flow generates multiple templates",
            f"Got {len(seq_templates)} templates"
        )
        
        # Find sentence flow template
        sentence_template = None
        for template in seq_templates:
            if all(g == GranularityLevel.SENTENCE for g in template["granularities"]):
                sentence_template = template
                break
        
        self.assert_test(
            sentence_template is not None,
            "Sequential flow includes sentence template"
        )
        
        if sentence_template:
            self.assert_test(
                sentence_template["hops"] >= 5,
                "Sentence flow template has minimum 5 hops",
                f"Got {sentence_template['hops']} hops"
            )
    
    def test_traversal_planning(self):
        """Test traversal strategy planning."""
        self.logger.info("🗺️  Testing Traversal Planning...")
        
        # Test planning for different complexities
        complexities = ["simple", "medium", "hard", "expert"]
        
        for complexity in complexities:
            plans = self.planner.plan_traversal(
                start_node_granularity=GranularityLevel.CHUNK,
                target_complexity=complexity,
                traversal_pattern="raw_similarity"
            )
            
            self.assert_test(
                len(plans) > 0,
                f"Plans generated for {complexity} complexity"
            )
            
            if plans:
                best_plan = plans[0]
                self.assert_test(
                    "viability_score" in best_plan,
                    f"Best plan for {complexity} has viability score"
                )
                
                self.assert_test(
                    best_plan["viability_score"] > 0.0,
                    f"Best plan for {complexity} has positive viability",
                    f"Score: {best_plan['viability_score']:.2f}"
                )
        
        # Test cross-document requirement filtering
        cross_doc_plans = self.planner.plan_traversal(
            start_node_granularity=GranularityLevel.CHUNK,
            target_complexity="medium",
            traversal_pattern="theme_bridge",
            cross_document_required=True
        )
        
        self.assert_test(
            len(cross_doc_plans) > 0,
            "Cross-document plans generated for theme bridge"
        )
        
        if cross_doc_plans:
            for plan in cross_doc_plans:
                self.assert_test(
                    plan["template"].get("cross_document", False),
                    "Cross-document plan templates support cross-document navigation"
                )
    
    def test_multi_dimensional_validation(self):
        """Test multi-dimensional traversal validation."""
        self.logger.info("🌟 Testing Multi-Dimensional Validation...")
        
        # Test valid multi-dimensional path
        valid_multi_dim = self.validator.validate_path(
            path_nodes=["doc_1", "chunk_1", "chunk_2", "sentence_2"],
            connection_types=[ConnectionType.HIERARCHICAL, ConnectionType.THEME_BRIDGE, ConnectionType.SEQUENTIAL],
            node_granularities=[GranularityLevel.DOCUMENT, GranularityLevel.CHUNK, GranularityLevel.CHUNK, GranularityLevel.SENTENCE],
            node_documents=["doc_a", "doc_a", "doc_b", "doc_b"],
            traversal_pattern="multi_dimensional"
        )
        
        self.assert_test(
            valid_multi_dim.is_valid,
            "Valid multi-dimensional path passes validation",
            f"Errors: {valid_multi_dim.validation_errors}"
        )
        
        # Test invalid: wrong hop count
        invalid_hop_count = self.validator.validate_path(
            path_nodes=["doc_1", "chunk_1"],  # Only 1 hop
            connection_types=[ConnectionType.HIERARCHICAL],
            node_granularities=[GranularityLevel.DOCUMENT, GranularityLevel.CHUNK],
            node_documents=["doc_a", "doc_a"],
            traversal_pattern="multi_dimensional"
        )
        
        self.assert_test(
            not invalid_hop_count.is_valid,
            "Multi-dimensional with wrong hop count rejected",
            "Should require exactly 3 hops"
        )
        
        # Test navigation logic constraints
        connection_types = [ConnectionType.HIERARCHICAL, ConnectionType.THEME_BRIDGE, ConnectionType.SEQUENTIAL]
        is_valid, errors = NavigationLogic.validate_multi_dimensional_constraints(connection_types)
        
        self.assert_test(
            is_valid,
            "Valid multi-dimensional connection types pass validation"
        )
        
        # Test invalid: repeated connection types
        repeated_types = [ConnectionType.HIERARCHICAL, ConnectionType.HIERARCHICAL, ConnectionType.HIERARCHICAL]
        is_valid, errors = NavigationLogic.validate_multi_dimensional_constraints(repeated_types)
        
        self.assert_test(
            not is_valid,
            "Repeated connection types in multi-dimensional rejected",
            f"Errors: {errors}"
        )
    
    def test_real_knowledge_graph(self):
        """Test against real knowledge graph data if available."""
        if not self.sample_kg_data:
            self.logger.info("📝 Skipping real knowledge graph tests - no data loaded")
            return
        
        self.logger.info("🔍 Testing Against Real Knowledge Graph...")
        
        # Extract nodes and relationships from knowledge graph
        nodes = self.sample_kg_data.get('nodes', [])
        relationships = self.sample_kg_data.get('relationships', [])
        
        self.logger.info(f"Found {len(nodes)} nodes and {len(relationships)} relationships")
        
        # Create node lookup
        node_lookup = {node['id']: node for node in nodes}
        
        # Test some real paths from the knowledge graph
        valid_paths = []
        invalid_paths = []
        
        # Sample some paths from relationships
        for rel in relationships[:20]:  # Test first 20 relationships
            source_id = rel.get('source')
            target_id = rel.get('target')
            rel_type = rel.get('type', 'unknown')
            
            if source_id in node_lookup and target_id in node_lookup:
                source_node = node_lookup[source_id]
                target_node = node_lookup[target_id]
                
                # Map relationship types to our ConnectionType enum
                connection_type = self._map_kg_relationship_type(rel_type)
                if not connection_type:
                    continue
                
                # Get granularities from node types
                source_granularity = self._map_kg_node_type(source_node.get('type', 'UNKNOWN'))
                target_granularity = self._map_kg_node_type(target_node.get('type', 'UNKNOWN'))
                
                if source_granularity and target_granularity:
                    # Get document information
                    source_doc = self._get_node_document(source_node)
                    target_doc = self._get_node_document(target_node)
                    
                    # Determine traversal pattern
                    traversal_pattern = self._determine_traversal_pattern(connection_type)
                    
                    # Validate the path
                    path = self.validator.validate_path(
                        path_nodes=[source_id, target_id],
                        connection_types=[connection_type],
                        node_granularities=[source_granularity, target_granularity],
                        node_documents=[source_doc, target_doc],
                        traversal_pattern=traversal_pattern,
                        node_properties=node_lookup
                    )
                    
                    if path.is_valid:
                        valid_paths.append(path)
                    else:
                        invalid_paths.append(path)
        
        self.logger.info(f"Real KG validation: {len(valid_paths)} valid, {len(invalid_paths)} invalid paths")
        
        # Assert some basic expectations
        self.assert_test(
            len(valid_paths) + len(invalid_paths) > 0,
            "Successfully processed real knowledge graph paths"
        )
        
        # If we have invalid paths, check if they violate our expected rules
        if invalid_paths:
            sentence_to_sentence_raw = 0
            cross_doc_sentence_sequential = 0
            
            for path in invalid_paths:
                for error in path.validation_errors:
                    if "raw_similarity" in error and "SENTENCE" in error:
                        sentence_to_sentence_raw += 1
                    elif "sequential" in error and "cannot cross documents" in error:
                        cross_doc_sentence_sequential += 1
            
            self.logger.info(f"Expected violations found: {sentence_to_sentence_raw} sentence-to-sentence raw, {cross_doc_sentence_sequential} cross-doc sentence sequential")
    
    def _map_kg_relationship_type(self, kg_rel_type: str) -> Optional[ConnectionType]:
        """Map knowledge graph relationship type to our ConnectionType."""
        mapping = {
            'cosine_similarity': ConnectionType.RAW_SIMILARITY,
            'cosine_similarity_intra': ConnectionType.RAW_SIMILARITY,
            'cosine_similarity_inter': ConnectionType.RAW_SIMILARITY,
            'parent': ConnectionType.HIERARCHICAL,
            'child': ConnectionType.HIERARCHICAL,
            'sentence_similarity_sequential': ConnectionType.SEQUENTIAL,
            'sentence_similarity_semantic': ConnectionType.RAW_SIMILARITY,  # Map to raw similarity
            'theme_bridge': ConnectionType.THEME_BRIDGE,
            'entity_overlap': ConnectionType.ENTITY_OVERLAP,
        }
        return mapping.get(kg_rel_type)
    
    def _map_kg_node_type(self, kg_node_type: str) -> Optional[GranularityLevel]:
        """Map knowledge graph node type to our GranularityLevel."""
        mapping = {
            'DOCUMENT': GranularityLevel.DOCUMENT,
            'CHUNK': GranularityLevel.CHUNK,
            'SENTENCE': GranularityLevel.SENTENCE,
        }
        return mapping.get(kg_node_type)
    
    def _get_node_document(self, node: Dict[str, Any]) -> str:
        """Extract document name from node properties."""
        props = node.get('properties', {})
        return props.get('source_article', props.get('title', f"doc_{node['id'][:8]}"))
    
    def _determine_traversal_pattern(self, connection_type: ConnectionType) -> str:
        """Determine traversal pattern from connection type."""
        mapping = {
            ConnectionType.RAW_SIMILARITY: "raw_similarity",
            ConnectionType.HIERARCHICAL: "hierarchical",
            ConnectionType.SEQUENTIAL: "sequential_flow",
            ConnectionType.THEME_BRIDGE: "theme_bridge",
            ConnectionType.ENTITY_OVERLAP: "raw_similarity",  # Treat as similarity-based
        }
        return mapping.get(connection_type, "raw_similarity")
    
    def run_all_tests(self):
        """Run all test suites."""
        self.logger.info("🚀 Starting Traversal Rules Engine Test Suite")
        self.logger.info("=" * 60)
        
        # Core rule testing
        self.test_granularity_rules()
        self.test_scope_rules()
        
        # Path validation testing
        self.test_valid_path_validation()
        self.test_invalid_path_validation()
        self.test_constraint_validation()
        
        # Advanced features testing
        self.test_context_sufficiency_assessment()
        self.test_path_template_generation()
        self.test_traversal_planning()
        self.test_multi_dimensional_validation()
        
        # Real data testing
        self.test_real_knowledge_graph()
        
        # Final results
        self.print_test_results()
    
    def print_test_results(self):
        """Print comprehensive test results."""
        self.logger.info("=" * 60)
        self.logger.info("🏁 Test Results Summary")
        self.logger.info("=" * 60)
        
        results = self.test_results
        total_tests = results["tests_run"]
        passed = results["tests_passed"]
        failed = results["tests_failed"]
        
        success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
        
        self.logger.info(f"Total Tests: {total_tests}")
        self.logger.info(f"Passed: {passed} ✅")
        self.logger.info(f"Failed: {failed} ❌")
        self.logger.info(f"Success Rate: {success_rate:.1f}%")
        
        if failed > 0:
            self.logger.error("\nFailed Tests:")
            for i, failure in enumerate(results["failures"], 1):
                self.logger.error(f"{i}. {failure}")
        
        if success_rate >= 90:
            self.logger.info("\n🎉 Excellent! Traversal rules engine is working well.")
        elif success_rate >= 75:
            self.logger.info("\n👍 Good! Minor issues to address.")
        else:
            self.logger.error("\n⚠️  Issues detected. Review failed tests.")
        
        return success_rate >= 90


def main():
    """Main entry point for test suite."""
    print("🧪 Semantic RAG Traversal Rules Test Suite")
    print("=" * 50)
    
    # Run comprehensive test suite
    test_suite = TraversalRulesTestSuite()
    success = test_suite.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
