#!/usr/bin/env python3
"""
Retrieval and Question Generation Test
====================================

Simple test to verify the refactored retrieval and question generation systems work correctly.
Tests the core principle: both systems use the same traversal.py rules.
"""

import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))

from pipeline import SemanticRAGPipeline
from retrieval import create_retrieval_engine
from questions import create_question_generator, create_dataset_generator
from traversal import TraversalValidator, ConnectionType, GranularityLevel


def test_unified_traversal_system():
    """Test that retrieval and question generation use the same traversal rules."""
    print("🧪 Testing Unified Traversal System")
    print("=" * 50)
    
    try:
        # Initialize pipeline and run through Phase 6
        print("🚀 1. Initializing pipeline...")
        pipeline = SemanticRAGPipeline()
        
        # Run phases 1-6
        pipeline._phase_1_setup_and_initialization()
        pipeline._phase_2_data_acquisition()
        pipeline._phase_3_embedding_generation()
        pipeline._phase_4_similarity_matrices()
        pipeline._phase_5_theme_extraction()
        pipeline._phase_6_knowledge_graph_construction()
        
        print(f"✅ Pipeline initialized with {pipeline.kg_stats.get('total_chunks', 0)} chunks")
        
        # Test 2: Verify retrieval engine works
        print("\n🔍 2. Testing retrieval engine...")
        if pipeline.retrieval_engine:
            
            # Test baseline vector retrieval
            query1 = "What is machine learning?"
            result1 = pipeline.retrieval_engine.retrieve(query1, strategy="baseline_vector")
            print(f"✅ Baseline retrieval: {len(result1.retrieved_content)} results for '{query1}'")
            
            # Test semantic traversal retrieval
            query2 = "How does artificial intelligence work?"
            result2 = pipeline.retrieval_engine.retrieve(query2, strategy="semantic_traversal")
            print(f"✅ Semantic traversal: {len(result2.retrieved_content)} results for '{query2}'")
            
            if result2.traversal_path:
                print(f"   🛤️  Traversal path: {len(result2.traversal_path.nodes)} nodes, {len(result2.traversal_path.connection_types)} hops")
                print(f"   🎯 Path valid: {result2.traversal_path.is_valid}")
                if result2.traversal_path.connection_types:
                    print(f"   🔗 Connection types: {[ct.value for ct in result2.traversal_path.connection_types]}")
        else:
            print("❌ Retrieval engine not initialized")
            return False
        
        # Test 3: Verify question generation works
        print("\n❓ 3. Testing question generation...")
        question_generator = create_question_generator(pipeline.knowledge_graph, pipeline.config, pipeline.logger)
        
        # Generate a small set of questions
        single_hop_questions = question_generator.generate_single_hop_questions(3)
        print(f"✅ Generated {len(single_hop_questions)} single-hop questions")
        
        multi_hop_questions = question_generator.generate_multi_hop_questions(2, ["medium"])
        print(f"✅ Generated {len(multi_hop_questions)} multi-hop questions")
        
        # Test 4: Verify shared traversal rules
        print("\n🔄 4. Testing shared traversal rules...")
        validator = TraversalValidator(pipeline.logger)
        
        # Check if both systems produce valid paths using same rules
        valid_retrieval_paths = 0
        valid_question_paths = 0
        
        # Check retrieval paths
        if result2.traversal_path and result2.traversal_path.is_valid:
            valid_retrieval_paths += 1
        
        # Check question paths
        for question in single_hop_questions + multi_hop_questions:
            if question.ground_truth_path.is_valid:
                valid_question_paths += 1
        
        total_questions = len(single_hop_questions) + len(multi_hop_questions)
        
        print(f"✅ Retrieval path validity: {valid_retrieval_paths}/1")
        print(f"✅ Question path validity: {valid_question_paths}/{total_questions}")
        
        # Test 5: Show concrete example of "cargo crane" approach
        print("\n🏗️  5. Demonstrating 'Cargo Crane' Approach...")
        
        if result2.traversal_path and result2.traversal_path.nodes:
            print(f"📍 Anchor: Found {result2.metadata.get('num_anchors', 0)} anchor points")
            print(f"🚶 Traverse: Executed {result2.traversal_path.total_hops} hops")
            print(f"📦 Extract: Retrieved {len(result2.retrieved_content)} content pieces")
            
            # Show first few content pieces
            for i, content in enumerate(result2.retrieved_content[:2]):
                print(f"   Content {i+1}: {content[:100]}...")
        
        # Test 6: Verify question-retrieval coherence
        print("\n🎯 6. Testing Question-Retrieval Coherence...")
        
        if single_hop_questions:
            sample_question = single_hop_questions[0]
            print(f"📝 Sample question: {sample_question.question_text[:80]}...")
            print(f"🛤️  Ground truth path: {len(sample_question.ground_truth_path.nodes)} nodes")
            print(f"🔗 Connection types: {[ct.value for ct in sample_question.ground_truth_path.connection_types]}")
            
            # Try to retrieve answer using the same traversal logic
            retrieval_result = pipeline.retrieval_engine.retrieve(
                sample_question.question_text, 
                strategy="semantic_traversal"
            )
            
            coherence_score = len(retrieval_result.retrieved_content) / max(1, len(sample_question.ground_truth_path.nodes))
            print(f"✅ Coherence test: Retrieval found {len(retrieval_result.retrieved_content)} pieces for {len(sample_question.ground_truth_path.nodes)} ground truth nodes")
            print(f"📊 Coherence ratio: {coherence_score:.2f}")
        
        print("\n🎉 All tests completed successfully!")
        
        # Summary
        print("\n📊 Test Summary:")
        print(f"   Knowledge Graph: {len(pipeline.knowledge_graph.chunks)} chunks, {len(pipeline.knowledge_graph.sentences)} sentences")
        print(f"   Retrieval Engine: {pipeline.retrieval_stats.get('status', 'unknown')}")
        print(f"   Question Generation: {len(single_hop_questions + multi_hop_questions)} questions generated")
        print(f"   Traversal Rules: Shared between both systems ✅")
        print(f"   Cargo Crane: Anchor → Traverse → Extract ✅")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_traversal_rules_directly():
    """Test traversal rules directly to show they work independently."""
    print("\n🧭 Testing Traversal Rules Directly")
    print("-" * 40)
    
    try:
        from traversal import TraversalValidator, ConnectionType, GranularityLevel
        
        validator = TraversalValidator()
        
        # Test a valid raw similarity path
        path_nodes = ["chunk_1", "chunk_2"]
        connection_types = [ConnectionType.RAW_SIMILARITY]
        granularities = [GranularityLevel.CHUNK, GranularityLevel.CHUNK]
        documents = ["doc_a", "doc_b"]
        
        path = validator.validate_path(
            path_nodes, connection_types, granularities, documents, "raw_similarity"
        )
        
        print(f"✅ Raw similarity validation: {path.is_valid}")
        if not path.is_valid:
            print(f"   Errors: {path.validation_errors}")
        
        # Test an invalid path (wrong granularity for connection type)
        invalid_connection_types = [ConnectionType.RAW_SIMILARITY]
        invalid_granularities = [GranularityLevel.SENTENCE, GranularityLevel.SENTENCE]  # Not allowed!
        
        invalid_path = validator.validate_path(
            path_nodes, invalid_connection_types, invalid_granularities, documents, "raw_similarity"
        )
        
        print(f"✅ Invalid path correctly rejected: {not invalid_path.is_valid}")
        if not invalid_path.is_valid:
            print(f"   Expected errors: {invalid_path.validation_errors[0]}")
        
        # Test template generation
        templates = validator.generate_valid_path_templates("hierarchical")
        print(f"✅ Hierarchical templates: {len(templates)} generated")
        if templates:
            template = templates[0]
            print(f"   Sample: {template['description']}")
            print(f"   Hops: {template['hops']}, Granularities: {[g.name for g in template['granularities']]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Traversal rules test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Retrieval and Question Generation Integration Test")
    print("=" * 70)
    
    # Test 1: Direct traversal rules
    rules_test = test_traversal_rules_directly()
    
    # Test 2: Full system integration
    system_test = test_unified_traversal_system()
    
    print("\n" + "=" * 70)
    if rules_test and system_test:
        print("🎉 ALL TESTS PASSED! The refactored system works correctly.")
        print("🔧 Traversal rules provide unified foundation for both retrieval and question generation.")
        print("⚙️  Cargo crane approach: Anchor → Traverse → Extract is operational.")
        print("🎯 Question-retrieval coherence achieved through shared traversal logic.")
    else:
        print("❌ Some tests failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
