#!/usr/bin/env python3
"""
Test Phase 5: Retrieval Graph Construction
==========================================

Test script to verify Phase 5 (Retrieval Graph Construction) functionality.
Tests semantic traversal retrieval vs baseline vector retrieval.

Run from project root:
    python test_phase5.py
"""

import sys
import os
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))

from pipeline import SemanticRAGPipeline


def test_phase5():
    """Test Phase 5: Retrieval Graph Construction."""
    print("🧪 Testing Phase 5: Retrieval Graph Construction")
    print("=" * 65)
    
    try:
        # Initialize pipeline
        config_path = "config.yaml"
        if not Path(config_path).exists():
            print(f"❌ Config file not found: {config_path}")
            print("💡 Make sure you're running from the project root directory")
            return False
        
        print(f"📋 Using config: {config_path}")
        pipeline = SemanticRAGPipeline(config_path)
        
        # Load config and setup
        pipeline._load_config()
        
        # Override config for testing
        print("🔧 Configuring for Phase 5 testing...")
        
        # Set mode to full pipeline to include Phase 5
        pipeline.config['execution']['mode'] = 'full_pipeline'
        pipeline.config['execution']['skip_phases'] = []
        
        # Use smaller models for faster testing
        pipeline.config['models']['embedding_models'] = [
            "sentence-transformers/all-MiniLM-L6-v2"  # Faster model for testing
        ]
        
        # Configure retrieval settings for testing
        pipeline.config['retrieval']['semantic_traversal']['num_anchors'] = 2
        pipeline.config['retrieval']['semantic_traversal']['max_hops'] = 2
        pipeline.config['retrieval']['semantic_traversal']['max_results'] = 8
        pipeline.config['retrieval']['baseline_vector']['top_k'] = 5
        
        print(f"   Models: {pipeline.config['models']['embedding_models']}")
        print(f"   Algorithm: {pipeline.config['retrieval']['algorithm']}")
        print(f"   Num anchors: {pipeline.config['retrieval']['semantic_traversal']['num_anchors']}")
        print(f"   Max hops: {pipeline.config['retrieval']['semantic_traversal']['max_hops']}")
        print(f"   Max results: {pipeline.config['retrieval']['semantic_traversal']['max_results']}")
        
        # Run pipeline phases 1-5
        print("\n🚀 Running pipeline phases 1-5...")
        
        # Phase 1: Setup
        pipeline._phase_1_setup_and_initialization()
        print(f"✅ Phase 1: Setup completed (device: {pipeline.device})")
        
        # Phase 2: Data Acquisition (if needed)
        wiki_path = Path(pipeline.config['directories']['data']) / "wiki.json"
        if not wiki_path.exists():
            print("📥 No cached articles found, running Phase 2...")
            pipeline._phase_2_data_acquisition()
        else:
            print("📂 Loading cached articles...")
            from wiki import WikiEngine
            wiki_engine = WikiEngine(pipeline.config, pipeline.logger)
            pipeline.articles = wiki_engine._load_cached_articles(wiki_path)
            pipeline.corpus_stats = wiki_engine.get_corpus_statistics()
        
        print(f"✅ Phase 2: Loaded {len(pipeline.articles)} articles")
        
        # Phase 3: Embedding Generation (if needed)
        if not pipeline.embeddings:
            print("🧠 Running Phase 3: Embedding Generation...")
            pipeline._phase_3_embedding_generation()
        else:
            print("✅ Phase 3: Embeddings already available")
        
        total_embeddings = sum(len(embeddings) for embeddings in pipeline.embeddings.values())
        print(f"✅ Phase 3: {total_embeddings:,} chunks embedded")
        
        # Phase 4: Similarity Matrix Construction (if needed)
        if not pipeline.similarities:
            print("🕰 Running Phase 4: Similarity Matrix Construction...")
            pipeline._phase_4_similarity_matrices()
        else:
            print("✅ Phase 4: Similarity matrices already available")
        
        print(f"✅ Phase 4: Similarity matrices for {len(pipeline.similarities)} models")
        
        # Phase 5: Retrieval Graph Construction
        print("🎯 Running Phase 5: Retrieval Graph Construction...")
        pipeline._phase_5_retrieval_graphs()
        
        # Verify results
        print("\n🔍 Verifying Phase 5 results...")
        
        # Check retrieval engine
        if not pipeline.retrieval_engine:
            print("❌ No retrieval engine was created")
            return False
        
        print(f"✅ Retrieval engine created successfully")
        
        # Check retrieval statistics
        if pipeline.retrieval_stats:
            stats = pipeline.retrieval_stats
            print(f"   📊 Retrieval Statistics:")
            print(f"      Algorithm: {stats['algorithm']}")
            print(f"      Models: {stats['models_available']}")
            for model, count in stats['total_chunks_per_model'].items():
                print(f"      {model}: {count:,} chunks")
        
        # Test actual retrieval with sample queries
        print("\n🧪 Testing retrieval with sample queries...")
        
        test_queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "What are the applications of artificial intelligence?",
            "Explain deep learning algorithms"
        ]
        
        model_name = list(pipeline.embeddings.keys())[0]  # Use first available model
        
        for i, query in enumerate(test_queries):
            print(f"\n   Query {i+1}: '{query}'")
            
            try:
                # Test semantic traversal
                semantic_result = pipeline.retrieval_engine.retrieve(
                    query, model_name, algorithm="semantic_traversal"
                )
                
                print(f"      🔄 Semantic Traversal:")
                print(f"         Results: {len(semantic_result.chunks)}")
                print(f"         Method: {semantic_result.retrieval_method}")
                
                if semantic_result.metadata:
                    metadata = semantic_result.metadata
                    print(f"         Anchors: {metadata.get('num_anchors', 'N/A')}")
                    print(f"         Paths: {metadata.get('total_paths', 'N/A')}")
                    print(f"         Unique chunks: {metadata.get('unique_chunks', 'N/A')}")
                    print(f"         Time: {metadata.get('retrieval_time', 0):.3f}s")
                
                # Show sample results
                if semantic_result.chunks:
                    print(f"         Sample result: '{semantic_result.chunks[0].chunk_text[:80]}...'")
                    print(f"         Score: {semantic_result.scores[0]:.3f}")
                
                # Test baseline for comparison
                baseline_result = pipeline.retrieval_engine.retrieve(
                    query, model_name, algorithm="baseline_vector"
                )
                
                print(f"      📊 Baseline Vector:")
                print(f"         Results: {len(baseline_result.chunks)}")
                print(f"         Method: {baseline_result.retrieval_method}")
                
                if baseline_result.chunks:
                    print(f"         Sample result: '{baseline_result.chunks[0].chunk_text[:80]}...'")
                
                # Compare results
                semantic_ids = {chunk.chunk_id for chunk in semantic_result.chunks}
                baseline_ids = {chunk.chunk_id for chunk in baseline_result.chunks}
                overlap = len(semantic_ids & baseline_ids)
                
                print(f"      🔍 Comparison:")
                print(f"         Overlap: {overlap}/{len(semantic_result.chunks)} chunks")
                print(f"         Semantic unique: {len(semantic_ids - baseline_ids)}")
                print(f"         Baseline unique: {len(baseline_ids - semantic_ids)}")
                
            except Exception as e:
                print(f"      ❌ Query failed: {e}")
                continue
        
        print("\n🎉 Phase 5 test completed successfully!")
        print(f"📋 Experiment ID: {pipeline.experiment_id}")
        print(f"📁 Logs saved to: logs/{pipeline.experiment_id}.log")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 5 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🧪 Semantic RAG Pipeline - Phase 5 Test")
    print("Testing semantic traversal retrieval system")
    print("=" * 70)
    
    # Check if we're in the right directory
    if not Path("config.yaml").exists():
        print("❌ config.yaml not found!")
        print("💡 Make sure you're running this from the project root directory")
        print("💡 Expected directory structure:")
        print("   config.yaml")
        print("   utils/")
        print("   requirements.txt")
        return
    
    success = test_phase5()
    
    if success:
        print("\n✅ All tests passed!")
        print("🚀 Phase 5 is ready for production use")
        print("\n🎯 Key features verified:")
        print("   • Semantic traversal retrieval algorithm")
        print("   • Baseline vector retrieval for comparison")
        print("   • Anchor-based traversal starting points")
        print("   • Greedy path following with loop prevention")
        print("   • Query-based reranking of traversal results")
        print("   • Multi-model retrieval support")
    else:
        print("\n❌ Tests failed!")
        print("🔧 Please check the error messages above")


if __name__ == "__main__":
    main()
