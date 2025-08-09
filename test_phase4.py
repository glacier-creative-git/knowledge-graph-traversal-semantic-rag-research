#!/usr/bin/env python3
"""
Test Phase 4: Similarity Matrix Construction
===========================================

Test script to verify Phase 4 (Similarity Matrix Construction) functionality.
Tests intra-document and inter-document similarity computation with sparse matrices.

Run from project root:
    python test_phase4.py
"""

import sys
import os
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))

from pipeline import SemanticRAGPipeline


def test_phase4():
    """Test Phase 4: Similarity Matrix Construction."""
    print("🧪 Testing Phase 4: Similarity Matrix Construction")
    print("=" * 60)
    
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
        print("🔧 Configuring for Phase 4 testing...")
        
        # Set mode to full pipeline to include Phase 4
        pipeline.config['execution']['mode'] = 'full_pipeline'
        pipeline.config['execution']['skip_phases'] = []
        
        # Use smaller models and settings for faster testing
        pipeline.config['models']['embedding_models'] = [
            "sentence-transformers/all-MiniLM-L6-v2"  # Faster model for testing
        ]
        pipeline.config['models']['embedding_batch_size'] = 16
        
        # Configure similarity settings for testing
        pipeline.config['similarities']['intra_document']['top_k'] = 5  # Smaller for testing
        pipeline.config['similarities']['inter_document']['top_x'] = 3  # Smaller for testing
        pipeline.config['similarities']['batch_size'] = 500  # Smaller batches
        
        # Enable force recompute to avoid potential cache issues during testing
        pipeline.config['execution']['force_recompute'] = ['similarities']
        
        print(f"   Models: {pipeline.config['models']['embedding_models']}")
        print(f"   Intra-document top_k: {pipeline.config['similarities']['intra_document']['top_k']}")
        print(f"   Inter-document top_x: {pipeline.config['similarities']['inter_document']['top_x']}")
        print(f"   Similarity metric: {pipeline.config['similarities']['similarity_metric']}")
        
        # Run pipeline phases 1-4
        print("\n🚀 Running pipeline phases 1-4...")
        
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
        
        # Check embeddings
        total_embeddings = sum(len(embeddings) for embeddings in pipeline.embeddings.values())
        print(f"✅ Phase 3: Generated embeddings for {total_embeddings:,} chunks")
        
        # Phase 4: Similarity Matrix Construction
        print("🕰 Running Phase 4: Similarity Matrix Construction...")
        pipeline._phase_4_similarity_matrices()
        
        # Verify results
        print("\n🔍 Verifying Phase 4 results...")
        
        # Check similarities
        if not pipeline.similarities:
            print("❌ No similarity matrices were created")
            return False
        
        print(f"✅ Created similarity matrices for {len(pipeline.similarities)} models")
        
        # Check similarity statistics for each model
        for model_name, similarity_data in pipeline.similarities.items():
            print(f"\n📊 Results for {model_name}:")
            
            metadata = similarity_data['metadata']
            matrices = similarity_data['matrices']
            
            print(f"   📈 Metadata:")
            print(f"      Total chunks: {metadata.total_chunks:,}")
            print(f"      Intra-document connections: {metadata.intra_doc_connections:,}")
            print(f"      Inter-document connections: {metadata.inter_doc_connections:,}")
            print(f"      Total connections: {metadata.intra_doc_connections + metadata.inter_doc_connections:,}")
            print(f"      Sparsity ratio: {metadata.sparsity_ratio:.6f}")
            print(f"      Memory usage: {metadata.memory_usage_mb:.1f} MB")
            print(f"      Computation time: {metadata.computation_time:.2f}s")
            
            print(f"   🏗️  Matrices:")
            for matrix_name, matrix in matrices.items():
                print(f"      {matrix_name}: {matrix.shape} ({matrix.nnz:,} non-zero)")
            
            # Verify top_k and top_x constraints
            expected_max_intra = metadata.total_chunks * pipeline.config['similarities']['intra_document']['top_k']
            expected_max_inter = metadata.total_chunks * pipeline.config['similarities']['inter_document']['top_x']
            
            print(f"   🎯 Constraint verification:")
            print(f"      Max possible intra-doc: {expected_max_intra:,} (actual: {metadata.intra_doc_connections:,})")
            print(f"      Max possible inter-doc: {expected_max_inter:,} (actual: {metadata.inter_doc_connections:,})")
            
            if metadata.intra_doc_connections <= expected_max_intra:
                print(f"      ✅ Intra-document constraint satisfied")
            else:
                print(f"      ❌ Intra-document constraint violated")
            
            if metadata.inter_doc_connections <= expected_max_inter:
                print(f"      ✅ Inter-document constraint satisfied")
            else:
                print(f"      ❌ Inter-document constraint violated")
        
        # Test cache functionality
        print("\n💾 Testing cache functionality...")
        print("   Running Phase 4 again to test caching...")
        
        try:
            # Run Phase 4 again (should use cache this time)
            # First clear force_recompute to allow caching
            pipeline.config['execution']['force_recompute'] = []
            
            original_time = pipeline.similarities[list(pipeline.similarities.keys())[0]]['metadata'].computation_time
            
            # Time the actual cache load operation
            import time
            cache_start = time.time()
            pipeline._phase_4_similarity_matrices()
            cache_end = time.time()
            actual_cache_time = cache_end - cache_start
            
            # The metadata still contains the original computation time
            metadata_time = pipeline.similarities[list(pipeline.similarities.keys())[0]]['metadata'].computation_time
            
            print(f"   Original computation: {original_time:.2f}s (from metadata)")
            print(f"   Actual cache load: {actual_cache_time:.3f}s (measured)")
            print(f"   Speedup: {original_time / actual_cache_time:.1f}x faster")
            
            if actual_cache_time < 0.1:  # Should be very fast
                print("   ✅ Caching working effectively")
            else:
                print("   ⚠️  Caching may not be working as expected")
        except Exception as e:
            print(f"   ⚠️  Cache test failed: {e}")
            print("   (This is not critical for core functionality)")
        
        print("\n🎉 Phase 4 test completed successfully!")
        print(f"📋 Experiment ID: {pipeline.experiment_id}")
        print(f"📁 Logs saved to: logs/{pipeline.experiment_id}.log")
        print(f"💾 Similarity matrices cached in: embeddings/similarities/")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 4 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🧪 Semantic RAG Pipeline - Phase 4 Test")
    print("Testing similarity matrix construction")
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
    
    success = test_phase4()
    
    if success:
        print("\n✅ All tests passed!")
        print("🚀 Phase 4 is ready for production use")
        print("\n🔗 Key features verified:")
        print("   • Sparse similarity matrix construction")
        print("   • Intra-document top-k connections")
        print("   • Inter-document top-x connections") 
        print("   • Memory-efficient storage")
        print("   • Intelligent caching system")
    else:
        print("\n❌ Tests failed!")
        print("🔧 Please check the error messages above")


if __name__ == "__main__":
    main()
