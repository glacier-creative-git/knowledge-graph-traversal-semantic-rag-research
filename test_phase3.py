#!/usr/bin/env python3
"""
Test Phase 3: Embedding Generation
=================================

Test script to verify Phase 3 (Embedding Generation) functionality.
Tests chunking and embedding generation with multiple models.

Run from project root:
    python test_phase3.py
"""

import sys
import os
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))

from utils.pipeline import SemanticRAGPipeline


def test_phase3():
    """Test Phase 3: Embedding Generation."""
    print("🧪 Testing Phase 3: Embedding Generation")
    print("=" * 50)
    
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
        print("🔧 Configuring for Phase 3 testing...")
        
        # Set mode to only run through Phase 3
        pipeline.config['execution']['mode'] = 'embedding_only'
        pipeline.config['execution']['skip_phases'] = []
        
        # Use smaller models and batch sizes for testing
        original_models = pipeline.config['models']['embedding_models']
        pipeline.config['models']['embedding_models'] = [
            "sentence-transformers/all-MiniLM-L6-v2"  # Smaller, faster model for testing
        ]
        pipeline.config['models']['embedding_batch_size'] = 16  # Smaller batch for testing
        
        # Use smaller chunking window for testing
        pipeline.config['chunking']['window_size'] = 3
        pipeline.config['chunking']['overlap'] = 1
        
        # Enable caching but allow fresh computation
        pipeline.config['wikipedia']['use_cached_articles'] = True
        
        print(f"   Models: {pipeline.config['models']['embedding_models']}")
        print(f"   Batch size: {pipeline.config['models']['embedding_batch_size']}")
        print(f"   Chunking: {pipeline.config['chunking']['strategy']} (window={pipeline.config['chunking']['window_size']}, overlap={pipeline.config['chunking']['overlap']})")
        
        # Run pipeline phases 1-3
        print("\n🚀 Running pipeline phases 1-3...")
        
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
            from utils.wiki import WikiEngine
            wiki_engine = WikiEngine(pipeline.config, pipeline.logger)
            pipeline.articles = wiki_engine._load_cached_articles(wiki_path)
            pipeline.corpus_stats = wiki_engine.get_corpus_statistics()
        
        print(f"✅ Phase 2: Loaded {len(pipeline.articles)} articles")
        
        # Phase 3: Embedding Generation
        print("🧠 Running Phase 3: Embedding Generation...")
        pipeline._phase_3_embedding_generation()
        
        # Verify results
        print("\n🔍 Verifying Phase 3 results...")
        
        # Check chunks
        if not pipeline.chunks:
            print("❌ No chunks were created")
            return False
        
        print(f"✅ Created {len(pipeline.chunks)} chunks")
        
        # Check chunk statistics
        if pipeline.chunk_stats:
            stats = pipeline.chunk_stats
            print(f"   📊 Chunk stats: {stats['total_chunks']} chunks from {stats['total_articles']} articles")
            print(f"   📏 Avg chunk length: {stats['chunk_length_stats']['mean_words']:.1f} words")
            print(f"   📝 Avg sentences per chunk: {stats['sentence_count_stats']['mean_sentences']:.1f}")
        
        # Check embeddings
        if not pipeline.embeddings:
            print("❌ No embeddings were generated")
            return False
        
        for model_name, embeddings in pipeline.embeddings.items():
            print(f"✅ Generated {len(embeddings)} embeddings for {model_name}")
            
            if embeddings:
                # Verify embedding properties
                first_embedding = embeddings[0]
                print(f"   🔢 Embedding dimension: {len(first_embedding.embedding)}")
                print(f"   📄 Sample chunk: '{first_embedding.chunk_text[:100]}...'")
                print(f"   🏷️  Sample chunk ID: {first_embedding.chunk_id}")
                print(f"   📚 Source article: {first_embedding.source_article}")
                print(f"   🎯 Anchor sentence: {first_embedding.anchor_sentence_idx}")
        
        # Check embedding statistics
        if pipeline.embedding_stats:
            print(f"   📈 Embedding statistics available for {len(pipeline.embedding_stats)} models")
            for model_name, stats in pipeline.embedding_stats.items():
                print(f"      {model_name}: {stats['total_chunks']} chunks, {stats['embedding_dimension']}D")
        
        # Test specific chunking behavior
        print("\n🔍 Testing chunking behavior...")
        if len(pipeline.chunks) >= 3:
            # Show first few chunks to verify sliding window
            for i, chunk in enumerate(pipeline.chunks[:3]):
                print(f"   Chunk {i+1}: sentences {chunk['source_sentences']} (anchor: {chunk['anchor_sentence_idx']})")
                print(f"      Text: '{chunk['text'][:80]}...'")
        
        print("\n🎉 Phase 3 test completed successfully!")
        print(f"📋 Experiment ID: {pipeline.experiment_id}")
        print(f"📁 Logs saved to: logs/{pipeline.experiment_id}.log")
        print(f"💾 Embeddings cached in: embeddings/raw/")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 3 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🧪 Semantic RAG Pipeline - Phase 3 Test")
    print("Testing embedding generation and chunking")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("config.yaml").exists():
        print("❌ config.yaml not found!")
        print("💡 Make sure you're running this from the project root directory")
        print("💡 Expected directory structure:")
        print("   config.yaml")
        print("   utils/")
        print("   requirements.txt")
        return
    
    success = test_phase3()
    
    if success:
        print("\n✅ All tests passed!")
        print("🚀 Phase 3 is ready for production use")
    else:
        print("\n❌ Tests failed!")
        print("🔧 Please check the error messages above")


if __name__ == "__main__":
    main()
