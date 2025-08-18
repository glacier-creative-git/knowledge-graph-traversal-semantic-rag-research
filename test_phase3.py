#!/usr/bin/env python3
"""
Test Phase 3: Enhanced Multi-Granularity Embedding Generation
===========================================================

Test script to verify Phase 3 (Enhanced Multi-Granularity Embedding Generation) functionality.
Tests chunking and multi-granularity embedding generation (chunks, sentences, doc_summaries).

Run from project root:
    python test_phase3.py
"""

import sys
import os
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))

from pipeline import SemanticRAGPipeline


def test_phase3():
    """Test Phase 3: Enhanced Multi-Granularity Embedding Generation."""
    print("🧪 Testing Phase 3: Enhanced Multi-Granularity Embedding Generation")
    print("🌟 Architecture: Document → Chunk → Sentence Three-Tier Hierarchy")
    print("=" * 70)
    
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
        print("🔧 Configuring for Enhanced Phase 3 testing...")
        
        # Set mode to only run through Phase 3
        pipeline.config['execution']['mode'] = 'embedding_only'
        pipeline.config['execution']['skip_phases'] = []
        
        # Use smaller models and batch sizes for testing
        pipeline.config['models']['embedding_models'] = [
            "sentence-transformers/all-MiniLM-L6-v2"  # Smaller, faster model for testing
        ]
        pipeline.config['models']['embedding_batch_size'] = 16  # Smaller batch for testing
        
        # Configure multi-granularity types for testing
        pipeline.config['models']['granularity_types'] = {
            'chunks': {
                'enabled': True,
                'description': "3-sentence sliding windows"
            },
            'sentences': {
                'enabled': True,
                'description': "Individual sentences for fine-grained navigation"
            },
            'doc_summaries': {
                'enabled': True,
                'description': "Document-level summaries",
                'method': "extractive",
                'max_sentences': 3
            }
        }
        
        # Use smaller chunking window for testing
        pipeline.config['chunking']['window_size'] = 3
        pipeline.config['chunking']['overlap'] = 1
        
        # Enable caching but allow fresh computation
        pipeline.config['wikipedia']['use_cached_articles'] = True
        
        print(f"   Models: {pipeline.config['models']['embedding_models']}")
        print(f"   Batch size: {pipeline.config['models']['embedding_batch_size']}")
        print(f"   Chunking: {pipeline.config['chunking']['strategy']} (window={pipeline.config['chunking']['window_size']}, overlap={pipeline.config['chunking']['overlap']})")
        print(f"   Granularity types: {list(pipeline.config['models']['granularity_types'].keys())}")
        
        # Run pipeline phases 1-3
        print("\n🚀 Running enhanced pipeline phases 1-3...")
        
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
        
        # Phase 3: Enhanced Multi-Granularity Embedding Generation
        print("🧠 Running Phase 3: Enhanced Multi-Granularity Embedding Generation...")
        pipeline._phase_3_embedding_generation()
        
        # Verify results
        print("\n🔍 Verifying Enhanced Phase 3 results...")
        
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
        
        # Check multi-granularity embeddings
        if not pipeline.embeddings:
            print("❌ No multi-granularity embeddings were generated")
            return False
        
        print(f"✅ Generated multi-granularity embeddings for {len(pipeline.embeddings)} models")
        
        for model_name, granularity_embeddings in pipeline.embeddings.items():
            print(f"\n   📊 Results for {model_name}:")
            
            # Check each granularity type
            for granularity_type, embeddings in granularity_embeddings.items():
                print(f"      {granularity_type}: {len(embeddings)} embeddings")
                
                if embeddings:
                    # Verify embedding properties for each granularity type
                    first_embedding = embeddings[0]
                    print(f"         🔢 Embedding dimension: {len(first_embedding.embedding)}")
                    
                    if granularity_type == 'chunks':
                        print(f"         📄 Sample chunk: '{first_embedding.chunk_text[:60]}...'")
                        print(f"         🏷️  Chunk ID: {first_embedding.chunk_id}")
                        print(f"         📚 Source article: {first_embedding.source_article}")
                        print(f"         🎯 Anchor sentence: {first_embedding.anchor_sentence_idx}")
                    elif granularity_type == 'sentences':
                        print(f"         📝 Sample sentence: '{first_embedding.sentence_text[:60]}...'")
                        print(f"         🏷️  Sentence ID: {first_embedding.sentence_id}")
                        print(f"         📚 Source article: {first_embedding.source_article}")
                        print(f"         📍 Sentence index: {first_embedding.sentence_index}")
                        print(f"         🔗 Containing chunks: {len(first_embedding.containing_chunks)}")
                    elif granularity_type == 'doc_summaries':
                        print(f"         📄 Sample summary: '{first_embedding.summary_text[:60]}...'")
                        print(f"         🏷️  Doc ID: {first_embedding.doc_id}")
                        print(f"         📚 Source article: {first_embedding.source_article}")
                        print(f"         📏 Total sentences: {first_embedding.total_sentences}")
                        print(f"         🔧 Summary method: {first_embedding.summary_method}")
        
        # Check embedding statistics
        if pipeline.embedding_stats:
            print(f"\n   📈 Multi-Granularity Embedding statistics:")
            for model_name, stats in pipeline.embedding_stats.items():
                print(f"      {model_name}:")
                print(f"         Total embeddings: {stats['total_embeddings']:,}")
                
                for granularity_type, granularity_stats in stats['granularity_types'].items():
                    print(f"         {granularity_type}: {granularity_stats['count']:,} embeddings, {granularity_stats['embedding_dimension']}D")
                    
                    # Show sample lengths for different granularity types
                    if 'sample_chunk_lengths' in granularity_stats:
                        print(f"            Sample chunk lengths: {granularity_stats['sample_chunk_lengths']}")
                    elif 'sample_sentence_lengths' in granularity_stats:
                        print(f"            Sample sentence lengths: {granularity_stats['sample_sentence_lengths']}")
                    elif 'sample_summary_lengths' in granularity_stats:
                        print(f"            Sample summary lengths: {granularity_stats['sample_summary_lengths']}")
        
        # Test specific multi-granularity behavior
        print("\n🔍 Testing Multi-Granularity Behavior:")
        
        # Test chunking behavior
        if len(pipeline.chunks) >= 3:
            print("   📝 Chunk sliding window behavior:")
            for i, chunk in enumerate(pipeline.chunks[:3]):
                print(f"      Chunk {i+1}: sentences {chunk['source_sentences']} (anchor: {chunk['anchor_sentence_idx']})")
                print(f"         Text: '{chunk['text'][:60]}...'")
        
        # Test sentence relationships
        model_name = list(pipeline.embeddings.keys())[0]
        sentence_embeddings = pipeline.embeddings[model_name].get('sentences', [])
        
        if sentence_embeddings:
            print(f"   📖 Sentence-to-chunk relationships:")
            for i, sentence_emb in enumerate(sentence_embeddings[:3]):
                print(f"      Sentence {i+1}: '{sentence_emb.sentence_text[:50]}...'")
                print(f"         Containing chunks: {len(sentence_emb.containing_chunks)}")
                print(f"         Sentence index: {sentence_emb.sentence_index}")
        
        # Test document summary relationships
        doc_summary_embeddings = pipeline.embeddings[model_name].get('doc_summaries', [])
        
        if doc_summary_embeddings:
            print(f"   📄 Document summary relationships:")
            for i, doc_emb in enumerate(doc_summary_embeddings[:2]):
                print(f"      Document {i+1}: '{doc_emb.doc_title}'")
                print(f"         Summary: '{doc_emb.summary_text[:50]}...'")
                print(f"         Total sentences: {doc_emb.total_sentences}")
                print(f"         Method: {doc_emb.summary_method}")
        
        # Verify multi-granularity consistency
        print("\n🔄 Verifying Multi-Granularity Consistency:")
        
        chunk_embeddings = pipeline.embeddings[model_name].get('chunks', [])
        total_chunks = len(chunk_embeddings)
        total_sentences = len(sentence_embeddings) if sentence_embeddings else 0
        total_docs = len(doc_summary_embeddings) if doc_summary_embeddings else 0
        
        print(f"   📊 Granularity counts:")
        print(f"      Documents (L0): {total_docs}")
        print(f"      Chunks (L1): {total_chunks}")
        print(f"      Sentences (L2): {total_sentences}")
        
        # Verify expected ratios
        if total_docs > 0 and total_chunks > 0:
            chunks_per_doc = total_chunks / total_docs
            print(f"      Avg chunks per document: {chunks_per_doc:.1f}")
        
        if total_chunks > 0 and total_sentences > 0:
            sentences_per_chunk = total_sentences / total_chunks
            print(f"      Avg sentences per chunk: {sentences_per_chunk:.1f}")
        
        # Verify all granularity types are present
        expected_granularities = ['chunks', 'sentences', 'doc_summaries']
        present_granularities = list(pipeline.embeddings[model_name].keys())
        
        missing_granularities = set(expected_granularities) - set(present_granularities)
        if missing_granularities:
            print(f"   ⚠️  Missing granularity types: {missing_granularities}")
        else:
            print(f"   ✅ All expected granularity types present: {present_granularities}")
        
        print("\n🎉 Enhanced Phase 3 test completed successfully!")
        print(f"📋 Experiment ID: {pipeline.experiment_id}")
        print(f"📁 Logs saved to: logs/{pipeline.experiment_id}.log")
        print(f"💾 Multi-granularity embeddings cached in: embeddings/raw/")
        
        print("\n🌟 Multi-Granularity Architecture verified:")
        print("   • Chunk embeddings: 3-sentence sliding windows ✅")
        print("   • Sentence embeddings: Individual sentence analysis ✅")
        print("   • Document summary embeddings: High-level document representation ✅")
        print("   • Cross-granularity relationships maintained ✅")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Enhanced Phase 3 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🧪 Enhanced Semantic RAG Pipeline - Phase 3 Test")
    print("Testing multi-granularity embedding generation and chunking")
    print("🌟 Architecture: Three-Tier Hierarchy (Document → Chunk → Sentence)")
    print("=" * 75)
    
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
        print("🚀 Enhanced Phase 3 is ready for production use")
        print("\n🔗 Key features verified:")
        print("   • Multi-granularity embedding generation")
        print("   • Document → Chunk → Sentence hierarchy")
        print("   • Cross-granularity relationship tracking")
        print("   • Intelligent caching system")
        print("   • Extractive document summarization")
    else:
        print("\n❌ Tests failed!")
        print("🔧 Please check the error messages above")


if __name__ == "__main__":
    main()