#!/usr/bin/env python3
"""
Test Script for Phases 1-6
===========================

Simple test script to verify the simplified architecture works correctly.
Tests data acquisition through knowledge graph construction.
"""

import sys
import traceback
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))

from pipeline import SemanticRAGPipeline


def test_phases_1_6():
    """Test phases 1-6 of the simplified pipeline."""
    print("🧪 Testing Simplified Semantic RAG Pipeline (Phases 1-6)")
    print("=" * 60)

    try:
        # Initialize pipeline
        print("🚀 Initializing pipeline...")
        pipeline = SemanticRAGPipeline()

        # Phase 1: Setup
        print("\n📋 Phase 1: Setup & Initialization")
        pipeline._phase_1_setup_and_initialization()
        print(f"✅ Experiment ID: {pipeline.experiment_id}")

        # Phase 2: Data Acquisition
        print("\n🌐 Phase 2: Data Acquisition")
        pipeline._phase_2_data_acquisition()
        print(f"✅ Articles acquired: {len(pipeline.articles)}")
        if pipeline.corpus_stats:
            print(f"   Total sentences: {pipeline.corpus_stats.get('total_sentences', 0):,}")
            print(f"   Total words: {pipeline.corpus_stats.get('total_words', 0):,}")

        # Phase 3: Embedding Generation
        print("\n🧠 Phase 3: Multi-Granularity Embedding Generation")
        pipeline._phase_3_embedding_generation()
        print(f"✅ Models processed: {len(pipeline.embeddings)}")
        for model_name, granularity_embeddings in pipeline.embeddings.items():
            print(f"   {model_name}:")
            for granularity_type, embeddings in granularity_embeddings.items():
                print(f"      {granularity_type}: {len(embeddings)} embeddings")

        # Phase 4: Simplified Similarities
        print("\n🔗 Phase 4: Simplified Similarity Matrices")
        pipeline._phase_4_similarity_matrices()
        print(f"✅ Similarity matrices computed for {len(pipeline.similarities)} models")
        for model_name, similarity_data in pipeline.similarities.items():
            metadata = similarity_data['metadata']
            print(f"   {model_name}:")
            print(f"      Total connections: {metadata.total_connections:,}")
            print(f"      Intra-document: {metadata.intra_doc_connections:,}")
            print(f"      Inter-document: {metadata.inter_doc_connections:,}")

        # Phase 5: Theme Extraction
        print("\n🎨 Phase 5: Theme Extraction")
        pipeline._phase_5_theme_extraction()
        print(f"✅ Theme extraction completed")
        if pipeline.theme_data and 'metadata' in pipeline.theme_data:
            metadata = pipeline.theme_data['metadata']
            print(f"   Total themes: {metadata.total_themes_extracted}")
            print(f"   Documents processed: {metadata.document_count}")
            print(f"   Method: {'Ollama' if metadata.ollama_available else 'Fallback'}")

        # Phase 6: Simplified Knowledge Graph
        print("\n🏗️  Phase 6: Simplified Knowledge Graph Construction")
        pipeline._phase_6_knowledge_graph_construction()
        print(f"✅ Knowledge graph constructed")
        if pipeline.knowledge_graph and pipeline.knowledge_graph.metadata:
            metadata = pipeline.knowledge_graph.metadata
            print(f"   Architecture: {metadata.get('architecture', 'unknown')}")
            print(f"   Documents: {metadata.get('total_documents', 0)}")
            print(f"   Chunks: {metadata.get('total_chunks', 0)}")
            print(f"   Sentences: {metadata.get('total_sentences', 0)}")
            print(f"   Chunk connections: {metadata.get('total_chunk_connections', 0)}")

        print("\n🎉 All phases completed successfully!")
        print("=" * 60)

        # Test knowledge graph functionality
        print("\n🔍 Testing Knowledge Graph Functionality:")
        kg = pipeline.knowledge_graph

        if kg.chunks:
            # Test getting a chunk and its connections
            first_chunk_id = list(kg.chunks.keys())[0]
            first_chunk = kg.chunks[first_chunk_id]
            connections = kg.get_chunk_connections(first_chunk_id)

            print(f"   Sample chunk: {first_chunk_id}")
            print(f"   Chunk text preview: {first_chunk.chunk_text[:100]}...")
            print(f"   Inherited themes: {first_chunk.inherited_themes}")
            print(f"   Total connections: {len(connections)}")
            print(f"   Intra-doc connections: {len(first_chunk.intra_doc_connections)}")
            print(f"   Inter-doc connections: {len(first_chunk.inter_doc_connections)}")

            # Test embedding retrieval
            embedding = kg.get_chunk_embedding(first_chunk_id)
            if embedding is not None:
                print(f"   Embedding shape: {embedding.shape}")
                print(f"   Embedding norm: {float(embedding.dot(embedding) ** 0.5):.3f}")

        if kg.sentences:
            # Test sentence functionality
            first_sentence_id = list(kg.sentences.keys())[0]
            first_sentence = kg.sentences[first_sentence_id]

            print(f"   Sample sentence: {first_sentence_id}")
            print(f"   Sentence text: {first_sentence.sentence_text}")
            print(f"   Source chunk: {first_sentence.source_chunk}")

            # Test sentence embedding
            sent_embedding = kg.get_sentence_embedding(first_sentence_id)
            if sent_embedding is not None:
                print(f"   Sentence embedding shape: {sent_embedding.shape}")

        return True

    except Exception as e:
        print(f"\n❌ Pipeline failed at phase: {e}")
        print(f"Error details: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False


def test_knowledge_graph_speed(pipeline):
    """Test the speed of embedding lookups in the knowledge graph."""
    print("\n⚡ Testing Knowledge Graph Speed:")

    try:
        if not hasattr(pipeline, 'knowledge_graph') or not pipeline.knowledge_graph:
            print("   ⚠️  No knowledge graph available for speed test")
            return

        kg = pipeline.knowledge_graph

        import time
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        if not kg.chunks:
            print("   ⚠️  No chunks available for speed test")
            return

        # Debug: Check embedding cache status
        print(f"   🔍 Debug - Embedding cache models: {list(kg._embedding_cache.keys())}")
        for model_name, model_cache in kg._embedding_cache.items():
            print(f"   🔍 Debug - {model_name}: {len(model_cache)} cached embeddings")

        # Get a sample of chunk embeddings
        chunk_ids = list(kg.chunks.keys())[:10]  # Test with 10 chunks
        embeddings = []

        print(f"   🔍 Debug - Testing {len(chunk_ids)} chunk IDs")

        start_time = time.time()
        for i, chunk_id in enumerate(chunk_ids):
            chunk = kg.chunks.get(chunk_id)
            if chunk:
                print(f"   🔍 Debug - Chunk {i}: {chunk_id}")
                print(f"   🔍 Debug - Embedding ref: {chunk.embedding_ref}")

            embedding = kg.get_chunk_embedding(chunk_id)
            if embedding is not None:
                embeddings.append(embedding)
                print(f"   ✅ Debug - Embedding found for chunk {i}: shape {embedding.shape}")
            else:
                print(f"   ❌ Debug - No embedding found for chunk {i}: {chunk_id}")

            # Only debug first 3 to avoid spam
            if i >= 2:
                break

        lookup_time = time.time() - start_time

        if len(embeddings) < 2:
            print(
                f"   ⚠️  Not enough embeddings for speed test (found {len(embeddings)} out of {len(chunk_ids)} tested)")
            print(f"   🔍 Debug - First few chunks and their embedding refs:")
            for chunk_id in chunk_ids[:3]:
                chunk = kg.chunks.get(chunk_id)
                if chunk:
                    print(f"      {chunk_id}: {chunk.embedding_ref}")
            return

        # Test batch cosine similarity computation
        embeddings_matrix = np.array(embeddings)
        query_embedding = embeddings[0]  # Use first as query

        start_time = time.time()
        similarities = cosine_similarity([query_embedding], embeddings_matrix)[0]
        similarity_time = time.time() - start_time

        print(f"   ✅ Embedding lookup for {len(chunk_ids)} chunks: {lookup_time * 1000:.2f}ms")
        print(f"   ✅ Batch similarity computation: {similarity_time * 1000:.2f}ms")
        print(f"   ✅ Total time for comparison: {(lookup_time + similarity_time) * 1000:.2f}ms")
        print(f"   📊 Sample similarities: {similarities[:5]}")

    except Exception as e:
        print(f"   ❌ Speed test failed: {e}")


if __name__ == "__main__":
    success = test_phases_1_6()

    if success:
        # Create pipeline instance and run phases 1-6 to have knowledge graph available
        pipeline = SemanticRAGPipeline()
        pipeline._phase_1_setup_and_initialization()
        pipeline._phase_2_data_acquisition()
        pipeline._phase_3_embedding_generation()
        pipeline._phase_4_similarity_matrices()
        pipeline._phase_5_theme_extraction()
        pipeline._phase_6_knowledge_graph_construction()

        # Now test speed with the built pipeline
        test_knowledge_graph_speed(pipeline)
        print("\n🎯 Simplified architecture test completed successfully!")
    else:
        print("\n💥 Test failed - check the errors above")
        sys.exit(1)