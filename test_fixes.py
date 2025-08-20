#!/usr/bin/env python3
"""
Quick Test Script for Chunk ID Determinism Fix
==============================================

Test script to verify that chunk IDs are now deterministic 
and that Phase 5 caching works properly.
"""

import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))

from chunking import ChunkEngine
from wiki import WikipediaArticle, ArticleMetadata


def test_chunk_id_determinism():
    """Test that chunk IDs are deterministic across multiple runs."""
    print("🧪 Testing Chunk ID Determinism Fix")
    print("=" * 50)
    
    # Create a simple test article
    metadata = ArticleMetadata(
        title="Test Article",
        url="test://url", 
        word_count=100,
        char_count=500,
        sentence_count=5,
        scraped_at="2024-01-01T00:00:00",
        language="en",
        processing_time=1.0,
        source_hash="test_hash"
    )
    
    test_article = WikipediaArticle(
        title="Test Article",
        url="test://url",
        raw_text="Test raw text",
        cleaned_text="Test cleaned text", 
        sentences=[
            "This is the first sentence.",
            "This is the second sentence.",
            "This is the third sentence.",
            "This is the fourth sentence.",
            "This is the fifth sentence."
        ],
        metadata=metadata
    )
    
    # Create chunk engine with test config
    config = {
        'chunking': {
            'strategy': 'sliding_window',
            'window_size': 3,
            'overlap': 1
        }
    }
    
    chunk_engine = ChunkEngine(config)
    
    # Generate chunks twice
    print("🔄 Generating chunks - Run 1...")
    chunks_run1 = chunk_engine.create_chunks([test_article])
    
    print("🔄 Generating chunks - Run 2...")  
    chunks_run2 = chunk_engine.create_chunks([test_article])
    
    # Check if chunk IDs are identical
    print(f"\n📊 Results:")
    print(f"   Run 1: {len(chunks_run1)} chunks")
    print(f"   Run 2: {len(chunks_run2)} chunks")
    
    if len(chunks_run1) != len(chunks_run2):
        print("❌ Different number of chunks generated")
        return False
    
    ids_match = True
    for i, (chunk1, chunk2) in enumerate(zip(chunks_run1, chunks_run2)):
        chunk1_id = chunk1['chunk_id'] 
        chunk2_id = chunk2['chunk_id']
        
        print(f"   Chunk {i+1}: {chunk1_id}")
        
        if chunk1_id != chunk2_id:
            print(f"      ❌ Mismatch: {chunk2_id}")
            ids_match = False
        else:
            print(f"      ✅ Deterministic")
    
    if ids_match:
        print(f"\n✅ SUCCESS: All chunk IDs are deterministic!")
        print(f"🔧 This should fix both Phase 5 caching and Phase 6 similarity relationships")
        return True
    else:
        print(f"\n❌ FAILURE: Chunk IDs are still non-deterministic")
        return False


def main():
    """Main test function."""
    print("🔧 Testing Semantic RAG Pipeline Fixes")
    print("Testing chunk ID determinism fix")
    print("=" * 60)
    
    success = test_chunk_id_determinism()
    
    if success:
        print("\n🎉 Chunk ID fix is working!")
        print("📋 Next steps:")
        print("   1. Run test_phase6.py to see if similarity relationships are now created")
        print("   2. Check if Phase 5 caching works (should see 'Loading cached' messages)")
        print("   3. Verify entity relationships are built with debug logging")
    else:
        print("\n❌ Chunk ID fix needs more work")
    
    return success


if __name__ == "__main__":
    main()
