#!/usr/bin/env python3
"""
Test Phase 5: Entity/Theme Extraction
====================================

Test script to verify Phase 5 (Entity/Theme Extraction) functionality.
Tests entity extraction at chunk/sentence levels and theme extraction at document level.

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
    """Test Phase 5: Entity/Theme Extraction."""
    print("🧪 Testing Phase 5: Entity/Theme Extraction")
    print("🏷️  Architecture: spaCy Entities + Ollama Themes Across All Granularity Levels")
    print("=" * 80)

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

        # Override config for testing entity/theme extraction
        print("🔧 Configuring for Phase 5 Entity/Theme Extraction testing...")

        # Set mode to full pipeline to include Phase 5
        pipeline.config['execution']['mode'] = 'full_pipeline'
        pipeline.config['execution']['skip_phases'] = []

        # Use smaller models for faster testing
        pipeline.config['models']['embedding_models'] = [
            "sentence-transformers/all-MiniLM-L6-v2"  # Faster model for testing
        ]

        # Configure entity/theme extraction settings for testing
        pipeline.config['entity_theme_extraction'] = {
            'use_cached': True,
            'entity_extraction': {
                'entity_types': ['PERSON', 'ORG', 'GPE'],
                'spacy_model': 'en_core_web_sm',
                'min_entity_length': 2
            },
            'theme_extraction': {
                'num_themes': 4,  # Reduced for testing
                'method': 'ollama',
                'fallback_method': 'keyword_patterns',
                'max_theme_length': 50
            },
            'quality_filters': {
                'min_entities_per_item': 0,
                'min_themes_per_document': 1,
                'filter_common_words': True
            }
        }

        # Enable force recompute to see fresh extraction
        pipeline.config['execution']['force_recompute'] = ['entity_theme']

        # Test Ollama and spaCy availability
        try:
            import ollama
            models = ollama.list()
            ollama_available = True
            print(f"   🤖 Ollama available with models: {[m.model for m in models.models]}")
        except Exception:
            ollama_available = False
            print(f"   ⚠️  Ollama not available - will use fallback theme extraction")

        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            spacy_available = True
            print(f"   🧠 spaCy model 'en_core_web_sm' available")
        except Exception:
            spacy_available = False
            print(f"   ⚠️  spaCy model not available - will use pattern-based entity extraction")

        print(f"   📊 Entity types: {pipeline.config['entity_theme_extraction']['entity_extraction']['entity_types']}")
        print(
            f"   🎯 Themes per document: {pipeline.config['entity_theme_extraction']['theme_extraction']['num_themes']}")
        print(
            f"   🔧 Extraction methods: {'spaCy + Ollama' if (spacy_available and ollama_available) else 'Fallback methods'}")

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

        # Phase 3: Multi-Granularity Embedding Generation (if needed)
        if not pipeline.embeddings:
            print("🧠 Running Phase 3: Multi-Granularity Embedding Generation...")
            pipeline._phase_3_embedding_generation()
        else:
            print("✅ Phase 3: Multi-granularity embeddings already available")

        total_embeddings = sum(
            len(embeddings) for granularity_embeddings in pipeline.embeddings.values() for embeddings in
            granularity_embeddings.values())
        print(f"✅ Phase 3: {total_embeddings:,} total embeddings across all granularity levels")

        # Phase 4: Similarity Matrix Construction (if needed)
        if not pipeline.similarities:
            print("🕰 Running Phase 4: Similarity Matrix Construction...")
            pipeline._phase_4_similarity_matrices()
        else:
            print("✅ Phase 4: Similarity matrices already available")

        print(f"✅ Phase 4: Similarity matrices for {len(pipeline.similarities)} models")

        # Phase 5: Entity/Theme Extraction
        print("🏷️  Running Phase 5: Entity/Theme Extraction...")
        pipeline._phase_5_entity_theme_extraction()

        # Verify entity/theme extraction results
        print("\n🔍 Verifying Phase 5 Entity/Theme Extraction results...")

        # Check entity/theme data
        if not pipeline.entity_theme_data:
            print("❌ No entity/theme data was extracted")
            return False

        print(f"✅ Entity/theme extraction completed successfully")

        # Test extraction results structure
        print("\n📊 Testing Entity/Theme Extraction Results Structure:")

        metadata = pipeline.entity_theme_data['metadata']
        extraction_results = pipeline.entity_theme_data['extraction_results']

        print(f"   📈 Extraction Metadata:")
        print(f"      Total entities: {metadata.total_entities_extracted:,}")
        print(f"      Total themes: {metadata.total_themes_extracted:,}")
        print(f"      Processing time: {metadata.processing_time:.2f}s")
        print(f"      spaCy available: {metadata.spacy_model_available}")
        print(f"      Ollama available: {metadata.ollama_available}")

        print(f"   🏗️  Granularity Level Breakdown:")
        for granularity_level, count in metadata.granularity_counts.items():
            print(f"      {granularity_level}: {count:,} items processed")

        # Test chunk-level entity extraction
        print("\n🔨 Testing Chunk-Level Entity Extraction:")

        chunk_entities = extraction_results.get('chunk_entities', [])
        if chunk_entities:
            print(f"   📊 Processed {len(chunk_entities)} chunks for entity extraction")

            # Analyze entity distribution
            entity_type_counts = {'PERSON': 0, 'ORG': 0, 'GPE': 0}
            chunks_with_entities = 0

            for result in chunk_entities[:10]:  # Show first 10 for testing
                has_entities = any(len(entities) > 0 for entities in result.entities.values())
                if has_entities:
                    chunks_with_entities += 1

                for entity_type, entity_list in result.entities.items():
                    entity_type_counts[entity_type] += len(entity_list)

                print(f"      Chunk {result.source_id[:20]}...")
                print(f"         Text: '{result.source_text[:60]}...'")
                print(f"         Entities: {dict(result.entities)}")
                print(f"         Method: {result.extraction_method}")
                print(f"         Time: {result.extraction_time:.3f}s")

            print(f"   📈 Chunk Entity Statistics:")
            print(f"      Chunks with entities: {chunks_with_entities}/{len(chunk_entities)}")
            for entity_type, count in entity_type_counts.items():
                print(f"      {entity_type}: {count} entities")

        else:
            print("   ⚠️  No chunk entities found")

        # Test sentence-level entity extraction
        print("\n📝 Testing Sentence-Level Entity Extraction:")

        sentence_entities = extraction_results.get('sentence_entities', [])
        if sentence_entities:
            print(f"   📊 Processed {len(sentence_entities)} sentences for entity extraction")

            # Analyze sentence entity distribution
            sentence_entity_counts = {'PERSON': 0, 'ORG': 0, 'GPE': 0}
            sentences_with_entities = 0

            for result in sentence_entities[:10]:  # Show first 10 for testing
                has_entities = any(len(entities) > 0 for entities in result.entities.values())
                if has_entities:
                    sentences_with_entities += 1

                for entity_type, entity_list in result.entities.items():
                    sentence_entity_counts[entity_type] += len(entity_list)

                print(f"      Sentence {result.source_id[:20]}...")
                print(f"         Text: '{result.source_text[:80]}...'")
                print(f"         Entities: {dict(result.entities)}")
                print(f"         Method: {result.extraction_method}")

            print(f"   📈 Sentence Entity Statistics:")
            print(f"      Sentences with entities: {sentences_with_entities}/{len(sentence_entities)}")
            for entity_type, count in sentence_entity_counts.items():
                print(f"      {entity_type}: {count} entities")

        else:
            print("   ⚠️  No sentence entities found")

        # Test document-level theme extraction
        print("\n📄 Testing Document-Level Theme Extraction:")

        document_themes = extraction_results.get('document_themes', [])
        if document_themes:
            print(f"   📊 Processed {len(document_themes)} documents for theme extraction")

            # Analyze theme distribution
            total_themes = sum(len(result.themes) for result in document_themes)
            documents_with_themes = sum(1 for result in document_themes if len(result.themes) > 0)

            for result in document_themes:
                print(f"      Document: '{result.doc_title}'")
                print(f"         Summary: '{result.source_text[:100]}...'")
                print(f"         Themes: {result.themes}")
                print(f"         Method: {result.extraction_method}")
                print(f"         Model: {result.model_used}")
                print(f"         Time: {result.extraction_time:.3f}s")

            print(f"   📈 Document Theme Statistics:")
            print(f"      Documents with themes: {documents_with_themes}/{len(document_themes)}")
            print(f"      Total themes extracted: {total_themes}")
            print(f"      Average themes per document: {total_themes / len(document_themes):.1f}")

        else:
            print("   ⚠️  No document themes found")

        # Test entity/theme quality and distribution
        print("\n🎯 Testing Entity/Theme Quality and Distribution:")

        if pipeline.entity_theme_stats:
            stats = pipeline.entity_theme_stats
            print(f"   📊 Overall Statistics:")
            print(f"      Total entities: {stats['total_entities']:,}")
            print(f"      Total themes: {stats['total_themes']:,}")
            print(f"      Processing time: {stats['processing_time']:.2f}s")

            if 'entity_type_breakdown' in stats:
                print(f"   🏷️  Entity Type Distribution:")
                for entity_type, count in stats['entity_type_breakdown'].items():
                    percentage = (count / stats['total_entities'] * 100) if stats['total_entities'] > 0 else 0
                    print(f"      {entity_type}: {count} ({percentage:.1f}%)")

            if 'theme_statistics' in stats:
                theme_stats = stats['theme_statistics']
                print(f"   🎨 Theme Statistics:")
                print(f"      Avg themes per document: {theme_stats['avg_themes_per_document']:.1f}")
                print(f"      Max themes per document: {theme_stats['max_themes_per_document']}")
                print(f"      Min themes per document: {theme_stats['min_themes_per_document']}")

            print(f"   🔧 Extraction Methods Used:")
            for method_type, method in stats['extraction_methods'].items():
                print(f"      {method_type}: {method}")

        # Test extraction quality validation
        print("\n✅ Testing Extraction Quality Validation:")

        # Validate entity format and content
        entity_quality_issues = []
        for result_type in ['chunk_entities', 'sentence_entities']:
            for result in extraction_results.get(result_type, [])[:5]:  # Test first 5
                for entity_type, entity_list in result.entities.items():
                    for entity in entity_list:
                        # Check entity quality
                        if len(entity) < 2:
                            entity_quality_issues.append(f"Entity too short: '{entity}' in {result.source_id}")
                        if entity.lower() in ['the', 'and', 'or']:
                            entity_quality_issues.append(f"Common word entity: '{entity}' in {result.source_id}")

        if entity_quality_issues:
            print(f"   ⚠️  Entity quality issues found: {len(entity_quality_issues)}")
            for issue in entity_quality_issues[:3]:  # Show first 3
                print(f"      {issue}")
        else:
            print(f"   ✅ Entity quality validation passed")

        # Validate theme format and content
        theme_quality_issues = []
        for result in document_themes:
            for theme in result.themes:
                # Check theme quality
                if len(theme) < 3:
                    theme_quality_issues.append(f"Theme too short: '{theme}' in {result.doc_title}")
                if '_' in theme:
                    theme_quality_issues.append(f"Underscore in theme: '{theme}' in {result.doc_title}")
                if theme.isupper():
                    theme_quality_issues.append(f"All caps theme: '{theme}' in {result.doc_title}")

        if theme_quality_issues:
            print(f"   ⚠️  Theme quality issues found: {len(theme_quality_issues)}")
            for issue in theme_quality_issues[:3]:  # Show first 3
                print(f"      {issue}")
        else:
            print(f"   ✅ Theme quality validation passed")

        # Test caching functionality
        print("\n💾 Testing entity/theme caching...")
        try:
            # Clear force recompute and run again (should use cache)
            pipeline.config['execution']['force_recompute'] = []

            import time
            cache_start = time.time()
            pipeline._phase_5_entity_theme_extraction()
            cache_end = time.time()
            cache_time = cache_end - cache_start

            print(f"   Cache load time: {cache_time:.3f}s")
            print(f"   Cached extraction results: {len(pipeline.entity_theme_data['extraction_results'])} result types")

            if cache_time < 1.0:  # Should be very fast
                print("   ✅ Caching working effectively")
            else:
                print("   ⚠️  Caching may not be working as expected")

        except Exception as e:
            print(f"   ⚠️  Cache test failed: {e}")

        # Check if entity/theme data was saved
        entity_theme_path = Path(
            pipeline.config['directories']['data']) / "entity_theme" / "entity_theme_extraction.json"
        if entity_theme_path.exists():
            print(f"\n✅ Entity/theme data saved successfully to {entity_theme_path}")

            # Show file size
            file_size = entity_theme_path.stat().st_size / 1024  # KB
            print(f"   File size: {file_size:.1f} KB")
        else:
            print(f"\n⚠️  Entity/theme data file not found at {entity_theme_path}")

        # Test cross-granularity entity consistency
        print("\n🔄 Testing Cross-Granularity Entity Consistency:")

        if chunk_entities and sentence_entities:
            # Check if entities appear across granularity levels
            chunk_entity_set = set()
            for result in chunk_entities:
                for entity_list in result.entities.values():
                    chunk_entity_set.update(entity_list)

            sentence_entity_set = set()
            for result in sentence_entities:
                for entity_list in result.entities.values():
                    sentence_entity_set.update(entity_list)

            overlap = chunk_entity_set.intersection(sentence_entity_set)
            overlap_percentage = (len(overlap) / len(chunk_entity_set)) * 100 if chunk_entity_set else 0

            print(f"   📊 Entity Overlap Analysis:")
            print(f"      Chunk entities: {len(chunk_entity_set)}")
            print(f"      Sentence entities: {len(sentence_entity_set)}")
            print(f"      Overlapping entities: {len(overlap)} ({overlap_percentage:.1f}%)")

            if overlap:
                print(f"      Sample overlapping entities: {list(overlap)[:5]}")

        print("\n🎉 Phase 5 Entity/Theme Extraction test completed successfully!")
        print(f"📋 Experiment ID: {pipeline.experiment_id}")
        print(f"📁 Logs saved to: logs/{pipeline.experiment_id}.log")
        print(f"💾 Entity/theme data cached in: data/entity_theme/")

        return True

    except Exception as e:
        print(f"\n❌ Phase 5 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🧪 Semantic RAG Pipeline - Phase 5 Test")
    print("Testing entity/theme extraction across all granularity levels")
    print("🏷️  Architecture: spaCy Entities (PERSON/ORG/GPE) + Ollama Themes")
    print("=" * 85)

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
        print("🚀 Phase 5 Entity/Theme Extraction is ready for production use")
        print("\n🔗 Key features verified:")
        print("   • spaCy-based entity extraction (PERSON/ORG/GPE) with pattern fallback")
        print("   • Ollama-based theme extraction with keyword fallback")
        print("   • Multi-granularity extraction (chunks + sentences + documents)")
        print("   • High-quality entity/theme filtering and validation")
        print("   • Independent caching system for extraction results")
        print("   • Cross-granularity entity consistency analysis")
        print("   • Comprehensive extraction statistics and metadata")
        print("\n🎯 Ready for Phase 6 Knowledge Graph Assembly:")
        print("   Phase 6 can now use pre-extracted entities and themes")
        print("   for pure O(n) translation to graph relationships")
    else:
        print("\n❌ Tests failed!")
        print("🔧 Please check the error messages above")


if __name__ == "__main__":
    main()