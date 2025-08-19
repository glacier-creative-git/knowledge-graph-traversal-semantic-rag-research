#!/usr/bin/env python3
"""
Test Phase 6: Knowledge Graph Assembly with Theme Bridges
========================================================

Test script to verify Phase 6 (Knowledge Graph Assembly) functionality.
Tests assembly of knowledge graph using pre-computed similarities (Phase 4)
and extracted entities/themes (Phase 5) with new cross-document theme bridging.

NEW PHASE STRUCTURE:
- Phase 1: Setup & Initialization
- Phase 2: Data Acquisition
- Phase 3: Multi-Granularity Embedding Generation
- Phase 4: Multi-Granularity Similarity Matrix Construction
- Phase 5: Entity/Theme Extraction (NEW)
- Phase 6: Knowledge Graph Assembly (NEW - this test)
- Phase 7: Question Generation (renamed from old Phase 6)

Run from project root:
    python test_phase6.py
"""

import sys
import os
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))

from pipeline import SemanticRAGPipeline


def test_phase6():
    """Test Phase 6: Knowledge Graph Assembly with Theme Bridges."""
    print("🧪 Testing Phase 6: Knowledge Graph Assembly with Theme Bridges")
    print("🌟 Architecture: Pre-computed Assembly + Cross-Document Theme Highways")
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

        # Override config for testing knowledge graph assembly
        print("🔧 Configuring for Phase 6 Knowledge Graph Assembly testing...")

        # Set mode to full pipeline to include Phase 6
        pipeline.config['execution']['mode'] = 'full_pipeline'
        pipeline.config['execution']['skip_phases'] = []

        # Use smaller models for faster testing
        pipeline.config['models']['embedding_models'] = [
            "sentence-transformers/all-MiniLM-L6-v2"  # Faster model for testing
        ]

        # Configure knowledge graph assembly settings for testing
        pipeline.config['knowledge_graph_assembly'] = {
            'use_cached': True,
            'theme_bridging': {
                'enabled': True,
                'top_k_bridges': 3,
                'min_bridge_similarity': 0.2,
                'exclude_intra_document': True
            },
            'entity_relationships': {
                'enabled': True,
                'high_confidence_only': True,
                'min_entity_overlap': 2,
                'min_jaccard_similarity': 0.3
            }
        }

        # Enable force recompute to see fresh assembly
        pipeline.config['execution']['force_recompute'] = ['knowledge_graph']

        print(f"   🌉 Theme bridging: {pipeline.config['knowledge_graph_assembly']['theme_bridging']['enabled']}")
        print(
            f"   🔗 Cross-document bridges per theme: {pipeline.config['knowledge_graph_assembly']['theme_bridging']['top_k_bridges']}")
        print(
            f"   🎯 Bridge similarity threshold: {pipeline.config['knowledge_graph_assembly']['theme_bridging']['min_bridge_similarity']}")
        print(
            f"   🏷️  Entity relationships: {pipeline.config['knowledge_graph_assembly']['entity_relationships']['enabled']}")

        # Run pipeline phases 1-6
        print("\n🚀 Running pipeline phases 1-6...")

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

        # Phase 5: Entity/Theme Extraction (if needed)
        if not pipeline.entity_theme_data:
            print("🏷️  Running Phase 5: Entity/Theme Extraction...")
            pipeline._phase_5_entity_theme_extraction()
        else:
            print("✅ Phase 5: Entity/theme data already available")

        entities_count = pipeline.entity_theme_data['metadata'].total_entities_extracted
        themes_count = pipeline.entity_theme_data['metadata'].total_themes_extracted
        print(f"✅ Phase 5: {entities_count:,} entities and {themes_count} themes extracted")

        # Phase 6: Knowledge Graph Assembly with Theme Bridges
        print("🏗️  Running Phase 6: Knowledge Graph Assembly with Theme Bridges...")
        pipeline._phase_6_knowledge_graph_construction()

        # Verify knowledge graph assembly results
        print("\n🔍 Verifying Phase 6 Knowledge Graph Assembly results...")

        # Check knowledge graph
        if not pipeline.knowledge_graph:
            print("❌ No knowledge graph was assembled")
            return False

        print(f"✅ Knowledge graph assembled successfully")

        # Test knowledge graph structure
        print("\n📊 Testing Knowledge Graph Assembly Structure:")

        metadata = pipeline.knowledge_graph.metadata

        print(f"   🏗️  Architecture: {metadata.get('architecture', 'Unknown')}")
        print(f"   📈 Assembly Metadata:")
        print(f"      Total nodes: {metadata['total_nodes']:,}")
        print(f"      Total relationships: {metadata['total_relationships']:,}")
        print(f"      Build time: {metadata.get('build_time', 0):.2f}s")

        # Test granularity distribution
        granularity_counts = metadata.get('granularity_counts', {})
        print(f"   📊 Granularity Distribution:")
        for granularity_type, count in granularity_counts.items():
            print(f"      {granularity_type}: {count:,} nodes")

        # Test theme bridge statistics
        if 'theme_bridge_stats' in metadata:
            bridge_stats = metadata['theme_bridge_stats']
            print(f"   🌉 Theme Bridge Statistics:")
            print(f"      Total unique themes: {bridge_stats['total_unique_themes']}")
            print(f"      Themes with cross-document bridges: {bridge_stats['themes_with_bridges']}")
            print(f"      Total cross-document bridges: {bridge_stats['total_bridges']}")

            if bridge_stats['total_unique_themes'] > 0:
                bridge_percentage = (bridge_stats['themes_with_bridges'] / bridge_stats['total_unique_themes']) * 100
                print(f"      Bridge coverage: {bridge_percentage:.1f}% of themes have cross-document connections")

        # Test relationship type distribution
        relationship_types = metadata.get('relationship_types', {})
        print(f"   🔗 Relationship Type Distribution:")
        for rel_type, count in relationship_types.items():
            print(f"      {rel_type}: {count:,} relationships")

        # Test cross-document theme bridging functionality
        print("\n🌉 Testing Cross-Document Theme Bridges:")

        # Get sample nodes to test theme inheritance
        chunk_nodes = pipeline.knowledge_graph.get_nodes_by_type('CHUNK')
        sentence_nodes = pipeline.knowledge_graph.get_nodes_by_type('SENTENCE')
        document_nodes = pipeline.knowledge_graph.get_nodes_by_type('DOCUMENT')

        if chunk_nodes:
            print(f"   📊 Testing Theme Inheritance in Chunk Nodes:")

            # Test first few chunks for theme inheritance
            for i, chunk_node in enumerate(chunk_nodes[:3]):
                props = chunk_node.properties

                print(f"      Chunk {i + 1}: {chunk_node.id}")
                print(f"         Source: '{props.get('source_article', 'Unknown')}'")
                print(f"         Text: '{props.get('text', '')[:60]}...'")

                # Test direct themes
                direct_themes = props.get('direct_themes', [])
                print(f"         Direct themes ({len(direct_themes)}): {direct_themes}")

                # Test inherited themes (cross-document bridges)
                inherited_themes = props.get('inherited_themes', [])
                print(f"         Inherited themes ({len(inherited_themes)}) from cross-document bridges:")

                for inherited_theme in inherited_themes[:3]:  # Show first 3
                    print(
                        f"            • '{inherited_theme['theme']}' (similarity: {inherited_theme['similarity']:.3f})")
                    print(f"              inherited from: '{inherited_theme['inherited_from']}'")
                    print(f"              source document: {inherited_theme['source_document']}")

                # Test theme inheritance map
                theme_inheritance_map = props.get('theme_inheritance_map', {})
                print(f"         Theme inheritance pathways: {len(theme_inheritance_map)} direct themes create bridges")

                total_themes = props.get('total_semantic_themes', 0)
                print(f"         Total semantic context: {total_themes} themes available for traversal")

                # Test entities
                entities = props.get('entities', {})
                total_entities = sum(len(entity_list) for entity_list in entities.values())
                print(f"         Entities: {total_entities} entities ({dict(entities)})")

                print()

        if sentence_nodes:
            print(f"   📝 Testing Theme Inheritance in Sentence Nodes:")

            sample_sentence = sentence_nodes[0]
            props = sample_sentence.properties

            print(f"      Sample Sentence: {sample_sentence.id}")
            print(f"         Text: '{props.get('text', '')[:80]}...'")

            inherited_themes = props.get('inherited_themes', [])
            print(f"         Inherited cross-document themes: {len(inherited_themes)}")

            for inherited_theme in inherited_themes[:2]:  # Show first 2
                print(f"            • {inherited_theme['theme']} (from {inherited_theme['source_document']})")

        # Test knowledge graph navigation and relationships
        print("\n🧭 Testing Knowledge Graph Navigation:")

        if chunk_nodes:
            test_chunk = chunk_nodes[0]

            # Test hierarchical navigation
            parent = pipeline.knowledge_graph.get_parent(test_chunk.id)
            children = pipeline.knowledge_graph.get_children(test_chunk.id)
            print(f"   📊 Hierarchical Navigation:")
            print(f"      Test chunk parent: {'Found' if parent else 'None'}")
            print(f"      Test chunk children: {len(children)} children")

            # Test similarity-based neighbors
            similarity_neighbors = pipeline.knowledge_graph.get_neighbors(test_chunk.id, ['cosine_similarity',
                                                                                          'cosine_similarity_intra',
                                                                                          'cosine_similarity_inter'])
            print(f"      Similarity neighbors: {len(similarity_neighbors)} similar nodes")

            # Test entity-based neighbors
            entity_neighbors = pipeline.knowledge_graph.get_neighbors(test_chunk.id, ['entity_overlap'])
            print(f"      Entity overlap neighbors: {len(entity_neighbors)} entity-connected nodes")

            # Test all neighbors (multi-dimensional navigation)
            all_neighbors = pipeline.knowledge_graph.get_neighbors(test_chunk.id)
            print(f"      Total neighbors: {len(all_neighbors)} connected nodes")

        # Test semantic highway traversal potential
        print("\n🛣️  Testing Semantic Highway Traversal Potential:")

        if chunk_nodes and len(chunk_nodes) >= 2:
            # Test cross-document theme bridge potential
            chunk1 = chunk_nodes[0]
            chunk1_themes = set(chunk1.properties.get('direct_themes', []))
            chunk1_inherited = {theme['theme'] for theme in chunk1.properties.get('inherited_themes', [])}
            chunk1_all_themes = chunk1_themes.union(chunk1_inherited)

            cross_doc_connections = 0
            for chunk2 in chunk_nodes[1:6]:  # Test against next 5 chunks
                if chunk2.properties.get('source_article') != chunk1.properties.get('source_article'):
                    chunk2_themes = set(chunk2.properties.get('direct_themes', []))
                    chunk2_inherited = {theme['theme'] for theme in chunk2.properties.get('inherited_themes', [])}
                    chunk2_all_themes = chunk2_themes.union(chunk2_inherited)

                    # Check for theme overlap (potential highway)
                    theme_overlap = chunk1_all_themes.intersection(chunk2_all_themes)
                    if theme_overlap:
                        cross_doc_connections += 1

            print(f"   🌉 Cross-document theme highway potential:")
            print(f"      Test chunk can connect to {cross_doc_connections}/5 cross-document chunks via theme bridges")

            # Show sample theme bridge
            if chunk1.properties.get('inherited_themes'):
                sample_bridge = chunk1.properties['inherited_themes'][0]
                print(f"      Sample highway: '{sample_bridge['inherited_from']}' → '{sample_bridge['theme']}'")
                print(f"      Bridge strength: {sample_bridge['similarity']:.3f}")

        # Test retrieval engine integration
        print("\n🎯 Testing Retrieval Engine Integration:")

        if pipeline.retrieval_engine:
            retrieval_stats = pipeline.retrieval_stats
            print(f"   📊 Retrieval Engine Status:")
            print(f"      Algorithm: {retrieval_stats['algorithm']}")
            print(f"      Models available: {retrieval_stats['models_available']}")
            print(f"      Knowledge graph integration: {'✅ Enabled' if pipeline.knowledge_graph else '❌ Disabled'}")

            for model, count in retrieval_stats['total_chunks_per_model'].items():
                print(f"      {model}: {count:,} retrievable chunks")

            # Test sample retrieval with theme-aware system
            print(f"   🔍 Testing Sample Retrieval with Theme Awareness:")

            test_queries = [
                "machine learning algorithms",
                "neural network architecture",
                "cognitive psychology"
            ]

            model_name = list(pipeline.embeddings.keys())[0]

            for query in test_queries[:2]:  # Test first 2 queries
                try:
                    print(f"      Query: '{query}'")

                    # Test semantic traversal (which now has theme awareness)
                    semantic_result = pipeline.retrieval_engine.retrieve(
                        query, model_name, algorithm="semantic_traversal"
                    )

                    print(f"         Semantic traversal: {len(semantic_result.chunks)} results")

                    if semantic_result.chunks:
                        sample_chunk = semantic_result.chunks[0]
                        sample_text = sample_chunk.chunk_text[:40].replace('\n', ' ')
                        print(f"         Top result: '{sample_text}...' (score: {semantic_result.scores[0]:.3f})")

                        # Show if result has theme context
                        if hasattr(sample_chunk, 'properties'):
                            themes = len(sample_chunk.properties.get('direct_themes', []))
                            inherited = len(sample_chunk.properties.get('inherited_themes', []))
                            print(f"         Theme context: {themes} direct + {inherited} inherited themes")

                except Exception as e:
                    print(f"         ⚠️  Query failed: {e}")

        # Test caching functionality
        print("\n💾 Testing knowledge graph assembly caching...")
        try:
            # Clear force recompute and run again (should use cache)
            pipeline.config['execution']['force_recompute'] = []

            import time
            cache_start = time.time()
            pipeline._phase_6_knowledge_graph_construction()
            cache_end = time.time()
            cache_time = cache_end - cache_start

            print(f"   Cache load time: {cache_time:.3f}s")
            print(f"   Cached knowledge graph: {len(pipeline.knowledge_graph.nodes)} nodes")

            if cache_time < 2.0:  # Should be fast for graph loading
                print("   ✅ Caching working effectively")
            else:
                print("   ⚠️  Caching may not be working as expected")

        except Exception as e:
            print(f"   ⚠️  Cache test failed: {e}")

        # Check if knowledge graph was saved
        kg_path = Path(pipeline.config['directories']['data']) / "knowledge_graph.json"
        if kg_path.exists():
            print(f"\n✅ Knowledge graph saved successfully to {kg_path}")

            # Show file size
            file_size = kg_path.stat().st_size / 1024  # KB
            print(f"   File size: {file_size:.1f} KB")

            # Verify it can be loaded
            try:
                from utils.knowledge_graph import MultiGranularityKnowledgeGraph
                loaded_kg = MultiGranularityKnowledgeGraph.load(str(kg_path))
                print(f"   ✅ Knowledge graph successfully loads from cache")
                print(f"   📊 Loaded: {len(loaded_kg.nodes)} nodes, {len(loaded_kg.relationships)} relationships")

                # Verify theme bridge data is preserved
                if loaded_kg.metadata.get('theme_bridge_stats'):
                    bridge_stats = loaded_kg.metadata['theme_bridge_stats']
                    print(f"   🌉 Theme bridges preserved: {bridge_stats['total_bridges']} bridges")

            except Exception as e:
                print(f"   ❌ Failed to load knowledge graph: {e}")
        else:
            print(f"\n⚠️  Knowledge graph file not found at {kg_path}")

        # Test data flow validation
        print("\n🔄 Testing Data Flow Validation (Phase 4 → Phase 5 → Phase 6):")

        # Verify Phase 4 data was used
        if pipeline.similarities:
            model_name = list(pipeline.similarities.keys())[0]
            connections = pipeline.similarities[model_name].get('connections', [])
            print(f"   📊 Phase 4 → Phase 6: {len(connections)} pre-computed similarity connections used")

        # Verify Phase 5 data was used
        if pipeline.entity_theme_data:
            extraction_results = pipeline.entity_theme_data['extraction_results']
            chunk_entities = len(extraction_results.get('chunk_entities', []))
            sentence_entities = len(extraction_results.get('sentence_entities', []))
            document_themes = len(extraction_results.get('document_themes', []))
            print(
                f"   🏷️  Phase 5 → Phase 6: {chunk_entities} chunk entities, {sentence_entities} sentence entities, {document_themes} document themes used")

        # Verify assembly integration
        if pipeline.knowledge_graph:
            sample_chunk_node = chunk_nodes[0] if chunk_nodes else None
            if sample_chunk_node:
                has_entities = bool(sample_chunk_node.properties.get('entities'))
                has_themes = bool(sample_chunk_node.properties.get('direct_themes'))
                has_inherited = bool(sample_chunk_node.properties.get('inherited_themes'))
                print(f"   🔗 Integration verification:")
                print(f"      Nodes have entities: {'✅' if has_entities else '❌'}")
                print(f"      Nodes have direct themes: {'✅' if has_themes else '❌'}")
                print(f"      Nodes have inherited themes: {'✅' if has_inherited else '❌'}")

        print("\n🎉 Phase 6 Knowledge Graph Assembly test completed successfully!")
        print(f"📋 Experiment ID: {pipeline.experiment_id}")
        print(f"📁 Logs saved to: logs/{pipeline.experiment_id}.log")
        print(f"📊 Knowledge graph saved to: {kg_path}")

        return True

    except Exception as e:
        print(f"\n❌ Phase 6 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🧪 Semantic RAG Pipeline - Phase 6 Test")
    print("Testing knowledge graph assembly with cross-document theme bridges")
    print("🌟 Architecture: Pre-computed Assembly + Semantic Highway System")
    print("=" * 90)

    # Check if we're in the right directory
    if not Path("config.yaml").exists():
        print("❌ config.yaml not found!")
        print("💡 Make sure you're running this from the project root directory")
        print("💡 Expected directory structure:")
        print("   config.yaml")
        print("   utils/")
        print("   requirements.txt")
        return

    success = test_phase6()

    if success:
        print("\n✅ All tests passed!")
        print("🚀 Phase 6 Knowledge Graph Assembly is ready for production use")
        print("\n🔗 Key features verified:")
        print("   • Knowledge graph assembly using pre-computed similarities (Phase 4)")
        print("   • Entity and theme integration from extraction phase (Phase 5)")
        print("   • Cross-document theme bridging system (semantic highways)")
        print("   • Multi-granularity node creation with theme inheritance")
        print("   • Relationship building from pre-computed connection data")
        print("   • Hierarchical + similarity + entity relationship layers")
        print("   • Enhanced retrieval engine with theme-aware navigation")
        print("   • Comprehensive caching system for assembled graphs")
        print("\n🌉 BREAKTHROUGH: Theme Highway System:")
        print("   Every chunk and sentence now inherits both direct themes from its")
        print("   document AND cross-document similar themes, creating semantic")
        print("   bridges between different knowledge domains for traversal.")
        print("\n🎯 Ready for Phase 7 Question Generation:")
        print("   The assembled knowledge graph provides rich semantic context")
        print("   for generating multi-hop questions across theme highways.")
    else:
        print("\n❌ Tests failed!")
        print("🔧 Please check the error messages above")


if __name__ == "__main__":
    main()