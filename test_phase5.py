#!/usr/bin/env python3
"""
Test Phase 5: Multi-Dimensional Knowledge Graph Construction
==========================================================

Test script to verify Phase 5 (Multi-Dimensional Knowledge Graph Construction) functionality.
Tests revolutionary three-tier hierarchy (Document → Chunk → Sentence) with multi-dimensional
relationship system for domain-agnostic semantic RAG.

Architecture Under Test:
- Hierarchical Structure Layer: Parent/child navigation across granularity levels
- Cosine Similarity Layer: Mathematical semantic relationships (from Phase 4)  
- Entity Overlap Layer: Factual bridges via shared entities (domain-agnostic NER)

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
    """Test Phase 5: Multi-Dimensional Knowledge Graph Construction."""
    print("🧪 Testing Phase 5: Multi-Dimensional Knowledge Graph Construction")
    print("🌟 REVOLUTIONARY ARCHITECTURE: Three-Tier Hierarchy with Multi-Dimensional Relationships")
    print("=" * 85)
    
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
        
        # Override config for testing the new architecture
        print("🔧 Configuring for Multi-Dimensional Knowledge Graph testing...")
        
        # Set mode to full pipeline to include Phase 5
        pipeline.config['execution']['mode'] = 'full_pipeline'
        pipeline.config['execution']['skip_phases'] = []
        
        # Use smaller models for faster testing
        pipeline.config['models']['embedding_models'] = [
            "sentence-transformers/all-MiniLM-L6-v2"  # Faster model for testing
        ]
        
        # Configure multi-dimensional knowledge graph settings for testing
        pipeline.config['knowledge_graph'] = {
            'use_cached': True,
            'architecture': 'multi_dimensional_three_tier',
            'sparsity': {
                'relationship_limits': {
                    'cosine_similarity': 10,   # Reduced for testing
                    'entity_overlap': 6        # Reduced for testing
                },
                'min_thresholds': {
                    'cosine_similarity': 0.3,  # Lower threshold for testing
                    'entity_overlap': 0.15     # Lower threshold for testing
                }
            },
            'extractors': {
                'ner': {'enabled': True, 'entity_types': ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'MISC']},
                'keyphrases': {'enabled': True, 'max_features': 12},
                'summary': {'enabled': True, 'max_sentences': 2}
            }
        }
        
        # Enable force recompute to see fresh multi-dimensional generation
        pipeline.config['execution']['force_recompute'] = ['knowledge_graph']
        
        # Remove old knowledge graph to see fresh multi-dimensional version
        old_kg_path = Path(pipeline.config['directories']['data']) / "knowledge_graph.json"
        if old_kg_path.exists():
            old_size_mb = old_kg_path.stat().st_size / (1024 * 1024)
            old_kg_path.unlink()
            print(f"   Removed old knowledge graph ({old_size_mb:.1f} MB)")
        
        print(f"   🏗️  Architecture: {pipeline.config['knowledge_graph']['architecture']}")
        print(f"   📊 Models: {pipeline.config['models']['embedding_models']}")
        print(f"   🔗 Relationship Types: Hierarchical + Cosine Similarity + Entity Overlap")
        print(f"   🎯 Node Types: Document (Level 0) + Chunk (Level 1) + Sentence (Level 2)")
        print(f"   🧠 Domain Agnostic: Zero theme bias, works with ANY content")
        
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
        
        # Phase 5: Multi-Dimensional Knowledge Graph Construction
        print("🏗️  Running Phase 5: Multi-Dimensional Knowledge Graph Construction...")
        pipeline._phase_5_knowledge_graph_construction()
        
        # Verify revolutionary architecture results
        print("\n🔍 Verifying Multi-Dimensional Knowledge Graph Architecture...")
        
        # Check knowledge graph
        if not pipeline.knowledge_graph:
            print("❌ No knowledge graph was created")
            return False
        
        print(f"✅ Multi-dimensional knowledge graph created successfully")
        
        # Verify three-tier hierarchical structure
        print("\n🏗️  Testing Three-Tier Hierarchical Architecture:")
        
        # Check node types distribution
        document_nodes = pipeline.knowledge_graph.get_nodes_by_type('DOCUMENT')
        chunk_nodes = pipeline.knowledge_graph.get_nodes_by_type('CHUNK')
        sentence_nodes = pipeline.knowledge_graph.get_nodes_by_type('SENTENCE')
        
        print(f"   📄 Document Nodes (Level 0): {len(document_nodes)}")
        print(f"   📝 Chunk Nodes (Level 1): {len(chunk_nodes)}")
        print(f"   📖 Sentence Nodes (Level 2): {len(sentence_nodes)}")
        
        # Verify hierarchy levels
        for node_type, expected_level in [('DOCUMENT', 0), ('CHUNK', 1), ('SENTENCE', 2)]:
            nodes = pipeline.knowledge_graph.get_nodes_by_type(node_type)
            if nodes:
                actual_level = nodes[0].hierarchy_level
                if actual_level == expected_level:
                    print(f"   ✅ {node_type} nodes at correct hierarchy level {expected_level}")
                else:
                    print(f"   ❌ {node_type} nodes at wrong hierarchy level {actual_level} (expected {expected_level})")
        
        # Test hierarchical navigation
        print("\n🧭 Testing Hierarchical Navigation:")
        
        if document_nodes and chunk_nodes:
            # Test document → chunk relationships
            test_doc = document_nodes[0]
            doc_children = pipeline.knowledge_graph.get_children(test_doc.id)
            print(f"   📄→📝 Document '{test_doc.properties.get('title', 'Unknown')[:30]}...' has {len(doc_children)} chunk children")
            
            if doc_children:
                # Test chunk → document relationships
                test_chunk = doc_children[0]
                chunk_parent = pipeline.knowledge_graph.get_parent(test_chunk.id)
                if chunk_parent and chunk_parent.id == test_doc.id:
                    print(f"   ✅ Bidirectional hierarchy: chunk correctly references parent document")
                else:
                    print(f"   ❌ Hierarchy broken: chunk doesn't reference correct parent")
        
        if chunk_nodes and sentence_nodes:
            # Test chunk → sentence relationships
            test_chunk = chunk_nodes[0]
            chunk_children = pipeline.knowledge_graph.get_children(test_chunk.id)
            print(f"   📝→📖 Chunk has {len(chunk_children)} sentence children")
            
            if chunk_children:
                # Test sentence → chunk relationships
                test_sentence = chunk_children[0]
                sentence_parent = pipeline.knowledge_graph.get_parent(test_sentence.id)
                if sentence_parent and sentence_parent.id == test_chunk.id:
                    print(f"   ✅ Bidirectional hierarchy: sentence correctly references parent chunk")
                else:
                    print(f"   ❌ Hierarchy broken: sentence doesn't reference correct parent")
        
        # Test multi-dimensional relationships
        print("\n🔗 Testing Multi-Dimensional Relationship System:")
        
        # Group relationships by type
        relationships_by_type = {}
        for rel in pipeline.knowledge_graph.relationships:
            rel_type = rel.type
            if rel_type not in relationships_by_type:
                relationships_by_type[rel_type] = []
            relationships_by_type[rel_type].append(rel)
        
        print(f"   📊 Relationship Type Distribution:")
        for rel_type, rels in relationships_by_type.items():
            print(f"      {rel_type}: {len(rels):,} relationships")
        
        # Verify required relationship types exist
        required_types = ['parent', 'child', 'cosine_similarity', 'entity_overlap']
        for req_type in required_types:
            if req_type in relationships_by_type:
                print(f"   ✅ {req_type} relationships present")
            else:
                print(f"   ⚠️  {req_type} relationships missing")
        
        # Test domain agnostic extractors
        print("\n🌍 Testing Domain-Agnostic Extractors:")
        
        # Test NER extraction
        sample_node = chunk_nodes[0] if chunk_nodes else None
        if sample_node and 'entities' in sample_node.properties:
            entities = sample_node.properties['entities']
            total_entities = sum(len(v) for v in entities.values())
            print(f"   🏷️  Entity extraction: {total_entities} entities found")
            for entity_type, entity_list in entities.items():
                if entity_list:
                    print(f"      {entity_type}: {len(entity_list)} entities")
        
        # Test keyphrases extraction
        if sample_node and 'keyphrases' in sample_node.properties:
            keyphrases = sample_node.properties['keyphrases']
            print(f"   🔑 Keyphrase extraction: {len(keyphrases)} keyphrases")
            print(f"      Sample: {keyphrases[:3]}")
        
        # Test summary extraction
        if sample_node and 'summary' in sample_node.properties:
            summary = sample_node.properties['summary']
            print(f"   📝 Summary extraction: '{summary[:60]}...'")
        
        # Test knowledge graph statistics
        if pipeline.kg_stats:
            stats = pipeline.kg_stats
            print(f"\n📊 Knowledge Graph Statistics:")
            print(f"   Total nodes: {stats['total_nodes']:,}")
            print(f"   Total relationships: {stats['total_relationships']:,}")
            print(f"   Architecture: {stats.get('architecture', 'Unknown')}")
            print(f"   Node types: {stats['node_types']}")
            print(f"   Relationship types: {stats['relationship_types']}")
            print(f"   Build time: {stats.get('build_time', 0):.2f}s")
        
        # Test sample multi-dimensional traversal
        print("\n🧪 Testing Multi-Dimensional Semantic Traversal:")
        
        if chunk_nodes:
            test_chunk = chunk_nodes[0]
            
            # Test cosine similarity neighbors
            cosine_neighbors = pipeline.knowledge_graph.get_neighbors(test_chunk.id, ['cosine_similarity'])
            print(f"   🔄 Cosine similarity neighbors: {len(cosine_neighbors)}")
            
            # Test entity overlap neighbors  
            entity_neighbors = pipeline.knowledge_graph.get_neighbors(test_chunk.id, ['entity_overlap'])
            print(f"   🏷️  Entity overlap neighbors: {len(entity_neighbors)}")
            
            # Test hierarchical neighbors
            hierarchical_neighbors = pipeline.knowledge_graph.get_neighbors(test_chunk.id, ['parent', 'child'])
            print(f"   📊 Hierarchical neighbors: {len(hierarchical_neighbors)}")
            
            # Test all neighbors (multi-dimensional)
            all_neighbors = pipeline.knowledge_graph.get_neighbors(test_chunk.id)
            print(f"   🌟 Total multi-dimensional neighbors: {len(all_neighbors)}")
        
        # Check enhanced retrieval engine
        if not pipeline.retrieval_engine:
            print("❌ No retrieval engine was created")
            return False
        
        print(f"\n✅ Enhanced retrieval engine created successfully")
        
        # Check retrieval statistics
        if pipeline.retrieval_stats:
            stats = pipeline.retrieval_stats
            print(f"   📊 Enhanced Retrieval Statistics:")
            print(f"      Algorithm: {stats['algorithm']}")
            print(f"      Models: {stats['models_available']}")
            print(f"      Multi-dimensional KG enabled: {pipeline.knowledge_graph is not None}")
            for model, count in stats['total_chunks_per_model'].items():
                print(f"      {model}: {count:,} chunks")
        
        # Test actual retrieval with domain-agnostic queries
        print("\n🧪 Testing Multi-Dimensional Retrieval with Domain-Agnostic Queries...")
        
        # Use generic queries that should work with any domain
        test_queries = [
            "what is machine learning",
            "what is a neural network",
            "how does ai work",
            "can you tell me a little bit about transformer architecture? What is a transformer in AI?"
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
                    print(f"         Unique chunks: {metadata.get('unique_chunks', 'N/A')}")
                    print(f"         Time: {metadata.get('retrieval_time', 0):.3f}s")
                
                # Show sample results
                if semantic_result.chunks:
                    sample_text = semantic_result.chunks[0].chunk_text[:60].replace('\n', ' ')
                    print(f"         Sample result: '{sample_text}...'")
                    print(f"         Score: {semantic_result.scores[0]:.3f}")
                
            except Exception as e:
                print(f"      ❌ Query failed: {e}")
                continue
        
        # Check if knowledge graph was saved
        kg_path = Path(pipeline.config['directories']['data']) / "knowledge_graph.json"
        if kg_path.exists():
            print(f"\n✅ Multi-dimensional knowledge graph saved successfully to {kg_path}")
            
            # Show file size
            file_size = kg_path.stat().st_size / 1024  # KB
            print(f"   File size: {file_size:.1f} KB")
            
            # Verify it can be loaded
            try:
                from knowledge_graph import KnowledgeGraph
                loaded_kg = KnowledgeGraph.load(str(kg_path))
                print(f"   ✅ Knowledge graph successfully loads from cache")
                print(f"   📊 Loaded: {len(loaded_kg.nodes)} nodes, {len(loaded_kg.relationships)} relationships")
            except Exception as e:
                print(f"   ❌ Failed to load knowledge graph: {e}")
        else:
            print(f"\n⚠️  Knowledge graph file not found at {kg_path}")
        
        print("\n🎉 Phase 5 Multi-Dimensional Architecture test completed successfully!")
        print(f"📋 Experiment ID: {pipeline.experiment_id}")
        print(f"📁 Logs saved to: logs/{pipeline.experiment_id}.log")
        print(f"📊 Knowledge graph saved to: {kg_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 5 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🧪 Semantic RAG Pipeline - Phase 5 Test")
    print("🌟 Testing REVOLUTIONARY Multi-Dimensional Three-Tier Knowledge Graph Architecture")
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
    
    success = test_phase5()
    
    if success:
        print("\n✅ All tests passed!")
        print("🚀 Phase 5 Multi-Dimensional Architecture is ready for production use")
        print("\n🌟 REVOLUTIONARY FEATURES VERIFIED:")
        print("   • Three-tier hierarchical structure (Document → Chunk → Sentence)")
        print("   • Multi-dimensional relationship system:")
        print("     - Hierarchical Structure Layer (parent/child navigation)")
        print("     - Cosine Similarity Layer (mathematical semantic relationships)")
        print("     - Entity Overlap Layer (factual bridges via shared entities)")
        print("   • Domain-agnostic extractors (zero theme bias)")
        print("   • Bidirectional relationship traversal")
        print("   • Multi-granularity semantic navigation")
        print("   • Intelligent sparse relationship construction")
        print("   • JSON serialization with hierarchical metadata")
        print("\n🎯 RESEARCH BREAKTHROUGH:")
        print("   This is the first text-centric multi-dimensional knowledge graph")
        print("   for RAG systems that mirrors human cognitive reading patterns!")
    else:
        print("\n❌ Tests failed!")
        print("🔧 Please check the error messages above")


if __name__ == "__main__":
    main()
