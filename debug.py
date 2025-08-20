#!/usr/bin/env python3
"""
Direct Theme Inheritance Debug Script
====================================

Standalone script to debug theme inheritance issues without pipeline interference.
Run from project root: python debug_theme_inheritance.py
"""

import sys
import json
import logging
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))


def load_entity_theme_data():
    """Load entity/theme data directly from cache."""
    entity_theme_path = Path("data/entity_theme/entity_theme_extraction.json")

    if not entity_theme_path.exists():
        print(f"❌ Entity/theme data not found at {entity_theme_path}")
        return None

    with open(entity_theme_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✅ Loaded entity/theme data from {entity_theme_path}")
    return data


def load_knowledge_graph():
    """Load knowledge graph directly from JSON."""
    kg_path = Path("data/knowledge_graph.json")

    if not kg_path.exists():
        print(f"❌ Knowledge graph not found at {kg_path}")
        return None

    with open(kg_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✅ Loaded knowledge graph from {kg_path}")
    return data


def debug_entity_theme_structure(entity_theme_data):
    """Debug the structure of entity/theme data."""
    print("\n" + "=" * 60)
    print("🔍 DEBUGGING ENTITY/THEME DATA STRUCTURE")
    print("=" * 60)

    if not entity_theme_data:
        print("❌ No entity/theme data to debug")
        return

    extraction_results = entity_theme_data.get('extraction_results', {})
    document_themes = extraction_results.get('document_themes', [])

    print(f"📄 Document themes found: {len(document_themes)}")

    for i, theme_result in enumerate(document_themes[:3]):  # Show first 3
        print(f"\n   Document {i + 1}:")
        print(f"      doc_id: {theme_result.get('doc_id', 'MISSING')}")
        print(f"      doc_title: {theme_result.get('doc_title', 'MISSING')}")
        print(f"      themes: {theme_result.get('themes', [])}")
        print(f"      extraction_method: {theme_result.get('extraction_method', 'MISSING')}")


def debug_theme_bridge_builder():
    """Create and debug ThemeBridgeBuilder directly."""
    print("\n" + "=" * 60)
    print("🌉 DEBUGGING THEME BRIDGE BUILDER")
    print("=" * 60)

    # Load entity/theme data
    entity_theme_data = load_entity_theme_data()
    if not entity_theme_data:
        return

    # Create a minimal config
    config = {
        'models': {'embedding_batch_size': 32},
        'system': {'device': 'cpu'},
        'theme_bridging': {
            'top_k_bridges': 3,
            'min_bridge_similarity': 0.2
        }
    }

    # Import and create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(handler)

    try:
        # Import ThemeBridgeBuilder
        from knowledge_graph import ThemeBridgeBuilder
        from models import EmbeddingModel

        print("✅ Successfully imported ThemeBridgeBuilder")

        # Create embedding model (minimal setup)
        print("🔄 Creating minimal embedding model...")
        embedding_model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2", "cpu", logger)

        # Create ThemeBridgeBuilder
        print("🔄 Creating ThemeBridgeBuilder...")
        theme_bridge_builder = ThemeBridgeBuilder(entity_theme_data, embedding_model, config, logger)

        print(f"✅ ThemeBridgeBuilder created successfully")
        print(f"   Unique themes: {len(theme_bridge_builder.all_unique_themes)}")
        print(f"   Documents with themes: {len(theme_bridge_builder.themes_by_document)}")

        # Debug the themes_by_document structure
        print("\n📊 Debugging themes_by_document structure:")
        for i, (doc_key, themes) in enumerate(list(theme_bridge_builder.themes_by_document.items())[:3]):
            print(f"   Document {i + 1}:")
            print(f"      Key: '{doc_key}' (type: {type(doc_key)})")
            print(f"      Themes: {themes} (count: {len(themes)})")

        # Test theme bridge computation
        print("\n🔄 Computing theme bridges...")
        theme_bridges = theme_bridge_builder.compute_cross_document_theme_bridges()

        print(f"✅ Theme bridges computed: {len(theme_bridges)} themes have bridges")

        # Debug bridge structure
        print("\n🌉 Debugging bridge structure:")
        for i, (theme, bridges) in enumerate(list(theme_bridges.items())[:3]):
            print(f"   Theme {i + 1}: '{theme}'")
            print(f"      Bridges: {bridges}")
            print(f"      Bridge count: {len(bridges)}")
            if bridges:
                first_bridge = bridges[0]
                print(f"      First bridge: {first_bridge}")
                print(f"      First bridge type: {type(first_bridge)}")
                print(f"      First bridge element types: {[type(x) for x in first_bridge]}")

        # Test theme inheritance for a specific document
        print("\n🧪 Testing theme inheritance:")

        # Get a document title from the structure
        doc_titles = list(theme_bridge_builder.themes_by_document.keys())
        if doc_titles:
            test_doc = doc_titles[0]
            print(f"   Testing document: '{test_doc}'")

            # Call get_inherited_themes_for_node
            print("   Calling get_inherited_themes_for_node...")
            try:
                result = theme_bridge_builder.get_inherited_themes_for_node(test_doc, theme_bridges)

                print(f"   ✅ Method executed successfully!")
                print(f"   Direct themes: {len(result['direct_themes'])}")
                print(f"   Inherited themes: {len(result['inherited_themes'])}")

                # Show inherited theme details
                if result['inherited_themes']:
                    print(f"   First inherited theme sample:")
                    first_inherited = result['inherited_themes'][0]
                    for key, value in first_inherited.items():
                        print(f"      {key}: {value} (type: {type(value)})")

            except Exception as e:
                print(f"   ❌ Method failed: {e}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"❌ Failed to create ThemeBridgeBuilder: {e}")
        import traceback
        traceback.print_exc()


def debug_knowledge_graph_nodes():
    """Debug actual nodes in the knowledge graph to see theme data."""
    print("\n" + "=" * 60)
    print("📊 DEBUGGING KNOWLEDGE GRAPH NODES")
    print("=" * 60)

    kg_data = load_knowledge_graph()
    if not kg_data:
        return

    nodes = kg_data.get('nodes', [])
    print(f"📈 Total nodes in graph: {len(nodes)}")

    # Find chunk nodes and examine their theme properties
    chunk_nodes = [node for node in nodes if node.get('type') == 'CHUNK']
    print(f"🔨 Chunk nodes found: {len(chunk_nodes)}")

    if chunk_nodes:
        print("\n🔍 Examining first few chunk nodes:")
        for i, node in enumerate(chunk_nodes[:3]):
            props = node.get('properties', {})
            print(f"\n   Chunk {i + 1}: {node.get('id', 'NO_ID')}")
            print(f"      Source: {props.get('source_article', 'NO_SOURCE')}")
            print(f"      Direct themes: {props.get('direct_themes', 'MISSING')}")
            print(f"      Inherited themes count: {len(props.get('inherited_themes', []))}")
            print(f"      Total semantic themes: {props.get('total_semantic_themes', 'MISSING')}")

            # Check inherited themes structure
            inherited = props.get('inherited_themes', [])
            if inherited:
                print(f"      First inherited theme: {inherited[0]}")


def main():
    """Main debug function."""
    print("🔍 DIRECT THEME INHERITANCE DEBUG SCRIPT")
    print("=" * 60)
    print("This script bypasses the pipeline to debug theme inheritance directly.")

    try:
        # Debug entity/theme data structure
        entity_theme_data = load_entity_theme_data()
        debug_entity_theme_structure(entity_theme_data)

        # Debug ThemeBridgeBuilder
        debug_theme_bridge_builder()

        # Debug knowledge graph nodes
        debug_knowledge_graph_nodes()

        print("\n" + "=" * 60)
        print("🎉 Debug script completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Debug script failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()