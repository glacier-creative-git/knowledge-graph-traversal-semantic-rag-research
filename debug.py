#!/usr/bin/env python3
"""
Knowledge Graph Architecture Diagnostic
======================================

Analyzes the exported knowledge graph to understand:
1. What types of connections actually exist (explicit vs implicit)
2. How theme bridges are stored and accessed
3. The true distribution of raw vs theme-based connections
4. Node property analysis for theme inheritance

Run from project root:
    python kg_diagnostics.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))


def load_knowledge_graph():
    """Load the knowledge graph from the standard cache location."""
    kg_path = Path("data/knowledge_graph.json")

    if not kg_path.exists():
        print(f"❌ Knowledge graph not found at {kg_path}")
        print("💡 Make sure you've run Phase 6 to generate the knowledge graph")
        return None

    try:
        with open(kg_path, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)

        print(f"✅ Loaded knowledge graph from {kg_path}")
        return kg_data

    except Exception as e:
        print(f"❌ Failed to load knowledge graph: {e}")
        return None


def analyze_node_structure(kg_data):
    """Analyze the structure and properties of nodes."""
    print("\n🔍 NODE STRUCTURE ANALYSIS")
    print("=" * 50)

    nodes = kg_data.get('nodes', [])

    # Count nodes by type
    node_type_counts = Counter(node.get('type', 'UNKNOWN') for node in nodes)
    print(f"Node type distribution: {dict(node_type_counts)}")

    # Analyze node properties by type
    for node_type in ['DOCUMENT', 'CHUNK', 'SENTENCE']:
        type_nodes = [n for n in nodes if n.get('type') == node_type]
        if not type_nodes:
            continue

        print(f"\n📊 {node_type} Node Analysis ({len(type_nodes)} nodes):")

        # Get sample properties
        sample_node = type_nodes[0]
        properties = sample_node.get('properties', {})

        print(f"   Available properties: {list(properties.keys())}")

        # Theme-specific analysis
        if 'direct_themes' in properties:
            theme_counts = [len(node.get('properties', {}).get('direct_themes', [])) for node in type_nodes]
            avg_direct_themes = sum(theme_counts) / len(theme_counts) if theme_counts else 0
            print(f"   Average direct themes per node: {avg_direct_themes:.2f}")

        if 'inherited_themes' in properties:
            inherited_counts = [len(node.get('properties', {}).get('inherited_themes', [])) for node in type_nodes]
            avg_inherited = sum(inherited_counts) / len(inherited_counts) if inherited_counts else 0
            print(f"   Average inherited themes per node: {avg_inherited:.2f}")

            # Sample inherited theme structure
            sample_inherited = properties.get('inherited_themes', [])
            if sample_inherited:
                print(f"   Sample inherited theme: {sample_inherited[0]}")

        if 'theme_inheritance_map' in properties:
            inheritance_maps = [node.get('properties', {}).get('theme_inheritance_map', {}) for node in type_nodes]
            total_bridge_themes = sum(len(theme_map) for theme_map in inheritance_maps)
            print(f"   Total themes with bridges: {total_bridge_themes}")


def analyze_relationship_structure(kg_data):
    """Analyze the structure and types of relationships."""
    print("\n🔗 RELATIONSHIP STRUCTURE ANALYSIS")
    print("=" * 50)

    relationships = kg_data.get('relationships', [])

    # Count relationships by type
    rel_type_counts = Counter(rel.get('type', 'UNKNOWN') for rel in relationships)
    print(f"Relationship type distribution:")
    for rel_type, count in sorted(rel_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {rel_type}: {count:,}")

    # Count by granularity type
    granularity_counts = Counter(rel.get('granularity_type', 'UNKNOWN') for rel in relationships)
    print(f"\nGranularity type distribution:")
    for gran_type, count in sorted(granularity_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {gran_type}: {count:,}")

    # Look for theme-related relationships
    theme_relationships = [rel for rel in relationships if 'theme' in rel.get('type', '').lower()]
    print(f"\nTheme-related relationships found: {len(theme_relationships)}")

    if theme_relationships:
        theme_rel_types = Counter(rel.get('type') for rel in theme_relationships)
        print(f"   Theme relationship types: {dict(theme_rel_types)}")


def analyze_theme_bridge_implementation(kg_data):
    """Deep dive into how theme bridges are actually implemented."""
    print("\n🌉 THEME BRIDGE IMPLEMENTATION ANALYSIS")
    print("=" * 50)

    nodes = kg_data.get('nodes', [])

    # Find nodes with theme inheritance
    nodes_with_inheritance = []
    for node in nodes:
        properties = node.get('properties', {})
        inherited_themes = properties.get('inherited_themes', [])
        theme_inheritance_map = properties.get('theme_inheritance_map', {})

        if inherited_themes or theme_inheritance_map:
            nodes_with_inheritance.append({
                'node_id': node.get('id'),
                'node_type': node.get('type'),
                'inherited_themes': inherited_themes,
                'theme_inheritance_map': theme_inheritance_map,
                'source_article': properties.get('source_article', 'Unknown')
            })

    print(f"Nodes with theme inheritance: {len(nodes_with_inheritance)}")

    if nodes_with_inheritance:
        # Analyze inheritance patterns
        total_inherited_themes = sum(len(node['inherited_themes']) for node in nodes_with_inheritance)
        total_bridge_mappings = sum(len(node['theme_inheritance_map']) for node in nodes_with_inheritance)

        print(f"Total inherited themes across all nodes: {total_inherited_themes}")
        print(f"Total theme bridge mappings: {total_bridge_mappings}")

        # Show sample theme bridge
        sample_node = nodes_with_inheritance[0]
        print(f"\nSample theme inheritance structure:")
        print(f"   Node: {sample_node['node_id']} ({sample_node['node_type']})")
        print(f"   Source article: {sample_node['source_article']}")

        if sample_node['inherited_themes']:
            print(f"   Inherited themes ({len(sample_node['inherited_themes'])}):")
            for inherited in sample_node['inherited_themes'][:3]:  # Show first 3
                print(f"      • {inherited}")

        if sample_node['theme_inheritance_map']:
            print(f"   Theme inheritance map ({len(sample_node['theme_inheritance_map'])} mappings):")
            for direct_theme, bridges in list(sample_node['theme_inheritance_map'].items())[:2]:
                print(f"      {direct_theme} → {bridges}")


def analyze_connection_pathways(kg_data):
    """Analyze actual connection pathways available for traversal."""
    print("\n🧭 CONNECTION PATHWAY ANALYSIS")
    print("=" * 50)

    nodes = kg_data.get('nodes', [])
    relationships = kg_data.get('relationships', [])

    # Build connection index
    connections_by_source = defaultdict(list)
    connections_by_target = defaultdict(list)

    for rel in relationships:
        source = rel.get('source')
        target = rel.get('target')
        rel_type = rel.get('type')

        connections_by_source[source].append({
            'target': target,
            'type': rel_type,
            'weight': rel.get('weight', 1.0)
        })

        connections_by_target[target].append({
            'source': source,
            'type': rel_type,
            'weight': rel.get('weight', 1.0)
        })

    # Analyze connectivity
    node_connectivity = []
    for node in nodes[:100]:  # Sample first 100 nodes
        node_id = node.get('id')
        outgoing = len(connections_by_source.get(node_id, []))
        incoming = len(connections_by_target.get(node_id, []))
        total_connections = outgoing + incoming

        node_connectivity.append({
            'node_id': node_id,
            'node_type': node.get('type'),
            'outgoing': outgoing,
            'incoming': incoming,
            'total': total_connections
        })

    # Sort by connectivity
    node_connectivity.sort(key=lambda x: x['total'], reverse=True)

    print(f"Most connected nodes (sample of first 100):")
    for i, node_info in enumerate(node_connectivity[:5]):
        print(f"   {i + 1}. {node_info['node_type']} {node_info['node_id'][:20]}...")
        print(
            f"      Outgoing: {node_info['outgoing']}, Incoming: {node_info['incoming']}, Total: {node_info['total']}")

    # Analyze connection types for highly connected nodes
    highly_connected = node_connectivity[0]
    node_id = highly_connected['node_id']

    print(f"\nConnection type breakdown for most connected node:")
    outgoing_types = Counter(conn['type'] for conn in connections_by_source.get(node_id, []))
    incoming_types = Counter(conn['type'] for conn in connections_by_target.get(node_id, []))

    print(f"   Outgoing connections: {dict(outgoing_types)}")
    print(f"   Incoming connections: {dict(incoming_types)}")


def analyze_raw_vs_theme_connections(kg_data):
    """Determine the actual ratio of raw similarity vs theme-based connections."""
    print("\n⚖️  RAW SIMILARITY vs THEME-BASED CONNECTION ANALYSIS")
    print("=" * 60)

    relationships = kg_data.get('relationships', [])

    # Categorize relationships
    raw_similarity_types = [
        'chunk_to_chunk_intra', 'chunk_to_chunk_inter',
        'sentence_to_sentence_semantic', 'sentence_to_chunk',
        'chunk_to_doc', 'doc_to_doc'
    ]

    theme_based_types = [
        'theme_bridge', 'theme_similarity', 'cross_document_theme'
    ]

    structural_types = [
        'parent', 'child', 'contains', 'hierarchical'
    ]

    sequential_types = [
        'sentence_to_sentence_sequential', 'sequential_flow'
    ]

    raw_similarity_count = 0
    theme_based_count = 0
    structural_count = 0
    sequential_count = 0
    other_count = 0

    for rel in relationships:
        rel_type = rel.get('type', '')

        if rel_type in raw_similarity_types:
            raw_similarity_count += 1
        elif rel_type in theme_based_types or 'theme' in rel_type.lower():
            theme_based_count += 1
        elif rel_type in structural_types or rel_type in ['parent', 'child']:
            structural_count += 1
        elif rel_type in sequential_types or 'sequential' in rel_type.lower():
            sequential_count += 1
        else:
            other_count += 1

    total_relationships = len(relationships)

    print(f"Connection type breakdown:")
    print(
        f"   Raw cosine similarity: {raw_similarity_count:,} ({raw_similarity_count / total_relationships * 100:.1f}%)")
    print(f"   Theme-based: {theme_based_count:,} ({theme_based_count / total_relationships * 100:.1f}%)")
    print(f"   Structural/Hierarchical: {structural_count:,} ({structural_count / total_relationships * 100:.1f}%)")
    print(f"   Sequential/Narrative: {sequential_count:,} ({sequential_count / total_relationships * 100:.1f}%)")
    print(f"   Other: {other_count:,} ({other_count / total_relationships * 100:.1f}%)")

    print(f"\n🎯 KEY FINDING:")
    if raw_similarity_count > theme_based_count * 10:
        print(f"   Your knowledge graph is OVERWHELMINGLY raw similarity-based!")
        print(
            f"   Raw connections outnumber theme connections by {raw_similarity_count // max(theme_based_count, 1)}:1")
        print(f"   Theme bridges appear to work through NODE PROPERTIES, not explicit edges")
    else:
        print(f"   Your knowledge graph has significant theme-based connections")


def main():
    """Main diagnostic function."""
    print("🔍 Knowledge Graph Architecture Diagnostic")
    print("🧬 Analyzing the true structure of your semantic lattice")
    print("=" * 80)

    # Load knowledge graph
    kg_data = load_knowledge_graph()
    if not kg_data:
        return

    # Run all analyses
    analyze_node_structure(kg_data)
    analyze_relationship_structure(kg_data)
    analyze_theme_bridge_implementation(kg_data)
    analyze_connection_pathways(kg_data)
    analyze_raw_vs_theme_connections(kg_data)

    print("\n" + "=" * 80)
    print("🎯 DIAGNOSTIC COMPLETE")
    print("\nKey Questions Answered:")
    print("1. Are theme bridges explicit edges or implicit node properties?")
    print("2. What's the true ratio of raw similarity vs theme-based connections?")
    print("3. How should question generation be rebalanced?")
    print("\nReview the analysis above to understand your architecture! 🧠")


if __name__ == "__main__":
    main()