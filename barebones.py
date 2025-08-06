#!/usr/bin/env python3
"""
Barebones WikiEval Demo
======================

Minimal script to load WikiEval, run semantic graph traversal on one question,
and create both 2D and 3D visualizations.
"""

import sys
import os

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
from utils import SemanticGraphRAG, create_2d_visualization, create_3d_visualization


def main():
    print("🚀 WikiEval Demo - Semantic Graph Traversal")
    print("=" * 50)

    # Load WikiEval dataset
    print("📚 Loading WikiEval...")
    dataset = load_dataset("explodinggradients/WikiEval")

    # Get first example
    split_name = list(dataset.keys())[0]
    example = dataset[split_name][0]

    print(f"📊 Dataset loaded: {len(dataset[split_name])} examples")
    print(f"🔍 Using first example from '{split_name}' split")

    # Extract question and context from WikiEval format
    # WikiEval stores actual Wikipedia content in context_v1/context_v2 (as lists)
    question = example.get('question', example.get('query', ''))

    # Get the actual Wikipedia content (not just the title)
    context = None
    if 'context_v1' in example and example['context_v1']:
        context = example['context_v1'][0]  # Get first item from list
        print("📄 Using context_v1 for Wikipedia content")
    elif 'context_v2' in example and example['context_v2']:
        context = example['context_v2'][0]  # Get first item from list
        print("📄 Using context_v2 for Wikipedia content")
    else:
        # Fallback to other possible fields
        context = example.get('context', example.get('passage', ''))
        if isinstance(context, list) and context:
            context = context[0]

    if not question or not context or len(context) < 100:
        print("❌ Could not find adequate question/context in example")
        print(f"Available fields: {list(example.keys())}")
        print(f"Question length: {len(question) if question else 0}")
        print(f"Context length: {len(context) if context else 0}")

        # Try to find the longest text field as fallback
        if len(context) < 100:
            print("🔍 Searching for longer content...")
            longest_content = ""
            for key, value in example.items():
                if isinstance(value, str) and len(value) > len(longest_content):
                    longest_content = value
                elif isinstance(value, list) and value and isinstance(value[0], str):
                    if len(value[0]) > len(longest_content):
                        longest_content = value[0]

            if len(longest_content) > 100:
                context = longest_content
                print(f"📄 Using longest field with {len(context)} characters")
            else:
                return

    print(f"❓ Question: {question}")
    print(f"📄 Context: {len(context)} chars, ~{len(context.split('.'))} sentences")

    # Prepare context for RAG system
    contexts = [{
        'context': context,
        'question': question,
        'id': 'wikieval_demo',
        'title': 'WikiEval Document'
    }]

    # Create RAG system
    print("\n🧠 Creating RAG system...")
    rag_system = SemanticGraphRAG(
        top_k_per_sentence=25,
        cross_doc_k=15,
        similarity_threshold=0.4,
        use_sliding_window=True
    )

    # Ingest and retrieve
    print("📚 Ingesting context...")
    rag_system.ingest_contexts(contexts)

    print("🔍 Running traversal...")
    retrieved_texts, traversal_steps, analysis = rag_system.retrieve(question, top_k=8)

    print(f"✅ Found {len(retrieved_texts)} results via {len(traversal_steps)} traversal steps")

    # 2D Visualization
    print("\n🎨 Creating 2D visualization...")
    fig_2d = create_2d_visualization(rag_system, question, traversal_steps,
                                     save_path="wikieval_2d.png")
    print("💾 Saved: wikieval_2d.png")

    # 3D Visualization
    print("🎨 Creating 3D visualization...")
    fig_3d = create_3d_visualization(question, traversal_steps, method="pca")
    fig_3d.write_html("wikieval_3d.html")
    print("💾 Saved: wikieval_3d.html")

    try:
        fig_3d.show()
    except:
        print("   (Open wikieval_3d.html in browser)")

    print(f"\n✅ Demo complete! Check the visualization files.")
    print(f"   Cross-document rate: {analysis['cross_document_rate']:.1f}%")


if __name__ == "__main__":
    main()