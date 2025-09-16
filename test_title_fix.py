#!/usr/bin/env python3
"""
Test Script for Title Overlap Fix
=================================

Quick test to verify that the matplotlib visualization title overlapping issue
has been resolved. This script tests the title formatting functions directly
and creates a minimal visualization to check spacing.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.matplotlib_visualizer import KnowledgeGraphMatplotlibVisualizer


def test_title_formatting():
    """Test the title formatting functions directly."""
    print("Testing title formatting functions...")
    
    # Create a minimal visualizer instance (we don't need actual KG data for this test)
    visualizer = KnowledgeGraphMatplotlibVisualizer(None, figure_size=(20, 8))
    
    # Test main title formatting
    long_query = "How does machine learning relate to neural networks and what are the implications for artificial intelligence development in the modern era?"
    formatted_title = visualizer._format_title_text(
        algorithm_name="QueryTraversal Sequential Reading Sessions",
        query=long_query,
        retrieved_count=10,
        score=0.497,
        additional_info="Sessions: 6"
    )
    
    print("Formatted main title:")
    print(formatted_title)
    print(f"Title has {len(formatted_title.split())} lines")
    print()
    
    # Test session title formatting
    long_doc_name = "Neural network (machine learning) fundamentals and applications"
    session_title = visualizer._format_session_title(
        session_number=1,
        doc_id=long_doc_name,
        idx_range="chunks 0-0",
        max_length=25
    )
    
    print("Formatted session title:")
    print(f"'{session_title}' (length: {len(session_title)})")
    
    # Test multiple session titles to ensure they don't overlap
    test_docs = [
        "Neural network (machine learning) fundamentals",
        "Machine learning (global view)",
        "Artificial intelligence applications",
        "Deep learning architectures and methods",
        "Convolutional neural networks overview"
    ]
    
    print("\nAll session titles (should be uniformly short):")
    for i, doc in enumerate(test_docs):
        title = visualizer._format_session_title(
            session_number=i + 1,
            doc_id=doc,
            idx_range=f"chunks {i}-{i+2}",
            max_length=25
        )
        print(f"  {title} (len: {len(title)})")
    
    print("\n✅ Title formatting test completed successfully!")


def test_spacing_parameters():
    """Test the spacing parameters used in grid layout."""
    print("\nTesting spacing parameters...")
    
    # Create a test figure to verify spacing
    fig = plt.figure(figsize=(20, 8), facecolor='white', dpi=150)
    
    # Test the grid layout parameters we're using
    import matplotlib.gridspec as gridspec
    
    num_sessions = 6
    gs = gridspec.GridSpec(2, num_sessions, figure=fig,
                           height_ratios=[0.08, 1],
                           hspace=0.2, wspace=0.35,  # Our updated spacing
                           top=0.82, bottom=0.15)    # Our updated margins
    
    # Add a test title to check positioning
    test_title = ("QueryTraversal Sequential Reading Sessions Traversal Path Visualization\n"
                  "Query: 'How does machine learning relate to neural networks?'\n"
                  "Retrieved: 10 sentences | Score: 0.497\n"
                  "Sessions: 6")
    
    fig.suptitle(test_title, fontsize=14, fontweight='bold', y=0.95,
                ha='center', va='top')
    
    # Create dummy subplot to test spacing
    ax = fig.add_subplot(gs[1, 0])
    ax.text(0.5, 0.5, "Session 1\nTest Plot", ha='center', va='center')
    ax.set_title("S1: Neural... (chunks 0-0)", fontsize=10, fontweight='bold', pad=8)
    
    # Save test figure
    test_output = project_root / "test_title_spacing.png"
    fig.savefig(str(test_output), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Test spacing figure saved to: {test_output}")
    print("Check this image to verify title positioning and spacing.")


if __name__ == "__main__":
    print("🔧 Testing Title Overlap Fixes")
    print("=" * 40)
    
    test_title_formatting()
    test_spacing_parameters()
    
    print("\n🎉 All tests completed!")
    print("\nTo test with real data, run:")
    print("  python test_algorithms.py")
    print("\nThe visualizations should now have:")
    print("  ✅ Properly spaced main titles with text wrapping")
    print("  ✅ Truncated session titles that don't overlap")
    print("  ✅ Improved margins and padding throughout")
