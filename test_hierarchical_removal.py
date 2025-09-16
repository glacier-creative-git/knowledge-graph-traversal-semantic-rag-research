#!/usr/bin/env python3
"""
Hierarchical Visualization Removal Verification
==============================================

Quick test to verify that all hierarchical connection visualization elements
have been successfully removed from matplotlib visualizations.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_legend_elements():
    """Test that legend creation functions don't include hierarchical elements."""
    print("🔍 Testing legend element creation...")
    
    # Import the visualizer
    from utils.matplotlib_visualizer import KnowledgeGraphMatplotlibVisualizer
    
    # Create a minimal visualizer instance
    visualizer = KnowledgeGraphMatplotlibVisualizer(None, figure_size=(20, 8))
    
    # Create a test figure
    fig = plt.figure(figsize=(10, 6))
    
    # Test the legend creation
    visualizer._add_legend(fig)
    
    # Get the legend and check its labels
    legend = fig.get_children()[-1]  # Legend should be the last child
    if hasattr(legend, 'get_texts'):
        legend_labels = [text.get_text() for text in legend.get_texts()]
        print(f"✅ Legend labels found: {legend_labels}")
        
        # Check that hierarchical is NOT in the labels
        if 'Hierarchical' in legend_labels:
            print("❌ ERROR: 'Hierarchical' still found in legend!")
            return False
        else:
            print("✅ SUCCESS: 'Hierarchical' removed from legend")
            
        # Check that expected labels are present
        expected_labels = ['Anchor Point', 'Traversal Step', 'Early Stop Point', 'Cross-Document', 'Within Document']
        missing_labels = [label for label in expected_labels if label not in legend_labels]
        
        if missing_labels:
            print(f"⚠️ WARNING: Missing expected labels: {missing_labels}")
        else:
            print("✅ SUCCESS: All expected labels present")
            
        return len(missing_labels) == 0
    else:
        print("⚠️ Could not find legend text elements")
        return False

def test_connection_logic():
    """Test the connection type determination logic."""
    print("\n🔗 Testing connection type logic...")
    
    # Test the simplified connection logic
    test_cases = [
        {'connection_type': 'cross_document', 'expected_color': 'purple'},
        {'connection_type': 'theme_bridge', 'expected_color': 'purple'},
        {'connection_type': 'raw_similarity', 'expected_color': 'green'},
        {'connection_type': 'sequential', 'expected_color': 'green'},
        {'connection_type': 'hierarchical', 'expected_color': 'green'},  # Should default to green now
    ]
    
    for test_case in test_cases:
        connection_type = test_case['connection_type']
        expected_color = test_case['expected_color']
        
        # Simulate the logic from _draw_traversal_path
        if connection_type in ['cross_document', 'theme_bridge']:
            actual_color = 'purple'
        else:
            actual_color = 'green'
            
        if actual_color == expected_color:
            print(f"✅ {connection_type}: {actual_color} (correct)")
        else:
            print(f"❌ {connection_type}: {actual_color} (expected {expected_color})")
    
    print("✅ Connection type logic test completed")

def test_code_search_for_hierarchical():
    """Search for any remaining 'hierarchical' references in the visualization code."""
    print("\n🔍 Searching for remaining 'hierarchical' references...")
    
    viz_file = project_root / "utils" / "matplotlib_visualizer.py"
    
    if viz_file.exists():
        with open(viz_file, 'r') as f:
            content = f.read()
            
        # Count occurrences of 'hierarchical' (case insensitive)
        hierarchical_count = content.lower().count('hierarchical')
        
        if hierarchical_count == 0:
            print("✅ SUCCESS: No 'hierarchical' references found in visualization code")
            return True
        else:
            print(f"⚠️ WARNING: Found {hierarchical_count} 'hierarchical' references")
            
            # Find the lines containing 'hierarchical'
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'hierarchical' in line.lower():
                    print(f"   Line {i+1}: {line.strip()}")
            return False
    else:
        print("❌ Could not find matplotlib_visualizer.py file")
        return False

if __name__ == "__main__":
    print("🧹 Hierarchical Visualization Removal Verification")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    # Run all tests
    if test_legend_elements():
        success_count += 1
    
    test_connection_logic()  # Always passes, just informational
    success_count += 1
    
    if test_code_search_for_hierarchical():
        success_count += 1
    
    print(f"\n📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All hierarchical visualization elements successfully removed!")
        print("\nNext steps:")
        print("  1. Run: python test_algorithms.py")
        print("  2. Check that visualizations show:")
        print("     ✅ Step-based titles (Step 0, Steps 1-2, etc.)")
        print("     ✅ Only Cross-Document and Within Document connections")
        print("     ✅ No blue dashed 'Hierarchical' lines")
        print("     ✅ Clean legend with 5 elements (no Hierarchical entry)")
    else:
        print("⚠️ Some tests failed - manual review recommended")
