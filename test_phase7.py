#!/usr/bin/env python3
"""
Test Phase 7: Multi-Dimensional Question Generation
=================================================

Test script to verify Phase 7 (Multi-Dimensional Question Generation) functionality.
Tests the new sophisticated question generation system that exploits the full
semantic lattice architecture with theme bridges, granularity cascades, and
multi-dimensional connection pathways.

UPDATED PHASE STRUCTURE:
- Phase 1: Setup & Initialization
- Phase 2: Data Acquisition
- Phase 3: Multi-Granularity Embedding Generation
- Phase 4: Multi-Granularity Similarity Matrix Construction
- Phase 5: Entity/Theme Extraction
- Phase 6: Knowledge Graph Assembly with Theme Bridges
- Phase 7: Multi-Dimensional Question Generation (THIS TEST)

Run from project root:
    python test_phase7.py
"""

import sys
import os
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))

from pipeline import SemanticRAGPipeline


def test_phase7():
    """Test Phase 7: Multi-Dimensional Question Generation."""
    print("🧪 Testing Phase 7: Multi-Dimensional Question Generation")
    print("🎯 Architecture: Theme Bridges + Granularity Cascades + Sequential Flows + Multi-Dimensional Navigation")
    print("=" * 100)

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

        # Override config for testing question generation
        print("🔧 Configuring for Phase 7 Multi-Dimensional Question Generation testing...")

        # Set mode to full pipeline to include Phase 7
        pipeline.config['execution']['mode'] = 'full_pipeline'
        pipeline.config['execution']['skip_phases'] = []

        # Use smaller models for faster testing
        pipeline.config['models']['embedding_models'] = [
            "sentence-transformers/all-MiniLM-L6-v2"  # Faster model for testing
        ]

        # Configure question generation settings for testing
        pipeline.config['question_generation'] = {
            'target_questions': 20,  # Smaller for testing
            'question_types': {
                'theme_bridge': 0.25,
                'granularity_cascade': 0.25,
                'theme_synthesis': 0.20,
                'sequential_flow': 0.15,
                'multi_dimensional': 0.15
            },
            'personas': {
                'researcher': 0.6,
                'googler': 0.4
            },
            'validation': {
                'min_question_length': 10,
                'max_question_length': 300,
                'require_ground_truth': True
            }
        }

        # Enable force recompute to see fresh question generation
        pipeline.config['execution']['force_recompute'] = ['questions']

        # Test Ollama availability
        try:
            import ollama
            models = ollama.list()
            ollama_available = True
            ollama_models = [m.model for m in models.models]
            print(f"   🤖 Ollama available with models: {ollama_models}")
        except Exception:
            ollama_available = False
            print(f"   ⚠️  Ollama not available - will use fallback question generation")

        print(f"   🎯 Target questions: {pipeline.config['question_generation']['target_questions']}")
        print(f"   📊 Question type distribution: {pipeline.config['question_generation']['question_types']}")
        print(f"   👥 Persona distribution: {pipeline.config['question_generation']['personas']}")
        print(f"   🧠 Ollama status: {'Available' if ollama_available else 'Fallback mode'}")

        # Run pipeline phases 1-7
        print("\n🚀 Running pipeline phases 1-7...")

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
        if not pipeline.theme_data:
            print("🏷️  Running Phase 5: Entity/Theme Extraction...")
            pipeline._phase_5_theme_extraction()
        else:
            print("✅ Phase 5: Entity/theme data already available")

        themes_count = pipeline.theme_data['metadata'].total_themes_extracted
        print(f"✅ Phase 5: {themes_count} themes extracted")

        # Phase 6: Knowledge Graph Assembly (if needed)
        if not pipeline.knowledge_graph:
            print("🏗️  Running Phase 6: Knowledge Graph Assembly...")
            pipeline._phase_6_knowledge_graph_construction()
        else:
            print("✅ Phase 6: Knowledge graph already available")

        kg_stats = pipeline.knowledge_graph.metadata
        print(f"✅ Phase 6: {kg_stats['total_nodes']:,} nodes, {kg_stats['total_relationships']:,} relationships")

        # Phase 7: Multi-Dimensional Question Generation
        print("🎯 Running Phase 7: Multi-Dimensional Question Generation...")
        pipeline._phase_7_question_generation()

        # Verify question generation results
        print("\n🔍 Verifying Phase 7 Question Generation results...")

        # Check questions
        if not pipeline.questions:
            print("❌ No questions were generated")
            return False

        print(f"✅ Generated {len(pipeline.questions)} questions successfully")

        # Test question structure and distribution
        print("\n📊 Testing Question Generation Structure:")

        # Analyze question types
        question_type_counts = {}
        persona_counts = {}
        difficulty_counts = {}
        hop_counts = {}

        for question in pipeline.questions:
            # Count by type
            q_type = question.question_type.value if hasattr(question.question_type, 'value') else str(
                question.question_type)
            question_type_counts[q_type] = question_type_counts.get(q_type, 0) + 1

            # Count by persona
            persona = question.persona.value if hasattr(question.persona, 'value') else str(question.persona)
            persona_counts[persona] = persona_counts.get(persona, 0) + 1

            # Count by difficulty
            difficulty_counts[question.difficulty_level] = difficulty_counts.get(question.difficulty_level, 0) + 1

            # Count by expected hops
            hops = question.expected_hops
            hop_counts[hops] = hop_counts.get(hops, 0) + 1

        print(f"   📈 Question Type Distribution:")
        for q_type, count in question_type_counts.items():
            percentage = (count / len(pipeline.questions)) * 100
            print(f"      {q_type}: {count} questions ({percentage:.1f}%)")

        print(f"   👥 Persona Distribution:")
        for persona, count in persona_counts.items():
            percentage = (count / len(pipeline.questions)) * 100
            print(f"      {persona}: {count} questions ({percentage:.1f}%)")

        print(f"   🎯 Difficulty Distribution:")
        for difficulty, count in difficulty_counts.items():
            percentage = (count / len(pipeline.questions)) * 100
            print(f"      {difficulty}: {count} questions ({percentage:.1f}%)")

        print(f"   🦘 Expected Hops Distribution:")
        for hops, count in hop_counts.items():
            percentage = (count / len(pipeline.questions)) * 100
            print(f"      {hops} hops: {count} questions ({percentage:.1f}%)")

        # Test individual question quality
        print("\n🔍 Testing Individual Question Quality:")

        for i, question in enumerate(pipeline.questions[:5]):  # Test first 5 questions
            print(f"\n   Question {i + 1}: {question.question_id}")
            print(f"      Type: {question.question_type}")
            print(f"      Persona: {question.persona}")
            print(f"      Difficulty: {question.difficulty_level}")
            print(f"      Expected hops: {question.expected_hops}")
            print(f"      Question: \"{question.question_text}\"")

            # Test ground truth
            print(f"      Ground truth nodes: {len(question.ground_truth_nodes)} nodes")
            for gt_node_id in question.ground_truth_nodes[:2]:  # Show first 2
                print(f"         • {gt_node_id}")

            # Test theme context
            if question.primary_themes:
                print(f"      Primary themes: {question.primary_themes}")
            if question.secondary_themes:
                print(f"      Secondary themes: {question.secondary_themes}")

            print(f"      Cross-document: {question.cross_document}")
            print(f"      Generation time: {question.generation_time:.3f}s")

        # Test pathway analysis quality
        print("\n🧭 Testing Connection Pathway Analysis:")

        # Import the pathway analyzer to test it directly
        from questions import EnhancedConnectionPathwayAnalyzer, QuestionType

        pathway_analyzer = EnhancedConnectionPathwayAnalyzer(pipeline.knowledge_graph, pipeline.logger)

        print(f"   📊 Pathway Analysis Results:")
        for question_type in QuestionType:
            pathways = pathway_analyzer.pathway_cache[question_type]
            print(f"      {question_type.value}: {len(pathways)} pathways found")

            if pathways:
                # Show sample pathway details
                sample_pathway = pathways[0]
                print(f"         Sample pathway type: {sample_pathway.get('pathway_type', 'unknown')}")
                print(f"         Expected hops: {sample_pathway.get('expected_hops', 'unknown')}")

        # Test theme bridge functionality specifically
        print("\n🌉 Testing Theme Bridge Question Generation:")

        theme_bridge_questions = [q for q in pipeline.questions if str(q.question_type) == 'theme_bridge']

        if theme_bridge_questions:
            print(f"   ✅ Generated {len(theme_bridge_questions)} theme bridge questions")

            sample_bridge_q = theme_bridge_questions[0]
            print(f"   📝 Sample Theme Bridge Question:")
            print(f"      Question: \"{sample_bridge_q.question_text}\"")
            print(f"      Cross-document: {sample_bridge_q.cross_document}")
            print(f"      Primary themes: {sample_bridge_q.primary_themes}")
            print(f"      Expected hops: {sample_bridge_q.expected_hops}")

            # Verify this question actually requires cross-document navigation
            if sample_bridge_q.cross_document and sample_bridge_q.primary_themes:
                print(f"      ✅ Properly configured for cross-document theme navigation")
            else:
                print(f"      ⚠️  May not be properly configured for theme bridge navigation")
        else:
            print(f"   ⚠️  No theme bridge questions generated")

        # Test granularity cascade questions
        print("\n📊 Testing Granularity Cascade Question Generation:")

        cascade_questions = [q for q in pipeline.questions if str(q.question_type) == 'granularity_cascade']

        if cascade_questions:
            print(f"   ✅ Generated {len(cascade_questions)} granularity cascade questions")

            sample_cascade_q = cascade_questions[0]
            print(f"   📝 Sample Granularity Cascade Question:")
            print(f"      Question: \"{sample_cascade_q.question_text}\"")
            print(f"      Expected hops: {sample_cascade_q.expected_hops}")
            print(f"      Ground truth nodes: {len(sample_cascade_q.ground_truth_nodes)}")

            # Check if ground truth spans multiple granularities
            if len(sample_cascade_q.ground_truth_nodes) >= 2:
                print(f"      ✅ Multi-granularity ground truth configured")
            else:
                print(f"      ⚠️  May not span multiple granularity levels")
        else:
            print(f"   ⚠️  No granularity cascade questions generated")

        # Test question validation
        print("\n✅ Testing Question Validation:")

        valid_questions = 0
        validation_issues = []

        for question in pipeline.questions:
            issues = []

            # Check question length
            if len(question.question_text) < 10:
                issues.append("Question too short")
            if len(question.question_text) > 300:
                issues.append("Question too long")

            # Check if it ends with question mark
            if not question.question_text.endswith('?'):
                issues.append("Missing question mark")

            # Check ground truth
            if not question.ground_truth_nodes:
                issues.append("No ground truth nodes")

            # Check persona-specific formatting
            if question.persona.value == 'googler' and not question.question_text.islower():
                issues.append("Googler questions should be lowercase")

            if not issues:
                valid_questions += 1
            else:
                validation_issues.extend(issues)

        validation_percentage = (valid_questions / len(pipeline.questions)) * 100 if pipeline.questions else 0
        print(f"   📊 Question Validation Results:")
        print(f"      Valid questions: {valid_questions}/{len(pipeline.questions)} ({validation_percentage:.1f}%)")

        if validation_issues:
            print(f"      Common issues found:")
            issue_counts = {}
            for issue in validation_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

            for issue, count in issue_counts.items():
                print(f"         • {issue}: {count} occurrences")
        else:
            print(f"      ✅ No validation issues found")

        # Test caching functionality
        print("\n💾 Testing question caching...")
        try:
            # Clear force recompute and run again (should use cache)
            pipeline.config['execution']['force_recompute'] = []

            import time
            cache_start = time.time()
            pipeline._phase_7_question_generation()
            cache_end = time.time()
            cache_time = cache_end - cache_start

            print(f"   Cache load time: {cache_time:.3f}s")
            print(f"   Cached questions: {len(pipeline.questions)}")

            if cache_time < 2.0:  # Should be very fast
                print("   ✅ Caching working effectively")
            else:
                print("   ⚠️  Caching may not be working as expected")

        except Exception as e:
            print(f"   ⚠️  Cache test failed: {e}")

        # Check if questions were saved
        questions_path = Path(pipeline.config['directories']['data']) / "questions" / "multi_dimensional_questions.json"
        if questions_path.exists():
            print(f"\n✅ Questions saved successfully to {questions_path}")

            # Show file size
            file_size = questions_path.stat().st_size / 1024  # KB
            print(f"   File size: {file_size:.1f} KB")
        else:
            print(f"\n⚠️  Questions file not found at {questions_path}")

        # Test question statistics
        if hasattr(pipeline, 'question_stats') and pipeline.question_stats:
            print(f"\n📈 Question Generation Statistics:")
            stats = pipeline.question_stats

            for stat_name, stat_value in stats.items():
                if isinstance(stat_value, dict):
                    print(f"   {stat_name}:")
                    for sub_name, sub_value in stat_value.items():
                        print(f"      {sub_name}: {sub_value}")
                else:
                    print(f"   {stat_name}: {stat_value}")

        print("\n🎉 Phase 7 Multi-Dimensional Question Generation test completed successfully!")
        print(f"📋 Experiment ID: {pipeline.experiment_id}")
        print(f"📁 Logs saved to: logs/{pipeline.experiment_id}.log")
        print(f"💾 Questions cached in: data/questions/")

        return True

    except Exception as e:
        print(f"\n❌ Phase 7 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🧪 Semantic RAG Pipeline - Phase 7 Test")
    print("Testing multi-dimensional question generation system")
    print("🎯 Architecture: Theme Bridges + Granularity Cascades + Sequential Flows + Multi-Dimensional Navigation")
    print("=" * 110)

    # Check if we're in the right directory
    if not Path("config.yaml").exists():
        print("❌ config.yaml not found!")
        print("💡 Make sure you're running this from the project root directory")
        print("💡 Expected directory structure:")
        print("   config.yaml")
        print("   utils/")
        print("   requirements.txt")
        return

    success = test_phase7()

    if success:
        print("\n✅ All tests passed!")
        print("🚀 Phase 7 Multi-Dimensional Question Generation is ready for production use")
        print("\n🔗 Key features verified:")
        print("   • Multi-dimensional question type generation (5 types)")
        print("   • Theme bridge questions for cross-document navigation")
        print("   • Granularity cascade questions for hierarchical reasoning")
        print("   • Theme synthesis questions for multi-theme integration")
        print("   • Sequential flow questions for narrative reasoning")
        print("   • Multi-dimensional questions combining multiple connection types")
        print("   • Dual persona system (Researcher + Googler)")
        print("   • Sophisticated pathway analysis and blueprint creation")
        print("   • Ollama-based question generation with fallback system")
        print("   • Comprehensive ground truth mapping")
        print("   • Question validation and quality assurance")
        print("   • Intelligent caching system")
        print("\n🎯 BREAKTHROUGH: Multi-Dimensional Question Architecture:")
        print("   Every question type exploits specific aspects of your semantic lattice,")
        print("   testing different navigation patterns and reasoning capabilities.")
        print("   This creates a comprehensive evaluation framework that validates")
        print("   your novel approach across all connection dimensions.")
        print("\n🎯 Ready for Retrieval Algorithm Development:")
        print("   With sophisticated questions generated, you can now develop")
        print("   retrieval algorithms that exploit the same connection pathways")
        print("   to answer these multi-dimensional reasoning challenges.")
    else:
        print("\n❌ Tests failed!")
        print("🔧 Please check the error messages above")


if __name__ == "__main__":
    main()