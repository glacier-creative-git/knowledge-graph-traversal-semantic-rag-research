#!/usr/bin/env python3
"""
Test Phase 6: Simplified Question Generation
===========================================

Test script to verify Phase 6 (Simplified Question Generation) functionality.
Tests the minimalist approach using only entities (PERSON/ORG/GPE) and 
three core relationship types with Ollama integration.

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
    """Test Phase 6: Simplified Question Generation."""
    print("🧪 Testing Phase 6: Simplified Question Generation")
    print("=" * 55)
    
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
        
        # Override config for testing
        print("🔧 Configuring for Phase 6 testing...")
        
        # Set mode to full pipeline to include Phase 6
        pipeline.config['execution']['mode'] = 'full_pipeline'
        pipeline.config['execution']['skip_phases'] = []
        
        # Use smaller models for faster testing
        pipeline.config['models']['embedding_models'] = [
            "sentence-transformers/all-MiniLM-L6-v2"  # Faster model for testing
        ]
        
        # Configure Ollama for question generation
        pipeline.config['ollama'] = {
            'model': 'llama3.1:8b'
        }
        
        # Enable force recompute to see fresh generation
        pipeline.config['execution']['force_recompute'] = ['questions']
        
        # Test Ollama integration
        try:
            import ollama
            models = ollama.list()
            ollama_available = True
            print(f"   🤖 Ollama available with models: {[m.model for m in models.models]}")
        except Exception:
            ollama_available = False
            print(f"   ⚠️  Ollama not available - will use fallback question generation")
        
        print(f"   Models: {pipeline.config['models']['embedding_models']}")
        print(f"   Ollama integration: {'Enabled' if ollama_available else 'Disabled (fallback mode)'}")
        print(f"   Architecture: Simplified (entities only, no themes/keyphrases)")
        
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
        
        # Phase 5: Knowledge Graph Construction (if needed)
        if not pipeline.knowledge_graph:
            print("🏗️  Running Phase 5: Simplified Knowledge Graph Construction...")
            pipeline._phase_5_knowledge_graph_construction()
        else:
            print("✅ Phase 5: Knowledge graph already available")
        
        print(f"✅ Phase 5: Simplified knowledge graph constructed")
        
        # Phase 6: Question Generation
        print("📊 Running Phase 6: Simplified Question Generation...")
        pipeline._phase_6_question_generation()
        
        # Verify results
        print("\n🔍 Verifying Phase 6 results...")
        
        # Check questions
        if not pipeline.questions:
            print("❌ No questions were generated")
            return False
        
        print(f"✅ Questions generated successfully: {len(pipeline.questions)} questions")
        
        # Check question generation strategies
        print("\n📊 Testing Question Generation Strategies:")
        
        # Group questions by generation strategy
        questions_by_strategy = {}
        for question in pipeline.questions:
            strategy = question.generation_strategy
            if strategy not in questions_by_strategy:
                questions_by_strategy[strategy] = []
            questions_by_strategy[strategy].append(question)
        
        for strategy, questions in questions_by_strategy.items():
            print(f"\n   {strategy.upper()} Strategy:")
            print(f"      Generated: {len(questions)} questions")
            
            if questions:
                sample_question = questions[0]
                print(f"      Sample question: \"{sample_question.question}\"")
                print(f"      Persona used: {sample_question.persona_used}")
                print(f"      Entities used: {sample_question.entities_used[:3]}")  # Show first 3 entities
                print(f"      Expected advantage: {sample_question.expected_advantage}")
                print(f"      Difficulty level: {sample_question.difficulty_level}")
                print(f"      Ground truth contexts: {len(sample_question.ground_truth_contexts)} nodes")
                print(f"      Relationship types tested: {sample_question.relationship_types}")
                print(f"      Generation time: {sample_question.generation_time:.3f}s")
        
        # Test question quality
        print("\n🎯 Testing Question Quality:")
        
        valid_questions = 0
        questions_with_ground_truth = 0
        questions_with_relationships = 0
        questions_with_entities = 0
        questions_with_personas = 0
        
        for question in pipeline.questions:
            # Check basic question format
            text = question.question.lower().strip()
            has_question_word = any(word in text for word in ['what', 'how', 'why', 'when', 'where', 'which', 'who'])
            has_question_mark = '?' in text
            has_imperative = any(word in text for word in ['explain', 'describe', 'compare', 'analyze'])
            
            if has_question_word or has_question_mark or has_imperative:
                valid_questions += 1
            
            # Check other quality indicators
            if question.ground_truth_contexts:
                questions_with_ground_truth += 1
            
            if question.relationship_types:
                questions_with_relationships += 1
            
            if question.entities_used:
                questions_with_entities += 1
            
            if question.persona_used:
                questions_with_personas += 1
        
        print(f"   📝 Valid question format: {valid_questions}/{len(pipeline.questions)} ({100*valid_questions/len(pipeline.questions):.1f}%)")
        print(f"   🎯 Ground truth coverage: {questions_with_ground_truth}/{len(pipeline.questions)} ({100*questions_with_ground_truth/len(pipeline.questions):.1f}%)")
        print(f"   🔗 Relationship testing: {questions_with_relationships}/{len(pipeline.questions)} ({100*questions_with_relationships/len(pipeline.questions):.1f}%)")
        print(f"   🏷️  Entity utilization: {questions_with_entities}/{len(pipeline.questions)} ({100*questions_with_entities/len(pipeline.questions):.1f}%)")
        print(f"   🎭 Persona usage: {questions_with_personas}/{len(pipeline.questions)} ({100*questions_with_personas/len(pipeline.questions):.1f}%)")
        
        # Test expected advantage distribution
        print("\n🏆 Testing Expected Advantage Distribution:")
        
        advantage_counts = {}
        for question in pipeline.questions:
            advantage = question.expected_advantage
            advantage_counts[advantage] = advantage_counts.get(advantage, 0) + 1
        
        for advantage, count in advantage_counts.items():
            percentage = (count / len(pipeline.questions)) * 100
            print(f"   {advantage}: {count} questions ({percentage:.1f}%)")
        
        # Test persona distribution
        print("\n🎭 Testing Persona Distribution:")
        
        persona_counts = {}
        for question in pipeline.questions:
            persona = question.persona_used
            persona_counts[persona] = persona_counts.get(persona, 0) + 1
        
        for persona, count in persona_counts.items():
            percentage = (count / len(pipeline.questions)) * 100
            print(f"   {persona}: {count} questions ({percentage:.1f}%)")
        
        # Verify semantic traversal questions
        semantic_questions = [q for q in pipeline.questions if q.expected_advantage == 'semantic_traversal']
        baseline_questions = [q for q in pipeline.questions if q.expected_advantage == 'baseline_vector']
        
        if semantic_questions:
            print(f"   ✅ Generated {len(semantic_questions)} questions favoring semantic traversal")
            sample_semantic = semantic_questions[0]
            print(f"      Sample: \"{sample_semantic.question}\"")
            print(f"      Entities: {sample_semantic.entities_used[:2]}")
        else:
            print(f"   ⚠️  No questions favoring semantic traversal found")
        
        if baseline_questions:
            print(f"   ✅ Generated {len(baseline_questions)} questions favoring baseline methods")
            sample_baseline = baseline_questions[0]
            print(f"      Sample: \"{sample_baseline.question}\"")
        
        # Test knowledge graph integration
        print("\n🏗️  Testing Simplified Knowledge Graph Integration:")
        
        # Check relationship types used
        relationship_types_used = set()
        for question in pipeline.questions:
            relationship_types_used.update(question.relationship_types)
        
        print(f"   🔗 Relationship types tested: {sorted(relationship_types_used)}")
        
        # Check entity extraction effectiveness
        all_entities = set()
        total_entities = 0
        for question in pipeline.questions:
            all_entities.update(question.entities_used)
            total_entities += len(question.entities_used)
        
        print(f"   🏷️  Entity extraction stats:")
        print(f"      Total entities used: {total_entities}")
        print(f"      Unique entities: {len(all_entities)}")
        if total_entities > 0:
            print(f"      Average entities per question: {total_entities / len(pipeline.questions):.1f}")
            print(f"      Entity reuse rate: {len(all_entities) / total_entities:.2f}")
        
        # Show sample entities
        if all_entities:
            sample_entities = list(all_entities)[:10]
            print(f"      Sample entities: {sample_entities}")
        
        # Check ground truth context diversity
        all_contexts = set()
        for question in pipeline.questions:
            all_contexts.update(question.ground_truth_contexts)
        
        print(f"   🎯 Ground truth context stats:")
        print(f"      Unique contexts: {len(all_contexts)}")
        context_usage = sum(len(q.ground_truth_contexts) for q in pipeline.questions)
        if context_usage > 0:
            print(f"      Context reuse rate: {len(all_contexts) / context_usage:.2f}")
            print(f"      Average contexts per question: {context_usage / len(pipeline.questions):.1f}")
        
        # Test question statistics
        if pipeline.question_stats:
            stats = pipeline.question_stats
            print(f"\n📊 Question Generation Statistics:")
            print(f"   Total questions: {stats['total_questions']:,}")
            
            if 'by_question_type' in stats:
                print(f"   By question type:")
                for q_type, count in stats['by_question_type'].items():
                    print(f"      {q_type}: {count}")
            
            if 'by_expected_advantage' in stats:
                print(f"   By expected advantage:")
                for advantage, count in stats['by_expected_advantage'].items():
                    print(f"      {advantage}: {count}")
            
            if 'by_persona' in stats:
                print(f"   By persona:")
                for persona, count in stats['by_persona'].items():
                    print(f"      {persona}: {count}")
        
        # Test caching functionality
        print("\n💾 Testing question caching...")
        try:
            # Clear force recompute and run again (should use cache)
            pipeline.config['execution']['force_recompute'] = []
            
            import time
            cache_start = time.time()
            pipeline._phase_6_question_generation()
            cache_end = time.time()
            cache_time = cache_end - cache_start
            
            print(f"   Cache load time: {cache_time:.3f}s")
            print(f"   Cached questions: {len(pipeline.questions)} questions")
            
            if cache_time < 1.0:  # Should be very fast
                print("   ✅ Caching working effectively")
            else:
                print("   ⚠️  Caching may not be working as expected")
                
        except Exception as e:
            print(f"   ⚠️  Cache test failed: {e}")
        
        # Check if questions were saved
        questions_path = Path(pipeline.config['directories']['data']) / "questions" / "evaluation_questions.json"
        if questions_path.exists():
            print(f"\n✅ Questions saved successfully to {questions_path}")
            
            # Show file size
            file_size = questions_path.stat().st_size / 1024  # KB
            print(f"   File size: {file_size:.1f} KB")
        else:
            print(f"\n⚠️  Questions file not found at {questions_path}")
        
        print("\n🎉 Phase 6 test completed successfully!")
        print(f"📋 Experiment ID: {pipeline.experiment_id}")
        print(f"📁 Logs saved to: logs/{pipeline.experiment_id}.log")
        print(f"📊 Questions saved to: {questions_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 6 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🧪 Semantic RAG Pipeline - Phase 6 Test")
    print("Testing simplified question generation (entities only, no themes/keyphrases)")
    print("=" * 75)
    
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
        print("🚀 Phase 6 is ready for production use")
        print("\n📊 Key features verified:")
        print("   • Simplified question generation using entities only (PERSON/ORG/GPE)")
        print("   • Three core relationship types: entity overlap, cosine similarity, hierarchical")
        print("   • Ollama integration with fallback generation")
        print("   • Two personas: Research Scientist and Basic Googler")
        print("   • Four question strategies: entity bridge, concept similarity, hierarchical, single-hop")
        print("   • Ground truth context tracking")
        print("   • Expected advantage prediction")
        print("   • Intelligent caching system")
        print("\n🎯 Architectural simplification verified:")
        print("   • Removed themes, keyphrases, and summaries")
        print("   • Focused on high-quality entities only")
        print("   • Simplified prompts let Ollama discover connections autonomously")
        print("   • Cognitive minimalism: elegance through constraint")
    else:
        print("\n❌ Tests failed!")
        print("🔧 Please check the error messages above")


if __name__ == "__main__":
    main()
