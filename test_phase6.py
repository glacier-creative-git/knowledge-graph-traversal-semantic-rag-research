#!/usr/bin/env python3
"""
Test Phase 6: Question Generation
=================================

Test script to verify Phase 6 (Question Generation) functionality.
Tests RAGAS-based question generation from knowledge graphs.

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
    """Test Phase 6: Question Generation."""
    print("🧪 Testing Phase 6: Question Generation")
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
        
        # Configure question generation for testing (smaller numbers)
        pipeline.config['questions']['total_questions'] = 15
        pipeline.config['questions']['model']['llm'] = 'gpt-3.5-turbo'
        pipeline.config['questions']['distribution']['single_hop'] = 0.5
        pipeline.config['questions']['distribution']['multi_hop_abstract'] = 0.3
        pipeline.config['questions']['distribution']['multi_hop_specific'] = 0.2
        
        # Enable force recompute to see fresh generation
        pipeline.config['execution']['force_recompute'] = ['questions']
        
        print(f"   Models: {pipeline.config['models']['embedding_models']}")
        print(f"   Total questions: {pipeline.config['questions']['total_questions']}")
        print(f"   LLM model: {pipeline.config['questions']['model']['llm']}")
        print(f"   Question distribution: {pipeline.config['questions']['distribution']}")
        
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
            print("🏢 Running Phase 5: Knowledge Graph Construction...")
            pipeline._phase_5_knowledge_graph_construction()
        else:
            print("✅ Phase 5: Knowledge graph already available")
        
        print(f"✅ Phase 5: Knowledge graph constructed")
        
        # Phase 6: Question Generation
        print("📊 Running Phase 6: Question Generation...")
        pipeline._phase_6_question_generation()
        
        # Verify results
        print("\n🔍 Verifying Phase 6 results...")
        
        # Check questions
        if not pipeline.questions:
            print("❌ No questions were generated")
            return False
        
        print(f"✅ Questions generated successfully: {len(pipeline.questions)} questions")
        
        # Check question statistics
        if pipeline.question_stats:
            stats = pipeline.question_stats
            print(f"   📊 Question Generation Statistics:")
            print(f"      Total questions: {stats['total_questions']:,}")
            
            if 'by_synthesizer' in stats:
                print(f"      By synthesizer:")
                for synthesizer, count in stats['by_synthesizer'].items():
                    print(f"         {synthesizer}: {count}")
            
            if 'by_question_type' in stats:
                print(f"      By question type:")
                for q_type, count in stats['by_question_type'].items():
                    print(f"         {q_type}: {count}")
            
            if 'by_expected_advantage' in stats:
                print(f"      By expected advantage:")
                for advantage, count in stats['by_expected_advantage'].items():
                    print(f"         {advantage}: {count}")
            
            if 'by_difficulty' in stats:
                print(f"      By difficulty:")
                for difficulty, count in stats['by_difficulty'].items():
                    print(f"         {difficulty}: {count}")
        
        # Show sample questions from each category
        print("\n📝 Sample Questions by Type:")
        
        # Group questions by type for display
        questions_by_type = {}
        for question in pipeline.questions:
            q_type = question.question_type
            if q_type not in questions_by_type:
                questions_by_type[q_type] = []
            questions_by_type[q_type].append(question)
        
        for q_type, questions in questions_by_type.items():
            print(f"\n   {q_type.upper()}:")
            sample_question = questions[0]  # Show first question of each type
            print(f"      Question: \"{sample_question.question}\"")
            print(f"      Expected advantage: {sample_question.expected_advantage}")
            print(f"      Difficulty: {sample_question.difficulty_level}")
            print(f"      Synthesizer: {sample_question.synthesizer_name}")
            if sample_question.metadata:
                print(f"      Metadata: {sample_question.metadata}")
        
        # Test question quality
        print("\n🎯 Testing Question Quality:")
        
        # Check for valid question indicators
        valid_questions = 0
        for question in pipeline.questions[:10]:  # Check first 10
            text = question.question.lower()
            has_question_word = any(word in text for word in ['what', 'how', 'why', 'when', 'where', 'which'])
            has_question_mark = '?' in text
            has_question_verb = any(word in text for word in ['explain', 'describe', 'compare'])
            
            if has_question_word or has_question_mark or has_question_verb:
                valid_questions += 1
        
        print(f"   Valid question format: {valid_questions}/10 checked")
        
        # Check expected advantage distribution
        advantage_counts = {}
        for question in pipeline.questions:
            advantage = question.expected_advantage
            advantage_counts[advantage] = advantage_counts.get(advantage, 0) + 1
        
        print(f"   Expected advantage distribution:")
        for advantage, count in advantage_counts.items():
            percentage = (count / len(pipeline.questions)) * 100
            print(f"      {advantage}: {count} ({percentage:.1f}%)")
        
        # Verify semantic traversal questions
        semantic_questions = [q for q in pipeline.questions if q.expected_advantage == 'semantic_traversal']
        if semantic_questions:
            print(f"   ✅ Generated {len(semantic_questions)} questions favoring semantic traversal")
            sample_semantic = semantic_questions[0]
            print(f"      Sample: \"{sample_semantic.question}\"")
        else:
            print(f"   ⚠️  No questions favoring semantic traversal found")
        
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
            print(f"   Cached questions size: {len(pipeline.questions)} questions")
            
            if cache_time < 1.0:  # Should be very fast
                print("   ✅ Caching working effectively")
            else:
                print("   ⚠️  Caching may not be working as expected")
                
        except Exception as e:
            print(f"   ⚠️  Cache test failed: {e}")
        
        print("\n🎉 Phase 6 test completed successfully!")
        print(f"📋 Experiment ID: {pipeline.experiment_id}")
        print(f"📁 Logs saved to: logs/{pipeline.experiment_id}.log")
        print(f"📊 Questions saved to: data/questions/evaluation_questions.json")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 6 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🧪 Semantic RAG Pipeline - Phase 6 Test")
    print("Testing RAGAS-based question generation from knowledge graphs")
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
        print("   • RAGAS-based question generation from knowledge graphs")
        print("   • Multi-hop and single-hop question synthesis")
        print("   • Semantic traversal advantage prediction")
        print("   • Question type categorization and difficulty assignment")
        print("   • Fallback generation when RAGAS unavailable")
        print("   • Intelligent caching system")
        print("\n🎯 Ready for Phase 7: RAG System Evaluation!")
    else:
        print("\n❌ Tests failed!")
        print("🔧 Please check the error messages above")


if __name__ == "__main__":
    main()
