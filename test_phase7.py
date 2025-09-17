#!/usr/bin/env python3
"""
Phase 7 Question Generation Test Script
======================================

Test script for validating traversal-based question generation.
Demonstrates the new Path → Question methodology where questions
are generated from validated traversal paths.
"""

import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.pipeline import SemanticRAGPipeline


def setup_logging() -> logging.Logger:
    """Setup logging for the test script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("Phase7Tester")


def test_question_generation_only():
    """Test just Phase 7 question generation assuming previous phases are complete."""
    logger = setup_logging()
    
    logger.info("🧪 Testing Phase 7: Traversal-Based Question Generation")
    logger.info("=" * 70)
    
    try:
        # Initialize pipeline
        pipeline = SemanticRAGPipeline()
        
        # Skip to Phase 7 by loading existing data
        logger.info("📂 Loading existing pipeline data...")
        
        # Load configuration
        pipeline._load_config()
        pipeline._initialize_experiment_tracker()
        pipeline._initialize_logging()
        
        # Try to load existing knowledge graph
        kg_path = Path(pipeline.config['directories']['data']) / "knowledge_graph.json"
        if not kg_path.exists():
            logger.error(f"❌ Knowledge graph not found at {kg_path}")
            logger.error("Please run the full pipeline first (Phases 1-6) to generate the knowledge graph")
            return False
        
        logger.info(f"📂 Loading knowledge graph from {kg_path}")
        from utils.knowledge_graph import KnowledgeGraph
        pipeline.knowledge_graph = KnowledgeGraph.load(str(kg_path))
        
        # Load embeddings if they exist
        try:
            from utils.models import MultiGranularityEmbeddingEngine
            embedding_engine = MultiGranularityEmbeddingEngine(pipeline.config, pipeline.logger)
            pipeline.embeddings = embedding_engine.load_cached_embeddings()
            
            if pipeline.embeddings:
                pipeline.knowledge_graph.load_phase3_embeddings(pipeline.embeddings)
                logger.info("✅ Loaded cached embeddings into knowledge graph")
        except Exception as e:
            logger.warning(f"⚠️  Could not load embeddings: {e}")
        
        logger.info(f"✅ Knowledge graph loaded: {len(pipeline.knowledge_graph.chunks)} chunks, "
                   f"{len(pipeline.knowledge_graph.sentences)} sentences, "
                   f"{len(pipeline.knowledge_graph.documents)} documents")
        
        # Test Phase 7 question generation
        logger.info("\\n🎯 Testing Phase 7 Question Generation...")
        
        # Override config for testing
        pipeline.config['question_generation'] = {
            'generator_model_type': 'ollama',
            'critic_model_type': 'ollama',
            'question_distribution': {
                'single_hop': 0.4,
                'sequential_flow': 0.2,
                'multi_hop': 0.2,
                'theme_hop': 0.1,
                'hierarchical': 0.1
            },
            'max_hops': 3,
            'max_sentences': 10,
            'cache_questions': True,
            'num_questions': 20  # Test with small number
        }
        
        # Run Phase 7
        pipeline._phase_7_question_generation()
        
        # Analyze results
        if pipeline.evaluation_dataset:
            logger.info("\\n📊 Question Generation Results:")
            logger.info("=" * 50)
            
            questions = pipeline.evaluation_dataset.questions
            logger.info(f"Total questions generated: {len(questions)}")
            
            # Analyze by type
            type_counts = {}
            for q in questions:
                q_type = q.question_type
                type_counts[q_type] = type_counts.get(q_type, 0) + 1
            
            logger.info("\\nQuestions by type:")
            for q_type, count in type_counts.items():
                logger.info(f"  {q_type}: {count}")
            
            # Show sample questions
            logger.info("\\n📝 Sample Generated Questions:")
            logger.info("-" * 50)
            
            for i, question in enumerate(questions[:5]):  # Show first 5
                logger.info(f"\\n{i+1}. [{question.question_type.upper()}]")
                logger.info(f"   Q: {question.question_text}")
                logger.info(f"   Path: {len(question.ground_truth_path.nodes)} nodes, "
                           f"{question.ground_truth_path.total_hops} hops")
                logger.info(f"   Valid: {question.ground_truth_path.is_valid}")
                if not question.ground_truth_path.is_valid:
                    logger.info(f"   Errors: {question.ground_truth_path.validation_errors}")
            
            # Show detailed path for one example
            if questions:
                example = questions[0]
                logger.info(f"\\n🗺️  Detailed Path Example ({example.question_type}):")
                logger.info(f"   Question: {example.question_text}")
                logger.info(f"   Path nodes: {example.ground_truth_path.nodes}")
                logger.info(f"   Connection types: {[ct.value for ct in example.ground_truth_path.connection_types]}")
                logger.info(f"   Granularities: {[gl.value for gl in example.ground_truth_path.granularity_levels]}")
                logger.info(f"   Expected answer: {example.expected_answer[:200]}...")
            
            # Validation summary
            valid_questions = pipeline.question_stats.get('valid_questions', 0)
            total_questions = pipeline.question_stats.get('total_questions', 0)
            validity_rate = pipeline.question_stats.get('validity_rate', 0.0)
            
            logger.info(f"\\n✅ Validation Summary:")
            logger.info(f"   Valid questions: {valid_questions}/{total_questions}")
            logger.info(f"   Validity rate: {validity_rate:.2%}")
            
            return True
        else:
            logger.error("❌ No evaluation dataset was generated")
            return False
            
    except Exception as e:
        logger.error(f"❌ Phase 7 test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_full_pipeline_with_phase7():
    """Test the complete pipeline including Phase 7."""
    logger = setup_logging()
    
    logger.info("🧪 Testing Full Pipeline with Phase 7")
    logger.info("=" * 70)
    
    try:
        # Initialize and run full pipeline
        pipeline = SemanticRAGPipeline()
        
        # Override question generation config for testing
        config_override = {
            'question_generation': {
                'generator_model_type': 'ollama',
                'critic_model_type': 'ollama', 
                'question_distribution': {
                    'single_hop': 0.5,
                    'sequential_flow': 0.2,
                    'multi_hop': 0.15,
                    'theme_hop': 0.1,
                    'hierarchical': 0.05
                },
                'num_questions': 15,  # Small number for testing
                'cache_questions': True
            }
        }
        
        # Merge with existing config
        pipeline.config.update(config_override)
        
        # Run full pipeline
        results = pipeline.pipe()
        
        logger.info("\\n🎉 Full Pipeline Results:")
        logger.info(f"   Experiment ID: {results['experiment_id']}")
        logger.info(f"   Execution time: {results['execution_time']}")
        logger.info(f"   Status: {results['status']}")
        
        # Check Phase 7 results
        if hasattr(pipeline, 'evaluation_dataset') and pipeline.evaluation_dataset:
            questions = pipeline.evaluation_dataset.questions
            logger.info(f"\\n📝 Phase 7 Results:")
            logger.info(f"   Questions generated: {len(questions)}")
            logger.info(f"   Question types: {list(set(q.question_type for q in questions))}")
            
            return True
        else:
            logger.warning("⚠️  Phase 7 did not generate questions")
            return False
            
    except Exception as e:
        logger.error(f"❌ Full pipeline test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_model_interfaces():
    """Test that both Ollama and OpenAI interfaces work."""
    logger = setup_logging()
    
    logger.info("🧪 Testing Model Interfaces")
    logger.info("=" * 50)
    
    # Test Ollama interface
    try:
        from utils.traversal_question_generator import OllamaInterface
        
        config = {
            'ollama': {
                'model': 'llama3.1:8b',
                'base_url': 'http://localhost:11434',
                'options': {'temperature': 0.1, 'num_predict': 50}
            }
        }
        
        ollama = OllamaInterface(config, logger)
        
        test_prompt = "Generate a simple question about machine learning."
        response = ollama.generate_question(test_prompt)
        
        logger.info(f"✅ Ollama test successful:")
        logger.info(f"   Prompt: {test_prompt}")
        logger.info(f"   Response: {response}")
        
        # Test critique
        critique = ollama.critique_question(response, "Machine learning context")
        logger.info(f"   Critique: {critique}")
        
    except Exception as e:
        logger.warning(f"⚠️  Ollama test failed: {e}")
    
    # Test OpenAI interface (if API key available)
    try:
        import os
        if os.getenv('OPENAI_API_KEY'):
            from utils.traversal_question_generator import OpenAIInterface
            
            config = {
                'openai': {
                    'generator_model': 'gpt-4o-mini',
                    'critic_model': 'gpt-4o',
                    'api_key': os.getenv('OPENAI_API_KEY'),
                    'options': {'temperature': 0.1, 'max_tokens': 50}
                }
            }
            
            openai_interface = OpenAIInterface(config, logger)
            
            test_prompt = "Generate a simple question about artificial intelligence."
            response = openai_interface.generate_question(test_prompt)
            
            logger.info(f"✅ OpenAI test successful:")
            logger.info(f"   Response: {response}")
        else:
            logger.info("⏭️  Skipping OpenAI test (no API key)")
            
    except Exception as e:
        logger.warning(f"⚠️  OpenAI test failed: {e}")


def main():
    """Main test function."""
    logger = setup_logging()
    
    logger.info("🚀 Starting Phase 7 Question Generation Tests")
    logger.info("=" * 70)
    
    # Test 1: Model interfaces
    logger.info("\\n" + "=" * 70)
    test_model_interfaces()
    
    # Test 2: Question generation only (assuming existing KG)
    logger.info("\\n" + "=" * 70)
    success = test_question_generation_only()
    
    if success:
        logger.info("\\n✅ Phase 7 test completed successfully!")
        logger.info("\\n💡 Next steps:")
        logger.info("   1. Review generated questions in data/questions/")
        logger.info("   2. Test with your retrieval algorithms using test_algorithms.py")
        logger.info("   3. Compare performance across the 4 algorithms")
        logger.info("   4. Validate that graph-based algorithms outperform basic retrieval")
    else:
        logger.error("\\n❌ Phase 7 test failed!")
        logger.info("\\n🔧 Troubleshooting:")
        logger.info("   1. Ensure Ollama is running: ollama serve")
        logger.info("   2. Run full pipeline first: python utils/pipeline.py")
        logger.info("   3. Check that knowledge graph exists in data/")


if __name__ == "__main__":
    main()
