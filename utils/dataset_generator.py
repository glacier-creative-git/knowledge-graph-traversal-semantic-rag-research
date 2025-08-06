#!/usr/bin/env python3
"""
Corrected RAGAS Dataset Generator with Knowledge Graph
====================================================

Uses the NEW RAGAS knowledge graph architecture for dataset generation.
This approach is very similar to your semantic graph traversal algorithm!

Usage:
    python corrected_ragas_generator.py --topics medical --num-questions 200
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from dataclasses import dataclass, asdict
import logging

# CORRECT RAGAS imports for new architecture
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import (
    default_transforms,
    apply_transforms,
    HeadlinesExtractor,
    HeadlineSplitter,
    KeyphrasesExtractor,
    OverlapScoreBuilder
)
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import default_query_distribution
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WikipediaLoader, DirectoryLoader
from ragas.testset.transforms.extractors import NERExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class KnowledgeGraphConfig:
    """Configuration for knowledge graph-based dataset generation"""
    # Model settings
    generator_model: str = "gpt-3.5-turbo"
    critic_model: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-ada-002"
    openai_api_key: str = "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"

    # Dataset settings
    num_questions: int = 200
    min_sentences: int = 50
    articles_per_topic: int = 5

    # Knowledge Graph settings
    use_custom_transforms: bool = True
    extract_headlines: bool = True
    extract_keyphrases: bool = True
    extract_entities: bool = True
    build_overlap_relationships: bool = True

    # Chunking settings for knowledge graph
    min_chunk_tokens: int = 300
    max_chunk_tokens: int = 1000

    # Topic settings
    topic_set: str = "medical"
    custom_topics: List[str] = None
    custom_doc_path: str = ""

    # Output settings
    output_file: str = "kg_ragas_dataset.json"
    save_knowledge_graph: bool = True

    def __post_init__(self):
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY", "")


class KnowledgeGraphRAGASGenerator:
    """
    Generate synthetic datasets using RAGAS Knowledge Graph approach.

    This is very similar to your semantic graph traversal algorithm:
    - Creates knowledge graph from documents
    - Establishes relationships between nodes
    - Traverses graph for multi-hop question generation
    """

    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config

        if not self.config.openai_api_key:
            raise ValueError("OpenAI API key required")

        os.environ["OPENAI_API_KEY"] = self.config.openai_api_key

        logger.info("🧠 Initializing Knowledge Graph RAGAS Generator")
        logger.info(f"   This uses graph traversal similar to your algorithm!")
        logger.info(f"   Generator Model: {self.config.generator_model}")

        # Initialize models
        self._initialize_models()

        # Initialize knowledge graph
        self.knowledge_graph = KnowledgeGraph()

        logger.info("✅ Knowledge Graph Generator initialized")

    def _initialize_models(self):
        """Initialize LLM and embedding models"""
        self.generator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=self.config.generator_model,
                temperature=0.7,
                api_key=self.config.openai_api_key
            )
        )

        self.critic_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=self.config.critic_model,
                temperature=0.3,
                api_key=self.config.openai_api_key
            )
        )

        self.embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=self.config.embedding_model,
                api_key=self.config.openai_api_key
            )
        )

    def load_documents(self) -> List:
        """Load documents for knowledge graph construction"""
        logger.info("📚 Loading documents for knowledge graph...")

        if self.config.custom_doc_path:
            return self._load_custom_documents()
        else:
            return self._load_wikipedia_documents()

    def _load_custom_documents(self) -> List:
        """Load documents from custom directory"""
        logger.info(f"📁 Loading from: {self.config.custom_doc_path}")

        loader = DirectoryLoader(
            self.config.custom_doc_path,
            glob="**/*.{txt,md,pdf,docx}",
            show_progress=True
        )
        docs = loader.load()

        return self._filter_documents_by_length(docs)

    def _load_wikipedia_documents(self) -> List:
        """Load Wikipedia documents"""
        topics = self._get_topics()

        logger.info(f"📖 Loading Wikipedia articles: {topics}")

        all_docs = []

        for topic in topics:
            logger.info(f"   Processing topic: {topic}")
            try:
                # Method 1: Try direct WikipediaLoader
                try:
                    loader = WikipediaLoader(query=topic, load_max_docs=self.config.articles_per_topic)
                    docs = loader.load()
                    logger.info(f"     Direct load: {len(docs)} documents")
                    all_docs.extend(docs)
                    continue
                except Exception as e:
                    logger.warning(f"     Direct load failed: {e}")

                # Method 2: Try wikipedia search + individual loading
                try:
                    import wikipedia
                    search_results = wikipedia.search(topic, results=self.config.articles_per_topic * 2)
                    logger.info(f"     Search found: {search_results}")

                    for result in search_results[:self.config.articles_per_topic]:
                        try:
                            loader = WikipediaLoader(query=result, load_max_docs=1)
                            docs = loader.load()
                            if docs:
                                all_docs.extend(docs)
                                logger.info(f"     Loaded: {result} ({len(docs[0].page_content)} chars)")
                        except Exception as e:
                            logger.warning(f"     Failed to load '{result}': {e}")

                except Exception as e:
                    logger.warning(f"     Wikipedia search failed: {e}")

            except Exception as e:
                logger.warning(f"   Complete failure for topic '{topic}': {e}")

        logger.info(f"📚 Total documents loaded: {len(all_docs)}")

        # If we got no documents, try some fallback topics
        if len(all_docs) == 0:
            logger.warning("No documents loaded, trying fallback topics...")
            fallback_topics = ["Machine learning", "Artificial intelligence", "Computer science"]
            for topic in fallback_topics:
                try:
                    loader = WikipediaLoader(query=topic, load_max_docs=2)
                    docs = loader.load()
                    all_docs.extend(docs)
                    logger.info(f"   Fallback loaded: {topic} ({len(docs)} docs)")
                    if len(all_docs) >= 3:  # Got some docs, break
                        break
                except:
                    continue

        return self._filter_documents_by_length(all_docs)

    def _filter_documents_by_length(self, docs: List) -> List:
        """Filter documents by sentence count (50+ sentences)"""
        substantial_docs = []

        for doc in docs:
            # More robust sentence counting
            import re
            # Remove multiple spaces and clean text
            clean_text = re.sub(r'\s+', ' ', doc.page_content.strip())
            # Count sentences more accurately
            sentences = re.split(r'[.!?]+', clean_text)
            # Filter out very short "sentences" (likely not real sentences)
            real_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            sentence_count = len(real_sentences)

            logger.debug(f"Document length: {sentence_count} sentences")

            if sentence_count >= self.config.min_sentences:
                substantial_docs.append(doc)
            else:
                logger.debug(f"Rejected doc: {sentence_count} < {self.config.min_sentences} sentences")

        logger.info(f"✅ Filtered to {len(substantial_docs)} substantial documents")

        # If no substantial docs, lower the threshold temporarily
        if len(substantial_docs) == 0 and self.config.min_sentences > 20:
            logger.warning(f"No docs found with {self.config.min_sentences}+ sentences, trying 20+...")
            for doc in docs:
                clean_text = re.sub(r'\s+', ' ', doc.page_content.strip())
                sentences = re.split(r'[.!?]+', clean_text)
                real_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
                if len(real_sentences) >= 20:
                    substantial_docs.append(doc)

        return substantial_docs

    def build_knowledge_graph(self, docs: List) -> KnowledgeGraph:
        """
        Build knowledge graph from documents.

        This is similar to your algorithm's graph construction:
        1. Create nodes from document chunks
        2. Extract relationships (entities, keyphrases, overlaps)
        3. Build connections between semantically related nodes
        """
        logger.info("🕸️  Building Knowledge Graph (similar to your semantic graph!)")

        # Step 1: Create document nodes
        logger.info("   Step 1: Creating document nodes...")
        for doc in docs:
            node = Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata if hasattr(doc, 'metadata') else {}
                }
            )
            self.knowledge_graph.nodes.append(node)

        logger.info(f"   Created {len(self.knowledge_graph.nodes)} document nodes")

        # Step 2: Apply transformations to build relationships
        logger.info("   Step 2: Building relationships between nodes...")

        if self.config.use_custom_transforms:
            transforms = self._build_custom_transforms()
        else:
            # Use default transforms
            transforms = default_transforms(
                documents=docs,
                llm=self.generator_llm,
                embedding_model=self.embeddings
            )

        # Apply transformations (this builds the graph structure)
        apply_transforms(self.knowledge_graph, transforms)

        logger.info("✅ Knowledge graph construction complete!")
        logger.info(f"   Total nodes: {len(self.knowledge_graph.nodes)}")

        # Save knowledge graph if requested
        if self.config.save_knowledge_graph:
            kg_path = self.config.output_file.replace('.json', '_knowledge_graph.json')
            self.knowledge_graph.save(kg_path)
            logger.info(f"💾 Knowledge graph saved to: {kg_path}")

        return self.knowledge_graph

    def _build_custom_transforms(self) -> List:
        """Build custom transformation pipeline for knowledge graph"""
        transforms = []

        # 1. Extract headlines (section structure)
        if self.config.extract_headlines:
            transforms.append(
                HeadlinesExtractor(llm=self.generator_llm, max_num=20)
            )

        # 2. Split by headlines (hierarchical chunking)
        transforms.append(
            HeadlineSplitter(
                min_tokens=self.config.min_chunk_tokens,
                max_tokens=self.config.max_chunk_tokens
            )
        )

        # 3. Extract keyphrases (semantic concepts)
        if self.config.extract_keyphrases:
            transforms.append(
                KeyphrasesExtractor(
                    llm=self.generator_llm,
                    max_num=10,
                    property_name="keyphrases"
                )
            )

        # 4. Extract named entities
        if self.config.extract_entities:
            transforms.append(NERExtractor())

        # 5. Build overlap relationships (semantic connections)
        if self.config.build_overlap_relationships:
            transforms.append(
                OverlapScoreBuilder(
                    property_name="keyphrases",
                    new_property_name="overlap_score",
                    threshold=0.1,  # Minimum overlap for connection
                    distance_threshold=0.8  # Similarity threshold
                )
            )

        logger.info(f"   Using {len(transforms)} custom transforms:")
        for t in transforms:
            logger.info(f"     - {t.__class__.__name__}")

        return transforms

    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate synthetic dataset using knowledge graph traversal.

        This leverages the graph structure (similar to your algorithm) to create
        multi-hop questions that require traversing connected nodes.
        """
        logger.info("🔬 Generating dataset using knowledge graph traversal...")

        # Load and process documents
        docs = self.load_documents()
        if not docs:
            raise ValueError("No documents loaded")

        # Build knowledge graph
        self.build_knowledge_graph(docs)

        # Create manual personas to avoid the filtering error
        personas = self._create_manual_personas()

        # Create TestsetGenerator with knowledge graph and personas
        generator = TestsetGenerator(
            llm=self.generator_llm,
            embedding_model=self.embeddings,
            knowledge_graph=self.knowledge_graph,
            persona_list=personas  # Provide personas manually
        )

        # Use default query distribution (optimized for multi-hop)
        query_distribution = default_query_distribution(self.generator_llm)

        logger.info("   Query distribution (optimized for graph traversal):")
        for synthesizer, weight in query_distribution:
            logger.info(f"     {synthesizer.__class__.__name__}: {weight}")

        # Generate testset using graph traversal
        start_time = time.time()

        try:
            testset = generator.generate(
                testset_size=self.config.num_questions,
                query_distribution=query_distribution
            )
        except Exception as e:
            logger.error(f"Knowledge graph generation failed: {e}")
            logger.info("Trying fallback approach with simpler distribution...")

            # Fallback: Try with simplified approach
            try:
                # Use only single hop queries if multi-hop fails
                from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer
                simple_distribution = [(SingleHopSpecificQuerySynthesizer(llm=self.generator_llm), 1.0)]

                testset = generator.generate(
                    testset_size=min(50, self.config.num_questions),  # Reduced size for fallback
                    query_distribution=simple_distribution
                )
                logger.info("✅ Fallback approach succeeded with single-hop queries")
            except Exception as e2:
                logger.error(f"Even fallback approach failed: {e2}")
                raise RuntimeError(f"Both primary and fallback generation failed. Primary: {e}, Fallback: {e2}")

        generation_time = time.time() - start_time
        logger.info(f"✅ Dataset generated in {generation_time:.1f}s using graph traversal")

        # Convert to DataFrame and add metadata
        df = testset.to_pandas()
        df = self._add_metadata(df)

        self._analyze_dataset(df)

        return df

    def _create_manual_personas(self) -> List:
        """Create manual personas to avoid knowledge graph filtering issues"""
        from ragas.testset.persona import Persona

        personas_by_topic = {
            "medical": [
                Persona(
                    name="Medical Researcher",
                    role_description="A medical researcher studying disease mechanisms and treatment approaches"
                ),
                Persona(
                    name="Healthcare Professional",
                    role_description="A healthcare professional seeking evidence-based information for patient care"
                ),
                Persona(
                    name="Medical Student",
                    role_description="A medical student learning about pathophysiology and clinical applications"
                )
            ],
            "tech": [
                Persona(
                    name="Software Engineer",
                    role_description="A software engineer implementing machine learning and AI systems"
                ),
                Persona(
                    name="Data Scientist",
                    role_description="A data scientist analyzing algorithms and methodologies for research"
                ),
                Persona(
                    name="Computer Science Student",
                    role_description="A computer science student learning about advanced algorithms and systems"
                )
            ],
            "science": [
                Persona(
                    name="Research Scientist",
                    role_description="A research scientist investigating fundamental scientific principles"
                ),
                Persona(
                    name="Graduate Student",
                    role_description="A graduate student conducting research in their scientific field"
                ),
                Persona(
                    name="Academic Researcher",
                    role_description="An academic researcher publishing scientific papers and conducting experiments"
                )
            ]
        }

        personas = personas_by_topic.get(self.config.topic_set, personas_by_topic["medical"])

        logger.info(f"   Created {len(personas)} manual personas for {self.config.topic_set} domain")
        for persona in personas:
            logger.info(f"     - {persona.name}: {persona.role_description}")

        return personas

    def _add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add metadata about knowledge graph generation"""
        logger.info("📊 Adding knowledge graph metadata...")

        # Standard metadata
        if 'contexts' in df.columns:
            df['num_contexts'] = df['contexts'].apply(len)
            df['total_context_length'] = df['contexts'].apply(
                lambda contexts: sum(len(str(ctx)) for ctx in contexts)
            )

        df['question_length'] = df['question'].str.len()
        df['generated_at'] = pd.Timestamp.now()
        df['generator_model'] = self.config.generator_model
        df['topic_set'] = self.config.topic_set
        df['generation_method'] = 'knowledge_graph_traversal'
        df['knowledge_graph_nodes'] = len(self.knowledge_graph.nodes)

        return df

    def _analyze_dataset(self, df: pd.DataFrame):
        """Analyze the generated dataset"""
        logger.info("📈 Knowledge Graph Dataset Analysis:")
        logger.info(f"   Total questions: {len(df)}")
        logger.info(f"   Knowledge graph nodes: {len(self.knowledge_graph.nodes)}")
        logger.info(f"   Average question length: {df['question_length'].mean():.0f} chars")

        if 'num_contexts' in df.columns:
            logger.info(f"   Average contexts per question: {df['num_contexts'].mean():.1f}")

    def save_dataset(self, df: pd.DataFrame) -> str:
        """Save the dataset and metadata"""
        output_path = Path(self.config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"💾 Saving knowledge graph dataset to: {output_path}")

        # Save as JSON
        df.to_json(output_path, orient='records', indent=2)

        # Save configuration
        config_path = output_path.with_name(f"{output_path.stem}_config.json")
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)

        logger.info(f"✅ Dataset and config saved")

        return str(output_path)

    def _get_topics(self) -> List[str]:
        """Get topics for document loading"""
        if self.config.custom_topics:
            return self.config.custom_topics

        topic_sets = {
            "medical": [
                "Alzheimer's disease pathophysiology",
                "Cancer immunotherapy mechanisms",
                "Cardiovascular disease epidemiology",
                "Neurodegenerative disease research",
                "Clinical trial methodology",
                "Pharmacokinetics drug metabolism"
            ],
            "tech": [
                "Machine learning algorithms",
                "Computer vision deep learning",
                "Natural language processing",
                "Distributed systems architecture",
                "Quantum computing principles",
                "Cybersecurity cryptography"
            ],
            "science": [
                "Quantum mechanics principles",
                "Molecular biology techniques",
                "Climate change science",
                "Astrophysics cosmology",
                "Materials science engineering",
                "Renewable energy technologies"
            ]
        }

        return topic_sets.get(self.config.topic_set, topic_sets["medical"])


def create_kg_preset_configs():
    """Create preset configurations for knowledge graph generation"""
    configs = {
        "kg_medical_config.json": KnowledgeGraphConfig(
            generator_model="gpt-3.5-turbo",
            topic_set="medical",
            num_questions=200,
            output_file="datasets/kg_medical_dataset.json"
        ),
        "kg_tech_config.json": KnowledgeGraphConfig(
            generator_model="gpt-3.5-turbo",
            topic_set="tech",
            num_questions=150,
            output_file="datasets/kg_tech_dataset.json"
        ),
        "kg_premium_config.json": KnowledgeGraphConfig(
            generator_model="gpt-4-turbo-preview",
            critic_model="gpt-4-turbo-preview",
            topic_set="medical",
            num_questions=300,
            output_file="datasets/kg_premium_dataset.json"
        )
    }

    Path("configs").mkdir(exist_ok=True)

    for filename, config in configs.items():
        config_path = Path("configs") / filename
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        print(f"Created: {config_path}")


def simple_ragas_generation(topics: List[str], num_questions: int = 100, api_key: str = None):
    """
    Simplified RAGAS generation that bypasses knowledge graph issues.
    Use this if the main approach keeps failing.
    """
    print("🔧 Using simplified RAGAS generation...")

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    # Load documents
    from langchain_community.document_loaders import WikipediaLoader

    docs = []
    for topic in topics:
        try:
            loader = WikipediaLoader(query=topic, load_max_docs=2)
            topic_docs = loader.load()
            docs.extend(topic_docs)
            print(f"   Loaded {len(topic_docs)} docs for: {topic}")
        except Exception as e:
            print(f"   Failed to load {topic}: {e}")

    if not docs:
        print("❌ No documents loaded")
        return None

    # Use the older, simpler RAGAS approach
    try:
        from ragas.testset import TestsetGenerator
        from ragas import simple, reasoning, multi_context
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        generator_llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)
        critic_llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)
        embeddings = OpenAIEmbeddings(api_key=api_key)

        generator = TestsetGenerator.from_langchain(
            generator_llm, critic_llm, embeddings
        )

        # Simple distribution
        distributions = {
            simple: 0.4,
            multi_context: 0.4,  # Still emphasize multi-context
            reasoning: 0.2
        }

        testset = generator.generate_with_langchain_docs(
            docs,
            test_size=num_questions,
            distributions=distributions
        )

        df = testset.to_pandas()
        output_path = "simple_ragas_dataset.json"
        df.to_json(output_path, orient='records', indent=2)

        print(f"✅ Generated {len(df)} questions using simple approach")
        print(f"💾 Saved to: {output_path}")

        return df

    except Exception as e:
        print(f"❌ Simple generation also failed: {e}")
        return None


def test_document_loading():
    """Test function to debug document loading issues"""
    print("🧪 Testing document loading...")

    # Test Wikipedia loading
    try:
        from langchain_community.document_loaders import WikipediaLoader

        print("Testing WikipediaLoader...")
        loader = WikipediaLoader(query="Machine learning", load_max_docs=1)
        docs = loader.load()

        if docs:
            doc = docs[0]
            print(f"✅ Loaded document: {len(doc.page_content)} characters")

            # Test sentence counting
            import re
            clean_text = re.sub(r'\s+', ' ', doc.page_content.strip())
            sentences = re.split(r'[.!?]+', clean_text)
            real_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

            print(f"   Sentence analysis:")
            print(f"     Raw sentence count: {len(sentences)}")
            print(f"     Real sentence count: {len(real_sentences)}")
            print(f"     First few sentences: {real_sentences[:3]}")

            return True
        else:
            print("❌ No documents loaded")
            return False

    except Exception as e:
        print(f"❌ Error testing document loading: {e}")
        return False


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Generate datasets using RAGAS Knowledge Graph")

    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Generator model")
    parser.add_argument("--topics", type=str, choices=["medical", "tech", "science"],
                        default="medical", help="Topic set")
    parser.add_argument("--num-questions", type=int, default=200, help="Number of questions")
    parser.add_argument("--output", type=str, default="kg_dataset.json", help="Output file")
    parser.add_argument("--create-configs", action="store_true", help="Create preset configs")
    parser.add_argument("--test-loading", action="store_true", help="Test document loading")
    parser.add_argument("--simple", action="store_true", help="Use simple RAGAS generation (bypass knowledge graph)")

    args = parser.parse_args()

    if args.create_configs:
        create_kg_preset_configs()
        return

    if args.test_loading:
        test_document_loading()
        return

    if args.simple:
        # Use simple generation approach
        topic_sets = {
            "medical": ["Alzheimer's disease", "Cancer immunotherapy", "Cardiovascular disease"],
            "tech": ["Machine learning", "Computer vision", "Natural language processing"],
            "science": ["Quantum mechanics", "Climate change", "Molecular biology"]
        }
        topics = topic_sets.get(args.topics, topic_sets["medical"])

        df = simple_ragas_generation(topics, args.num_questions)
        if df is not None:
            print("🎉 Simple generation completed successfully!")
        return

    # Create or load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = KnowledgeGraphConfig(**config_dict)
    else:
        config = KnowledgeGraphConfig(
            generator_model=args.model,
            topic_set=args.topics,
            num_questions=args.num_questions,
            output_file=args.output
        )

    # Generate dataset
    try:
        generator = KnowledgeGraphRAGASGenerator(config)
        df = generator.generate_dataset()
        output_path = generator.save_dataset(df)

        print(f"\n🎉 SUCCESS!")
        print(f"Generated {len(df)} questions using knowledge graph traversal")
        print(f"Saved to: {output_path}")
        print(f"Knowledge graph approach is similar to your semantic graph algorithm! 🧠")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise


if __name__ == "__main__":
    main()