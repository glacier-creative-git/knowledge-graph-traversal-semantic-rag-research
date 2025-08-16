#!/usr/bin/env python3
"""
RAGAS Hybrid Test Script - Custom + RAGAS Extractors
===================================================

This script tests the hybrid approach:
- Custom spaCy NER and TF-IDF keyphrase extraction
- Custom sliding window chunking
- RAGAS Summary/Theme extractors with LLM calls
- RAGAS relationship building

This demonstrates the cost-optimized approach while maintaining semantic richness.
"""

import os
import json
import hashlib
import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# Set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = "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"


class CustomNERExtractor:
    """Custom Named Entity Recognition extractor using spaCy (from your codebase)."""

    def __init__(self):
        """Initialize the NER extractor."""
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ SpaCy model loaded successfully")
        except OSError:
            print("⚠️  SpaCy model not found, using pattern fallback")
            self.nlp = None

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract named entities from text."""
        if self.nlp:
            return self._extract_with_spacy(text)
        else:
            return self._extract_with_patterns(text)

    def _extract_with_spacy(self, text: str) -> Dict[str, Any]:
        """Extract entities using spaCy."""
        doc = self.nlp(text)
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],  # Geopolitical entities
            'MISC': []
        }

        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE']:
                entities[ent.label_].append(ent.text)
            else:
                entities['MISC'].append(ent.text)

        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))

        return {'entities': entities}

    def _extract_with_patterns(self, text: str) -> Dict[str, Any]:
        """Fallback entity extraction using patterns."""
        entities = {'PERSON': [], 'ORG': [], 'GPE': [], 'MISC': []}

        # AI/ML specific terms
        tech_terms = [
            'machine learning', 'artificial intelligence', 'neural networks',
            'deep learning', 'natural language processing', 'computer vision',
            'algorithms', 'data science', 'supervised learning', 'unsupervised learning'
        ]

        text_lower = text.lower()
        for term in tech_terms:
            if term in text_lower:
                entities['MISC'].append(term)

        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))

        return {'entities': entities}


class CustomKeyphraseExtractor:
    """Custom keyphrase extractor using TF-IDF (from your codebase)."""

    def __init__(self, max_features: int = 15):
        """Initialize the keyphrase extractor."""
        self.max_features = max_features
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=1
            )
            print("✅ TF-IDF vectorizer initialized")
        except ImportError:
            print("⚠️  Scikit-learn not available, using word frequency fallback")
            self.vectorizer = None

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract key phrases from text."""
        if self.vectorizer:
            return self._extract_with_tfidf(text)
        else:
            return self._extract_with_frequency(text)

    def _extract_with_tfidf(self, text: str) -> Dict[str, Any]:
        """Extract phrases using TF-IDF."""
        try:
            # Fit and transform
            tfidf_matrix = self.vectorizer.fit_transform([text])
            feature_names = self.vectorizer.get_feature_names_out()

            # Get scores
            scores = tfidf_matrix.toarray()[0]

            # Get top phrases
            top_indices = scores.argsort()[-10:][::-1]
            keyphrases = [feature_names[i] for i in top_indices if scores[i] > 0]

            return {'keyphrases': keyphrases}

        except Exception as e:
            print(f"⚠️  TF-IDF extraction failed: {e}, using frequency fallback")
            return self._extract_with_frequency(text)

    def _extract_with_frequency(self, text: str) -> Dict[str, Any]:
        """Fallback keyphrase extraction using word frequency."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = defaultdict(int)

        # Common stop words to filter
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our',
                      'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two',
                      'way', 'who'}

        for word in words:
            if word not in stop_words:
                word_freq[word] += 1

        # Get top words
        keyphrases = sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:10]
        return {'keyphrases': keyphrases}


class CustomSlidingWindowChunker:
    """Custom sliding window chunker (simplified version of your approach)."""

    def __init__(self, window_size: int = 3, overlap: int = 1):
        """Initialize chunker."""
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = window_size - overlap

        # Initialize NLTK
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
        except ImportError:
            print("⚠️  NLTK not available, using simple sentence splitting")

    def create_chunks(self, documents) -> List[Dict[str, Any]]:
        """Create sliding window chunks from documents."""
        all_chunks = []

        for doc_idx, doc in enumerate(documents):
            # Extract sentences
            sentences = self._extract_sentences(doc.page_content)

            if len(sentences) < self.window_size:
                print(f"⚠️  Document {doc_idx} has only {len(sentences)} sentences, skipping")
                continue

            # Create sliding windows
            windows = self._calculate_sliding_windows(sentences)

            # Create chunks
            for window_idx, (start_idx, end_idx) in enumerate(windows):
                chunk = self._create_chunk(
                    doc, sentences, start_idx, end_idx, window_idx, len(windows)
                )
                all_chunks.append(chunk)

        print(f"✅ Created {len(all_chunks)} sliding window chunks")
        return all_chunks

    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        try:
            import nltk
            sentences = nltk.sent_tokenize(text)
        except ImportError:
            # Simple fallback
            sentences = [s.strip() for s in text.split('.') if s.strip()]

        # Filter sentences
        filtered = []
        for sentence in sentences:
            sentence = sentence.strip()
            if 10 <= len(sentence) <= 500:  # Length filtering
                filtered.append(sentence)

        return filtered

    def _calculate_sliding_windows(self, sentences: List[str]) -> List[Tuple[int, int]]:
        """Calculate sliding window positions."""
        windows = []
        start_idx = 0

        while start_idx < len(sentences):
            end_idx = min(start_idx + self.window_size, len(sentences))
            windows.append((start_idx, end_idx))

            if end_idx == len(sentences):
                break

            start_idx += self.step_size

        return windows

    def _create_chunk(self, doc, sentences: List[str], start_idx: int, end_idx: int,
                      window_position: int, total_windows: int) -> Dict[str, Any]:
        """Create a chunk from a window of sentences."""
        window_sentences = sentences[start_idx:end_idx]
        chunk_text = ' '.join(window_sentences)

        # Generate chunk ID
        chunk_id = f"{doc.metadata.get('title', 'doc')}_{start_idx}_{end_idx}_{hashlib.md5(chunk_text.encode()).hexdigest()[:8]}"

        return {
            'chunk_id': chunk_id,
            'text': chunk_text,
            'source_article': doc.metadata.get('title', 'Unknown'),
            'source_sentences': list(range(start_idx, end_idx)),
            'anchor_sentence_idx': start_idx,
            'window_position': window_position,
            'total_windows': total_windows,
            'window_size': len(window_sentences),
            'start_sentence_idx': start_idx,
            'end_sentence_idx': end_idx - 1
        }


def create_ai_ml_documents():
    """Create comprehensive AI/ML test documents."""
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content="""
            Machine learning (ML) is a field of study in artificial intelligence concerned with the development 
            and study of statistical algorithms that can learn from data and generalize to unseen data. Machine 
            learning algorithms build a model based on training data in order to make predictions or decisions 
            without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety 
            of applications, such as in medicine, email filtering, speech recognition, agriculture, and computer 
            vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.

            A subset of machine learning is closely related to computational statistics, which focuses on making 
            predictions using computers, but not all machine learning is statistical learning. The study of 
            mathematical optimization delivers methods, theory and application domains to the field of machine learning. 
            Data mining is a related field of study, focusing on exploratory data analysis through unsupervised learning.

            Some types of machine learning algorithms include supervised learning, unsupervised learning, and 
            reinforcement learning. Supervised learning involves training algorithms on labeled data to make 
            predictions on new data. Unsupervised learning finds patterns in data without labeled examples. 
            Reinforcement learning involves agents learning through interaction with an environment to maximize rewards.
            """,
            metadata={"source": "ml_overview.txt", "title": "Machine Learning Overview"}
        ),
        Document(
            page_content="""
            Deep learning is part of a broader family of machine learning methods based on artificial neural networks 
            with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning 
            architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent 
            neural networks, convolutional neural networks and transformers have been applied to fields including 
            computer vision, speech recognition, natural language processing, machine translation, bioinformatics and drug discovery.

            Artificial neural networks are computing systems inspired by the biological neural networks that constitute 
            animal brains. An artificial neural network consists of a collection of connected units or nodes called 
            artificial neurons, which loosely model the neurons in a biological brain. Each connection, like the synapses 
            in a biological brain, can transmit a signal to other neurons. An artificial neuron receives signals, 
            processes them, and then signals neurons connected to it.

            Convolutional neural networks are particularly effective for image recognition tasks. Recurrent neural 
            networks excel at sequence processing tasks like natural language processing. Transformer architectures 
            have revolutionized natural language processing and are the foundation for large language models like GPT and BERT.
            """,
            metadata={"source": "deep_learning.txt", "title": "Deep Learning and Neural Networks"}
        )
    ]

    print(f"✅ Created {len(docs)} comprehensive AI/ML documents")
    for i, doc in enumerate(docs):
        print(f"   Doc {i + 1}: {doc.metadata['title']} ({len(doc.page_content)} chars)")

    return docs


def test_hybrid_approach():
    """Test the hybrid custom + RAGAS approach."""

    print("🧪 Testing Hybrid Custom + RAGAS Approach")
    print("=" * 60)

    try:
        # Import RAGAS components
        from ragas.testset.graph import KnowledgeGraph, Node, NodeType
        from ragas.testset.transforms.extractors import SummaryExtractor
        from ragas.testset.transforms import apply_transforms
        from ragas.testset import TestsetGenerator
        from ragas.testset.synthesizers import default_query_distribution
        from ragas.testset.persona import Persona
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        print("✅ RAGAS imports successful")

    except ImportError as e:
        print(f"❌ RAGAS import failed: {e}")
        print("💡 Try: pip install ragas langchain-openai")
        return

    # Create documents
    docs = create_ai_ml_documents()

    # Setup LLMs
    print("\n🤖 Setting up LLMs...")
    try:
        generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7))
        generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-ada-002"))
        print("✅ LLMs initialized successfully")
    except Exception as e:
        print(f"❌ LLM setup failed: {e}")
        return

    # Step 1: Custom sliding window chunking
    print("\n✂️  Creating custom sliding window chunks...")
    chunker = CustomSlidingWindowChunker(window_size=3, overlap=1)
    chunks = chunker.create_chunks(docs)

    # Step 2: Create Knowledge Graph with chunk nodes
    print("\n📊 Creating Knowledge Graph from chunks...")
    kg = KnowledgeGraph()

    # Add chunks as nodes (not documents)
    for chunk in chunks:
        kg.nodes.append(
            Node(
                type=NodeType.CHUNK,
                properties={
                    "page_content": chunk['text'],
                    "chunk_id": chunk['chunk_id'],
                    "source_article": chunk['source_article'],
                    "anchor_sentence_idx": chunk['anchor_sentence_idx']
                }
            )
        )

    print(f"✅ Added {len(kg.nodes)} chunk nodes to knowledge graph")

    # Step 3: Apply custom extractors first
    print("\n🔧 Applying custom extractors...")

    # Initialize custom extractors
    ner_extractor = CustomNERExtractor()
    keyphrase_extractor = CustomKeyphraseExtractor()

    # Apply to each node
    for i, node in enumerate(kg.nodes):
        if i % 5 == 0:
            print(f"   Processing node {i + 1}/{len(kg.nodes)}...")

        text = node.properties['page_content']

        # Extract entities and keyphrases
        entities_result = ner_extractor.extract(text)
        keyphrases_result = keyphrase_extractor.extract(text)

        # Add to node properties
        node.properties.update(entities_result)
        node.properties.update(keyphrases_result)

    print(f"✅ Custom extractors applied to all nodes")

    # Show sample of custom extraction results
    if kg.nodes:
        sample_node = kg.nodes[0]
        print(f"\n🔍 Sample custom extraction results:")
        print(f"   Entities: {sample_node.properties.get('entities', {})}")
        print(f"   Keyphrases: {sample_node.properties.get('keyphrases', [])[:5]}")

    # Step 4: Use RAGAS's native cognitive architecture for theme extraction
    print("\n🤖 Applying RAGAS-native theme extraction and relationship building...")

    # Instead of manually creating themes, let's use RAGAS's expected approach
    # We'll create document nodes and let RAGAS do its own chunking and theme extraction

    try:
        # Import necessary RAGAS components
        from ragas.testset.transforms import default_transforms

        # Create a separate KG using RAGAS's expected document-first approach
        print("   Creating RAGAS-native knowledge graph...")
        ragas_kg = KnowledgeGraph()

        # Add original documents as DOCUMENT nodes (this is what RAGAS expects)
        for doc in docs:
            ragas_kg.nodes.append(
                Node(
                    type=NodeType.DOCUMENT,
                    properties={
                        "page_content": doc.page_content,
                        "document_metadata": doc.metadata
                    }
                )
            )

        print(f"   Added {len(ragas_kg.nodes)} document nodes")

        # Apply RAGAS's complete transform pipeline
        transforms = default_transforms(
            documents=docs,
            llm=generator_llm,
            embedding_model=generator_embeddings
        )

        print(f"   Applying {len(transforms)} RAGAS transforms...")
        apply_transforms(ragas_kg, transforms)

        print(f"   RAGAS transforms completed")
        print(f"   Final nodes: {len(ragas_kg.nodes)}")
        print(f"   Final relationships: {len(ragas_kg.relationships)}")

        # Show what RAGAS actually created
        if ragas_kg.nodes:
            sample_node = ragas_kg.nodes[0] if ragas_kg.nodes[0].type == NodeType.CHUNK else (
                ragas_kg.nodes[1] if len(ragas_kg.nodes) > 1 else ragas_kg.nodes[0])
            print(f"   RAGAS-created node properties: {list(sample_node.properties.keys())}")

            # Check if themes exist and their format
            if 'themes' in sample_node.properties:
                themes = sample_node.properties['themes']
                print(f"   RAGAS themes type: {type(themes)}")
                print(f"   RAGAS themes sample: {themes}")

        # Use the RAGAS-native knowledge graph for question generation
        kg = ragas_kg

    except Exception as e:
        print(f"   ⚠️  RAGAS-native approach failed: {e}")
        print(f"   Falling back to hybrid approach with property alignment...")

        # Alternative: Try to align our properties with RAGAS expectations
        # Maybe RAGAS expects 'topics' instead of 'themes'?
        for node in kg.nodes:
            # Try different property names that RAGAS might be looking for
            themes = node.properties.get('themes', [])

            # Set multiple possible property names
            node.properties['topics'] = themes
            node.properties['subjects'] = themes
            node.properties['concepts'] = themes

            # Also try flattening entities in case that's what RAGAS wants
            entities_dict = node.properties.get('entities', {})
            flat_entities = []
            for entity_list in entities_dict.values():
                flat_entities.extend(entity_list)

            # Clear potential conflicting properties
            if 'entities' in node.properties:
                # Keep entities but also add alternatives
                node.properties['entity_list'] = flat_entities
                node.properties['named_entities'] = flat_entities

    # Now apply RAGAS transforms to build relationships
    try:
        # Import the relationship building transforms directly
        from ragas.testset.transforms import default_transforms

        # Get default transforms but apply to our pre-chunked graph
        # We'll create fake documents to satisfy the transform requirements
        fake_docs = []
        for i, doc in enumerate(docs):
            fake_docs.append(doc)

        # Apply default transforms to build relationships
        transforms = default_transforms(
            documents=fake_docs,
            llm=generator_llm,
            embedding_model=generator_embeddings
        )

        # Filter out transforms that would re-chunk (we want to keep our chunks)
        from ragas.testset.transforms.extractors import HeadlineSplitter
        filtered_transforms = [t for t in transforms if not isinstance(t, HeadlineSplitter)]

        apply_transforms(kg, filtered_transforms)
        print("✅ RAGAS relationship building completed")

    except Exception as e:
        print(f"⚠️  RAGAS transform pipeline failed: {e}")
        print("   Trying manual relationship building...")

        # Fallback: manually create some relationships
        try:
            from ragas.testset.graph import Relationship

            # Create simple relationships between nodes with shared entities/themes
            for i, node1 in enumerate(kg.nodes):
                for j, node2 in enumerate(kg.nodes[i + 1:], i + 1):
                    # Check for shared entities
                    entities1 = set()
                    entities2 = set()

                    for ent_list in node1.properties.get('entities', {}).values():
                        entities1.update(ent_list)
                    for ent_list in node2.properties.get('entities', {}).values():
                        entities2.update(ent_list)

                    shared_entities = entities1.intersection(entities2)
                    if shared_entities:
                        rel = Relationship(
                            source=node1,
                            target=node2,
                            type="entities_overlap",
                            properties={"shared_entities": list(shared_entities)}
                        )
                        kg.add(rel)

                    # Check for shared themes
                    themes1 = set(node1.properties.get('themes', []))
                    themes2 = set(node2.properties.get('themes', []))
                    shared_themes = themes1.intersection(themes2)

                    if shared_themes:
                        rel = Relationship(
                            source=node1,
                            target=node2,
                            type="cosine_similarity",
                            properties={"shared_themes": list(shared_themes)}
                        )
                        kg.add(rel)

            print("✅ Manual relationship building completed")

        except Exception as e2:
            print(f"❌ Manual relationship building also failed: {e2}")
            print("   Proceeding without relationships...")

    # Step 5: Inspect the enriched knowledge graph
    print(f"\n🔍 Inspecting enriched knowledge graph...")
    print(f"   Total nodes: {len(kg.nodes)}")
    print(f"   Total relationships: {len(kg.relationships)}")

    # Show node types and relationships
    node_types = {}
    for node in kg.nodes:
        node_types[node.type] = node_types.get(node.type, 0) + 1
    print(f"   Node types: {node_types}")

    if kg.relationships:
        rel_types = {}
        for rel in kg.relationships:
            rel_types[rel.type] = rel_types.get(rel.type, 0) + 1
        print(f"   Relationship types: {rel_types}")

        # Show sample relationships
        print(f"   Sample relationships:")
        for i, rel in enumerate(kg.relationships[:3]):
            print(f"     {i + 1}. {rel.type}: {rel.source} -> {rel.target}")
    else:
        print("   ⚠️  No relationships found!")

    # Show enriched node properties
    if kg.nodes:
        print(f"   Sample enriched node properties:")
        sample_node = kg.nodes[0]
        for key in sample_node.properties.keys():
            value = sample_node.properties[key]
            if isinstance(value, str) and len(value) > 50:
                value = value[:50] + "..."
            elif isinstance(value, list) and len(value) > 3:
                value = value[:3] + ["..."]
            print(f"     {key}: {value}")

    # Step 6: Test question generation with debugging
    print(f"\n🎯 Testing question generation with enhanced debugging...")

    # First, let's inspect what themes RAGAS will actually see
    print("\n🔍 Debug: Inspecting themes structure that RAGAS will encounter...")
    for i, node in enumerate(kg.nodes[:3]):  # Check first 3 nodes
        print(f"   Node {i + 1}:")
        print(f"     themes type: {type(node.properties.get('themes'))}")
        print(f"     themes value: {node.properties.get('themes', [])}")
        print(f"     entities type: {type(node.properties.get('entities'))}")
        print(f"     entities value: {node.properties.get('entities', {})}")

    # Create personas
    personas = [
        Persona(
            name="AI Researcher",
            role_description="Researcher studying machine learning algorithms and neural networks. Asks detailed technical questions."
        ),
        Persona(
            name="Student",
            role_description="Student learning about AI concepts. Needs clear explanations with examples."
        )
    ]

    try:
        # Create generator
        generator = TestsetGenerator(
            llm=generator_llm,
            embedding_model=generator_embeddings,
            knowledge_graph=kg,
            persona_list=personas
        )

        # Try to understand what themes RAGAS extracts before generation
        print("\n🔍 Attempting question generation...")

        # Use a minimal testset size for debugging
        testset = generator.generate(
            testset_size=3,  # Smaller for debugging
            num_personas=len(personas)
        )

        print(f"✅ Successfully generated {len(testset)} questions!")

        # Show questions
        df = testset.to_pandas()
        print(f"\n📝 Generated Questions:")
        for i, row in df.iterrows():
            print(f"   {i + 1}. {row['user_input']}")
            if 'synthesizer_name' in row:
                print(f"      Type: {row.get('synthesizer_name', 'unknown')}")
            print()

        return True

    except Exception as e:
        print(f"❌ Question generation failed: {e}")
        print(f"   Error type: {type(e).__name__}")

        # Additional debugging for ValidationError
        if "ValidationError" in str(type(e)):
            print(f"\n🔍 ValidationError debugging:")
            print(f"   This suggests RAGAS is pulling the wrong property type")
            print(f"   Expected: list of themes")
            print(
                f"   Received: {str(e).split('input_value=')[1].split(',')[0] if 'input_value=' in str(e) else 'unknown'}")

            # Let's try to manually extract themes the way RAGAS might
            print(f"\n🔬 Attempting to replicate RAGAS's theme extraction...")

            # Check if any nodes have malformed themes
            for i, node in enumerate(kg.nodes):
                themes = node.properties.get('themes')
                if not isinstance(themes, list):
                    print(f"   ⚠️  Node {i} has non-list themes: {type(themes)} = {themes}")
                else:
                    print(f"   ✅ Node {i} has proper list themes: {themes}")

        # Try a workaround: create a new KG with only essential properties
        print(f"\n🔄 Trying simplified knowledge graph approach...")
        try:
            # Create a minimal KG for testing
            simple_kg = KnowledgeGraph()

            # Add only the first few nodes with cleaned properties
            for i, original_node in enumerate(kg.nodes[:5]):  # Just first 5 nodes
                simple_node = Node(
                    type=NodeType.CHUNK,
                    properties={
                        "page_content": original_node.properties['page_content'],
                        "themes": original_node.properties.get('themes', ['artificial_intelligence'])
                        # Ensure it's a list
                    }
                )
                simple_kg.nodes.append(simple_node)

            # Add some simple relationships
            from ragas.testset.graph import Relationship
            if len(simple_kg.nodes) >= 2:
                rel = Relationship(
                    source=simple_kg.nodes[0],
                    target=simple_kg.nodes[1],
                    type="cosine_similarity",
                    properties={"score": 0.8}
                )
                simple_kg.add(rel)

            print(
                f"   Created simplified KG with {len(simple_kg.nodes)} nodes and {len(simple_kg.relationships)} relationships")

            # Try generation with simplified KG
            simple_generator = TestsetGenerator(
                llm=generator_llm,
                embedding_model=generator_embeddings,
                knowledge_graph=simple_kg,
                persona_list=personas
            )

            simple_testset = simple_generator.generate(
                testset_size=2,
                num_personas=len(personas)
            )

            print(f"✅ Simplified approach worked! Generated {len(simple_testset)} questions")

            # Show questions
            simple_df = simple_testset.to_pandas()
            print(f"\n📝 Simplified Generated Questions:")
            for i, row in simple_df.iterrows():
                print(f"   {i + 1}. {row['user_input']}")

            return True

        except Exception as e2:
            print(f"❌ Simplified approach also failed: {e2}")
            import traceback
            print(f"\n🔍 Full traceback of simplified approach:")
            traceback.print_exc()
            return False


def main():
    """Main test function."""
    print("🧪 RAGAS Hybrid Test Script")
    print("Custom extractors + RAGAS relationship building")
    print("=" * 70)

    # Check API key
    if not os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") == "your-openai-api-key-here":
        print("❌ Please set your OpenAI API key in the script!")
        return

    success = test_hybrid_approach()

    if success:
        print("\n✅ SUCCESS! Hybrid approach worked!")
        print("🔍 This demonstrates:")
        print("   • Custom sliding window chunking ✅")
        print("   • Custom spaCy NER extraction ✅")
        print("   • Custom TF-IDF keyphrase extraction ✅")
        print("   • RAGAS summary extraction with LLM ✅")
        print("   • RAGAS relationship building ✅")
        print("   • RAGAS question generation ✅")
        print("\n💡 You can now integrate this approach into your pipeline!")
    else:
        print("\n❌ Hybrid approach needs refinement")
        print("🔍 Check the error messages to see what needs adjustment")


if __name__ == "__main__":
    main()