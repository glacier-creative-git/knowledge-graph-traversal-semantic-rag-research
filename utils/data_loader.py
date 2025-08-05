"""
Data Loading Utilities for Research Pipeline
===========================================

Handles SQuAD dataset loading, question selection, and context preparation
for the semantic graph traversal research pipeline.
"""

import random
import nltk
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
from .config import DataConfig

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class SQuADDataLoader:
    """Handles loading and preparing SQuAD datasets for research"""

    def __init__(self, config: DataConfig):
        self.config = config
        self.squad_data = None

    def load_squad_data(self, version: str = "2.0") -> bool:
        """
        Load SQuAD dataset

        Args:
            version: "1.1" or "2.0"

        Returns:
            True if successful, False otherwise
        """
        dataset_name = "squad_v2" if version == "2.0" else "squad"

        try:
            print(f"📚 Loading SQuAD {version} dataset ({self.config.num_samples} samples)...")
            self.squad_data = load_dataset(dataset_name, split=f"validation[:{self.config.num_samples}]")
            print(f"✅ Loaded {len(self.squad_data)} SQuAD {version} samples")
            return True
        except Exception as e:
            print(f"❌ Failed to load SQuAD {version}: {e}")
            if version == "2.0":
                print("🔄 Falling back to SQuAD 1.1...")
                return self.load_squad_data("1.1")
            return False

    def select_random_question_with_contexts(self, num_contexts: int = 5) -> Tuple[str, List[Dict]]:
        """
        Select a random question and gather multiple related contexts

        Args:
            num_contexts: Number of contexts to include

        Returns:
            Tuple of (question, contexts_list)
        """
        if not self.squad_data:
            raise ValueError("SQuAD data not loaded. Call load_squad_data() first.")

        # Filter for substantial contexts
        substantial_samples = [
            sample for sample in self.squad_data
            if len(sample['context']) >= self.config.min_context_length
        ]

        if not substantial_samples:
            # Fallback to shorter contexts
            substantial_samples = [
                sample for sample in self.squad_data
                if len(sample['context']) >= 200
            ]

        if not substantial_samples:
            raise ValueError("No suitable contexts found in dataset")

        # Add randomness
        random.shuffle(substantial_samples)

        # Select primary sample
        primary_sample = random.choice(substantial_samples)
        selected_question = primary_sample['question']

        print(f"🎯 Selected Question: {selected_question}")

        # Get the primary context
        primary_context = {
            'context': primary_sample['context'],
            'question': primary_sample['question'],
            'id': primary_sample['id'],
            'title': f"Document 1: {primary_sample['id']}",
            'answers': primary_sample.get('answers', {'text': [], 'answer_start': []})
        }

        # Find related contexts using keyword overlap
        question_keywords = set(selected_question.lower().split())
        context_scores = []
        seen_contexts = {primary_sample['context'].strip()}

        for sample in substantial_samples:
            if sample['id'] == primary_sample['id']:
                continue

            context_text = sample['context'].strip()
            if context_text in seen_contexts:
                continue

            # Calculate relevance score
            context_keywords = set(sample['context'].lower().split())
            overlap_score = len(question_keywords.intersection(context_keywords))
            question_overlap = len(set(sample['question'].lower().split()).intersection(question_keywords))
            total_score = overlap_score + (question_overlap * 2)

            if total_score > 0:
                context_scores.append((total_score, sample))
                seen_contexts.add(context_text)

        # Sort by relevance and select top contexts
        context_scores.sort(key=lambda x: x[0], reverse=True)
        contexts = [primary_context]

        for i, (score, sample) in enumerate(context_scores[:num_contexts - 1]):
            context = {
                'context': sample['context'],
                'question': sample['question'],
                'id': sample['id'],
                'title': f"Document {i + 2}: {sample['id']}",
                'answers': sample.get('answers', {'text': [], 'answer_start': []})
            }
            contexts.append(context)
            print(f"📄 Added Related Context: {len(sample['context'])} chars (score: {score})")

        print(f"✅ Created dataset with {len(contexts)} contexts")
        return selected_question, contexts

    def create_focused_context_set(self, topic_keywords: List[str], num_contexts: int = 5) -> Tuple[str, List[Dict]]:
        """
        Create a focused set of contexts around specific topics

        Args:
            topic_keywords: List of keywords to focus on
            num_contexts: Number of contexts to include

        Returns:
            Tuple of (question, contexts_list)
        """
        if not self.squad_data:
            raise ValueError("SQuAD data not loaded. Call load_squad_data() first.")

        print(f"🔍 Searching for contexts containing: {topic_keywords}")

        # Find contexts with topic keywords
        topic_contexts = []
        seen_contexts = set()

        for sample in self.squad_data:
            context_lower = sample['context'].lower()
            question_lower = sample['question'].lower()
            context_text = sample['context'].strip()

            if context_text in seen_contexts:
                continue

            # Count keyword matches
            keyword_matches = sum(1 for keyword in topic_keywords
                                  if keyword in context_lower or keyword in question_lower)

            if keyword_matches >= 2 and len(sample['context']) >= 200:
                topic_contexts.append((keyword_matches, sample))
                seen_contexts.add(context_text)

        if not topic_contexts:
            print("⚠️ No focused contexts found, falling back to random selection")
            return self.select_random_question_with_contexts(num_contexts)

        # Sort by relevance
        topic_contexts.sort(key=lambda x: x[0], reverse=True)
        selected_samples = [sample for _, sample in topic_contexts[:num_contexts]]

        # Use most relevant question
        primary_sample = selected_samples[0]
        selected_question = primary_sample['question']

        print(f"🎯 Selected Focused Question: {selected_question}")

        # Convert to context format
        contexts = []
        for i, sample in enumerate(selected_samples):
            context = {
                'context': sample['context'],
                'question': sample['question'],
                'id': sample['id'],
                'title': f"Document {i + 1}: {sample['id']}",
                'answers': sample.get('answers', {'text': [], 'answer_start': []})
            }
            contexts.append(context)
            print(f"📄 Context {i + 1}: {len(sample['context'])} chars")

        print(f"✅ Created focused dataset with {len(contexts)} contexts")
        return selected_question, contexts

    def create_demo_dataset(self) -> Tuple[str, List[Dict]]:
        """
        Create a simple demo dataset for testing

        Returns:
            Tuple of (question, contexts_list)
        """
        demo_contexts = [
            {
                'context': """Artificial intelligence has revolutionized many fields in recent years. Machine learning algorithms 
                can now process vast amounts of data to identify complex patterns that were previously invisible to 
                human analysts. Deep learning networks use multiple layers of artificial neurons to extract features 
                from raw input data in ways that mimic human cognition. Natural language processing enables computers 
                to understand and generate human text with remarkable accuracy.""",
                'question': 'How has artificial intelligence changed data analysis?',
                'id': 'demo_ai_1',
                'title': 'Document 1: AI Revolution',
                'answers': {'text': [
                    'Machine learning algorithms can now process vast amounts of data to identify complex patterns'],
                            'answer_start': [89]}
            },
            {
                'context': """The field of computer vision has made tremendous advances with the introduction of convolutional 
                neural networks. These sophisticated models can now recognize objects, faces, and even emotions in images 
                with superhuman accuracy. Medical imaging has particularly benefited from these advances, with AI systems 
                now capable of detecting cancers and other diseases earlier than human doctors in many cases.""",
                'question': 'What advances have been made in computer vision?',
                'id': 'demo_cv_1',
                'title': 'Document 2: Computer Vision',
                'answers': {'text': ['convolutional neural networks'], 'answer_start': [85]}
            }
        ]

        demo_question = "How is AI being used in medical applications?"

        print(f"🎯 Demo Question: {demo_question}")
        print(f"📄 Created demo dataset with {len(demo_contexts)} contexts")

        return demo_question, demo_contexts

    def get_evaluation_dataset(self, max_samples: int = 50) -> Dict:
        """
        Get a dataset suitable for RAGAS evaluation

        Args:
            max_samples: Maximum number of samples to include

        Returns:
            Dictionary with documents, queries, and ground truths
        """
        if not self.squad_data:
            if not self.load_squad_data():
                # Fallback to demo data
                question, contexts = self.create_demo_dataset()
                return {
                    'documents': contexts,
                    'queries': [question],
                    'ground_truths': [
                        contexts[0]['answers']['text'][0] if contexts[0]['answers']['text'] else "Demo answer"],
                    'name': 'Demo Dataset'
                }

        # Use actual SQuAD data
        documents = []
        queries = []
        ground_truths = []

        sample_count = min(max_samples, len(self.squad_data))

        for i in range(sample_count):
            item = self.squad_data[i]
            documents.append({
                'context': item['context'],
                'question': item['question'],
                'id': item['id']
            })
            queries.append(item['question'])

            if item['answers']['text']:
                ground_truths.append(item['answers']['text'][0])
            else:
                ground_truths.append("No answer can be determined from the given context")

        return {
            'documents': documents,
            'queries': queries,
            'ground_truths': ground_truths,
            'name': f'SQuAD Dataset ({sample_count} samples)'
        }


# Convenience functions for easy notebook usage
def load_demo_data() -> Tuple[str, List[Dict]]:
    """Quick function to load demo data"""
    from .config import DataConfig
    loader = SQuADDataLoader(DataConfig())
    return loader.create_demo_dataset()


def load_random_squad_data(num_contexts: int = 5) -> Tuple[str, List[Dict]]:
    """Quick function to load random SQuAD data"""
    from .config import DataConfig
    loader = SQuADDataLoader(DataConfig())
    if loader.load_squad_data():
        return loader.select_random_question_with_contexts(num_contexts)
    else:
        print("⚠️ Falling back to demo data")
        return loader.create_demo_dataset()


def load_focused_squad_data(topics: List[str] = None, num_contexts: int = 5) -> Tuple[str, List[Dict]]:
    """Quick function to load focused SQuAD data"""
    if topics is None:
        topics = ['technology', 'computer', 'science', 'research', 'data', 'system']

    from .config import DataConfig
    loader = SQuADDataLoader(DataConfig())
    if loader.load_squad_data():
        return loader.create_focused_context_set(topics, num_contexts)
    else:
        print("⚠️ Falling back to demo data")
        return loader.create_demo_dataset()