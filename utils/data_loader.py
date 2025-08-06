"""
Enhanced Data Loading System for RAG Research
============================================

Modular data loading system supporting WikiEval, Natural Questions, and extensible
for future datasets. Replaces SQuAD-dependent loading with a universal interface.
"""

import random
import nltk
from typing import List, Dict, Tuple, Optional, Any
from datasets import load_dataset
from abc import ABC, abstractmethod
from .config import DataConfig
import logging

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class BaseDataLoader(ABC):
    """
    Abstract base class for all dataset loaders.

    This ensures all loaders provide the same interface regardless of underlying dataset,
    enabling seamless swapping between WikiEval, Natural Questions, and future datasets.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.dataset = None
        self.dataset_name = self.get_dataset_name()

    @abstractmethod
    def get_dataset_name(self) -> str:
        """Return the name of this dataset for identification"""
        pass

    @abstractmethod
    def load_dataset(self) -> bool:
        """Load the dataset. Returns True if successful, False otherwise"""
        pass

    @abstractmethod
    def select_random_question_with_contexts(self, num_contexts: int = 5) -> Tuple[str, List[Dict]]:
        """Select a random question and gather related contexts"""
        pass

    @abstractmethod
    def create_focused_context_set(self, topic_keywords: List[str], num_contexts: int = 5) -> Tuple[str, List[Dict]]:
        """Create focused contexts around specific topics"""
        pass

    def create_demo_dataset(self) -> Tuple[str, List[Dict]]:
        """
        Create a simple demo dataset for testing.
        Default implementation - can be overridden by specific loaders.
        """
        demo_contexts = [
            {
                'context': """Artificial intelligence has revolutionized many fields in recent years. Machine learning algorithms 
                can now process vast amounts of data to identify complex patterns that were previously invisible to 
                human analysts. Deep learning networks use multiple layers of artificial neurons to extract features 
                from raw input data in ways that mimic human cognition.""",
                'question': 'How has artificial intelligence changed data analysis?',
                'id': 'demo_ai_1',
                'title': 'Document 1: AI Revolution',
                'answers': {'text': ['Machine learning algorithms can now process vast amounts of data'], 'answer_start': [89]}
            },
            {
                'context': """The field of computer vision has made tremendous advances with the introduction of convolutional 
                neural networks. These sophisticated models can now recognize objects, faces, and even emotions in images 
                with superhuman accuracy. Medical imaging has particularly benefited from these advances.""",
                'question': 'What advances have been made in computer vision?',
                'id': 'demo_cv_1',
                'title': 'Document 2: Computer Vision',
                'answers': {'text': ['convolutional neural networks'], 'answer_start': [85]}
            }
        ]

        demo_question = f"How is AI being used in {self.dataset_name} applications?"
        logger.info(f"🎯 Demo Question: {demo_question}")
        logger.info(f"📄 Created demo dataset with {len(demo_contexts)} contexts")
        return demo_question, demo_contexts

    @abstractmethod
    def get_evaluation_dataset(self, max_samples: int = 50) -> Dict[str, Any]:
        """Get dataset suitable for RAGAS evaluation"""
        pass


class WikiEvalDataLoader(BaseDataLoader):
    """
    Data loader for the WikiEval dataset from explodinggradients/ragas.

    WikiEval contains 50 Wikipedia pages with human-annotated question-answer pairs
    specifically designed for RAG system evaluation.
    """

    def get_dataset_name(self) -> str:
        return "WikiEval"

    def load_dataset(self) -> bool:
        """Load WikiEval dataset from HuggingFace"""
        try:
            logger.info(f"📚 Loading WikiEval dataset...")
            self.dataset = load_dataset("explodinggradients/WikiEval", split="train")
            logger.info(f"✅ Loaded {len(self.dataset)} WikiEval samples")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load WikiEval: {e}")
            return False

    def select_random_question_with_contexts(self, num_contexts: int = 5) -> Tuple[str, List[Dict]]:
        """
        Select random question from WikiEval with related contexts.
        WikiEval questions are already paired with their source context.
        """
        if not self.dataset:
            raise ValueError("WikiEval dataset not loaded. Call load_dataset() first.")

        # WikiEval has fewer samples (50), so we work with what we have
        available_samples = list(self.dataset)
        num_contexts = min(num_contexts, len(available_samples))

        # Select primary sample
        primary_sample = random.choice(available_samples)
        selected_question = primary_sample['question']

        logger.info(f"🎯 Selected WikiEval Question: {selected_question}")

        # Create contexts list starting with primary
        contexts = [{
            'context': primary_sample['source'],  # WikiEval uses 'source' field
            'question': primary_sample['question'],
            'id': f"wikieval_{hash(primary_sample['source']) % 100000}",
            'title': f"Document 1: WikiEval Context",
            'answers': {'text': [primary_sample.get('answer', '')], 'answer_start': [0]}
        }]

        # Add additional contexts if available
        remaining_samples = [s for s in available_samples if s != primary_sample]
        for i, sample in enumerate(random.sample(remaining_samples, min(num_contexts-1, len(remaining_samples)))):
            contexts.append({
                'context': sample['source'],
                'question': sample['question'],
                'id': f"wikieval_{hash(sample['source']) % 100000}",
                'title': f"Document {i+2}: WikiEval Context",
                'answers': {'text': [sample.get('answer', '')], 'answer_start': [0]}
            })

        logger.info(f"✅ Created WikiEval dataset with {len(contexts)} contexts")
        return selected_question, contexts

    def create_focused_context_set(self, topic_keywords: List[str], num_contexts: int = 5) -> Tuple[str, List[Dict]]:
        """
        Create focused contexts from WikiEval based on topic keywords.
        """
        if not self.dataset:
            raise ValueError("WikiEval dataset not loaded. Call load_dataset() first.")

        logger.info(f"🔍 Searching WikiEval for topics: {topic_keywords}")

        # Score samples based on keyword relevance
        scored_samples = []
        for sample in self.dataset:
            source_lower = sample['source'].lower()
            question_lower = sample['question'].lower()

            score = sum(1 for keyword in topic_keywords
                       if keyword.lower() in source_lower or keyword.lower() in question_lower)

            if score > 0:
                scored_samples.append((score, sample))

        if not scored_samples:
            logger.warning("⚠️ No focused WikiEval contexts found, using random selection")
            return self.select_random_question_with_contexts(num_contexts)

        # Sort by relevance and select best
        scored_samples.sort(key=lambda x: x[0], reverse=True)
        selected_samples = [sample for _, sample in scored_samples[:num_contexts]]

        # Use highest scoring sample for the question
        primary_sample = selected_samples[0]
        selected_question = primary_sample['question']

        logger.info(f"🎯 Selected Focused WikiEval Question: {selected_question}")

        # Build contexts
        contexts = []
        for i, sample in enumerate(selected_samples):
            contexts.append({
                'context': sample['source'],
                'question': sample['question'],
                'id': f"wikieval_focused_{hash(sample['source']) % 100000}",
                'title': f"Document {i+1}: WikiEval Focused",
                'answers': {'text': [sample.get('answer', '')], 'answer_start': [0]}
            })

        logger.info(f"✅ Created focused WikiEval dataset with {len(contexts)} contexts")
        return selected_question, contexts

    def get_evaluation_dataset(self, max_samples: int = 50) -> Dict[str, Any]:
        """Get WikiEval evaluation dataset"""
        if not self.dataset:
            if not self.load_dataset():
                return self._get_demo_evaluation_dataset()

        # WikiEval is small (50 samples), use all available
        sample_count = min(max_samples, len(self.dataset))

        documents = []
        queries = []
        ground_truths = []

        for i, item in enumerate(self.dataset):
            if i >= sample_count:
                break

            documents.append({
                'context': item['source'],
                'question': item['question'],
                'id': f"wikieval_{i}"
            })
            queries.append(item['question'])
            ground_truths.append(item.get('answer', 'Answer not available'))

        return {
            'documents': documents,
            'queries': queries,
            'ground_truths': ground_truths,
            'name': f'WikiEval Dataset ({sample_count} samples)'
        }

    def _get_demo_evaluation_dataset(self) -> Dict[str, Any]:
        """Fallback demo dataset for WikiEval"""
        question, contexts = self.create_demo_dataset()
        return {
            'documents': contexts,
            'queries': [question],
            'ground_truths': [contexts[0]['answers']['text'][0] if contexts[0]['answers']['text'] else "Demo answer"],
            'name': 'WikiEval Demo Dataset'
        }


class NaturalQuestionsDataLoader(BaseDataLoader):
    """
    Data loader for Google's Natural Questions dataset.

    Natural Questions contains real user queries from Google search with Wikipedia-based answers.
    Features both short and long answer annotations.
    """

    def get_dataset_name(self) -> str:
        return "Natural Questions"

    def load_dataset(self) -> bool:
        """Load Natural Questions dataset from HuggingFace with streaming to avoid disk usage"""
        # Set environment variable to avoid tokenizer warnings
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        try:
            # Use streaming to avoid downloading massive dataset to disk
            split_name = getattr(self.config, 'nq_split', 'validation')
            max_samples = getattr(self.config, 'nq_max_samples', 1000)
            use_streaming = getattr(self.config, 'nq_streaming', True)

            logger.info(f"📚 Loading Natural Questions dataset...")
            logger.info(f"   Split: {split_name}, Max samples: {max_samples}, Streaming: {use_streaming}")

            # First, try traditional loading with a small sample - most reliable
            logger.info("🔄 Trying traditional loading with small sample first...")
            try:
                small_sample_size = min(100, max_samples)
                split_spec = f"{split_name}[:{small_sample_size}]"

                dataset = load_dataset("natural_questions", split=split_spec)

                # Convert to list and filter
                if hasattr(dataset, '__iter__'):
                    dataset_samples = []
                    for sample in dataset:
                        # Basic validation
                        if (sample.get('document', {}).get('html') and
                            1000 <= len(sample['document']['html']) <= 30000):
                            dataset_samples.append(sample)

                        # Stop when we have enough good samples
                        if len(dataset_samples) >= min(50, max_samples):
                            break

                    if dataset_samples:
                        # Extend to requested size if needed and possible
                        if len(dataset_samples) < max_samples and len(dataset) > small_sample_size:
                            logger.info(f"📈 Expanding sample size to get {max_samples} samples...")
                            try:
                                larger_sample_size = min(max_samples * 3, 1000)  # Cap at 1000
                                split_spec_large = f"{split_name}[:{larger_sample_size}]"
                                larger_dataset = load_dataset("natural_questions", split=split_spec_large)

                                for sample in larger_dataset:
                                    if len(dataset_samples) >= max_samples:
                                        break
                                    if (sample.get('document', {}).get('html') and
                                        1000 <= len(sample['document']['html']) <= 30000 and
                                        sample not in dataset_samples):  # Avoid duplicates
                                        dataset_samples.append(sample)

                            except Exception as expand_error:
                                logger.warning(f"⚠️ Could not expand sample size: {expand_error}")
                                # Continue with smaller sample

                        self.dataset = dataset_samples
                        logger.info(f"✅ Successfully loaded {len(self.dataset)} Natural Questions samples via traditional loading")
                        return True
                    else:
                        raise ValueError("No valid samples found in traditional loading")

            except Exception as traditional_error:
                logger.warning(f"⚠️ Traditional loading failed: {traditional_error}")

                # Fallback to streaming if traditional fails
                if use_streaming:
                    logger.info("🌊 Falling back to streaming approach...")
                    try:
                        streaming_dataset = load_dataset(
                            "natural_questions",
                            split=split_name,
                            streaming=True
                        )

                        # Convert streaming dataset to list with better filtering
                        logger.info(f"📊 Processing samples from stream...")
                        dataset_samples = []
                        samples_processed = 0
                        max_process_attempts = max_samples * 5  # Reasonable limit

                        for sample in streaming_dataset:
                            samples_processed += 1

                            # Check if sample has required fields and reasonable size
                            doc = sample.get('document', {})
                            html = doc.get('html', '')

                            if html and 1000 <= len(html) <= 30000:  # Reasonable size range
                                dataset_samples.append(sample)

                                # Progress logging
                                if len(dataset_samples) % 25 == 0:
                                    logger.info(f"   ✅ Collected {len(dataset_samples)} valid samples (processed {samples_processed})...")

                            # Stop conditions
                            if len(dataset_samples) >= max_samples:
                                logger.info(f"   🎯 Reached target of {max_samples} samples")
                                break

                            if samples_processed >= max_process_attempts:
                                logger.info(f"   ⚠️ Processed {samples_processed} samples, stopping search")
                                break

                        if dataset_samples:
                            self.dataset = dataset_samples
                            logger.info(f"✅ Successfully loaded {len(self.dataset)} Natural Questions samples via streaming")
                            return True
                        else:
                            raise ValueError(f"No valid samples found after processing {samples_processed} examples via streaming")

                    except Exception as streaming_error:
                        logger.error(f"❌ Streaming also failed: {streaming_error}")

            # If we get here, both methods failed - try absolute minimal fallback
            logger.info("🆘 Trying minimal fallback...")
            try:
                minimal_dataset = load_dataset("natural_questions", split="validation[:10]")

                if minimal_dataset and len(minimal_dataset) > 0:
                    # Take whatever we can get
                    dataset_samples = []
                    for sample in minimal_dataset:
                        if sample.get('document', {}).get('html'):
                            dataset_samples.append(sample)

                    if dataset_samples:
                        self.dataset = dataset_samples
                        logger.warning(f"⚠️ Using minimal fallback: {len(self.dataset)} samples")
                        return True

            except Exception as minimal_error:
                logger.error(f"❌ Even minimal fallback failed: {minimal_error}")

            # Complete failure
            logger.error("❌ All Natural Questions loading methods failed")
            logger.error("💡 Troubleshooting suggestions:")
            logger.error("   - Check internet connection")
            logger.error("   - Try setting nq_streaming=False in config")
            logger.error("   - Use WikiEval dataset instead: dataset_name='wikieval'")
            return False

        except Exception as e:
            logger.error(f"❌ Unexpected error in load_dataset: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return False

    def select_random_question_with_contexts(self, num_contexts: int = 5) -> Tuple[str, List[Dict]]:
        """
        Select random question from Natural Questions with related contexts.
        """
        if not self.dataset:
            raise ValueError("Natural Questions dataset not loaded. Call load_dataset() first.")

        # Filter for samples with valid answers and reasonable size HTML
        valid_samples = []
        for sample in self.dataset:
            doc = sample.get('document', {})
            html_content = doc.get('html', '')

            # Skip samples with no HTML or extremely large HTML
            if html_content and 200 <= len(html_content) <= 50000:  # Reasonable size range
                valid_samples.append(sample)

            # Stop after we have enough valid samples to choose from
            if len(valid_samples) >= max(100, num_contexts * 5):
                break

        if not valid_samples:
            logger.warning("⚠️ No valid Natural Questions samples found, using demo dataset")
            return self.create_demo_dataset()

        # Select primary sample
        primary_sample = random.choice(valid_samples)
        selected_question = primary_sample['question']['text']

        logger.info(f"🎯 Selected Natural Questions Query: {selected_question}")

        # Extract context from HTML (simplified - in practice you'd want to parse HTML properly)
        primary_context = self._extract_text_from_html(primary_sample['document']['html'])

        contexts = [{
            'context': primary_context,
            'question': selected_question,
            'id': primary_sample.get('id', f"nq_{hash(primary_context) % 100000}"),
            'title': f"Document 1: {primary_sample['document'].get('title', 'Natural Questions Context')}",
            'answers': self._extract_answers(primary_sample)
        }]

        # Add related contexts by finding similar questions
        remaining_samples = [s for s in valid_samples if s != primary_sample]
        question_keywords = set(selected_question.lower().split())

        scored_samples = []
        for sample in remaining_samples[:100]:  # Limit search for performance
            other_question = sample['question']['text'].lower()
            overlap = len(question_keywords.intersection(set(other_question.split())))
            if overlap > 0:
                scored_samples.append((overlap, sample))

        # Select top related samples
        scored_samples.sort(key=lambda x: x[0], reverse=True)
        for i, (score, sample) in enumerate(scored_samples[:num_contexts-1]):
            context_text = self._extract_text_from_html(sample['document']['html'])
            contexts.append({
                'context': context_text,
                'question': sample['question']['text'],
                'id': sample.get('id', f"nq_{hash(context_text) % 100000}"),
                'title': f"Document {i+2}: {sample['document'].get('title', 'Natural Questions Context')}",
                'answers': self._extract_answers(sample)
            })
            logger.info(f"📄 Added Related Context: {len(context_text)} chars (relevance: {score})")

        logger.info(f"✅ Created Natural Questions dataset with {len(contexts)} contexts")
        return selected_question, contexts

    def create_focused_context_set(self, topic_keywords: List[str], num_contexts: int = 5) -> Tuple[str, List[Dict]]:
        """Create focused contexts from Natural Questions based on topic keywords"""
        if not self.dataset:
            raise ValueError("Natural Questions dataset not loaded. Call load_dataset() first.")

        logger.info(f"🔍 Searching Natural Questions for topics: {topic_keywords}")

        # Score samples based on keyword relevance - limit search for performance
        topic_samples = []
        samples_checked = 0
        max_samples_to_check = 1000  # Limit search scope for streaming datasets

        for sample in self.dataset:
            if samples_checked >= max_samples_to_check:
                break

            if not sample.get('document', {}).get('html'):
                continue

            question_text = sample['question']['text'].lower()
            # Use title for search instead of full HTML for performance
            document_title = sample['document'].get('title', '').lower()

            keyword_matches = sum(1 for keyword in topic_keywords
                                 if keyword.lower() in question_text or keyword.lower() in document_title)

            if keyword_matches >= 1 and 1000 <= len(sample['document']['html']) <= 30000:  # Reasonable size range
                topic_samples.append((keyword_matches, sample))

            samples_checked += 1

            # Stop early if we have plenty of good matches
            if len(topic_samples) >= num_contexts * 3:
                break

        if not topic_samples:
            logger.warning("⚠️ No focused Natural Questions contexts found, using random selection")
            return self.select_random_question_with_contexts(num_contexts)

        # Sort by relevance
        topic_samples.sort(key=lambda x: x[0], reverse=True)
        selected_samples = [sample for _, sample in topic_samples[:num_contexts]]

        # Use most relevant question
        primary_sample = selected_samples[0]
        selected_question = primary_sample['question']['text']

        logger.info(f"🎯 Selected Focused Natural Questions Query: {selected_question}")

        # Build contexts
        contexts = []
        for i, sample in enumerate(selected_samples):
            context_text = self._extract_text_from_html(sample['document']['html'])
            contexts.append({
                'context': context_text,
                'question': sample['question']['text'],
                'id': sample.get('id', f"nq_focused_{i}"),
                'title': f"Document {i+1}: {sample['document'].get('title', 'Natural Questions')}",
                'answers': self._extract_answers(sample)
            })
            logger.info(f"📄 Context {i+1}: {len(context_text)} chars")

        logger.info(f"✅ Created focused Natural Questions dataset with {len(contexts)} contexts")
        return selected_question, contexts

    def get_evaluation_dataset(self, max_samples: int = 50) -> Dict[str, Any]:
        """Get Natural Questions evaluation dataset"""
        if not self.dataset:
            if not self.load_dataset():
                return self._get_demo_evaluation_dataset()

        sample_count = min(max_samples, len(self.dataset))

        documents = []
        queries = []
        ground_truths = []

        processed = 0
        for item in self.dataset:
            if processed >= sample_count:
                break

            if not item.get('document', {}).get('html'):
                continue

            context_text = self._extract_text_from_html(item['document']['html'])

            documents.append({
                'context': context_text,
                'question': item['question']['text'],
                'id': item.get('id', f"nq_eval_{processed}")
            })
            queries.append(item['question']['text'])

            # Extract ground truth answer
            answers = self._extract_answers(item)
            ground_truth = answers['text'][0] if answers['text'] else "Answer not available in context"
            ground_truths.append(ground_truth)

            processed += 1

        return {
            'documents': documents,
            'queries': queries,
            'ground_truths': ground_truths,
            'name': f'Natural Questions Dataset ({processed} samples)'
        }

    def _extract_text_from_html(self, html_content: str) -> str:
        """
        Extract plain text from HTML content.

        This is a simplified extraction - in production you'd want more sophisticated
        HTML parsing to preserve structure and handle formatting properly.
        """
        import re

        # Remove HTML tags (basic approach)
        text = re.sub(r'<[^>]+>', ' ', html_content)

        # Clean up whitespace
        text = ' '.join(text.split())

        # Limit length for practical processing
        if len(text) > 10000:
            text = text[:10000] + "..."

        return text

    def _extract_answers(self, sample: Dict) -> Dict[str, List]:
        """Extract answers from Natural Questions sample format"""
        answers = {'text': [], 'answer_start': []}

        # Natural Questions has complex answer structure
        if 'annotations' in sample and sample['annotations']:
            annotation = sample['annotations'][0]  # Use first annotation

            # Extract short answers if available
            if 'short_answers' in annotation and annotation['short_answers']:
                for short_answer in annotation['short_answers']:
                    if 'text' in short_answer:
                        answers['text'].append(short_answer['text'])
                        answers['answer_start'].append(short_answer.get('start_byte', 0))

            # Fall back to long answer if no short answers
            elif 'long_answer' in annotation and annotation['long_answer'].get('candidate_index', -1) >= 0:
                # Long answer would require more complex extraction from HTML
                answers['text'].append("Long answer available - requires HTML parsing")
                answers['answer_start'].append(0)

        # If no answers found, provide placeholder
        if not answers['text']:
            answers['text'].append("Answer extraction requires further processing")
            answers['answer_start'].append(0)

        return answers

    def _get_demo_evaluation_dataset(self) -> Dict[str, Any]:
        """Fallback demo dataset for Natural Questions"""
        question, contexts = self.create_demo_dataset()
        return {
            'documents': contexts,
            'queries': [question],
            'ground_truths': [contexts[0]['answers']['text'][0] if contexts[0]['answers']['text'] else "Demo answer"],
            'name': 'Natural Questions Demo Dataset'
        }


# Factory function for dataset loader creation
def create_data_loader(dataset_name: str, config: DataConfig = None) -> BaseDataLoader:
    """
    Factory function to create appropriate data loader based on dataset name.

    This is the main interface for swapping between datasets - like changing
    Gameboy cartridges, just specify which dataset you want.

    Args:
        dataset_name: Name of dataset ("wikieval", "natural_questions")
        config: Data configuration object

    Returns:
        Appropriate data loader instance
    """
    if config is None:
        config = DataConfig()

    dataset_name = dataset_name.lower()

    if dataset_name == "wikieval":
        return WikiEvalDataLoader(config)
    elif dataset_name in ["natural_questions", "nq"]:
        return NaturalQuestionsDataLoader(config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported datasets: 'wikieval', 'natural_questions'")


# Convenience functions for easy notebook usage (updated interface)
def load_demo_data(dataset_name: str = "wikieval") -> Tuple[str, List[Dict]]:
    """Quick function to load demo data from specified dataset"""
    from .config import DataConfig
    loader = create_data_loader(dataset_name, DataConfig())
    return loader.create_demo_dataset()


def load_random_data(dataset_name: str = "wikieval", num_contexts: int = 5) -> Tuple[str, List[Dict]]:
    """Quick function to load random data from specified dataset"""
    from .config import DataConfig
    loader = create_data_loader(dataset_name, DataConfig())

    if loader.load_dataset():
        return loader.select_random_question_with_contexts(num_contexts)
    else:
        logger.warning("⚠️ Falling back to demo data")
        return loader.create_demo_dataset()


def load_focused_data(dataset_name: str = "wikieval", topics: List[str] = None, num_contexts: int = 5) -> Tuple[str, List[Dict]]:
    """Quick function to load focused data from specified dataset"""
    if topics is None:
        topics = ['technology', 'computer', 'science', 'research', 'data', 'system']

    from .config import DataConfig
    loader = create_data_loader(dataset_name, DataConfig())

    if loader.load_dataset():
        return loader.create_focused_context_set(topics, num_contexts)
    else:
        logger.warning("⚠️ Falling back to demo data")
        return loader.create_demo_dataset()