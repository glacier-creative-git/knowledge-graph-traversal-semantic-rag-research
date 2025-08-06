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
        """
        For Natural Questions, we DON'T pre-load the dataset.
        Instead, we just verify we can access the streaming dataset.
        """
        try:
            split_name = getattr(self.config, 'nq_split', 'validation')
            logger.info(f"📚 Preparing Natural Questions streaming access...")
            logger.info(f"   Split: {split_name} (streaming mode - no pre-loading)")

            # Just test that we can access the streaming dataset
            test_dataset = load_dataset("natural_questions", split=split_name, streaming=True)

            # Test by taking just one sample
            first_sample = next(iter(test_dataset))
            if first_sample:
                logger.info(f"✅ Natural Questions streaming access verified")
                # Don't store the dataset - we'll stream it fresh each time
                self.dataset = None  # Explicitly set to None to indicate streaming mode
                return True
            else:
                raise ValueError("Could not access any samples from streaming dataset")

        except Exception as e:
            logger.error(f"❌ Could not access Natural Questions streaming dataset: {e}")
            return False

    def select_random_question_with_contexts(self, num_contexts: int = 5) -> Tuple[str, List[Dict]]:
        """
        Select random question from Natural Questions using streaming + islice.
        Much more memory efficient than loading the full dataset.
        """
        from itertools import islice
        import random

        try:
            split_name = getattr(self.config, 'nq_split', 'validation')
            max_samples = getattr(self.config, 'nq_max_samples', 1000)

            logger.info(f"🎲 Starting Natural Questions streaming random selection...")
            logger.info(f"   Split: {split_name}")
            logger.info(f"   Target contexts: {num_contexts}")
            logger.info(f"   Max samples to process: {max_samples}")

            # Load streaming dataset
            logger.info("🔧 Loading streaming dataset...")
            streaming_dataset = load_dataset("natural_questions", split=split_name, streaming=True)
            logger.info("✅ Streaming dataset loaded")

            # MUCH more permissive filter - let's see what we're actually getting
            def is_valid_sample(sample):
                try:
                    # First, let's examine a few samples to understand the structure
                    doc = sample.get('document', {})
                    html = doc.get('html', '')
                    question = sample.get('question', {})

                    # Log the first few samples we see to understand structure
                    if not hasattr(is_valid_sample, 'samples_examined'):
                        is_valid_sample.samples_examined = 0

                    if is_valid_sample.samples_examined < 5:
                        logger.info(f"🔍 EXAMINING SAMPLE {is_valid_sample.samples_examined + 1}:")
                        logger.info(f"   Sample keys: {list(sample.keys())}")
                        if doc:
                            logger.info(f"   Document keys: {list(doc.keys())}")
                            logger.info(f"   HTML length: {len(html)} chars")
                            logger.info(f"   HTML preview: {html[:200]}..." if html else "   No HTML")
                        if question:
                            logger.info(
                                f"   Question keys: {list(question.keys()) if isinstance(question, dict) else 'not_dict'}")
                            if isinstance(question, dict) and 'text' in question:
                                logger.info(f"   Question text: {question['text']}")
                        is_valid_sample.samples_examined += 1

                    # VERY permissive validation - just check we have the basics
                    has_html = bool(html and len(html) > 50)  # Very low bar
                    has_question_text = bool(question and
                                             isinstance(question, dict) and
                                             question.get('text') and
                                             len(question['text']) > 5)

                    result = has_html and has_question_text

                    # Log rejections for the first few samples
                    if is_valid_sample.samples_examined <= 10 and not result:
                        logger.info(f"   ❌ REJECTED: has_html={has_html}, has_question={has_question_text}")
                    elif is_valid_sample.samples_examined <= 10 and result:
                        logger.info(f"   ✅ ACCEPTED: has_html={has_html}, has_question={has_question_text}")

                    return result

                except Exception as filter_error:
                    logger.warning(f"   Filter error: {filter_error}")
                    return False

            # Apply filter and collect samples
            logger.info("🔧 Starting to filter and collect samples...")
            filtered_dataset = filter(is_valid_sample, streaming_dataset)

            # Skip some samples for randomness, but be more conservative
            skip_samples = random.randint(0, min(50, max_samples // 20))
            if skip_samples > 0:
                logger.info(f"   🎲 Randomly skipping first {skip_samples} valid samples")
                skipped = 0
                for sample in filtered_dataset:
                    skipped += 1
                    logger.info(f"   📝 Skipped sample {skipped} (question: {sample['question']['text'][:50]}...)")
                    if skipped >= skip_samples:
                        break

            # Now collect the samples we need
            sample_target = min(num_contexts * 2, num_contexts)  # Conservative target
            logger.info(f"🔧 Collecting {sample_target} samples...")

            selected_samples = []
            processed_after_skip = 0

            for sample in filtered_dataset:
                selected_samples.append(sample)
                processed_after_skip += 1

                # Detailed progress logging
                logger.info(f"   📊 Collected sample {len(selected_samples)}: {sample['question']['text'][:80]}...")

                if len(selected_samples) >= sample_target:
                    logger.info(f"   🎯 Reached target of {sample_target} samples")
                    break

                if processed_after_skip >= max_samples // 2:  # Safety valve
                    logger.warning(f"   ⚠️ Hit safety limit, stopping collection")
                    break

            if not selected_samples:
                logger.error("❌ No valid Natural Questions samples found in streaming")
                logger.error("🔧 This suggests the filter is too strict or dataset structure changed")
                logger.warning("⚠️ Falling back to demo dataset")
                return self.create_demo_dataset()

            logger.info(f"✅ Successfully collected {len(selected_samples)} valid samples via streaming")

            # Select primary sample randomly from what we got
            primary_sample = random.choice(selected_samples)
            selected_question = primary_sample['question']['text']

            logger.info(f"🎯 Selected Natural Questions Query: {selected_question}")

            # Build contexts with detailed logging
            contexts = []

            # Primary context
            logger.info("🔧 Building primary context...")
            primary_context = self._extract_text_from_html(primary_sample['document']['html'])
            logger.info(f"   Primary context extracted: {len(primary_context)} chars")
            logger.info(f"   Primary context preview: {primary_context[:200]}...")

            contexts.append({
                'context': primary_context,
                'question': selected_question,
                'id': primary_sample.get('id', f"nq_{hash(primary_context) % 100000}"),
                'title': f"Document 1: {primary_sample['document'].get('title', 'Natural Questions Context')}",
                'answers': self._extract_answers(primary_sample)
            })

            # Add additional contexts from remaining samples
            remaining_samples = [s for s in selected_samples if s != primary_sample]
            contexts_needed = min(num_contexts - 1, len(remaining_samples))

            logger.info(f"🔧 Adding {contexts_needed} additional contexts...")

            for i, sample in enumerate(remaining_samples[:contexts_needed]):
                context_text = self._extract_text_from_html(sample['document']['html'])
                contexts.append({
                    'context': context_text,
                    'question': sample['question']['text'],
                    'id': sample.get('id', f"nq_{hash(context_text) % 100000}"),
                    'title': f"Document {i + 2}: {sample['document'].get('title', 'Natural Questions Context')}",
                    'answers': self._extract_answers(sample)
                })
                logger.info(
                    f"   📄 Added Context {i + 2}: {len(context_text)} chars - {sample['question']['text'][:50]}...")

            logger.info(f"✅ Created Natural Questions dataset with {len(contexts)} contexts")
            return selected_question, contexts

        except Exception as e:
            logger.error(f"❌ Streaming random selection failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.warning("⚠️ Falling back to demo dataset")
            return self.create_demo_dataset()

    def create_focused_context_set(self, topic_keywords: List[str], num_contexts: int = 5) -> Tuple[str, List[Dict]]:
        """
        Create focused contexts from Natural Questions using streaming + keyword filtering.
        Reports actual number found vs requested.
        """
        from itertools import islice

        try:
            split_name = getattr(self.config, 'nq_split', 'validation')
            max_samples_to_check = getattr(self.config, 'nq_max_samples', 1000) * 2  # Check more for focused search

            logger.info(f"🔍 Streaming Natural Questions for focused search...")
            logger.info(f"   Keywords: {topic_keywords}")
            logger.info(f"   Target contexts: {num_contexts}, Max samples to check: {max_samples_to_check}")

            # Load streaming dataset
            streaming_dataset = load_dataset("natural_questions", split=split_name, streaming=True)

            # Combined filter function for keywords AND validity
            def matches_topics_and_valid(sample):
                # Check validity first (faster)
                doc = sample.get('document', {})
                html = doc.get('html', '')
                if not (html and 1000 <= len(html) <= 30000):
                    return False

                # Check keyword matches
                question_text = sample['question']['text'].lower()
                document_title = doc.get('title', '').lower()

                # Count keyword matches
                keyword_matches = sum(1 for keyword in topic_keywords
                                      if keyword.lower() in question_text or keyword.lower() in document_title)

                return keyword_matches >= 1

            # Apply combined filter lazily
            filtered_dataset = filter(matches_topics_and_valid, streaming_dataset)

            # Limit how many samples we'll check to prevent infinite processing
            limited_filtered = islice(filtered_dataset, max_samples_to_check)

            # Take the samples we need
            target_samples = max(num_contexts, 20)  # Get at least 20 to choose the best from
            selected_samples = list(islice(limited_filtered, target_samples))

            if not selected_samples:
                logger.warning(f"⚠️ No focused Natural Questions contexts found for {topic_keywords}")
                logger.warning("⚠️ Falling back to random selection")
                return self.select_random_question_with_contexts(num_contexts)

            # Report what we found
            actual_found = len(selected_samples)
            logger.info(f"📊 Found {actual_found} samples matching topics (requested {num_contexts})")

            if actual_found < num_contexts:
                logger.warning(f"⚠️ Only found {actual_found} matching samples, using all available")
                num_contexts = actual_found

            # Score samples by keyword relevance and select the best
            scored_samples = []
            for sample in selected_samples:
                question_text = sample['question']['text'].lower()
                document_title = sample['document'].get('title', '').lower()

                score = sum(1 for keyword in topic_keywords
                            if keyword.lower() in question_text or keyword.lower() in document_title)
                scored_samples.append((score, sample))

            # Sort by relevance and take the top samples
            scored_samples.sort(key=lambda x: x[0], reverse=True)
            final_samples = [sample for _, sample in scored_samples[:num_contexts]]

            # Use the highest-scoring question
            primary_sample = final_samples[0]
            selected_question = primary_sample['question']['text']

            logger.info(f"🎯 Selected Focused Natural Questions Query: {selected_question}")

            # Build contexts
            contexts = []
            for i, sample in enumerate(final_samples):
                context_text = self._extract_text_from_html(sample['document']['html'])
                contexts.append({
                    'context': context_text,
                    'question': sample['question']['text'],
                    'id': sample.get('id', f"nq_focused_{i}"),
                    'title': f"Document {i + 1}: {sample['document'].get('title', 'Natural Questions')}",
                    'answers': self._extract_answers(sample)
                })

                # Show relevance score for debugging
                question_lower = sample['question']['text'].lower()
                title_lower = sample['document'].get('title', '').lower()
                matches = [kw for kw in topic_keywords if kw.lower() in question_lower or kw.lower() in title_lower]
                logger.info(f"📄 Context {i + 1}: {len(context_text)} chars, matches: {matches}")

            logger.info(f"✅ Created focused Natural Questions dataset with {len(contexts)} contexts")
            return selected_question, contexts

        except Exception as e:
            logger.error(f"❌ Streaming focused selection failed: {e}")
            logger.warning("⚠️ Falling back to random selection")
            return self.select_random_question_with_contexts(num_contexts)

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
        """Extract answers from Natural Questions - CORRECTED STRUCTURE VERSION"""
        answers = {'text': [], 'answer_start': []}

        try:
            # CORRECTED: annotations is a DICT, not a list!
            if ('annotations' in sample and
                    isinstance(sample['annotations'], dict) and
                    'short_answers' in sample['annotations']):

                short_answers_list = sample['annotations']['short_answers']

                # short_answers is a list of 5 annotator responses
                # Find the first non-empty answer
                for annotator_answer in short_answers_list:
                    if (isinstance(annotator_answer, dict) and
                            'text' in annotator_answer and
                            annotator_answer['text']):  # Non-empty list

                        # Got a good answer!
                        for text in annotator_answer['text']:
                            if text and text.strip():
                                answers['text'].append(text.strip())

                        # Get corresponding start positions
                        if 'start_byte' in annotator_answer:
                            answers['answer_start'].extend(annotator_answer['start_byte'])
                        else:
                            answers['answer_start'].extend([0] * len(annotator_answer['text']))

                        # Found good answers, stop looking
                        break

                # If no short answers, try long answers
                if not answers['text'] and 'long_answer' in sample['annotations']:
                    long_answers_list = sample['annotations']['long_answer']

                    for annotator_long_answer in long_answers_list:
                        if (isinstance(annotator_long_answer, dict) and
                                annotator_long_answer.get('candidate_index', -1) >= 0):
                            answers['text'].append("Long answer available in HTML")
                            answers['answer_start'].append(annotator_long_answer.get('start_byte', 0))
                            break

            # Fallback: Create a research-appropriate answer from the question
            if not answers['text']:
                question_text = sample.get('question', {}).get('text', 'Unknown question')
                answers['text'].append(f"Research context for: {question_text}")
                answers['answer_start'].append(0)

        except Exception as extract_error:
            # For research purposes, we don't need perfect answers
            question_text = sample.get('question', {}).get('text', 'Unknown question')
            answers['text'].append(f"Research context for: {question_text}")
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