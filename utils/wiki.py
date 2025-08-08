#!/usr/bin/env python3
"""
Wikipedia Engine
===============

Handles Wikipedia article acquisition, cleaning, and processing for the semantic RAG pipeline.
Uses LangChain's WikipediaLoader for reliable article acquisition with intelligent caching.
"""

import json
import re
import time
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

from langchain_community.document_loaders import WikipediaLoader
from tqdm import tqdm
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


@dataclass
class ArticleMetadata:
    """Metadata for a Wikipedia article."""
    title: str
    url: str
    word_count: int
    char_count: int
    sentence_count: int
    scraped_at: str
    language: str
    processing_time: float
    source_hash: str


@dataclass
class WikipediaArticle:
    """Container for a processed Wikipedia article."""
    title: str
    url: str
    raw_text: str
    cleaned_text: str
    sentences: List[str]
    metadata: ArticleMetadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'title': self.title,
            'url': self.url,
            'raw_text': self.raw_text,
            'cleaned_text': self.cleaned_text,
            'sentences': self.sentences,
            'metadata': asdict(self.metadata)
        }


class WikiEngine:
    """Engine for acquiring and processing Wikipedia articles using LangChain."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize the Wikipedia engine."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.wiki_config = config['wikipedia']
        self.text_config = config['text_processing']

        self.articles: List[WikipediaArticle] = []
        self.failed_articles: List[Dict[str, str]] = []

    def acquire_articles(self, save_path: Optional[str] = None) -> List[WikipediaArticle]:
        """
        Acquire articles from Wikipedia based on configured topics.

        Args:
            save_path: Optional path to save articles (defaults to data/wiki.json)

        Returns:
            List of processed WikipediaArticle objects
        """
        if save_path is None:
            save_path = Path(self.config['directories']['data']) / "wiki.json"
        else:
            save_path = Path(save_path)

        # Check for cached articles
        if self.wiki_config['use_cached_articles'] and save_path.exists():
            self.logger.info(f"Loading cached articles from {save_path}")
            return self._load_cached_articles(save_path)

        self.logger.info("Acquiring fresh articles from Wikipedia using LangChain")
        self._acquire_fresh_articles()

        # Save articles
        if self.articles:
            self._save_articles(save_path)

        return self.articles

    def _acquire_fresh_articles(self):
        """Acquire fresh articles from Wikipedia using LangChain WikipediaLoader."""
        topics = self.wiki_config['topics']
        articles_per_topic = self.wiki_config['articles_per_topic']

        self.logger.info(f"Fetching articles for {len(topics)} topics, {articles_per_topic} articles each")

        total_articles = 0

        for topic in tqdm(topics, desc="Processing topics"):
            self.logger.info(f"Searching for articles about: {topic}")

            try:
                # Method 1: Try direct WikipediaLoader
                articles = self._load_with_langchain_direct(topic, articles_per_topic)

                if articles:
                    self.articles.extend(articles)
                    total_articles += len(articles)
                    self.logger.info(f"Topic '{topic}': {len(articles)} articles acquired")
                else:
                    # Method 2: Try search-based approach
                    self.logger.info(f"Direct method failed, trying search-based approach for '{topic}'")
                    articles = self._load_with_search_fallback(topic, articles_per_topic)

                    if articles:
                        self.articles.extend(articles)
                        total_articles += len(articles)
                        self.logger.info(f"Topic '{topic}': {len(articles)} articles acquired (fallback)")
                    else:
                        self.logger.warning(f"Failed to acquire any articles for topic: {topic}")

            except Exception as e:
                self.logger.error(f"Failed to process topic '{topic}': {e}")
                continue

        self.logger.info(f"Total articles acquired: {total_articles}")
        if self.failed_articles:
            self.logger.warning(f"Failed to acquire {len(self.failed_articles)} articles")

    def _load_with_langchain_direct(self, topic: str, max_docs: int) -> List[WikipediaArticle]:
        """Load articles using direct LangChain WikipediaLoader."""
        try:
            max_chars = self.wiki_config['max_article_length']
            self.logger.info(f"   Trying direct WikipediaLoader for '{topic}' (max chars: {max_chars:,})...")

            loader = WikipediaLoader(
                query=topic,
                load_max_docs=max_docs,
                load_all_available_meta=False,  # Avoid metadata issues
                doc_content_chars_max=max_chars  # Use our config limit
            )

            docs = loader.load()
            self.logger.info(f"   LangChain loaded {len(docs)} documents")

            # Log actual document lengths for debugging
            for i, doc in enumerate(docs):
                actual_length = len(doc.page_content)
                self.logger.debug(f"   Document {i + 1}: {actual_length:,} characters")

            return self._process_langchain_documents(docs, topic)

        except Exception as e:
            self.logger.warning(f"   Direct LangChain method failed: {e}")
            return []

    def _load_with_search_fallback(self, topic: str, max_docs: int) -> List[WikipediaArticle]:
        """Fallback method using search terms related to the topic."""
        try:
            # Try alternative formulations of the topic
            alternative_topics = self._get_alternative_topics(topic)

            for alt_topic in alternative_topics:
                try:
                    self.logger.info(f"   Trying alternative topic: '{alt_topic}'")

                    loader = WikipediaLoader(
                        query=alt_topic,
                        load_max_docs=max_docs,
                        load_all_available_meta=False,
                        doc_content_chars_max=self.wiki_config['max_article_length']  # Use our config limit
                    )

                    docs = loader.load()

                    if docs:
                        self.logger.info(f"   Alternative '{alt_topic}' got {len(docs)} documents")
                        return self._process_langchain_documents(docs, topic)

                except Exception as e:
                    self.logger.warning(f"   Alternative topic '{alt_topic}' failed: {e}")
                    continue

            return []

        except Exception as e:
            self.logger.warning(f"   Search fallback failed: {e}")
            return []

    def _get_alternative_topics(self, topic: str) -> List[str]:
        """Generate alternative topic formulations."""
        alternatives = [topic]

        # Add some variations
        if " " in topic:
            # Try without spaces
            alternatives.append(topic.replace(" ", "_"))
            alternatives.append(topic.replace(" ", ""))

            # Try individual words
            words = topic.split()
            if len(words) > 1:
                alternatives.extend(words)

        # Add plurals/singulars
        if topic.endswith('s'):
            alternatives.append(topic[:-1])
        else:
            alternatives.append(topic + 's')

        # Some common transformations
        topic_lower = topic.lower()
        if 'machine learning' in topic_lower:
            alternatives.extend(['ML', 'artificial intelligence', 'deep learning'])
        elif 'artificial intelligence' in topic_lower:
            alternatives.extend(['AI', 'machine learning', 'neural networks'])
        elif 'natural language processing' in topic_lower:
            alternatives.extend(['NLP', 'computational linguistics', 'text processing'])

        return alternatives[:5]  # Limit to avoid too many requests

    def _process_langchain_documents(self, docs: List, topic: str) -> List[WikipediaArticle]:
        """Process documents from LangChain into our WikipediaArticle format."""
        articles = []

        for doc in docs:
            try:
                start_time = time.time()

                # Extract basic info
                title = doc.metadata.get('title', 'Unknown Title')
                url = doc.metadata.get('source', '')
                raw_text = doc.page_content

                # Validate article length
                if not self._validate_article_length(raw_text, title):
                    continue

                # Clean text
                cleaned_text = self._clean_text(raw_text)

                # Extract sentences
                sentences = self._extract_sentences(cleaned_text)

                # Skip if too few sentences
                if len(sentences) < 10:
                    self.logger.debug(f"Article '{title}' has too few sentences ({len(sentences)})")
                    continue

                # Create metadata
                processing_time = time.time() - start_time
                source_hash = hashlib.md5(raw_text.encode()).hexdigest()

                metadata = ArticleMetadata(
                    title=title,
                    url=url,
                    word_count=len(cleaned_text.split()),
                    char_count=len(cleaned_text),
                    sentence_count=len(sentences),
                    scraped_at=datetime.now().isoformat(),
                    language=self.wiki_config['language'],
                    processing_time=processing_time,
                    source_hash=source_hash
                )

                # Create article object
                article = WikipediaArticle(
                    title=title,
                    url=url,
                    raw_text=raw_text,
                    cleaned_text=cleaned_text,
                    sentences=sentences,
                    metadata=metadata
                )

                articles.append(article)
                self.logger.debug(
                    f"Processed article '{title}': {len(sentences)} sentences, {metadata.word_count} words, {metadata.char_count:,} chars")

                # Log if article might be truncated
                if len(raw_text) >= self.wiki_config['max_article_length'] * 0.95:
                    self.logger.warning(
                        f"Article '{title}' may be truncated (reached {len(raw_text):,} chars, limit: {self.wiki_config['max_article_length']:,})")
                else:
                    self.logger.info(f"Article '{title}': {len(raw_text):,} chars (full content)")

            except Exception as e:
                self.logger.warning(f"Failed to process document: {e}")
                self.failed_articles.append({
                    'title': doc.metadata.get('title', 'Unknown'),
                    'topic': topic,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                continue

        return articles

    def _validate_article_length(self, text: str, title: str) -> bool:
        """Validate that article meets length requirements."""
        length = len(text)
        min_length = self.wiki_config['min_article_length']
        max_length = self.wiki_config['max_article_length']

        if length < min_length:
            self.logger.debug(f"Article '{title}' too short: {length} chars (min: {min_length})")
            return False

        if length > max_length:
            self.logger.debug(f"Article '{title}' too long: {length} chars (max: {max_length})")
            return False

        return True

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        cleaned = text

        if self.text_config['remove_navigation']:
            # Remove everything after certain sections (more aggressive approach)
            # This removes all content from these sections to the end
            section_patterns = [
                r'==+\s*See also\s*==+.*',
                r'==+\s*References\s*==+.*',
                r'==+\s*External links\s*==+.*',
                r'==+\s*Further reading\s*==+.*',
                r'==+\s*Notes\s*==+.*',
                r'==+\s*Bibliography\s*==+.*',
                r'==+\s*Sources\s*==+.*'
            ]

            for pattern in section_patterns:
                cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Always remove section headers regardless of navigation setting
        # Handle different header levels: ==, ===, ====, etc.
        # More robust regex that handles various spacing patterns
        cleaned = re.sub(r'={2,}\s*[^=\n]+?\s*={2,}', '', cleaned)
        # Also handle cases where headers might not be perfectly balanced
        cleaned = re.sub(r'={2,}\s*[^=\n]+?\s*={2,}', '', cleaned)
        # Handle any remaining single equals that might be headers
        cleaned = re.sub(r'\n\s*=+[^=\n]*=+\s*\n', '\n', cleaned)
        # Clean up section headers that might be at start/end of lines
        cleaned = re.sub(r'^=+[^=\n]*=+\s*', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\s*=+[^=\n]*=+$', '', cleaned, flags=re.MULTILINE)

        if self.text_config['remove_references']:
            # Remove reference markers like [1], [citation needed], etc.
            cleaned = re.sub(r'\[[\d\s,\-]+\]', '', cleaned)
            cleaned = re.sub(r'\[citation needed\]', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\[clarification needed\]', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\[when\?\]', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\[who\?\]', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\[according to whom\?\]', '', cleaned, flags=re.IGNORECASE)

        if self.text_config['remove_tables']:
            # Remove table-like structures (simplified)
            cleaned = re.sub(r'\{\|.*?\|\}', '', cleaned, flags=re.DOTALL)

        if self.text_config['fix_encoding']:
            # Fix common encoding issues and escaped characters
            cleaned = cleaned.replace(''', "'").replace(''', "'")
            cleaned = cleaned.replace('"', '"').replace('"', '"')
            cleaned = cleaned.replace('–', '-').replace('—', '-')
            cleaned = cleaned.replace('…', '...')

            # Fix escaped quotes and other escape sequences
            cleaned = cleaned.replace('\\"', '"')  # Fix escaped quotes
            cleaned = cleaned.replace("\\'", "'")  # Fix escaped apostrophes
            cleaned = cleaned.replace('\\n', ' ')  # Fix escaped newlines
            cleaned = cleaned.replace('\\t', ' ')  # Fix escaped tabs

        if self.text_config['normalize_whitespace']:
            # Normalize whitespace (do this after other cleaning)
            cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces -> single space
            cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)  # Multiple newlines -> double newline
            cleaned = re.sub(r'^\s+|\s+$', '', cleaned, flags=re.MULTILINE)  # Trim line whitespace
            cleaned = cleaned.strip()

        return cleaned

    def _extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text with filtering.

        Args:
            text: Cleaned text

        Returns:
            List of sentences
        """
        # Use NLTK for sentence tokenization
        raw_sentences = nltk.sent_tokenize(text)

        sentences = []
        min_length = self.text_config['min_sentence_length']
        max_length = self.text_config['max_sentence_length']

        for sentence in raw_sentences:
            sentence = sentence.strip()

            # Length filtering
            if len(sentence) < min_length or len(sentence) > max_length:
                continue

            # Quality filtering
            if self._is_valid_sentence(sentence):
                sentences.append(sentence)

        return sentences

    def _is_valid_sentence(self, sentence: str) -> bool:
        """Check if a sentence is valid and worth keeping."""
        # Skip sentences that are mostly punctuation or numbers
        if re.match(r'^[\d\s\.,;:!?\-]+$', sentence):
            return False

        # Skip sentences with too many special characters
        if len(sentence) > 0:
            special_char_ratio = len(re.findall(r'[^\w\s]', sentence)) / len(sentence)
            if special_char_ratio > 0.3:
                return False

        # Skip sentences that look like headers (all caps and short)
        if sentence.isupper() and len(sentence) < 50:
            return False

        # Skip sentences with no alphabetic characters
        if not re.search(r'[a-zA-Z]', sentence):
            return False

        # Skip sentences that are mostly parentheses content
        if sentence.count('(') > 2 or sentence.count('[') > 2:
            return False

        return True

    def _save_articles(self, save_path: Path):
        """Save articles to JSON file."""
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_articles': len(self.articles),
                'failed_articles': len(self.failed_articles),
                'config': {
                    'wikipedia': self.wiki_config,
                    'text_processing': self.text_config
                }
            },
            'articles': [article.to_dict() for article in self.articles],
            'failed_articles': self.failed_articles
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved {len(self.articles)} articles to {save_path}")

        # Log statistics
        total_sentences = sum(len(article.sentences) for article in self.articles)
        total_words = sum(article.metadata.word_count for article in self.articles)

        self.logger.info(f"Corpus statistics:")
        self.logger.info(f"  Articles: {len(self.articles)}")
        self.logger.info(f"  Sentences: {total_sentences:,}")
        self.logger.info(f"  Words: {total_words:,}")
        self.logger.info(f"  Avg sentences per article: {total_sentences / len(self.articles):.1f}")

    def _load_cached_articles(self, cache_path: Path) -> List[WikipediaArticle]:
        """Load articles from cached JSON file."""
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            articles = []
            for article_data in data['articles']:
                metadata = ArticleMetadata(**article_data['metadata'])
                article = WikipediaArticle(
                    title=article_data['title'],
                    url=article_data['url'],
                    raw_text=article_data['raw_text'],
                    cleaned_text=article_data['cleaned_text'],
                    sentences=article_data['sentences'],
                    metadata=metadata
                )
                articles.append(article)

            self.articles = articles

            # Log cache statistics
            total_sentences = sum(len(article.sentences) for article in articles)
            total_words = sum(article.metadata.word_count for article in articles)

            self.logger.info(f"Loaded {len(articles)} cached articles")
            self.logger.info(f"  Sentences: {total_sentences:,}")
            self.logger.info(f"  Words: {total_words:,}")

            return articles

        except Exception as e:
            self.logger.error(f"Failed to load cached articles: {e}")
            self.logger.info("Will fetch fresh articles instead")
            return []

    def get_corpus_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current corpus."""
        if not self.articles:
            return {}

        total_sentences = sum(len(article.sentences) for article in self.articles)
        total_words = sum(article.metadata.word_count for article in self.articles)
        total_chars = sum(article.metadata.char_count for article in self.articles)

        return {
            'total_articles': len(self.articles),
            'total_sentences': total_sentences,
            'total_words': total_words,
            'total_characters': total_chars,
            'avg_sentences_per_article': total_sentences / len(self.articles),
            'avg_words_per_article': total_words / len(self.articles),
            'avg_words_per_sentence': total_words / total_sentences if total_sentences > 0 else 0
        }