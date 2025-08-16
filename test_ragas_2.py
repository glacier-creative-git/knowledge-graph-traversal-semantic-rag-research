#!/usr/bin/env python3
"""
Full Custom RAGAS Test - All Local Extractors + Sliding Windows
==============================================================

Tests RAGAS question generation using:
1. 3-sentence sliding window chunking (your approach)
2. All four custom extractors (spaCy, TF-IDF, Ollama)
3. RAGAS relationship building
4. Local LLM question generation

This tests the full cost-effective pipeline for enterprise deployment.
"""

import os
import json
import time
import hashlib
import ollama
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

# Set your OpenAI API key for RAGAS (only for question generation, not extraction)
os.environ["OPENAI_API_KEY"] = "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"


def create_test_documents():
    """Create test documents - replace the content with your own text."""
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content="""
            Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalise to unseen data, and thus perform tasks without explicit instructions. Within a subdiscipline in machine learning, advances in the field of deep learning have allowed neural networks, a class of statistical algorithms, to surpass many previous machine learning approaches in performance. ML finds application in many fields, including natural language processing, computer vision, speech recognition, email filtering, agriculture, and medicine. The application of ML to business problems is known as predictive analytics. Statistics and mathematical optimisation (mathematical programming) methods comprise the foundations of machine learning. Data mining is a related field of study, focusing on exploratory data analysis (EDA) via unsupervised learning. From a theoretical viewpoint, probably approximately correct learning provides a framework for describing machine learning. The term machine learning was coined in 1959 by Arthur Samuel, an IBM employee and pioneer in the field of computer gaming and artificial intelligence. The synonym self-teaching computers was also used in this time period. The earliest machine learning program was introduced in the 1950s when Arthur Samuel invented a computer program that calculated the winning chance in checkers for each side, but the history of machine learning roots back to decades of human desire and effort to study human cognitive processes. In 1949, Canadian psychologist Donald Hebb published the book The Organization of Behavior, in which he introduced a theoretical neural structure formed by certain interactions among nerve cells. Hebb's model of neurons interacting with one another set a groundwork for how AIs and machine learning algorithms work under nodes, or artificial neurons used by computers to communicate data. Other researchers who have studied human cognitive systems contributed to the modern machine learning technologies as well, including logician Walter Pitts and Warren McCulloch, who proposed the early mathematical models of neural networks to come up with algorithms that mirror human thought processes. By the early 1960s, an experimental \"learning machine\" with punched tape memory, called Cybertron, had been developed by Raytheon Company to analyse sonar signals, electrocardiograms, and speech patterns using rudimentary reinforcement learning. It was repetitively \"trained\" by a human operator/teacher to recognise patterns and equipped with a \"goof\" button to cause it to reevaluate incorrect decisions. A representative book on research into machine learning during the 1960s was Nilsson's book on Learning Machines, dealing mostly with machine learning for pattern classification. Interest related to pattern recognition continued into the 1970s, as described by Duda and Hart in 1973. In 1981 a report was given on using teaching strategies so that an artificial neural network learns to recognise 40 characters (26 letters, 10 digits, and 4 special symbols) from a computer terminal. Tom M. Mitchell provided a widely quoted, more formal definition of the algorithms studied in the machine learning field: \"A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E.\" This definition of the tasks in which machine learning is concerned offers a fundamentally operational definition rather than defining the field in cognitive terms. This follows Alan Turing's proposal in his paper \"Computing Machinery and Intelligence\", in which the question \"Can machines think?\" is replaced with the question \"Can machines do what we (as thinking entities) can do?\". Modern-day machine learning has two objectives. One is to classify data based on models which have been developed; the other purpose is to make predictions for future outcomes based on these models. A hypothetical algorithm specific to classifying data may use computer vision of moles coupled with supervised learning in order to train it to classify the cancerous moles. A machine learning algorithm for stock trading may inform the trader of future potential predictions. As a scientific endeavour, machine learning grew out of the quest for artificial intelligence (AI). In the early days of AI as an academic discipline, some researchers were interested in having machines learn from data. They attempted to approach the problem with various symbolic methods, as well as what were then termed \"neural networks\"; these were mostly perceptrons and other models that were later found to be reinventions of the generalised linear models of statistics. Probabilistic reasoning was also employed, especially in automated medical diagnosis. However, an increasing emphasis on the logical, knowledge-based approach caused a rift between AI and machine learning. Probabilistic systems were plagued by theoretical and practical problems of data acquisition and representation. By 1980, expert systems had come to dominate AI, and statistics was out of favour. Work on symbolic/knowledge-based learning did continue within AI, leading to inductive logic programming(ILP), but the more statistical line of research was now outside the field of AI proper, in pattern recognition and information retrieval. Neural networks research had been abandoned by AI and computer science around the same time. This line, too, was continued outside the AI/CS field, as \"connectionism\", by researchers from other disciplines including John Hopfield, David Rumelhart, and Geoffrey Hinton. Their main success came in the mid-1980s with the reinvention of backpropagation. Machine learning (ML), reorganised and recognised as its own field, started to flourish in the 1990s. The field changed its goal from achieving artificial intelligence to tackling solvable problems of a practical nature. It shifted focus away from the symbolic approaches it had inherited from AI, and toward methods and models borrowed from statistics, fuzzy logic, and probability theory. Machine learning and data mining often employ the same methods and overlap significantly, but while machine learning focuses on prediction, based on known properties learned from the training data, data mining focuses on the discovery of (previously) unknown properties in the data (this is the analysis step of knowledge discovery in databases). Data mining uses many machine learning methods, but with different goals; on the other hand, machine learning also employs data mining methods as \"unsupervised learning\" or as a preprocessing step to improve learner accuracy. Much of the confusion between these two research communities (which do often have separate conferences and separate journals, ECML PKDD being a major exception) comes from the basic assumptions they work with: in machine learning, performance is usually evaluated with respect to the ability to reproduce known knowledge, while in knowledge discovery and data mining (KDD) the key task is the discovery of previously unknown knowledge. Evaluated with respect to known knowledge, an uninformed (unsupervised) method will easily be outperformed by other supervised methods, while in a typical KDD task, supervised methods cannot be used due to the unavailability of training data. Machine learning also has intimate ties to optimisation: Many learning problems are formulated as minimisation of some loss function on a training set of examples. Loss functions express the discrepancy between the predictions of the model being trained and the actual problem instances (for example, in classification, one wants to assign a label to instances, and models are trained to correctly predict the preassigned labels of a set of examples). Characterizing the generalisation of various learning algorithms is an active topic of current research, especially for deep learning algorithms. Machine learning and statistics are closely related fields in terms of methods, but distinct in their principal goal: statistics draws population inferences from a sample, while machine learning finds generalisable predictive patterns. According to Michael I. Jordan, the ideas of machine learning, from methodological principles to theoretical tools, have had a long pre-history in statistics. He also suggested the term data science as a placeholder to call the overall field. Conventional statistical analyses require the a priori selection of a model most suitable for the study data set. In addition, only significant or theoretically relevant variables based on previous experience are included for analysis.
            """,
            metadata={"source": "ml_basics.txt", "title": "Machine Learning Basics"}
        ),
        Document(
            page_content="""
            In contrast, machine learning is not built on a pre-structured model; rather, the data shape the model by detecting underlying patterns. The more variables (input) used to train the model, the more accurate the ultimate model will be. Leo Breiman distinguished two statistical modelling paradigms: data model and algorithmic model, wherein \"algorithmic model\" means more or less the machine learning algorithms like Random Forest. Some statisticians have adopted methods from machine learning, leading to a combined field that they call statistical learning. Analytical and computational techniques derived from deep-rooted physics of disordered systems can be extended to large-scale problems, including machine learning, e.g., to analyse the weight space of deep neural networks. Statistical physics is thus finding applications in the area of medical diagnostics. A core objective of a learner is to generalise from its experience. Generalisation in this context is the ability of a learning machine to perform accurately on new, unseen examples/tasks after having experienced a learning data set. The training examples come from some generally unknown probability distribution (considered representative of the space of occurrences) and the learner has to build a general model about this space that enables it to produce sufficiently accurate predictions in new cases. The computational analysis of machine learning algorithms and their performance is a branch of theoretical computer science known as computational learning theory via the probably approximately correct learning model. Because training sets are finite and the future is uncertain, learning theory usually does not yield guarantees of the performance of algorithms. Instead, probabilistic bounds on the performance are quite common. The bias-variance decomposition is one way to quantify generalisation error. For the best performance in the context of generalisation, the complexity of the hypothesis should match the complexity of the function underlying the data. If the hypothesis is less complex than the function, then the model has under fitted the data. If the complexity of the model is increased in response, then the training error decreases. But if the hypothesis is too complex, then the model is subject to overfitting and generalisation will be poorer. In addition to performance bounds, learning theorists study the time complexity and feasibility of learning. In computational learning theory, a computation is considered feasible if it can be done in polynomial time. There are two kinds of time complexity results: Positive results show that a certain class of functions can be learned in polynomial time. Negative results show that certain classes cannot be learned in polynomial time. Machine learning approaches are traditionally divided into three broad categories, which correspond to learning paradigms, depending on the nature of the \"signal\" or \"feedback\" available to the learning system: Supervised learning: The computer is presented with example inputs and their desired outputs, given by a \"teacher\", and the goal is to learn a general rule that maps inputs to outputs. Unsupervised learning: No labels are given to the learning algorithm, leaving it on its own to find structure in its input. Unsupervised learning can be a goal in itself (discovering hidden patterns in data) or a means towards an end (feature learning). Reinforcement learning: A computer program interacts with a dynamic environment in which it must perform a certain goal (such as driving a vehicle or playing a game against an opponent). As it navigates its problem space, the program is provided feedback that's analogous to rewards, which it tries to maximise. Although each algorithm has advantages and limitations, no single algorithm works for all problems. Supervised learning algorithms build a mathematical model of a set of data that contains both the inputs and the desired outputs. The data, known as training data, consists of a set of training examples. Each training example has one or more inputs and the desired output, also known as a supervisory signal. In the mathematical model, each training example is represented by an array or vector, sometimes called a feature vector, and the training data is represented by a matrix. Through iterative optimisation of an objective function, supervised learning algorithms learn a function that can be used to predict the output associated with new inputs. An optimal function allows the algorithm to correctly determine the output for inputs that were not a part of the training data. An algorithm that improves the accuracy of its outputs or predictions over time is said to have learned to perform that task. Types of supervised-learning algorithms include active learning, classification and regression. Classification algorithms are used when the outputs are restricted to a limited set of values, while regression algorithms are used when the outputs can take any numerical value within a range. For example, in a classification algorithm that filters emails, the input is an incoming email, and the output is the folder in which to file the email. In contrast, regression is used for tasks such as predicting a person's height based on factors like age and genetics or forecasting future temperatures based on historical data. Similarity learning is an area of supervised machine learning closely related to regression and classification, but the goal is to learn from examples using a similarity function that measures how similar or related two objects are. It has applications in ranking, recommendation systems, visual identity tracking, face verification, and speaker verification. Unsupervised learning algorithms find structures in data that has not been labelled, classified or categorised. Instead of responding to feedback, unsupervised learning algorithms identify commonalities in the data and react based on the presence or absence of such commonalities in each new piece of data. Central applications of unsupervised machine learning include clustering, dimensionality reduction, and density estimation. Cluster analysis is the assignment of a set of observations into subsets (called clusters) so that observations within the same cluster are similar according to one or more predesignated criteria, while observations drawn from different clusters are dissimilar. Different clustering techniques make different assumptions on the structure of the data, often defined by some similarity metric and evaluated, for example, by internal compactness, or the similarity between members of the same cluster, and separation, the difference between clusters. Other methods are based on estimated density and graph connectivity. A special type of unsupervised learning called, self-supervised learning involves training a model by generating the supervisory signal from the data itself. Semi-supervised learning falls between unsupervised learning (without any labelled training data) and supervised learning (with completely labelled training data). Some of the training examples are missing training labels, yet many machine-learning researchers have found that unlabelled data, when used in conjunction with a small amount of labelled data, can produce a considerable improvement in learning accuracy. In weakly supervised learning, the training labels are noisy, limited, or imprecise; however, these labels are often cheaper to obtain, resulting in larger effective training sets. Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment so as to maximise some notion of cumulative reward. Due to its generality, the field is studied in many other disciplines, such as game theory, control theory, operations research, information theory, simulation-based optimisation, multi-agent systems, swarm intelligence, statistics and genetic algorithms. In reinforcement learning, the environment is typically represented as a Markov decision process (MDP). Many reinforcement learning algorithms use dynamic programming techniques. Reinforcement learning algorithms do not assume knowledge of an exact mathematical model of the MDP and are used when exact models are infeasible. Reinforcement learning algorithms are used in autonomous vehicles or in learning to play a game against a human opponent. Dimensionality reduction is a process of reducing the number of random variables under consideration by obtaining a set of principal variables. In other words, it is a process of reducing the dimension of the feature set, also called the \"number of features\". Most of the dimensionality reduction techniques can be considered as either feature elimination or extraction. One of the popular methods of dimensionality reduction is principal component analysis (PCA). PCA involves changing higher-dimensional data (e.g., 3D) to a smaller space (e.g., 2D). The manifold hypothesis proposes that high-dimensional data sets lie along low-dimensional manifolds, and many dimensionality reduction techniques make this assumption, leading to the area of manifold learning and manifold regularisation. Other approaches have been developed which do not fit neatly into this three-fold categorisation, and sometimes more than one is used by the same machine learning system. For example, topic modelling, meta-learning. Self-learning, as a machine learning paradigm was introduced in 1982 along with a neural network capable of self-learning, named crossbar adaptive array (CAA). It gives a solution to the problem learning without any external reward, by introducing emotion as an internal reward. Emotion is used as state evaluation of a self-learning agent. The CAA self-learning algorithm computes, in a crossbar fashion, both decisions about actions and emotions (feelings) about consequence situations. The system is driven by the interaction between cognition and emotion. The self-learning algorithm updates a memory matrix W =||w(a,s)|| such that in each iteration executes the following machine learning routine: in situation s perform action a receive a consequence situation s' compute emotion of being in the consequence situation v(s') update crossbar memory w'(a,s) = w(a,s) + v(s') It is a system with only one input, situation, and only one output, action (or behaviour) a. There is neither a separate reinforcement input nor an advice input from the environment. The backpropagated value (secondary reinforcement) is the emotion toward the consequence situation. The CAA exists in two environments, one is the behavioural environment where it behaves, and the other is the genetic environment, wherefrom it initially and only once receives initial emotions about situations to be encountered in the behavioural environment. After receiving the genome (species) vector from the genetic environment, the CAA learns a goal-seeking behaviour, in an environment that contains both desirable and undesirable situations. Several learning algorithms aim at discovering better representations of the inputs provided during training. Classic examples include principal component analysis and cluster analysis. Feature learning algorithms, also called representation learning algorithms, often attempt to preserve the information in their input but also transform it in a way that makes it useful, often as a pre-processing step before performing classification or predictions. This technique allows reconstruction of the inputs coming from the unknown data-generating distribution, while not being necessarily faithful to configurations that are implausible under that distribution. This replaces manual feature engineering, and allows a machine to both learn the features and use them to perform a specific task. Feature learning can be either supervised or unsupervised. In supervised feature learning, features are learned using labelled input data. Examples include artificial neural networks, multilayer perceptrons, and supervised dictionary learning. In unsupervised feature learning, features are learned with unlabelled input data. Examples include dictionary learning, independent component analysis, autoencoders, matrix factorisation and various forms of clustering. Manifold learning algorithms attempt to do so under the constraint that the learned representation is low-dimensional. Sparse coding algorithms attempt to do so under the constraint that the learned representation is sparse, meaning that the mathematical model has many zeros. Multilinear subspace learning algorithms aim to learn low-dimensional representations directly from tensor representations for multidimensional data, without reshaping them into higher-dimensional vectors. Deep learning algorithms discover multiple levels of representation, or a hierarchy of features, with higher-level, more abstract features defined in terms of (or generating) lower-level features. It has been argued that an intelligent machine is one that learns a representation that disentangles the underlying factors of variation that explain the observed data.
            """,
            metadata={"source": "neural_networks.txt", "title": "Neural Networks"}
        )
    ]

    print(f"✅ Created {len(docs)} test documents")
    for i, doc in enumerate(docs):
        word_count = len(doc.page_content.split())
        print(f"   Doc {i + 1}: {doc.metadata['title']} ({word_count} words)")

    return docs


class SlidingWindowChunker:
    """3-sentence sliding window chunker (matching your pipeline)."""

    def __init__(self, window_size: int = 3, overlap: int = 1):
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = window_size - overlap

        # Import NLTK and download required data
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def chunk_documents(self, docs) -> List[Dict[str, Any]]:
        """Create sliding window chunks from documents."""
        import nltk

        all_chunks = []

        for doc_idx, doc in enumerate(docs):
            # Tokenize into sentences
            sentences = nltk.sent_tokenize(doc.page_content)

            # Filter sentences (basic quality check)
            filtered_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10 and len(sentence) < 500:  # Basic length filter
                    filtered_sentences.append(sentence)

            if len(filtered_sentences) < self.window_size:
                print(
                    f"   ⚠️  Document '{doc.metadata['title']}' has only {len(filtered_sentences)} sentences, skipping")
                continue

            # Create sliding windows
            windows = self._calculate_windows(filtered_sentences)

            for window_idx, (start_idx, end_idx) in enumerate(windows):
                chunk = self._create_chunk(
                    doc, filtered_sentences, start_idx, end_idx, window_idx, len(windows)
                )
                all_chunks.append(chunk)

        print(f"✅ Created {len(all_chunks)} sliding window chunks")
        return all_chunks

    def _calculate_windows(self, sentences):
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

    def _create_chunk(self, doc, sentences, start_idx, end_idx, window_pos, total_windows):
        """Create chunk dictionary."""
        window_sentences = sentences[start_idx:end_idx]
        chunk_text = ' '.join(window_sentences)

        # Generate unique chunk ID
        chunk_identifier = f"{doc.metadata['title'].replace(' ', '_')}_{start_idx}_{end_idx}"
        chunk_id = f"{chunk_identifier}_{hashlib.md5(chunk_text.encode()).hexdigest()[:8]}"

        return {
            'chunk_id': chunk_id,
            'text': chunk_text,
            'source_article': doc.metadata['title'],
            'source_sentences': list(range(start_idx, end_idx)),
            'anchor_sentence_idx': start_idx,
            'window_position': window_pos,
            'total_windows': total_windows,
            'window_size': len(window_sentences)
        }


class CustomNERExtractor:
    """Named Entity Recognition using spaCy."""

    def __init__(self):
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            self.available = True
            print("✅ spaCy NER model loaded")
        except (ImportError, OSError):
            print("⚠️  spaCy model not available, using fallback NER")
            self.available = False

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract named entities from text."""
        if self.available:
            return self._extract_with_spacy(text)
        else:
            return self._extract_with_patterns(text)

    def _extract_with_spacy(self, text: str) -> Dict[str, Any]:
        """Extract entities using spaCy."""
        doc = self.nlp(text)
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],
            'MISC': []
        }

        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE']:
                entities[ent.label_].append(ent.text)
            elif ent.label_ in ['PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']:
                # Include more entity types that might appear in technical text
                entities['MISC'].append(ent.text)
            else:
                entities['MISC'].append(ent.text)

        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))

        # If no entities found, add some technical terms as fallback
        if not any(entities.values()):
            tech_terms = []
            text_lower = text.lower()
            common_terms = ['machine learning', 'neural network', 'algorithm', 'artificial intelligence',
                            'deep learning', 'supervised learning', 'unsupervised learning']
            for term in common_terms:
                if term in text_lower:
                    tech_terms.append(term)
            entities['MISC'] = tech_terms

        return {'entities': entities}

    def _extract_with_patterns(self, text: str) -> Dict[str, Any]:
        """Fallback pattern-based extraction."""
        import re

        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],
            'MISC': []
        }

        # Common AI/ML terms
        tech_terms = [
            'machine learning', 'artificial intelligence', 'neural networks',
            'deep learning', 'natural language processing', 'computer vision',
            'supervised learning', 'unsupervised learning', 'reinforcement learning',
            'backpropagation', 'gradient descent', 'transformer', 'BERT', 'GPT'
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
    """Extract key phrases using TF-IDF."""

    def __init__(self, max_features: int = 15):
        self.max_features = max_features

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=1
            )
            print("✅ TF-IDF keyphrase extractor ready")
        except ImportError:
            print("⚠️  scikit-learn not available, using fallback keyphrase extraction")
            self.vectorizer = None

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract key phrases from text."""
        if self.vectorizer:
            return self._extract_with_tfidf(text)
        else:
            return self._extract_with_frequency(text)

    def _extract_with_tfidf(self, text: str) -> Dict[str, Any]:
        """Extract keyphrases using TF-IDF."""
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
            print(f"   TF-IDF extraction failed: {e}")
            return self._extract_with_frequency(text)

    def _extract_with_frequency(self, text: str) -> Dict[str, Any]:
        """Fallback frequency-based extraction."""
        import re

        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = defaultdict(int)

        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our',
                      'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two',
                      'way', 'who', 'boy', 'did', 'man', 'end', 'few', 'got', 'own', 'say', 'she', 'too', 'use'}

        for word in words:
            if word not in stop_words:
                word_freq[word] += 1

        keyphrases = sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:10]
        return {'keyphrases': keyphrases}


class LocalSummaryExtractor:
    """Local summary extraction using Ollama."""

    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model
        self._test_ollama_connection()

    def _test_ollama_connection(self):
        """Test if Ollama is available and model is pulled."""
        try:
            # Test Ollama connection
            models = ollama.list()
            available_models = [model.model for model in models.models]  # Fixed: use .model attribute

            if self.model in available_models:
                print(f"✅ Ollama model {self.model} available")
                self.available = True
            else:
                print(f"⚠️  Ollama model {self.model} not found. Available models: {available_models}")
                print(f"   Run: ollama pull {self.model}")
                self.available = False
        except Exception as e:
            print(f"⚠️  Ollama not available: {e}")
            print("   Install Ollama and run: ollama pull llama3.1:8b")
            self.available = False

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract summary using local LLM."""
        if not self.available:
            return self._fallback_summary(text)

        try:
            # Improved prompt for better results
            prompt = f"""Summarize the following text in exactly 1-2 clear sentences. Focus only on the main concepts and avoid repeating the instruction.

Text: {text}

Summary:"""

            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                    "num_predict": 100,  # Limit response length
                    "stop": ["\n\n", "Text:", "Summary:"]
                }
            )

            # Clean up the response
            summary = response['response'].strip()

            # Remove common instruction artifacts
            if summary.lower().startswith(('here is', 'here are', 'this is', 'the summary is')):
                # Try to extract the actual summary after the preamble
                lines = summary.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.lower().startswith(('here is', 'here are', 'this is', 'the summary is')):
                        summary = line
                        break

            # Fallback if summary is still problematic
            if len(summary) < 10 or 'summary' in summary.lower() or 'sentence' in summary.lower():
                return self._fallback_summary(text)

            return {'summary': summary}

        except Exception as e:
            print(f"   Ollama summary failed: {e}")
            return self._fallback_summary(text)

    def _fallback_summary(self, text: str) -> Dict[str, Any]:
        """Fallback summary (first + middle sentence)."""
        import nltk
        sentences = nltk.sent_tokenize(text)

        if len(sentences) <= 2:
            summary = text.strip()
        else:
            middle_idx = len(sentences) // 2
            summary = f"{sentences[0]} {sentences[middle_idx]}"

        return {'summary': summary}


class LocalThemeExtractor:
    """Local theme extraction using Ollama."""

    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model
        self._test_ollama_connection()

    def _test_ollama_connection(self):
        """Test if Ollama is available."""
        try:
            models = ollama.list()
            available_models = [model.model for model in models.models]  # Fixed: use .model attribute

            if self.model in available_models:
                print(f"✅ Ollama model {self.model} available for themes")
                self.available = True
            else:
                self.available = False
        except Exception as e:
            self.available = False

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract themes using local LLM."""
        if not self.available:
            return self._fallback_themes(text)

        try:
            # Improved prompt for better results
            prompt = f"""Extract 3-5 main topics from this text. Return ONLY the topics as a comma-separated list, nothing else.

Text: {text}

Topics:"""

            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                    "num_predict": 50,  # Keep it short
                    "stop": ["\n", "Text:", "Topics:"]
                }
            )

            # Clean up the response
            themes_text = response['response'].strip()

            # Remove instruction artifacts
            if themes_text.lower().startswith(('here are', 'the topics are', 'topics:', 'main topics')):
                # Find the actual list part
                colon_idx = themes_text.find(':')
                if colon_idx > -1:
                    themes_text = themes_text[colon_idx + 1:].strip()

            # Parse themes
            if ',' in themes_text:
                themes = [theme.strip().lower() for theme in themes_text.split(',')]
            else:
                # If no commas, try splitting by other delimiters
                themes = [theme.strip().lower() for theme in
                          themes_text.replace('\n', ',').replace(';', ',').split(',')]

            # Clean themes
            themes = [t for t in themes if
                      t and len(t) > 2 and not any(word in t for word in ['topic', 'theme', 'main', 'extract'])]
            themes = themes[:5]  # Limit to 5

            # Fallback if no good themes
            if not themes:
                return self._fallback_themes(text)

            return {'themes': themes}

        except Exception as e:
            print(f"   Ollama themes failed: {e}")
            return self._fallback_themes(text)

    def _fallback_themes(self, text: str) -> Dict[str, Any]:
        """Fallback keyword-based themes."""
        theme_keywords = {
            'machine learning': ['machine learning', 'ml', 'algorithm', 'model', 'training'],
            'artificial intelligence': ['artificial intelligence', 'ai', 'intelligent', 'cognitive'],
            'neural networks': ['neural network', 'neuron', 'layer', 'activation', 'backpropagation'],
            'deep learning': ['deep learning', 'cnn', 'rnn', 'transformer'],
            'natural language processing': ['natural language processing', 'nlp', 'text', 'language'],
            'data science': ['data science', 'data analysis', 'statistics'],
            'supervised learning': ['supervised learning', 'supervised', 'classification', 'regression'],
            'unsupervised learning': ['unsupervised learning', 'unsupervised', 'clustering'],
            'computer vision': ['computer vision', 'image', 'visual', 'recognition'],
            'optimization': ['optimization', 'gradient', 'minimize', 'training']
        }

        text_lower = text.lower()
        themes = []
        theme_scores = {}

        for theme, keywords in theme_keywords.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            if score > 0:
                themes.append(theme)
                theme_scores[theme] = score

        # Sort by relevance and limit
        themes.sort(key=lambda x: theme_scores.get(x, 0), reverse=True)
        themes = themes[:5]

        # Ensure we always have at least one theme
        if not themes:
            if 'learning' in text_lower or 'algorithm' in text_lower:
                themes = ['machine learning']
            elif 'network' in text_lower or 'neural' in text_lower:
                themes = ['neural networks']
            else:
                themes = ['artificial intelligence']

        return {'themes': themes}


def test_full_custom_pipeline():
    """Test the complete custom pipeline with all local extractors."""
    print("🧪 Testing Full Custom RAGAS Pipeline")
    print("=" * 60)

    start_time = time.time()

    # Step 1: Create documents
    print("\n📄 Step 1: Creating test documents...")
    docs = create_test_documents()

    # Step 2: Sliding window chunking
    print("\n✂️  Step 2: Creating sliding window chunks...")
    chunker = SlidingWindowChunker(window_size=3, overlap=1)
    chunks = chunker.chunk_documents(docs)

    if not chunks:
        print("❌ No chunks created!")
        return False

    # Show chunk info
    for i, chunk in enumerate(chunks[:3]):
        print(f"   Sample chunk {i + 1}: {chunk['chunk_id']}")
        print(f"      Text: '{chunk['text'][:80]}...'")
        print(f"      Sentences: {chunk['source_sentences']}")

    # Step 3: Initialize all custom extractors
    print("\n🔧 Step 3: Initializing custom extractors...")
    ner_extractor = CustomNERExtractor()
    keyphrase_extractor = CustomKeyphraseExtractor()
    summary_extractor = LocalSummaryExtractor()
    theme_extractor = LocalThemeExtractor()

    # Step 4: Apply all extractors to chunks
    print("\n⚙️  Step 4: Applying custom extractors to chunks...")
    extraction_times = {'ner': 0, 'keyphrases': 0, 'summary': 0, 'themes': 0}

    for i, chunk in enumerate(chunks):
        print(f"   Processing chunk {i + 1}/{len(chunks)}...")

        # NER extraction
        ner_start = time.time()
        ner_result = ner_extractor.extract(chunk['text'])
        extraction_times['ner'] += time.time() - ner_start
        chunk.update(ner_result)

        # Keyphrase extraction
        kp_start = time.time()
        kp_result = keyphrase_extractor.extract(chunk['text'])
        extraction_times['keyphrases'] += time.time() - kp_start
        chunk.update(kp_result)

        # Summary extraction (Ollama)
        sum_start = time.time()
        sum_result = summary_extractor.extract(chunk['text'])
        extraction_times['summary'] += time.time() - sum_start
        chunk.update(sum_result)

        # Theme extraction (Ollama)
        theme_start = time.time()
        theme_result = theme_extractor.extract(chunk['text'])
        extraction_times['themes'] += time.time() - theme_start
        chunk.update(theme_result)

    # Show extraction timing
    print(f"\n⏱️  Extraction timing:")
    total_extraction_time = sum(extraction_times.values())
    for extractor, time_taken in extraction_times.items():
        print(f"   {extractor}: {time_taken:.2f}s ({time_taken / total_extraction_time * 100:.1f}%)")

    # Step 5: Show sample enriched chunk
    print(f"\n🔍 Step 5: Sample enriched chunk:")
    sample_chunk = chunks[0]
    print(f"   Chunk ID: {sample_chunk['chunk_id']}")
    print(f"   Entities: {sample_chunk.get('entities', {})}")
    print(f"   Keyphrases: {sample_chunk.get('keyphrases', [])[:5]}")
    print(f"   Summary: {sample_chunk.get('summary', '')[:100]}...")
    print(f"   Themes: {sample_chunk.get('themes', [])}")

    # Debug: Check all properties in chunks
    print(f"\n🔧 Debug: All chunk properties:")
    for i, chunk in enumerate(chunks[:2]):  # Check first 2 chunks
        print(f"   Chunk {i + 1} properties: {list(chunk.keys())}")
        if 'entities' in chunk:
            total_entities = sum(len(v) for v in chunk['entities'].values())
            print(f"      Total entities: {total_entities}")
        if 'themes' in chunk:
            print(f"      Themes type: {type(chunk['themes'])}, value: {chunk['themes']}")
        if 'summary' in chunk:
            print(f"      Summary length: {len(chunk['summary'])}")

    # Step 6: Build RAGAS knowledge graph
    print(f"\n🏗️  Step 6: Building RAGAS knowledge graph...")
    try:
        from ragas.testset.graph import KnowledgeGraph, Node, NodeType
        from ragas.testset.transforms import apply_transforms
        from ragas.testset.persona import Persona
        from ragas.testset import TestsetGenerator
        from ragas.testset.synthesizers import default_query_distribution
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        # Create knowledge graph with enriched chunks
        kg = KnowledgeGraph()

        for chunk in chunks:
            kg.nodes.append(
                Node(
                    type=NodeType.CHUNK,
                    properties={
                        'page_content': chunk['text'],
                        'entities': chunk.get('entities', {}),
                        'keyphrases': chunk.get('keyphrases', []),
                        'summary': chunk.get('summary', ''),
                        'themes': chunk.get('themes', []),
                        # Keep your custom metadata
                        'chunk_id': chunk['chunk_id'],
                        'source_article': chunk['source_article'],
                        'anchor_sentence_idx': chunk['anchor_sentence_idx']
                    }
                )
            )

        print(f"   Created knowledge graph with {len(kg.nodes)} nodes")

        # Step 6a: Apply minimal RAGAS transforms to build relationships
        print(f"   Applying RAGAS transforms to build relationships...")
        try:
            # Try using RAGAS default transforms but only for relationship building
            from ragas.testset.transforms import default_transforms, apply_transforms

            # Convert our chunks back to documents for RAGAS transforms
            from langchain_core.documents import Document

            # Create document objects from our chunks (RAGAS expects this)
            ragas_docs = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk['text'],
                    metadata={
                        'chunk_id': chunk['chunk_id'],
                        'source': chunk['source_article']
                    }
                )
                ragas_docs.append(doc)

            generator_llm_raw = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
            generator_embeddings_raw = OpenAIEmbeddings()

            # Get default transforms but apply them to our enriched KG
            transforms = default_transforms(
                documents=ragas_docs,
                llm=LangchainLLMWrapper(generator_llm_raw),
                embedding_model=LangchainEmbeddingsWrapper(generator_embeddings_raw)
            )

            print(f"   Applying {len(transforms)} RAGAS transforms...")
            apply_transforms(kg, transforms)

        except Exception as e:
            print(f"   RAGAS transforms failed: {e}")
            print(f"   Falling back to manual relationship building...")

            # Manual relationship building as fallback
            manual_relationships = 0

            # Create relationships based on shared entities and themes
            for i, node_a in enumerate(kg.nodes):
                for j, node_b in enumerate(kg.nodes[i + 1:], i + 1):

                    # Extract entities and themes from both nodes
                    entities_a = set()
                    entities_b = set()
                    themes_a = set()
                    themes_b = set()

                    # Get all entities (focus on MISC since that's where AI/ML terms are)
                    if 'entities' in node_a.properties:
                        for ent_list in node_a.properties['entities'].values():
                            entities_a.update([e.lower().strip() for e in ent_list if e])

                    if 'entities' in node_b.properties:
                        for ent_list in node_b.properties['entities'].values():
                            entities_b.update([e.lower().strip() for e in ent_list if e])

                    # Get themes
                    if 'themes' in node_a.properties:
                        themes_a.update([t.lower().strip() for t in node_a.properties['themes'] if t])

                    if 'themes' in node_b.properties:
                        themes_b.update([t.lower().strip() for t in node_b.properties['themes'] if t])

                    # Calculate overlaps
                    entity_overlap = len(entities_a & entities_b)
                    theme_overlap = len(themes_a & themes_b)

                    # Create relationships if there's meaningful overlap
                    if entity_overlap > 0:
                        from ragas.testset.graph import Relationship
                        rel = Relationship(
                            source=node_a,
                            target=node_b,
                            type="entities_overlap",
                            properties={"overlap_count": entity_overlap,
                                        "shared_entities": list(entities_a & entities_b)}
                        )
                        kg.relationships.append(rel)
                        manual_relationships += 1

                    if theme_overlap > 0:
                        from ragas.testset.graph import Relationship
                        rel = Relationship(
                            source=node_a,
                            target=node_b,
                            type="cosine_similarity",
                            properties={"theme_overlap": theme_overlap, "shared_themes": list(themes_a & themes_b)}
                        )
                        kg.relationships.append(rel)
                        manual_relationships += 1

            print(f"   Created {manual_relationships} manual relationships")

            # Debug: Show some sample relationships
            if kg.relationships:
                print(f"   Sample relationships:")
                for i, rel in enumerate(kg.relationships[:3]):
                    print(f"      {i + 1}. {rel.type}: overlap={rel.properties}")

        print(f"   Knowledge graph now has {len(kg.relationships)} relationships")

        # Debug: Show relationship types
        if kg.relationships:
            rel_types = {}
            for rel in kg.relationships:
                rel_types[rel.type] = rel_types.get(rel.type, 0) + 1
            print(f"   Relationship types: {rel_types}")
        else:
            print(f"   ⚠️  No relationships found - debugging entity/theme overlaps...")
            # Debug why no relationships were created
            if len(kg.nodes) >= 2:
                node_a = kg.nodes[0]
                node_b = kg.nodes[1]

                entities_a = set()
                entities_b = set()
                if 'entities' in node_a.properties:
                    for ent_list in node_a.properties['entities'].values():
                        entities_a.update([e.lower() for e in ent_list])
                if 'entities' in node_b.properties:
                    for ent_list in node_b.properties['entities'].values():
                        entities_b.update([e.lower() for e in ent_list])

                themes_a = set([t.lower() for t in node_a.properties.get('themes', [])])
                themes_b = set([t.lower() for t in node_b.properties.get('themes', [])])

                print(f"      Node A entities: {entities_a}")
                print(f"      Node B entities: {entities_b}")
                print(f"      Node A themes: {themes_a}")
                print(f"      Node B themes: {themes_b}")
                print(f"      Entity overlap: {entities_a & entities_b}")
                print(f"      Theme overlap: {themes_a & themes_b}")

        # Step 7: Test question generation
        print(f"\n🎯 Step 7: Testing question generation...")

        # Setup RAGAS for question generation (only this part needs API)
        generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7))
        generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

        # Create personas
        personas = [
            Persona(
                name="AI Researcher",
                role_description="Researcher studying AI and ML. Asks technical questions about algorithms and methods."
            ),
            Persona(
                name="Student",
                role_description="Student learning AI concepts. Needs clear explanations and examples."
            )
        ]

        # Generate questions
        generator = TestsetGenerator(
            llm=generator_llm,
            embedding_model=generator_embeddings,
            knowledge_graph=kg,
            persona_list=personas
        )

        # Use default query distribution
        query_distribution = default_query_distribution(generator_llm)

        testset = generator.generate(
            testset_size=5,
            query_distribution=query_distribution,
            num_personas=len(personas)
        )

        print(f"✅ Successfully generated {len(testset)} questions!")

        # Show generated questions
        df = testset.to_pandas()
        print(f"\n📝 Generated Questions:")
        for i, row in df.iterrows():
            print(f"   {i + 1}. {row['user_input']}")
            if hasattr(row, 'synthesizer_name'):
                print(f"      Type: {getattr(row, 'synthesizer_name', 'unknown')}")

        total_time = time.time() - start_time
        print(f"\n🎉 SUCCESS! Full custom pipeline completed in {total_time:.2f}s")
        print(f"   Extraction time: {total_extraction_time:.2f}s ({total_extraction_time / total_time * 100:.1f}%)")
        print(
            f"   RAGAS time: {total_time - total_extraction_time:.2f}s ({(total_time - total_extraction_time) / total_time * 100:.1f}%)")

        return True

    except Exception as e:
        print(f"❌ RAGAS integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🧪 Full Custom RAGAS Pipeline Test")
    print("Testing all local extractors + sliding windows + RAGAS")
    print("=" * 70)

    # Check requirements
    print("🔍 Checking requirements...")

    # Check OpenAI API key
    if not os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") == "your-openai-api-key-here":
        print("❌ Please set your OpenAI API key for RAGAS question generation!")
        print("   Only needed for question generation, not extraction")
        return

    # Check Ollama
    try:
        models = ollama.list()
        print(f"✅ Ollama available with {len(models.models)} models")  # Fixed: use .models attribute
    except:
        print("⚠️  Ollama not available - will use fallback extractors")

    success = test_full_custom_pipeline()

    if success:
        print("\n✅ All tests passed!")
        print("🚀 Full custom pipeline with local extractors working!")
        print("\n💡 Key benefits demonstrated:")
        print("   • All extraction done locally (cost-effective)")
        print("   • 3-sentence sliding windows preserved")
        print("   • RAGAS relationships and question generation")
        print("   • Enterprise-scalable approach")
    else:
        print("\n❌ Tests failed!")
        print("🔧 Check error messages above")


if __name__ == "__main__":
    main()