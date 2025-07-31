import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class MatrixRAG:
    """
    Revolutionary Matrix-Based RAG: Pre-compute ALL relationships, traverse from anchor points
    """

    def __init__(self, document_text: str):
        print("🔥 Building Matrix RAG System...")

        # Split document into sentences (no chunking!)
        self.sentences = nltk.sent_tokenize(document_text)
        print(f"   → Document split into {len(self.sentences)} sentences")

        # Get embeddings for ALL sentences
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("   → Computing embeddings for all sentences...")
        self.embeddings = self.model.encode(self.sentences)

        # Build the FULL similarity matrix (this is our secret weapon!)
        print("   → Building complete similarity matrix...")
        self.similarity_matrix = np.dot(self.embeddings, self.embeddings.T)
        print(f"   → Matrix shape: {self.similarity_matrix.shape}")
        print(f"   → Total pre-computed relationships: {len(self.sentences) ** 2:,}")

    def query(self, query_text: str, top_k: int = 5, traverse_depth: int = 2) -> Dict:
        """
        Query using Matrix RAG approach:
        1. Find anchor sentences most similar to query
        2. Traverse pre-computed relationships from anchors
        3. Return graduated relevance results
        """
        start_time = time.time()

        # Step 1: Embed the query
        query_embedding = self.model.encode([query_text])

        # Step 2: Find anchor points (most similar sentences to query)
        query_similarities = np.dot(query_embedding, self.embeddings.T)[0]
        anchor_indices = np.argsort(query_similarities)[-3:][::-1]  # Top 3 anchors

        # Step 3: Traverse from anchors using pre-computed matrix
        relevant_sentences = {}

        for anchor_idx in anchor_indices:
            anchor_score = query_similarities[anchor_idx]

            # Get all sentences similar to this anchor (using pre-computed matrix!)
            anchor_similarities = self.similarity_matrix[anchor_idx]

            # Find top related sentences to this anchor
            related_indices = np.argsort(anchor_similarities)[-10:][::-1]

            for related_idx in related_indices:
                if related_idx not in relevant_sentences:
                    # Combine anchor relevance with relationship strength
                    relationship_strength = anchor_similarities[related_idx]
                    combined_score = anchor_score * 0.7 + relationship_strength * 0.3

                    relevant_sentences[related_idx] = {
                        'sentence': self.sentences[related_idx],
                        'score': combined_score,
                        'anchor_via': anchor_idx,
                        'relationship_strength': relationship_strength
                    }

        # Sort by combined score and return top_k
        sorted_results = sorted(relevant_sentences.items(),
                                key=lambda x: x[1]['score'], reverse=True)[:top_k]

        query_time = time.time() - start_time

        return {
            'results': sorted_results,
            'query_time': query_time,
            'anchors_used': anchor_indices.tolist(),
            'total_relationships_traversed': len(relevant_sentences)
        }

    def visualize_query_traversal(self, query_text: str):
        """Visualize how the matrix traversal works"""
        result = self.query(query_text, top_k=10)
        anchors = result['anchors_used']

        # Create visualization showing query -> anchors -> related sentences
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Left plot: Show similarity matrix with anchors highlighted
        matrix_subset = self.similarity_matrix[:50, :50]  # Show subset for visibility
        sns.heatmap(matrix_subset, ax=ax1, cmap='viridis', cbar=True)
        ax1.set_title(f'Similarity Matrix (subset)\nAnchors: {anchors}')

        # Highlight anchor positions
        for anchor in anchors:
            if anchor < 50:
                ax1.axhline(y=anchor, color='red', linewidth=2, alpha=0.7)
                ax1.axvline(x=anchor, color='red', linewidth=2, alpha=0.7)

        # Right plot: Show traversal results
        scores = [item[1]['score'] for item in result['results']]
        ax2.bar(range(len(scores)), scores, color='skyblue')
        ax2.set_title('Matrix RAG Results by Score')
        ax2.set_xlabel('Result Rank')
        ax2.set_ylabel('Combined Score')

        plt.tight_layout()
        plt.show()

        return result


class TraditionalRAG:
    """
    Traditional RAG: Chunk document, find most similar chunks
    """

    def __init__(self, document_text: str, chunk_size: int = 3):
        print("📄 Building Traditional RAG System...")

        # Split into sentences then group into chunks
        sentences = nltk.sent_tokenize(document_text)
        self.chunks = []

        for i in range(0, len(sentences), chunk_size):
            chunk = ' '.join(sentences[i:i + chunk_size])
            self.chunks.append(chunk)

        print(f"   → Document split into {len(self.chunks)} chunks")

        # Get embeddings for chunks
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("   → Computing embeddings for chunks...")
        self.chunk_embeddings = self.model.encode(self.chunks)

    def query(self, query_text: str, top_k: int = 5) -> Dict:
        """Traditional RAG query: find most similar chunks"""
        start_time = time.time()

        # Embed query and find most similar chunks
        query_embedding = self.model.encode([query_text])
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]

        # Get top_k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'chunk': self.chunks[idx],
                'score': similarities[idx],
                'chunk_index': idx
            })

        query_time = time.time() - start_time

        return {
            'results': results,
            'query_time': query_time
        }


def compare_rag_systems(document_text: str, test_queries: List[str]):
    """
    Compare Matrix RAG vs Traditional RAG on the same document and queries
    """
    print("🆚 MATRIX RAG vs TRADITIONAL RAG COMPARISON")
    print("=" * 60)

    # Initialize both systems
    matrix_rag = MatrixRAG(document_text)
    traditional_rag = TraditionalRAG(document_text)

    print(f"\n📊 System Stats:")
    print(f"Matrix RAG: {len(matrix_rag.sentences)} sentences, {len(matrix_rag.sentences) ** 2:,} relationships")
    print(f"Traditional RAG: {len(traditional_rag.chunks)} chunks")

    # Test both systems on the same queries
    for i, query in enumerate(test_queries):
        print(f"\n" + "=" * 50)
        print(f"🔍 QUERY {i + 1}: '{query}'")
        print("=" * 50)

        # Matrix RAG Results
        print("\n🔥 MATRIX RAG RESULTS:")
        matrix_result = matrix_rag.query(query, top_k=3)
        print(f"Query time: {matrix_result['query_time']:.4f}s")
        print(f"Anchors used: {matrix_result['anchors_used']}")
        print(f"Relationships traversed: {matrix_result['total_relationships_traversed']}")

        for j, (idx, data) in enumerate(matrix_result['results']):
            print(f"\n  {j + 1}. Score: {data['score']:.3f}")
            print(f"     Sentence: {data['sentence'][:100]}...")
            print(f"     Via anchor: {data['anchor_via']}")

        # Traditional RAG Results
        print(f"\n📄 TRADITIONAL RAG RESULTS:")
        trad_result = traditional_rag.query(query, top_k=3)
        print(f"Query time: {trad_result['query_time']:.4f}s")

        for j, result in enumerate(trad_result['results']):
            print(f"\n  {j + 1}. Score: {result['score']:.3f}")
            print(f"     Chunk: {result['chunk'][:100]}...")

        # Performance comparison
        speedup = trad_result['query_time'] / matrix_result['query_time']
        print(f"\n⚡ PERFORMANCE: Matrix RAG is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")


# Example usage and test
if __name__ == "__main__":
    # Sample document about multiple topics for testing
    test_document = """
    Harry Potter was a young wizard who lived with his aunt and uncle in Little Whinging.
    He had messy black hair and wore round glasses that were held together with tape.
    Hogwarts School of Witchcraft and Wizardry was located in the Scottish Highlands.
    The Great Hall at Hogwarts was magnificent, with floating candles and a magical ceiling.
    Quidditch was the most popular sport in the wizarding world, played on flying broomsticks.

    Climate change represents one of the most pressing challenges of our time.
    Rising global temperatures are causing ice caps to melt at unprecedented rates.
    Sea levels are rising, threatening coastal communities around the world.
    Extreme weather events are becoming more frequent and more severe each year.
    Renewable energy sources like solar and wind power offer hope for the future.

    Machine learning algorithms can identify patterns in vast amounts of data.
    Neural networks are inspired by the structure of the human brain.
    Deep learning has revolutionized computer vision and natural language processing.
    Artificial intelligence is transforming industries from healthcare to finance.
    The future of AI promises both tremendous opportunities and significant challenges.

    The Mediterranean diet emphasizes fruits, vegetables, and olive oil.
    Regular exercise is essential for maintaining good physical and mental health.
    Sleep plays a crucial role in memory consolidation and immune function.
    Stress management techniques include meditation, yoga, and deep breathing exercises.
    Social connections and relationships are vital for overall well-being and longevity.
    """

    # Test queries spanning different topics
    test_queries = [
        "Tell me about Harry Potter's appearance",
        "What are the effects of climate change?",
        "How does machine learning work?",
        "What makes a healthy lifestyle?",
        "Flying and magical sports"  # This should test relationship traversal
    ]

    # Run the comparison
    compare_rag_systems(test_document, test_queries)

    # Optional: Visualize a specific query traversal
    print(f"\n🎨 VISUALIZING MATRIX TRAVERSAL:")
    matrix_rag = MatrixRAG(test_document)
    matrix_rag.visualize_query_traversal("magical flying sports")