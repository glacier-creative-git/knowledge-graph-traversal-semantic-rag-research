import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import time
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class MatrixRAGRetriever:
    """
    Pure Matrix RAG approach - no chunking, only sentence-level matrix relationships
    """

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentences = []
        self.embeddings = None
        self.similarity_matrix = None

    def ingest_document(self, text: str):
        """Ingest document by building complete similarity matrix"""
        print(f"🔥 Matrix RAG: Building similarity matrix...")
        start_time = time.time()

        # Split into sentences (no chunking!)
        self.sentences = nltk.sent_tokenize(text)

        # Get embeddings for ALL sentences
        self.embeddings = self.model.encode(self.sentences)

        # Build FULL similarity matrix
        self.similarity_matrix = np.dot(self.embeddings, self.embeddings.T)

        ingest_time = time.time() - start_time
        print(f"   ✅ Ingested {len(self.sentences)} sentences in {ingest_time:.2f}s")
        print(f"   ✅ Pre-computed {len(self.sentences) ** 2:,} relationships")

        return ingest_time

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], float]:
        """Retrieve using matrix traversal"""
        start_time = time.time()

        # Embed query
        query_embedding = self.model.encode([query])

        # Find anchor sentences most similar to query
        query_similarities = np.dot(query_embedding, self.embeddings.T)[0]
        anchor_indices = np.argsort(query_similarities)[-3:][::-1]  # Top 3 anchors

        # Traverse matrix from anchors
        relevant_sentences = {}

        for anchor_idx in anchor_indices:
            anchor_score = query_similarities[anchor_idx]

            # Get sentences related to this anchor using pre-computed matrix
            anchor_similarities = self.similarity_matrix[anchor_idx]
            related_indices = np.argsort(anchor_similarities)[-10:][::-1]

            for related_idx in related_indices:
                if related_idx not in relevant_sentences:
                    relationship_strength = anchor_similarities[related_idx]
                    combined_score = anchor_score * 0.7 + relationship_strength * 0.3
                    relevant_sentences[related_idx] = combined_score

        # Return top sentences
        sorted_results = sorted(relevant_sentences.items(), key=lambda x: x[1], reverse=True)[:top_k]
        retrieved_sentences = [self.sentences[idx] for idx, score in sorted_results]

        retrieval_time = time.time() - start_time
        return retrieved_sentences, retrieval_time


class TraditionalRAGRetriever:
    """
    Traditional RAG with chunking
    """

    def __init__(self, chunk_size: int = 400):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunk_size = chunk_size
        self.chunks = []
        self.chunk_embeddings = None

    def ingest_document(self, text: str):
        """Ingest document by creating chunks"""
        print(f"📄 Traditional RAG: Creating chunks...")
        start_time = time.time()

        # Split into sentences then group into chunks
        sentences = nltk.sent_tokenize(text)
        self.chunks = []

        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk + sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    self.chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            self.chunks.append(current_chunk.strip())

        # Get embeddings for chunks
        self.chunk_embeddings = self.model.encode(self.chunks)

        ingest_time = time.time() - start_time
        print(f"   ✅ Created {len(self.chunks)} chunks in {ingest_time:.2f}s")

        return ingest_time

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], float]:
        """Retrieve using traditional similarity search"""
        start_time = time.time()

        # Embed query and find most similar chunks
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]

        # Get top chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        retrieved_chunks = [self.chunks[idx] for idx in top_indices]

        retrieval_time = time.time() - start_time
        return retrieved_chunks, retrieval_time


def evaluate_retrieval_quality(retrieved_content: List[str], query: str, model) -> float:
    """
    Evaluate retrieval quality by measuring semantic similarity between
    query and retrieved content
    """
    if not retrieved_content:
        return 0.0

    # Embed query and retrieved content
    query_embedding = model.encode([query])
    content_embeddings = model.encode(retrieved_content)

    # Calculate similarities
    similarities = cosine_similarity(query_embedding, content_embeddings)[0]

    # Return average similarity (quality score)
    return np.mean(similarities)


def run_comprehensive_retrieval_evaluation():
    """
    Compare Matrix RAG vs Traditional RAG on actual retrieval tasks
    """
    print("🆚 MATRIX RAG vs TRADITIONAL RAG: RETRIEVAL EVALUATION")
    print("=" * 65)

    # Test document with multiple topics
    test_document = """
    Artificial intelligence has revolutionized many fields in recent years. Machine learning algorithms 
    can now process vast amounts of data to identify complex patterns that were previously invisible to 
    human analysts. Deep learning networks use multiple layers of artificial neurons to extract features 
    from raw input data in ways that mimic human cognition. Natural language processing enables computers 
    to understand and generate human text with remarkable accuracy. Computer vision allows machines to 
    interpret visual information from the world around them. These AI technologies are transforming 
    industries from healthcare to finance to transportation.

    Climate change poses significant challenges to our planet's future sustainability. Rising global 
    temperatures are causing ice caps to melt at unprecedented rates across both polar regions. Extreme 
    weather events such as hurricanes, droughts, and floods are becoming more frequent and more severe. 
    Sea levels are rising, threatening coastal communities worldwide with displacement and economic hardship. 
    However, renewable energy sources like solar and wind power offer hope for reducing carbon emissions. 
    Governments and corporations are increasingly investing in clean technology solutions.

    The human cardiovascular system consists of the heart, blood vessels, and blood working together 
    to circulate nutrients and oxygen throughout the body. The heart is a muscular organ divided into 
    four chambers that pump blood through a complex network of arteries and veins. Regular exercise 
    strengthens the heart muscle and improves circulation efficiency. A healthy diet low in saturated 
    fats and high in fruits and vegetables supports optimal cardiovascular function. High blood pressure, 
    if left untreated, can lead to serious complications including heart attack and stroke.

    Modern educational systems face numerous challenges in preparing students for the digital age. 
    Traditional teaching methods are being supplemented with technology-enhanced learning platforms 
    that provide interactive and personalized educational experiences. Teachers must adapt to new 
    pedagogical approaches that incorporate digital tools while maintaining focus on critical thinking 
    skills. Student engagement increases when learning activities connect to real-world applications 
    and career prospects. Educational institutions are investing in infrastructure and training to 
    support these technological transformations.
    """

    # Test queries across different topics and complexity levels
    test_queries = [
        # Direct topic queries
        "How does machine learning identify patterns in data?",
        "What are the effects of climate change on polar regions?",
        "How does the human heart pump blood?",
        "What challenges do modern schools face with technology?",

        # Cross-topic queries (these should show Matrix RAG's strength)
        "How are AI and education being combined?",
        "What technological solutions exist for climate problems?",
        "How does exercise benefit both heart health and learning?",

        # Specific detail queries
        "What are the four chambers of the heart?",
        "Which renewable energy sources are mentioned?",
        "What is deep learning?",
    ]

    # Initialize both systems
    matrix_rag = MatrixRAGRetriever()
    traditional_rag = TraditionalRAGRetriever(chunk_size=400)

    # Measure ingestion time
    print("📥 INGESTION COMPARISON:")
    print("-" * 25)
    matrix_ingest_time = matrix_rag.ingest_document(test_document)
    trad_ingest_time = traditional_rag.ingest_document(test_document)

    print(f"\n⚡ Ingestion Speed Comparison:")
    print(f"   Matrix RAG: {matrix_ingest_time:.3f}s")
    print(f"   Traditional: {trad_ingest_time:.3f}s")

    # Evaluation model
    eval_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Run retrieval comparison
    print(f"\n🔍 RETRIEVAL QUALITY COMPARISON:")
    print("-" * 35)

    matrix_scores = []
    traditional_scores = []
    matrix_times = []
    traditional_times = []

    for i, query in enumerate(test_queries):
        print(f"\n Query {i + 1}: '{query[:50]}...'")

        # Matrix RAG retrieval
        matrix_results, matrix_time = matrix_rag.retrieve(query, top_k=3)
        matrix_score = evaluate_retrieval_quality(matrix_results, query, eval_model)
        matrix_scores.append(matrix_score)
        matrix_times.append(matrix_time)

        # Traditional RAG retrieval
        trad_results, trad_time = traditional_rag.retrieve(query, top_k=3)
        trad_score = evaluate_retrieval_quality(trad_results, query, eval_model)
        traditional_scores.append(trad_score)
        traditional_times.append(trad_time)

        print(f"   Matrix RAG Score: {matrix_score:.3f} (time: {matrix_time:.4f}s)")
        print(f"   Traditional Score: {trad_score:.3f} (time: {trad_time:.4f}s)")

        if matrix_score > trad_score:
            improvement = ((matrix_score - trad_score) / trad_score) * 100
            print(f"   🔥 Matrix RAG wins by {improvement:.1f}%!")
        elif trad_score > matrix_score:
            gap = ((trad_score - matrix_score) / matrix_score) * 100
            print(f"   📈 Traditional leads by {gap:.1f}%")
        else:
            print(f"   🤝 Tie!")

    # Final analysis
    print(f"\n🏆 FINAL RESULTS:")
    print("=" * 20)

    avg_matrix_score = np.mean(matrix_scores)
    avg_trad_score = np.mean(traditional_scores)
    avg_matrix_time = np.mean(matrix_times)
    avg_trad_time = np.mean(traditional_times)

    print(f"Average Retrieval Quality:")
    print(f"   Matrix RAG: {avg_matrix_score:.3f} ± {np.std(matrix_scores):.3f}")
    print(f"   Traditional: {avg_trad_score:.3f} ± {np.std(traditional_scores):.3f}")

    print(f"\nAverage Retrieval Speed:")
    print(f"   Matrix RAG: {avg_matrix_time:.4f}s")
    print(f"   Traditional: {avg_trad_time:.4f}s")

    # Overall winner
    if avg_matrix_score > avg_trad_score:
        improvement = ((avg_matrix_score - avg_trad_score) / avg_trad_score) * 100
        print(f"\n🚀 BREAKTHROUGH: Matrix RAG beats Traditional RAG by {improvement:.1f}%!")
        print("   This validates the hypothesis that chunking boundaries are suboptimal!")

        matrix_wins = sum(1 for m, t in zip(matrix_scores, traditional_scores) if m > t)
        print(f"   Matrix RAG won {matrix_wins}/{len(test_queries)} queries")

    else:
        gap = ((avg_trad_score - avg_matrix_score) / avg_matrix_score) * 100
        print(f"\n📊 Traditional RAG leads by {gap:.1f}%")
        print("   Matrix RAG shows promise but needs refinement")

    # Speed analysis
    if avg_matrix_time < avg_trad_time:
        speedup = avg_trad_time / avg_matrix_time
        print(f"   ⚡ Matrix RAG is {speedup:.1f}x faster at retrieval!")

    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • Matrix RAG pre-computes {len(matrix_rag.sentences) ** 2:,} relationships")
    print(f"   • Traditional RAG creates {len(traditional_rag.chunks)} fixed chunks")
    print(f"   • Matrix approach enables semantic traversal vs boundary search")

    return {
        'matrix_avg_score': avg_matrix_score,
        'traditional_avg_score': avg_trad_score,
        'matrix_wins': sum(1 for m, t in zip(matrix_scores, traditional_scores) if m > t),
        'total_queries': len(test_queries)
    }


if __name__ == "__main__":
    results = run_comprehensive_retrieval_evaluation()

    if results['matrix_avg_score'] > results['traditional_avg_score']:
        print(f"\n🎉 MATRIX RAG VALIDATION SUCCESSFUL!")
        print(f"   Ready to challenge the chunking paradigm!")
    else:
        print(f"\n🔬 MATRIX RAG shows promise - ready for optimization!")