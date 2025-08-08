# Semantic RAG Pipeline

A novel semantic graph traversal approach to Retrieval-Augmented Generation (RAG) that uses dense similarity matrices and real-time semantic navigation instead of traditional entity-based knowledge graphs.

## Quick Start

### Installation

1. **Clone the repository** (or create the files in a new directory)

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Test Phase 2** (optional but recommended):
   ```bash
   python test_phase2.py
   ```

4. **Run the pipeline:**
   ```bash
   python main.py
   ```

### Current Status: Phases 1-2 Implementation

This implementation currently includes:

- ✅ **Phase 1:** Setup & Initialization
- ✅ **Phase 2:** Data Acquisition (Wikipedia scraping & processing)
- [ ] **Phase 3:** Embedding Generation (multi-model support)
- [ ] **Phase 4:** Similarity Matrix Construction  
- [ ] **Phase 5:** Retrieval Graph Construction (novel traversal)
- [ ] **Phase 6:** Dataset Generation (RAGAS integration)
- [ ] **Phase 7:** RAG System Evaluation
- [ ] **Phase 8:** Visualization Generation
- [ ] **Phase 9:** Results Analysis & Export
- [ ] **Phase 10:** Cleanup & Validation

**Phase 1 features:**
- ✅ Configuration loading and validation
- ✅ Experiment tracking with unique IDs
- ✅ Directory structure creation
- ✅ Logging system initialization
- ✅ System resource checking
- ✅ M1 Mac / Apple Silicon compatibility
- ✅ Device detection (CPU/CUDA/MPS)

**Phase 2 features:**
- ✅ Wikipedia article acquisition using LangChain (robust & reliable)
- ✅ **BeautifulSoup-based HTML parsing** for superior text cleaning
- ✅ Intelligent fallback system (BeautifulSoup → LangChain → Search)
- ✅ Removes Wikipedia artifacts (infoboxes, tables, navigation, references)
- ✅ Sentence extraction and validation with NLTK
- ✅ Multi-strategy error handling and retries
- ✅ Intelligent caching system
- ✅ Corpus statistics and analysis

### Usage Examples

```bash
# Run with default configuration (Phases 1-2)
python main.py

# Use custom config file
python main.py --config my_config.yaml

# Run only data acquisition
python main.py --mode data_only

# Force fresh article acquisition (ignore cache)
python main.py --force-recompute data

# Force MPS device (Apple Silicon)
python main.py --device mps

# Run in quiet mode
python main.py --quiet

# Show verbose output
python main.py --verbose
```

## Configuration

The pipeline is controlled by `config.yaml`. Key sections:

```yaml
# Experiment settings
experiment:
  name: "semantic_rag_pipeline"
  description: "Semantic graph traversal RAG system"

# System configuration (M1 Mac compatible)
system:
  device: "auto"  # auto, cpu, cuda, mps
  max_memory_gb: 8

# Wikipedia data acquisition
wikipedia:
  use_cached_articles: true
  use_html_parsing: true    # Use BeautifulSoup for superior text cleaning
  topics:
    - "Machine learning"
    - "Artificial intelligence"
    - "Natural language processing"
  articles_per_topic: 5
  min_article_length: 1000
  max_article_length: 50000

# Text processing
text_processing:
  clean_html: true
  remove_references: true
  normalize_whitespace: true
  min_sentence_length: 10
  max_sentence_length: 500
  
# Models to use
models:
  embedding_models:
    - "sentence-transformers/all-mpnet-base-v2"
    - "sentence-transformers/all-MiniLM-L6-v2"
```

## Architecture Overview

```
utils/
├── pipeline.py           # Main orchestrator (Phase 1 complete)
├── models.py            # EmbeddingModel abstraction (TODO)
├── chunking.py          # ChunkEngine (TODO)
├── retrieval.py         # RetrievalEngine - traversal algorithm (TODO)
├── wiki.py              # WikiEngine (TODO)
├── datasets.py          # DatasetEngine - RAGAS integration (TODO)
├── benchmark.py         # RAGASEngine (TODO)
└── visualization.py     # VisualEngine (TODO)

data/                    # Generated automatically
embeddings/              # Generated automatically  
visualizations/          # Generated automatically
logs/                    # Generated automatically

config.yaml              # Main configuration
main.py                  # CLI entry point
```

## Research Innovation

This pipeline implements a novel approach to RAG that differs from current methods:

**Traditional Approaches:**
- Extract entities and relationships
- Build discrete knowledge graphs
- Navigate predefined relationship types

**Our Approach:**
- Compute dense similarity matrices between all sentences
- Use real-time semantic traversal during retrieval
- Navigate through continuous semantic space

Key advantages:
- No lossy entity extraction
- Captures gradient semantic relationships
- Robust to extraction failures
- Natural reading flow through documents

## System Requirements

- **Python:** 3.8+
- **Memory:** 8GB+ recommended
- **Disk:** 5GB+ free space
- **GPU:** Optional but recommended (CUDA/MPS supported)

### Apple Silicon Support

Fully compatible with M1/M2 Macs using MPS (Metal Performance Shaders):
```bash
python main.py --device mps
```

## Development Status

- [x] **Phase 1:** Setup & Initialization
- [x] **Phase 2:** Data Acquisition (Wikipedia + text processing)
- [ ] **Phase 3:** Embedding Generation (multi-model support)
- [ ] **Phase 4:** Similarity Matrix Construction  
- [ ] **Phase 5:** Retrieval Graph Construction (novel traversal)
- [ ] **Phase 6:** Dataset Generation (RAGAS integration)
- [ ] **Phase 7:** RAG System Evaluation
- [ ] **Phase 8:** Visualization Generation
- [ ] **Phase 9:** Results Analysis & Export
- [ ] **Phase 10:** Cleanup & Validation

## Next Steps

The next development priorities are:

1. **Phase 3: Embedding Generation** - Multi-model embedding pipeline with caching
2. **Phase 4: Similarity Matrices** - Core semantic similarity computation  
3. **Phase 5: Retrieval Graph** - Novel semantic traversal algorithm

## License

[Your License Here]

## Contributing

[Contributing Guidelines Here]