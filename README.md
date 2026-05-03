# Natural Language Processing

This repository contains comprehensive assignments from a Natural Language Processing course, covering fundamental NLP concepts from tokenization to embeddings and beyond.

## Installation Instructions

### Prerequisites
- **Python 3.12** (recommended - avoid newer versions as library support isn't fully rolled out)
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd nlp
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
   - **Windows:**
   ```bash
   .venv\Scripts\activate
   ```
   - **macOS/Linux:**
   ```bash
   source .venv/bin/activate
   ```

4. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Repository Structure

### Week 1 - Tokenization & Text Processing
- **Topics:** Text tokenization, sentence segmentation, word normalization
- **Key Concepts:** Regular expressions, NLTK tokenizers, corpus preprocessing

### Week 2 - Text Statistics & Analysis
- **Topics:** Frequency analysis, concordances, lexical diversity
- **Key Concepts:** Vocabulary size, type-token ratio, collocation analysis

### Week 3 - N-grams & Language Models
- **Topics:** Bigrams, trigrams, language modeling fundamentals
- **Key Concepts:** Probability estimation, smoothing techniques

### Week 4 - POS Tagging & Parsing
- **Topics:** Part-of-speech tagging, syntactic analysis, parsing
- **Key Concepts:** HMM taggers, dependency parsing, parse trees

### Week 5 - Sentiment Analysis & Classification
- **Topics:** Text classification, sentiment analysis, feature extraction
- **Key Concepts:** Naive Bayes, TF-IDF, supervised learning

### Week 6 - Word Embeddings & Semantic Analysis
- **Topics:** GloVe embeddings, Word2Vec, word vectors
- **Key Concepts:** Polysemous word analysis, Synonym vs antonym relationships, Analogy solving with word vectors, Gender bias detection in embeddings, Custom embeddings from literary corpora (Gutenberg corpus)
- **Models Analyzed:** Pre-trained GloVe embeddings (6B tokens, 300d), Custom Word2Vec models trained on: Jane Austen (Emma), Lewis Carroll (Alice in Wonderland), William Shakespeare (Hamlet), Herman Melville (Moby Dick), Walt Whitman (Leaves of Grass)

## Dependencies

All required libraries are listed in `requirements.txt`. Key packages include:
- `nltk` - Natural Language Toolkit
- `gensim` - Word2Vec and embeddings
- `scikit-learn` - Machine learning utilities
- `pandas` - Data manipulation
- `numpy` - Numerical computing

## Usage

Run assignments by week:
```bash
python Week1/AssignmentWeek1.py
python Week2/AssignmentWeek2.py
# ... and so on
```

For Week 6 embeddings analysis, ensure the GloVe embeddings file is downloaded:
```bash
# Download from: https://github.com/stanfordnlp/GloVe
# Extract to: Week6/Corpora/glove.6B.300d.txt
```

## Notes

- Assignment files may include both `.py` scripts and `.ipynb` Jupyter notebooks
- Large corpus files (like GloVe embeddings) are excluded from version control via `.gitignore`
- Some assignments require downloading external datasets (NLTK corpora, Gutenberg texts)

## Course Topics Covered

- Text preprocessing and normalization
- Statistical analysis of text
- Language modeling and probability
- Syntactic and semantic analysis
- Machine learning for NLP
- Word embeddings and vector semantics
- Bias detection in NLP models

---

Enjoy exploring NLP!