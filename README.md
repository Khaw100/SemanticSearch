# Semantic Search for Course Datasets Using Feature Expansion

## 🎓 Thesis Research & Project Overview

This repository contains the codebase for an Information Retrieval research project focused on building a Semantic Search engine for a dataset of courses. 

The core research objective of this project is to solve the **Vocabulary Mismatch Problem**—where a user's search query (e.g., "vehicle") does not match the exact keywords used in the underlying document (e.g., "car"). To achieve this, the project leverages **Zero-Shot Pre-Trained Transformers** augmented with **FastText-based Feature Expansion (Query Expansion)**.

### Why Feature Expansion?
Instead of computationally expensive model fine-tuning, this project manipulates the search input at runtime using word embeddings.

**Example Pipeline:**
1. **Original Query:** `"Fast Car"`
2. **FastText Feature Expansion:** For each term in the query, the system retrieves the top-k semantically similar words.
   - *fast* → `Fast`, `super-fast`, `fast-`
   - *car* → `cars`, `vehicle`, `automobile`
3. **Expanded Query:** `"fast super car cars vehicle automobile"`

By expanding the semantic footprint of the query using FastText before calculating similarity, the system can significantly improve document retrieval relevance and handle diverse vocabulary without requiring domain-specific model retraining.

---

## 🛠 Tech Stack
- **Language**: Python 3
- **Data Manipulation**: `pandas`, `numpy`
- **NLP & Text Processing**: `nltk` (tokenization, stopwords, Porter Stemming), Regular Expressions (`re`)
- **Embeddings & Modeling**: 
  - `fastText` (Word-level embeddings for Feature Expansion)
  - `sentence_transformers` (Document-level zero-shot embeddings)
  - `torch` (PyTorch backend)
- **Retrieval & Similarity**: `scikit-learn` (Cosine Similarity)

---

## 📁 Repository Structure

- **`DataMerge.ipynb`**: 
  Handles the ingestion and aggregation of partitioned dataset chunks (`translated_1.csv` through `translated_12.csv`) located in the `/Data` directory.
- **`EDA.ipynb`**: 
  Exploratory Data Analysis (EDA) notebook. Focuses on data cleaning, extracting summary statistics, and heavily preprocessing text (casing, stopwords, stemming) to prepare the `courses_dataset.csv` for embedding.
- **`FastText.ipynb` (Core Methodology)**: 
  The core notebook demonstrating the **Feature Expansion** technique. Uses FastText word embeddings to find semantic neighbors for query terms, expanding the user's intent.
- **`modeling.ipynb`**: 
  Generates dense vector representations of the preprocessed course descriptions using zero-shot `SentenceTransformer` models, utilizing Cosine Similarity to rank and retrieve documents against the expanded queries.
- **`test.ipynb` & `testa.ipynb`**: 
  Validation notebooks used to test the semantic search logic, vectorization pipelines, and metric evaluations.
- **`requirements.txt`**: 
  Project dependencies.

---

## 🚀 Pipeline & Reproducibility

### 1. Installation
Clone the repository and set up your virtual environment:
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
```
*(Note: GPU versions of PyTorch are recommended for faster embedding generation).*

### 2. Execution Flow
To reproduce the research pipeline, execute the Jupyter Notebooks in the following order:
1. **`DataMerge.ipynb`**: Aggregates raw text data.
2. **`EDA.ipynb`**: Cleans text and standardizes the vocabulary (`preprocessed_courses.csv`).
3. **`FastText.ipynb`**: Run the Feature Expansion experiments on the vocabulary.
4. **`modeling.ipynb`**: Generates document embeddings and performs the final Cosine Similarity retrieval.

---

## 🔬 Key Takeaways
- **Efficiency over Fine-Tuning:** Proves that query modification (Feature Expansion) via FastText is a lightweight, highly effective alternative to expensive domain-specific model fine-tuning.
- **Zero-Shot Viability:** Demonstrates that general-purpose Sentence Transformers can perform excellently on specialized course datasets when assisted by word-level semantic expansion.