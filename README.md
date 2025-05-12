Embedding101: Sentence Embeddings and Similarity Analysis
This repository contains a Jupyter notebook (`Embedding101.ipynb`) that demonstrates how to generate and compare sentence embeddings using pre-trained models, evaluate similarity scores, and visualize results with UMAP and FAISS.
📝 Notebook Overview
The notebook covers the following steps:
- **Installation**: Setup of necessary libraries (`sentence-transformers`, `faiss-cpu`, `umap-learn`, `scikit-learn`, `datasets`).
- **Dataset**: Uses the `stsb_multi_mt` dataset (English split) for sentence pairs and similarity scores.
- **Embedding Models**:
  - `all-MiniLM-L6-v2`
  - `BAAI/bge-small-en-v1.5`
  - `distilbert-base-nli-stsb-mean-tokens`
### Key Steps:
1. **Load and Preprocess Data**: Load the `stsb_multi_mt` dataset for sentence pairs.
2. **Generate Embeddings**: Use pre-trained models to generate embeddings for sentence pairs.
3. **Compute Cosine Similarity**: Calculate the cosine similarity between the generated sentence embeddings.
4. **Index Embeddings with FAISS**: Use FAISS for efficient similarity search and indexing.
5. **Visualize with UMAP**: Reduce dimensionality and visualize the embeddings.
### Results:
- **Model Comparison**: Compares model performance against human-annotated similarity scores to evaluate the effectiveness of each embedding model.
🛠 Usage
📦 Clone the Repository
Clone the repository using the command below:
```bash
git clone https://github.com/ihebbenalii/Embeddings-101
```
💾 Install Dependencies
Install the required libraries using pip:
```bash
pip install sentence-transformers faiss-cpu umap-learn scikit-learn datasets
```
🔓 Open and Run the Notebook
Open and run the notebook with the following command:
```bash
jupyter notebook Embedding101.ipynb
```
