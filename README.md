# AI Doubt Solving + Personalized Learning System

This project is a practical end-to-end learning assistant built with Python, Sentence Transformers, FAISS, Pandas, NumPy, and Streamlit.

It supports:

- Semantic search over study material using embeddings and FAISS
- Retrieval-augmented answering for student doubts
- Query history tracking for topic-level weak-area detection
- Personalized study recommendations and quiz prompts
- A simple Streamlit interface for exploration

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- data/
|   `-- knowledge_base.json
`-- src/
    |-- __init__.py
    |-- answer_generator.py
    |-- knowledge_base.py
    |-- personalization.py
    `-- rag_pipeline.py
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Optional: set an OpenAI API key for model-generated answers.

```bash
export OPENAI_API_KEY=your_key_here
```

4. Run the app:

```bash
streamlit run app.py
```

## How It Works

1. Study material in `data/knowledge_base.json` is embedded with a Sentence Transformer model.
2. FAISS indexes the embeddings for fast semantic retrieval.
3. A student question is matched to the most relevant learning chunks.
4. The system generates an answer using:
   - OpenAI, if `OPENAI_API_KEY` is available
   - A local retrieval-based fallback otherwise
5. Query logs are stored in `data/query_history.csv`.
6. Pandas and NumPy summarize topic trends and weak areas to produce recommendations.

## Notes

- The first app run builds the FAISS index locally.
- The included dataset is intentionally small and easy to replace.
- To improve quality, expand the knowledge base with curriculum-specific material.
