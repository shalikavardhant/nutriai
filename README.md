# NutriAI — Automated Diet Plan Builder
BAX-423 Final Project | UC Davis GSM | Spring 2026

## What it does
Generates a personalised 7-day meal plan in under 60 seconds,
tailored to clinical conditions, allergens, diet type, and nutrition targets.

## Pipeline
Clinical filtering → Allergen exclusion (Bloom filter) →
FAISS retrieval → Ranking & diversity → Macro/micro analysis → Plan output

## BAX-423 Techniques
1. **Bloom Filter** — O(k) allergen exclusion (vs O(n) linear scan)
2. **FAISS** — Approximate nearest-neighbour food retrieval using
   combined text + nutrient embeddings (sentence-transformers all-MiniLM-L6-v2)

## Setup

### Local
```bash
pip install -r requirements.txt
# Place data files in data/ folder:
#   data/nutriai_foods.db
#   data/bloom_filters.pkl
#   data/nutriai.index
#   data/nutrient_col_max.npy
streamlit run app.py
```

### Deploy (Streamlit Community Cloud)
1. Push this repo to GitHub
2. Go to share.streamlit.io → New app → select repo → app.py
3. Add data files via Streamlit Cloud file upload or Git LFS

## Project structure
```
code/
  app.py                  ← Streamlit app (this file)
  requirements.txt
  README.md
  day1_data_pipeline.py   ← USDA data pipeline
  day2_filtering_engine.py← Bloom filter + clinical rules
  day3_faiss_ranking.py   ← FAISS index + ranking + diversity
data/
  nutriai_foods.db        ← 5,181 USDA foods with nutrient profiles
  bloom_filters.pkl       ← Pre-built Bloom filters (9 allergen tags)
  nutriai.index           ← FAISS index (384-dim, 5181 vectors)
  nutrient_col_max.npy    ← Normalisation constants
brief.pdf                 ← Technical brief (≤4 pages)
prompts.md                ← Key AI prompts used
```

## Test personas
| Persona | Conditions | Diet | Allergens |
|---------|-----------|------|-----------|
| Priya   | IBS       | Vegetarian | Dairy |
| Ravi    | GERD      | None | Gluten |
| Mei     | T2D       | Vegan | Tree nuts |
| James   | Hypertension | Pescatarian | Shellfish |
