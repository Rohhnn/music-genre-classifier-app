# Music Genre Classifier App

Streamlit deployment for a 10-class music genre classifier (AST full fine-tune).

Weights are downloaded at runtime from Hugging Face — this repo only holds the app.

| | |
|---|---|
| **Full project** | [Rohhnn/music-genre-classifier](https://github.com/Rohhnn/music-genre-classifier) |
| **Model weights** | [Rohhnn/music-genre-classifier](https://huggingface.co/Rohhnn/music-genre-classifier) |
| **Live demo** | [Streamlit app](https://music-genre-classifier-app-nxjyfsazbfkpqqzwpujfyw.streamlit.app/) |

## What this app does

Upload a WAV/MP3; the model predicts among blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock. Predictions average 5 time-spaced crops (TTA).

Reported Kaggle macro F1: **0.9314** (val F1 0.9763).

## Run locally

```bash
git clone https://github.com/Rohhnn/music-genre-classifier-app.git
cd music-genre-classifier-app
pip install -r requirements.txt streamlit huggingface_hub
streamlit run app.py
```

## Stack

- Streamlit + Plotly UI  
- `transformers` AST (`MIT/ast-finetuned-audioset-10-10-0.4593`)  
- Checkpoint via `huggingface_hub`

## Note

This repository exists **only for Streamlit Cloud**. Training code, notebooks, and the report live in the [full project repo](https://github.com/Rohhnn/music-genre-classifier).

Originally built in an academic setting (IIT Madras BS — Intro to DL & GenAI; Messy Mashup competition).
