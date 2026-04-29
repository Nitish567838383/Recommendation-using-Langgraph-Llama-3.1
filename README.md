# Crop Recommendation Chatbot

A Streamlit interface for a crop recommendation system built on a trained ML model and an agricultural knowledge base.

## What this app does

- Loads a trained `RandomForestClassifier` pipeline and label encoder from `crop_system.pkl`
- Accepts structured crop feature input (`N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`)
- Predicts the recommended crop and confidence score
- Provides a chat interface for natural queries
- Uses a static knowledge base for agricultural explanations
- Optionally integrates with Groq LLaMA 3.1 when `GROQ_API_KEY` is set

## Files

- `app.py` — Streamlit application
- `crop_system.pkl` — pickled trained model and label encoder
- `requirements.txt` — required Python packages
- `README.md` — this documentation

## Setup

1. Create and activate a Python virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

## Optional Groq / LLaMA integration

If you want the chatbot to answer questions using the LLM, set the `GROQ_API_KEY` environment variable before launching:

```bash
set GROQ_API_KEY=your_groq_api_key
streamlit run app.py
```

If the environment variable is not provided, the app falls back to the static knowledge base.

## Usage

- Use the chat input to ask questions or request crop recommendations.
- Use the direct prediction panel to provide all 7 crop features and get an immediate recommendation.
- The app maintains session chat history during the Streamlit session.

## Notes

- The model is loaded from `crop_system.pkl`.
- The app uses `DEFAULT_MEANS` for partial predictions when some features are missing.
- This solution provides a practical hybrid interface with ML-powered prediction and knowledge-based assistance.
