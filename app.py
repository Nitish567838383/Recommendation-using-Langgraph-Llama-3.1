import os
import pickle
import re
import pandas as pd
import streamlit as st

MODEL_PATH = "crop_system.pkl"
FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
DEFAULT_MEANS = {
    "N": 50.55,
    "P": 53.36,
    "K": 48.14,
    "temperature": 25.6,
    "humidity": 71.4,
    "ph": 6.47,
    "rainfall": 103.4,
}

KNOWLEDGE_BASE = """
Crop recommendation depends on:

- N (Nitrogen): Leaf growth and chlorophyll production.
- P (Phosphorus): Root development and flowering.
- K (Potassium): Overall plant strength and stress tolerance.
- Temperature & Humidity: Climate suitability and crop performance.
- pH: Soil acidity/alkalinity affects nutrient availability.
- Rainfall: Water availability for crops.

Examples:
- High rainfall → rice, sugarcane.
- Moderate climate → wheat, maize.
- Tropical climate → banana, coconut.
"""


def load_model():
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["encoder"]


@st.cache_resource
def get_model():
    return load_model()


def predict_crop(model, encoder, N, P, K, temperature, humidity, ph, rainfall):
    features = pd.DataFrame([
        {
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
        }
    ])
    prediction = model.predict(features)[0]
    confidence = float(model.predict_proba(features).max())
    crop_name = encoder.inverse_transform([prediction])[0]
    return crop_name, confidence


def extract_full_features(query: str):
    values = re.findall(r"\d+\.?\d*", query)
    if len(values) >= 7:
        selected = [float(v) for v in values[:7]]
        return dict(zip(FEATURE_NAMES, selected))
    return None


def extract_partial_features(query: str):
    matches = dict(
        re.findall(r"(n|p|k|temperature|humidity|ph|rainfall)\s*([0-9]+\.?[0-9]*)",
                   query,
                   re.IGNORECASE)
    )
    return {k.lower(): float(v) for k, v in matches.items()} if matches else {}


def get_llm_answer(question: str) -> str:
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return f"{KNOWLEDGE_BASE}\n\n(LLM not configured. Set GROQ_API_KEY to enable Groq/LLaMA answers.)"

    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(model="llama-3.1-8b-instant")
        prompt = f"Answer briefly using this:\n{KNOWLEDGE_BASE}\n\nQ: {question}"
        response = llm.invoke(prompt)
        answer = response.content.strip()
        return f"🤖 Model: LLaMA 3.1 (Groq)\n\n📚 {answer}"
    except Exception as exc:
        return f"LLM integration failed: {exc}\n\nKnowledge base:\n{KNOWLEDGE_BASE}"


def route_query(query: str, model, encoder) -> str:
    query_text = query.strip().lower()

    full_features = extract_full_features(query)
    if full_features is not None:
        crop, confidence = predict_crop(model, encoder, **full_features)
        return (
            f"🤖 Model: Random Forest\n\n🌱 Crop: {crop}\n📊 Confidence: {confidence:.2f}\n\n🧠 Decision: Rule-based → ML"
        )

    partial = extract_partial_features(query)
    if partial:
        merged = {k: partial.get(k, DEFAULT_MEANS[k]) for k in DEFAULT_MEANS}
        crop, confidence = predict_crop(model, encoder, **merged)
        missing = [k for k in merged if k not in partial]
        missing_text = ", ".join(missing) if missing else "none"
        return (
            f"🤖 Model: Random Forest\n\n🌱 Crop: {crop}\n📊 Confidence: {confidence:.2f}\n\n🧠 Decision: Rule-based → Partial ML"
        )

    return get_llm_answer(query)


def init_session():
    if "history" not in st.session_state:
        st.session_state.history = []


def add_message(role: str, content: str):
    st.session_state.history.append({"role": role, "content": content})


def main():
    st.set_page_config(page_title="Crop Recommendation Chatbot", layout="wide")
    st.title("🌾 Crop Recommendation Chatbot")
    st.write(
        "Interact with a hybrid crop recommendation system that combines ML predictions with a knowledge base. "
        "Ask questions or enter field values to get a recommendation."
    )

    init_session()
    model, encoder = get_model()

    with st.expander("How it works", expanded=True):
        st.markdown(
            "- Use the **chat input** for natural questions or partial data queries.\n"
            "- Use the **direct prediction panel** to provide all 7 crop features.\n"
            "- The app loads the trained `RandomForestClassifier` from `crop_system.pkl`.\n"
            "- If `GROQ_API_KEY` is configured, the chatbot can answer conceptual questions using LLaMA 3.1 via Groq."
        )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Chat interface")
        query = st.text_input("Ask a question or request a crop recommendation", key="user_query")
        if st.button("Send") and query:
            add_message("user", query)
            response = route_query(query, model, encoder)
            add_message("bot", response)
            st.session_state.latest_response = response

        if "latest_response" in st.session_state:
            st.markdown("**Bot:**")
            st.write(st.session_state.latest_response)

    with col2:
        st.subheader("Direct prediction panel")
        with st.form(key="prediction_form"):
            values = {
                feature: st.number_input(feature, value=DEFAULT_MEANS[feature] if feature != "ph" else 6.5, format="%.2f")
                for feature in FEATURE_NAMES
            }
            submit = st.form_submit_button("Predict crop")
            if submit:
                crop, confidence = predict_crop(model, encoder, **values)
                st.success(f"Recommended crop: {crop}")
                st.info(f"Confidence: {confidence:.2f}")
                add_message("user", f"Direct prediction: {values}")
                add_message("bot", f"Recommended crop: {crop} (confidence {confidence:.2f})")

        st.subheader("Knowledge base")
        st.write(KNOWLEDGE_BASE)

        st.markdown("**Sample queries:**")
        st.markdown(
            "- `Which crop is suitable for N 90 P 40 K 40 temperature 22 humidity 80 ph 6.5 rainfall 200?`\n"
            "- `How does pH affect crop growth?`\n"
            "- `Recommend a crop for moderate humidity and high rainfall.`"
        )

    if st.session_state.history:
        with st.expander("Chat History", expanded=False):
            for msg in st.session_state.history:
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['content']}")
                else:
                    # Extract model and key info for summary
                    content = msg['content']
                    if "🤖 Model: Random Forest" in content or "Recommended crop:" in content:
                        # Extract crop and confidence
                        if "🌱 Crop:" in content:
                            lines = content.split('\n')
                            crop_line = [l for l in lines if '🌱 Crop:' in l]
                            crop = crop_line[0].split('🌱 Crop:')[1].strip()
                            conf_line = [l for l in lines if '📊 Confidence:' in l]
                            conf = conf_line[0].split('📊 Confidence:')[1].strip() if conf_line else ''
                        else:
                            # Direct prediction format
                            crop = content.split('Recommended crop:')[1].split('(')[0].strip()
                            conf = content.split('confidence')[1].strip().strip(')')
                        st.markdown(f"**Predicted using Random Forest:** {crop} (confidence {conf})")
                    elif "🤖 Model: LLaMA 3.1 (Groq)" in content:
                        # Show first line of answer
                        lines = content.split('\n')
                        answer_line = [l for l in lines if l.startswith('📚')]
                        if answer_line:
                            answer = answer_line[0].replace('📚 ', '').strip()
                            if len(answer) > 100:
                                answer = answer[:100] + "..."
                            st.markdown(f"**Answered using LLaMA 3.1:** {answer}")
                        else:
                            st.markdown(f"**Answered using LLaMA 3.1:** {content[:100]}...")
                    else:
                        st.markdown(f"**Bot:** {content[:100]}...")

    st.markdown("---")
    st.write("Built with Streamlit, trained ML model, and a static agricultural knowledge base.")


if __name__ == "__main__":
    main()
