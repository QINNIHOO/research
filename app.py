import streamlit as st
import numpy as np
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Cyberbullying Detector",
    page_icon="🛡️",
    layout="centered",
)

@st.cache_resource
def get_model():
    #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    model = BertForSequenceClassification.from_pretrained("qinnihoo/Bert")
    return tokenizer,model


tokenizer,model = get_model()

# -------------------------------
# UI Layout
# -------------------------------
st.title("🛡️ Cyberbullying Detection App")
st.markdown(
    """
    Enter a sentence below and let the model check whether it contains **cyberbullying** content.  
    This demo uses a fine-tuned BERT model.
    """
)

# Input box
user_input = st.text_area("💬 Enter Text", height=150, placeholder="Type a social media comment...")

# Predict button
if st.button("🚀 Detect Now!"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before detection.")
    else:
        with st.spinner("Analyzing text..."):
            test_sample = tokenizer(
                [user_input],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            output = model(**test_sample)
            y_pred = np.argmax(output.logits.detach().numpy(), axis=1)

            # Result mapping
            d = {
                1: ("Oh no! 🚨 This looks like **Cyberbullying**", "red"),
                0: ("✅ Safe! This is **Non-Cyberbullying**", "green"),
            }

            message, color = d[y_pred[0]]

            # Show result as a colored card
            st.markdown(
                f"""
                <div style="padding:20px; border-radius:12px; background-color:{'mistyrose' if color=='red' else '#d4edda'}; border:1px solid {color};">
                    <h3 style="color:{color}; text-align:center;">{message}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Optional: Show raw logits
            with st.expander("🔎 View Model Details"):
                st.write("**Logits:**", output.logits)
                st.write("**Predicted Class:**", y_pred[0])