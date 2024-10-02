import streamlit as st
from transformers import pipeline, PegasusForConditionalGeneration, PegasusTokenizer

# Load the Pegasus Model and Tokenizer
model_name = "google/pegasus-xsum"
pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name)

# Streamlit App Title
st.title("Text Summarization Using NLP")

# Input Text Box
input_text = st.text_area("Enter the text you want to summarize", height=300)

# Parameters for Summary
min_len = st.slider("Minimum Summary Length", 10, 100, 30)
max_len = st.slider("Maximum Summary Length", 50, 300, 150)

# Summarization Function
def summarize_text(text, min_length, max_length):
    tokens = pegasus_tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
    summary_ids = pegasus_model.generate(**tokens, min_length=min_length, max_length=max_length)
    decoded_summary = pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return decoded_summary

# Button to Generate Summary
if st.button("Summarize"):
    if input_text:
        with st.spinner("Generating summary..."):
            summary = summarize_text(input_text, min_len, max_len)
            st.subheader("Summary:")
            st.write(summary)
    else:
        st.warning("Please input text for summarization.")

# Option to view the original text
if st.checkbox("Show Original Text"):
    st.subheader("Original Text:")
    st.write(input_text)