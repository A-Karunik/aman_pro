import streamlit as st
from nltk.tokenize import sent_tokenize
import PyPDF2
import tempfile
import os
from transformers import BertTokenizer, BertModel, BertForNextSentencePrediction

# Specify the directory location where you want to store temporary files
temp_dir = os.path.join(os.getcwd(), "temp_files")

# Create the directory if it doesn't exist
os.makedirs(temp_dir, exist_ok=True)

def extractive_summary(text, num_sentences):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    sentences = sent_tokenize(text)
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    probs = outputs.logits[:, 0]
    ranked_sentences = [sentence for _, sentence in sorted(zip(probs, sentences), reverse=True)]
    return " ".join(ranked_sentences[:num_sentences])

def extract_text_from_pdf(filename):
    with open(filename, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

def main():
    st.title("Extractive Summarization App")
    summary_type = st.selectbox("Summarize from:", ("Text Input", "PDF Upload"))
    if summary_type == "Text Input":
        user_text = st.text_area("Enter the text you want to summarize:", height=200)
        num_sentences = st.text_input("Enter the desired number of sentences in the summary (default 2): ", value="2")
        if st.button("Summarize"):
            try:
                num_sentences = int(num_sentences)
                summary = extractive_summary(user_text, num_sentences)
                st.subheader("Summary:")
                st.write(summary)
            except ValueError:
                st.error("Please enter a valid number for desired sentences.")
    else:
        uploaded_file = st.file_uploader("Upload your PDF file:", type="pdf")

        if uploaded_file is not None:
                pdf_bytes = uploaded_file.read()
                with tempfile.NamedTemporaryFile(suffix=".pdf", dir=temp_dir, delete=False) as temp_pdf:
                    temp_pdf.write(pdf_bytes)
                    extracted_text = extract_text_from_pdf(temp_pdf.name)
                    num_sentences = st.text_input("Enter the desired number of sentences in the summary (default 2): ", value="2")
                    print("num_sentences:", num_sentences)  # Debug statement
                    if st.button("Summarize"):
                        try:
                            num_sentences = int(num_sentences)
                            summary = extractive_summary(extracted_text, num_sentences)
                            st.subheader("Summary:")
                            st.write(summary)
                        except ValueError:
                            st.error("Please enter a valid number for desired sentences.")


if __name__ == "__main__":
    main()
