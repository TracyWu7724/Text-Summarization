import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import streamlit as st

# Streamlit app configuration
st.set_page_config(page_title="Company Profile")

# Header
st.header("Generate Summary")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Load data
    df_company = pd.read_csv(uploaded_file)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("TracyWu32/t5small-cnn-BO-finetuned")
    
    # Generate summaries
    summaries = []
    for profile in df_company.Profile:
        inputs = tokenizer(profile, return_tensors="pt", max_length=2000, truncation=True)
        summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=100, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    # Add summaries to DataFrame
    df_company["Summary"] = summaries

    # Display the DataFrame
    st.dataframe(df_company)

    # Provide option to download the results
    csv = df_company.to_csv(index=False).encode('utf-8')
    st.download_button("Download Summarized Data", csv, "summarized_data.csv", "text/csv")
