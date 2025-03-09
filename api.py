# -*- coding = utf-8 -*-
# @Time : 3/8/25 14:45
# @Author : Tracy
# @File : api.py
# @Software : PyCharm

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

import os
import pandas as pd
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()


# Load model directly
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("TracyWu32/t5small-cnn-BO-finetuned")
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


# define a path in fastapi app
@app.get("/") # path decorator
def root():
    return {"message": "Hello World"}


@app.post("/summarize/")
async def summarize_text(profile_file: UploadFile = File(...)):
    # Process the uploaded file
    df_desc = pd.read_csv(profile_file.file)

    summaries = []
    for p in df_desc.Profile:
        inputs = tokenizer(p, return_tensors="pt", max_length=500, truncation=True)
        summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=100, early_stopping=True)
        summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        summaries.append(summary)

    df_desc["Summary"] = summaries
    df_desc["T5_Rouge"] = df_desc.apply(lambda row: scorer.score(row['Profile'], row['Summary']), axis=1)

    # Save the result to a CSV file
    summary_path = "data_Summary.csv"
    df_desc.to_csv(summary_path, index=False)

    # Return the generated CSV file
    return FileResponse(summary_path, media_type='text/csv', filename="data_Summary.csv")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)



