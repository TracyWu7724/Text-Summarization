from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import boto3
import zipfile
import os
import uvicorn
import tempfile
from mangum import Mangum  

# Add root_path to avoid returning not found
app = FastAPI(root_path="/dev")

# AWS S3 Model Configuration
S3_BUCKET = "hf-t5small-ft-summary"
MODEL_ZIP = "HF_Model.zip"

MODEL_ZIP_PATH = "/tmp/HF_Model.zip"
MODEL_PATH = "/tmp/"
LOAD_PATH = "/tmp/HF_Model/HF_Model"


s3_client = boto3.client("s3")

#todo: initialize logger


def download_from_s3(bucket_name: str, key: str, download_path: str):
    """Download a file from S3 to a local path."""
    try:
        # download zip file
        print(f"Downloading {key} to {download_path}")
        s3_client.download_file(bucket_name, key, download_path)

        # extract model from the zip file
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)

        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(MODEL_PATH)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file from S3: {e}")

def load_model_from_s3():
    """Download and load the model and tokenizer from S3."""

    # ensure the model already exists
    if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
        print("Downloading model from S3...")
        download_from_s3(S3_BUCKET, MODEL_ZIP, MODEL_ZIP_PATH)

    print("Loading model and tokenizer from: ", os.listdir(LOAD_PATH))
    tokenizer = AutoTokenizer.from_pretrained(LOAD_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(LOAD_PATH)

    return tokenizer, model

tokenizer, model = load_model_from_s3()

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "FastAPI running on AWS Lambda!"}

@app.post("/summarize/")
async def summarize_text(input_data: TextInput):
    inputs = tokenizer(input_data.text, return_tensors="pt", max_length=2000, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=100, early_stopping=True)
    summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return {"summary": summary}

handler = Mangum(app)
