# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from groq import Groq
import os
import traceback

app = FastAPI(title="Quiz Backend")

# Allow all origins (you can lock this down later to your Wix domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq client (make sure GROQ_API_KEY is set in Render env)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Health / root route so / doesn't 404 ---
@app.get("/")
async def root():
    return {"status": "ok", "message": "Quiz backend is running."}

# --- Upload endpoint (POST /upload) ---
@app.post("/upload")
async def upload(pdf: UploadFile = File(...)):
    try:
        # Read and extract text from PDF
        reader = PdfReader(pdf.file)
        text = ""
        for page in reader.pages:
            ptext = page.extract_text()
            if ptext:
                text += ptext + "\n"

        if not text.strip():
            return {"error": "Could not extract any text from the PDF."}

        # Build prompt for the model
        prompt = f"""
Create 10 clear quiz questions (mix of multiple choice, short answer, true/false)
based on the text below. Return plain text only (no meta JSON).

Text:
{text}
"""

        # Call Groq (adjust model name if you prefer)
        response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        )


        # Extract the generated quiz text
        quiz = response.choices[0].message.content

        # Return the quiz under the "quiz" field (frontend expects this)
        return {"quiz": quiz_text}

    except Exception as e:
        # Return error message so frontend can show it instead of "undefined"
        tb = traceback.format_exc()
        return {"error": str(e), "traceback": tb}
