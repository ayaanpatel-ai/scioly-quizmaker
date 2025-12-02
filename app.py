from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from groq import Groq
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def generate_questions(text):
    prompt = f"""
    Create 10 quiz questions based on the following text.
    Include a mix of:
    - multiple choice
    - short answer
    - true/false
    Questions should be clear and understandable.

    Text:
    {text}
    """

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message["content"]

@app.post("/generate-quiz")
async def generate_quiz(pdf: UploadFile = File(...)):
    text = extract_text_from_pdf(pdf.file)
    if not text:
        return {"error": "Could not extract text"}
    questions = generate_questions(text)
    return {"questions": questions}
