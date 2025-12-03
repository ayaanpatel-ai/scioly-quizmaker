# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from groq import Groq
import os

app = FastAPI(title="Quiz Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


@app.get("/")
def root():
    return {"status": "ok"}


# ----------- QUIZ GENERATION -----------
@app.post("/generate_quiz")
async def generate_quiz(pdf: UploadFile = File(...)):
    try:
        # Read PDF
        reader = PdfReader(pdf.file)

        # Detect encryption (this is your PyCryptodome error)
        if reader.is_encrypted:
            raise HTTPException(
                status_code=400,
                detail="PDF is encrypted. Please upload a non-password-protected PDF."
            )

        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

        # Prompt with answer key
        prompt = f"""
You are a science quiz generator.

Create:
1. 10 multiple-choice questions
2. 5 short-answer questions
3. Provide an answer key at the end

Base everything STRICTLY on this PDF content:

{text}
"""

        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        quiz = res.choices[0].message.content

        return {"quiz": quiz}

    except HTTPException as e:
        raise e
    except Exception as e:
        return {"error": str(e)}


# ----------- ANSWER CHECKING -----------
@app.post("/check_answers")
async def check_answers(user_answers: dict):
    """
    Expected:
    {
      "quiz": "...",
      "answers": {
         "1": "B",
         "2": "C",
         ...
      }
    }
    """
    try:
        prompt = f"""
Grade the following answers.

QUIZ + ANSWER KEY:
{user_answers.get('quiz')}

USER ANSWERS:
{user_answers.get('answers')}

Return JSON in this format ONLY:
{{
  "scores": {{
     "1": true/false,
     "2": true/false,
     ...
  }},
  "total_correct": X,
  "total_questions": Y
}}
"""

        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        graded = res.choices[0].message.content
        return {"graded": graded}

    except Exception as e:
        return {"error": str(e)}
