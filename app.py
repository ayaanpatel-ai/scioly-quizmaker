# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from groq import Groq
import os
import traceback

app = FastAPI(title="Quiz Backend")

# Allow all origins so Wix can call the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.get("/")
async def root():
    return {"status": "ok", "message": "Quiz backend is live"}

# ------------------------------
#       PDF UPLOAD ENDPOINT
# ------------------------------
@app.post("/upload")
async def upload(pdf: UploadFile = File(...)):
    try:
        # Read file bytes
        pdf_bytes = await pdf.read()

        # Parse PDF
        try:
            reader = PdfReader(pdf_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Failed to open PDF. (Encrypted PDFs require PyCryptodome)")

        text = ""
        for page in reader.pages:
            extracted = page.extract_text() or ""
            text += extracted

        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        # PROMPT
        prompt = f"""
You are a quiz generator. Produce 20–40 questions based ONLY on this PDF text.
Format strictly as JSON:

{{
  "questions": [
    {{
      "id": 1,
      "type": "short" | "mc" | "tf",
      "question": "text",
      "options": ["a", "b", "c"]  // only for mc
    }},
    ...
  ]
}}

PDF text:
{text}
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        quiz = response.choices[0].message.content

        return {"quiz": quiz}

    except HTTPException as e:
        raise e

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------
#      GRADING ENDPOINT
# ------------------------------
@app.post("/grade")
async def grade(payload: dict):
    try:
        prompt = f"""
Grade the student's answers using the quiz below.
Return ONLY JSON. No backticks.

Quiz:
{payload["quiz"]}

Student Answers:
{payload["answers"]}

Format:
{{
  "scores": {{
      "1": true/false,
      "2": true/false
  }},
  "total_correct": X,
  "total_questions": Y
}}
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        result = response.choices[0].message.content
        return {"graded": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
