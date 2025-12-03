# app.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from groq import Groq
import os
import json
import traceback

app = FastAPI(title="QuizMaker Backend")

# --- CORS for Wix ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.get("/")
async def root():
    return {"status": "ok", "msg": "QuizMaker backend running"}

# --------------- QUIZ GENERATION ------------------

@app.post("/generate_quiz")
async def generate_quiz(pdf: UploadFile = File(...)):
    try:
        # Read PDF safely
        try:
            reader = PdfReader(pdf.file)
            text = ""
            for page in reader.pages:
                txt = page.extract_text() or ""
                text += txt
        except Exception:
            return {"error": "Failed to open PDF. (Encrypted PDFs require PyCryptodome)"}

        if not text.strip():
            return {"error": "PDF contained no readable text."}

        prompt = f"""
You are generating a quiz. 
Return ONLY valid JSON. No explanations. No code fences.

Create 20–30 mixed questions from the following content:

{text}

Each question must be formatted exactly like this:

{{
  "id": 1,
  "type": "short" | "mc" | "tf" | "yn",
  "question": "text",
  "options": ["a) ...", "b) ..."] OR null,
  "answer": "correct answer"
}}

Return a JSON object:
{{"questions": [ ... ]}}
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        content = response.choices[0].message.content

        # Remove accidental code fences
        content = content.replace("```json", "").replace("```", "")

        quiz_json = json.loads(content)

        return quiz_json

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

# ------------------- GRADING ENDPOINT -------------------

@app.post("/grade")
async def grade(questions_json: str = Form(...), answers_json: str = Form(...)):
    """
    questions_json = JSON returned by /generate_quiz
    answers_json = {"1": "user answer", "2": "B", ...}
    """

    try:
        questions = json.loads(questions_json)
        answers = json.loads(answers_json)

        prompt = f"""
Grade the following quiz.

QUESTIONS:
{json.dumps(questions)}

USER ANSWERS:
{json.dumps(answers)}

Return ONLY JSON:
{{
  "scores": {{"1": true/false, ...}},
  "total_correct": X,
  "total_questions": Y
}}
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        graded = response.choices[0].message.content
        graded = graded.replace("```json", "").replace("```", "")

        return json.loads(graded)

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
