from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from groq import Groq
import io
import os
import json
import traceback

app = FastAPI(title="QuizMaker Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.get("/")
async def root():
    return {"status": "ok", "message": "Quiz backend is live."}


# ---------- QUIZ GENERATION ----------
@app.post("/upload")
async def upload(pdf: UploadFile = File(...)):
    try:
        # Read PDF
        pdf_bytes = await pdf.read()
        reader = PdfReader(io.BytesIO(pdf_bytes))

        text = ""
        for page in reader.pages:
            extracted = page.extract_text() or ""
            text += extracted + "\n"

        if not text.strip():
            return {"error": "Failed to extract text from PDF."}

        prompt = f"""
You are a quiz generator. Based ONLY on the following content:

{text}

Produce EXACT JSON in this structure ONLY:

{{
  "questions": [
    {{
      "id": 1,
      "type": "short|mc|tf|yn",
      "question": "text",
      "options": ["a","b","c","d"] OR [],
      "answer": "correct answer"
    }}
  ]
}}
Only return JSON. No markdown. No code blocks.
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        content = response.choices[0].message.content

        # Fix: remove ```json or ``` wrappers if they appear
        if content.startswith("```"):
            content = content.split("```")[-2]

        data = json.loads(content)

        return data

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


# ---------- QUIZ GRADING ----------
@app.post("/grade")
async def grade(payload: dict):
    """
    payload structure:
    {
      "questions": [...],
      "answers": { "1": "user answer", ... }
    }
    """
    try:
        questions = payload["questions"]
        user_answers = payload["answers"]

        scores = {}
        correct_count = 0

        for q in questions:
            qid = str(q["id"])
            correct = str(q["answer"]).strip().lower()
            user = str(user_answers.get(qid, "")).strip().lower()
            is_correct = (user == correct)

            scores[qid] = is_correct
            if is_correct:
                correct_count += 1

        return {
            "scores": scores,
            "total_correct": correct_count,
            "total_questions": len(questions)
        }

    except Exception as e:
        return {"error": str(e)}
