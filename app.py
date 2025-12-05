# app.py
import os
import io
import json
import traceback
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from groq import Groq

app = FastAPI(title="SciOly Quiz Backend")

# ----- CORS (Wix compatible) -----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Groq -----
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------------------------------
# ROOT
# -----------------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "msg": "Quiz backend running"}


# -----------------------------------------------------
# QUIZ GENERATION (MC ONLY)
# -----------------------------------------------------
@app.post("/upload")
async def upload(pdf: UploadFile = File(...)):
    try:
        pdf_bytes = await pdf.read()
        stream = io.BytesIO(pdf_bytes)

        # Extract text
        try:
            reader = PdfReader(stream)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except Exception as e:
            return {"error": f"Failed to open PDF: {str(e)}"}

        if not text.strip():
            return {"error": "No extractable text found in PDF"}

        # Force MC-only JSON format
        prompt = f"""
        You are a quiz generator. Based ONLY on the following PDF content,
        generate exactly 10–20 **multiple-choice** questions.

        **REQUIRED JSON FORMAT (NO markdown, NO backticks):**
        {{
            "questions": [
                {{
                    "id": 1,
                    "type": "mc",
                    "question": "text",
                    "options": ["a) ...", "b) ...", "c) ...", "d) ..."],
                    "answer": "a"
                }}
            ]
        }}

        RULES:
        - ONLY multiple-choice, each with options a/b/c/d.
        - The correct answer MUST be ONLY a letter ("a", "b", "c", "d").
        - MUST return valid JSON and nothing else.
        - All answers must be deterministic and unambiguous.

        PDF CONTENT:
        {text}
        """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        content = response.choices[0].message.content.strip()

        # Strip markdown fences if model added them
        if content.startswith("```"):
            content = content.split("```")[1]

        try:
            quiz = json.loads(content)
        except Exception:
            return {
                "error": "Model returned invalid JSON.",
                "raw": content
            }

        return quiz

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# -----------------------------------------------------
# GRADING — MC ONLY + RETURNS CORRECT ANSWERS
# -----------------------------------------------------
@app.post("/grade")
async def grade(payload: dict):
    """
    Expected payload:
    {
      "questions": [...],
      "answers": { "1": "b", ... }
    }
    """
    try:
        questions = payload["questions"]
        user_answers = payload["answers"]

        results = {}
        correct_count = 0

        for q in questions:
            qid = str(q["id"])
            correct = q.get("answer", "").strip().lower()
            user = user_answers.get(qid, "").strip().lower()

            is_correct = (user == correct)

            if is_correct:
                correct_count += 1

            results[qid] = {
                "user_answer": user,
                "correct_answer": correct,
                "is_correct": is_correct
            }

        return {
            "results": results,
            "total_correct": correct_count,
            "total_questions": len(questions)
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
