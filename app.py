# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from groq import Groq
import os
import json
import traceback
from io import BytesIO

app = FastAPI(title="DV SciOly Quiz Backend")

# CORS for Wix
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.get("/")
async def root():
    return {"status": "ok"}


# ---------------------------
# 🔹 1) Upload PDF → Generate Quiz
# ---------------------------
@app.post("/generate-quiz")
async def generate_quiz(pdf: UploadFile = File(...)):
    try:
        pdf_bytes = await pdf.read()

        # Extract text
        reader = PdfReader(BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        if not text.strip():
            return {"error": "Could not extract text from PDF."}

        # LLM Prompt
        prompt = f"""
You are a Science Olympiad quiz generator.

Output STRICTLY in this JSON format:

{{
  "questions": [
    {{
      "id": number,
      "type": "short" | "mc" | "tf",
      "question": "text",
      "options": ["a)", "b)", ...] (only for MC),
      "answer": "correct answer"
    }}
  ]
}}

Create 10–20 questions based ONLY on this PDF:

{text}
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        raw = response.choices[0].message.content

        # Parse JSON from model output
        try:
            quiz_json = json.loads(raw)
        except:
            quiz_json = json.loads(raw[raw.find("{"): raw.rfind("}")+1])

        return quiz_json

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


# ---------------------------
# 🔹 2) Submit Answers → Grade them
# ---------------------------
@app.post("/grade")
async def grade(answers: dict):
    try:
        quiz = answers["questions"]
        user = answers["user_answers"]

        scores = {}
        correct = 0

        for q in quiz:
            qid = str(q["id"])
            correct_ans = q["answer"].strip().lower()
            user_ans = user.get(qid, "").strip().lower()

            is_correct = (correct_ans == user_ans)
            scores[qid] = is_correct

            if is_correct:
                correct += 1

        return {
            "scores": scores,
            "total_correct": correct,
            "total_questions": len(quiz)
        }

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
