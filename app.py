# app.py
import os
import io
import json
import traceback
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from groq import Groq

app = FastAPI(title="Quiz Backend")

# Allow Wix
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ---------------------------------------
# ROOT
# ---------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "Quiz backend running"}


# ---------------------------------------
# QUIZ GENERATOR (MC ONLY)
# ---------------------------------------
@app.post("/upload")
async def upload(pdf: UploadFile = File(...)):
    try:
        pdf_bytes = await pdf.read()
        stream = io.BytesIO(pdf_bytes)

        # Extract PDF text
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

        # -----------------------------
        # FORCE MC-ONLY QUESTIONS
        # -----------------------------
        prompt = f"""
        You are a strict quiz generator. Based ONLY on the PDF text below, generate
        a JSON object with 10–30 MULTIPLE-CHOICE questions only.

        JSON FORMAT (NO MARKDOWN, NO ```):
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
        - ALWAYS use multiple choice (mc).
        - ALWAYS include 4 options: a, b, c, d.
        - ALWAYS set answer to the correct letter only.
        - NO markdown, NO commentary, ONLY valid JSON.
        - Use information STRICTLY found in the PDF.

        PDF CONTENT:
        {text}
        """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        raw = response.choices[0].message.content.strip()

        # remove accidental triple-backticks
        if raw.startswith("```"):
            raw = raw.split("```")[1]
        if raw.endswith("```"):
            raw = raw.replace("```", "")

        # remove stray markdown labels
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            quiz = json.loads(raw)
        except Exception:
            return {
                "error": "Invalid JSON returned from model.",
                "raw": raw
            }

        return quiz

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ---------------------------------------
# GRADING ENDPOINT (MC ONLY + RETURN CORRECT ANSWER)
# ---------------------------------------
@app.post("/grade")
async def grade(payload: dict):
    """
    payload:
    {
      "questions": [...],
      "answers": { "1": "user's answer", ... }
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

            # Normalize MC like "a)", "A", "a) text"
            def normalize_mc(x):
                x = x.lower().strip()
                x = x.replace(")", "")
                x = x.split(" ")[0]
                return x
            user_norm = normalize_mc(user)
            correct_norm = normalize_mc(correct)

            is_correct = (user_norm == correct_norm)

            results[qid] = {
                "question": q["question"],
                "user_answer": user_norm,
                "correct_answer": correct_norm,
                "is_correct": is_correct
            }

            if is_correct:
                correct_count += 1

        return {
            "results": results,
            "total_correct": correct_count,
            "total_questions": len(questions)
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
