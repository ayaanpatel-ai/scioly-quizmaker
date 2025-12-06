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

# Allow Wix
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# -------------------------------
# ROOT
# -------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "Quiz backend running"}


# -------------------------------
# GENERATE QUIZ
# -------------------------------
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

        # MC ONLY prompt
        prompt = f"""
        Generate ONLY a JSON object in this exact structure:

        {{
          "questions": [
            {{
              "id": 1,
              "type": "mc",
              "question": "text",
              "options": ["a) ...","b) ...","c) ...","d) ..."],
              "answer": "a"
            }}
          ]
        }}

        RULES:
        - ALWAYS generate 10 multiple-choice questions.
        - EVERY question MUST have 4 answer choices.
        - The correct answer MUST be the letter only: "a", "b", "c", or "d".
        - NO markdown, NO ```.

        Base ALL questions strictly on this PDF content:
        {text}
        """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        content = response.choices[0].message.content.strip()

        # Remove accidental markdown
        if content.startswith("```"):
            content = content.split("```")[-2]

        try:
            quiz_json = json.loads(content)
        except Exception:
            return {"error": "Model returned invalid JSON", "raw": content}

        return quiz_json

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# -------------------------------
# GRADING (MC ONLY + RETURN CORRECT ANSWER)
# -------------------------------
@app.post("/grade")
async def grade(payload: dict):
    """
    payload:
    {
      "questions": [...],
      "answers": { "1": "b", ... }
    }
    """
    try:
        questions = payload["questions"]
        user_answers = payload["answers"]

        scores = {}
        correct_answers = {}
        correct_count = 0

        def clean_answer(ans: str) -> str:
            """
            Normalize answers like:
            'a)', 'A', 'a. ', 'a) text', 'a: ', 'A )'
            into just: 'a'
            """
            if not ans:
                return ""
            ans = ans.strip().lower()

            # First character must be a, b, c, or d
            if len(ans) > 0 and ans[0] in ["a", "b", "c", "d"]:
                return ans[0]

            return ans  # fallback

        for q in questions:
            qid = str(q["id"])

            # Clean both correct + user answers
            correct = clean_answer(q["answer"])
            user = clean_answer(user_answers.get(qid, ""))

            correct_answers[qid] = correct
            is_correct = (correct == user)
            scores[qid] = is_correct

            if is_correct:
                correct_count += 1

        return {
            "scores": scores,
            "correct_answers": correct_answers,
            "total_correct": correct_count,
            "total_questions": len(questions)
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
