import os
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from groq import Groq
from difflib import SequenceMatcher

app = FastAPI(title="DV QuizMaker Backend")

# -----------------------------
# CORS (Wix requires *)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# -----------------------------------
# Root health check
# -----------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "Quiz backend running"}


# -----------------------------------
# Upload + Generate Quiz
# -----------------------------------
@app.post("/upload")
async def upload(pdf: UploadFile = File(...)):
    try:
        if not pdf.filename.lower().endswith(".pdf"):
            return {"error": "File must be a PDF"}

        # Ensure large PDFs load properly (up to ~10MB)
        file_bytes = await pdf.read()
        pdf_stream = io.BytesIO(file_bytes)

        try:
            reader = PdfReader(pdf_stream)
        except Exception as e:
            return {"error": f"Failed to open PDF: {str(e)}"}

        # Extract text
        text = ""
        for page in reader.pages:
            extracted = page.extract_text() or ""
            text += extracted + "\n\n"

        if len(text.strip()) < 10:
            return {"error": "PDF text could not be extracted"}

        # Force the model to ALWAYS include an answer field
        prompt = f"""
You are generating a JSON quiz from this PDF content.
RULES:
- ALWAYS include an "answer" field for every question.
- Allowed types: "short", "mc", "tf", "yn"
- For MC, provide "options": ["A) ...", "B) ...", ...]
- The JSON MUST NOT include markdown.
- Produce exactly this structure:

{{
  "questions": [
    {{
      "id": 1,
      "type": "short",
      "question": "What is ...?",
      "answer": "the correct answer"
    }}
  ]
}}

PDF CONTENT BELOW:
{text}
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        quiz_text = response.choices[0].message.content

        # Parse JSON from model
        import json
        quiz = json.loads(quiz_text)

        return quiz

    except Exception as e:
        return {"error": str(e)}


# -----------------------------------
# Grading Helpers
# -----------------------------------

def normalize(s: str):
    return s.strip().lower()

def similar(a, b):
    """fuzzy match ratio"""
    return SequenceMatcher(None, a, b).ratio()


# -----------------------------------
# Grade Endpoint
# -----------------------------------
@app.post("/grade")
async def grade(payload: dict):
    """
    Payload:
    {
      "questions": [...],
      "answers": { "1": "user input", ... }
    }
    """
    try:
        questions = payload["questions"]
        user_answers = payload["answers"]

        scores = {}
        correct_count = 0

        for q in questions:
            qid = str(q["id"])
            correct = normalize(str(q.get("answer", "")))
            user = normalize(str(user_answers.get(qid, "")))

            # Empty user answer? Always incorrect.
            if user.strip() == "":
                scores[qid] = False
                continue

            is_correct = False

            # ---------- MC QUESTIONS ----------
            if q["type"] == "mc":
                # allow user to input "a" or "A" or "A)"
                user_letter = user[0]
                correct_letter = correct[0]
                if user_letter == correct_letter:
                    is_correct = True

            # ---------- TRUE/FALSE / YES/NO ----------
            elif q["type"] in ["tf", "yn"]:
                truth_map = {
                    "true": "true",
                    "t": "true",
                    "yes": "true",
                    "y": "true",
                    "false": "false",
                    "f": "false",
                    "no": "false",
                    "n": "false",
                }
                user_norm = truth_map.get(user, user)
                correct_norm = truth_map.get(correct, correct)
                is_correct = (user_norm == correct_norm)

            # ---------- SHORT ANSWER ----------
            else:
                # fuzzy match threshold
                if similar(user, correct) > 0.75:
                    is_correct = True

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
