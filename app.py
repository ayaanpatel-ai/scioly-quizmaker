# app.py
import os
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from groq import Groq
import traceback

app = FastAPI(title="Quiz Backend")

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
# ROOT ROUTE
# -------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "Quiz backend running"}


# -------------------------------
# UPLOAD + QUIZ GENERATION
# -------------------------------
@app.post("/upload")
async def upload(pdf: UploadFile = File(...)):
    try:
        # Read file into memory as bytes
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

        # Force model to always include answers
        prompt = f"""
        You are a quiz generator. Based strictly on the following PDF content, generate a JSON object ONLY in this format:

        {{
          "questions": [
            {{
              "id": 1,
              "type": "short/mc/tf/yn",
              "question": "text",
              "options": ["A", "B", ...]  (only for MC),
              "answer": "the correct answer ALWAYS filled in"
            }}
          ]
        }}

        RULES:
        - ALWAYS include an answer.
        - For MC questions, answers must be only the letter (e.g., "a").
        - For TF/YN questions, answers must be "true"/"false"/"yes"/"no".
        - Create 10–30 questions.
        - JSON only, no markdown, no backticks.

        PDF CONTENT:
        {text}
        """

        # Call Groq
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        content = response.choices[0].message.content

        # Clean JSON (remove accidental ```json or stray characters)
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[-2]

        # Try parsing
        import json
        try:
            quiz_json = json.loads(content)
        except Exception:
            return {
                "error": "Model returned invalid JSON. Raw output included.",
                "raw": content
            }

        return quiz_json

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# -------------------------------
# GRADING ENDPOINT (IMPROVED)
# -------------------------------
@app.post("/grade")
async def grade(payload: dict):
    """
    Expected:
    {
      "questions": [...],
      "answers": { "1": "user text", ... }
    }
    """
    try:
        import json

        questions = payload["questions"]
        user_answers = payload["answers"]

        scores = {}
        correct_count = 0

        def normalize(x):
            x = str(x).strip().lower()
            # Normalize T/F / Yes/No
            if x in ["true", "t", "yes", "y"]:
                return "true"
            if x in ["false", "f", "no", "n"]:
                return "false"
            return x

        for q in questions:
            qid = str(q["id"])
            correct = normalize(q.get("answer", ""))
            user = normalize(user_answers.get(qid, ""))

            # Multiple choice: allow "a", "a)", "A", "a) option text"
            if q["type"] == "mc":
                user_letter = user.replace(")", "").split(" ")[0]
                correct_letter = correct.replace(")", "")
                is_correct = (user_letter == correct_letter)

            # Short answer: require literal match
            elif q["type"] == "short":
                is_correct = (user == correct)

            # True/false & yes/no normalized
            elif q["type"] in ["tf", "yn"]:
                is_correct = (user == correct)

            else:
                is_correct = False

            scores[qid] = is_correct
            if is_correct:
                correct_count += 1

        return {
            "scores": scores,
            "total_correct": correct_count,
            "total_questions": len(questions)
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
