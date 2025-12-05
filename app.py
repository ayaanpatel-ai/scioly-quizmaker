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

# Allow Wix (wide open)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq client (make sure GROQ_API_KEY exists)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ---------------------------------------------------
# ROOT ROUTE
# ---------------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "Quiz backend running"}


# ---------------------------------------------------
# UPLOAD + QUIZ GENERATION
# ---------------------------------------------------
@app.post("/upload")
async def upload(pdf: UploadFile = File(...)):
    try:
        # Read bytes
        pdf_bytes = await pdf.read()
        stream = io.BytesIO(pdf_bytes)

        # Extract text using pypdf
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

        # Prompt that forces valid JSON and forces answers
        prompt = f"""
        You are a quiz generator. Based ONLY on the following PDF content,
        generate a JSON object EXACTLY in this format:

        {{
          "questions": [
            {{
              "id": 1,
              "type": "short" | "mc" | "tf" | "yn",
              "question": "string",
              "options": ["a) ...", "b) ...", "c) ..."]  (MC ONLY),
              "answer": "string (ALWAYS filled in)"
            }}
          ]
        }}

        RULES:
        - ALWAYS include an "answer" for every question.
        - MC answers must be only the LETTER: "a", "b", "c", etc.
        - TF answers must be "true"/"false".
        - Yes/No answers must be "yes"/"no".
        - Short answers must be a short phrase, not empty.
        - Create between 10 and 30 questions.
        - DO NOT include Markdown, DO NOT include backticks.
        - Response MUST be pure JSON only.

        PDF CONTENT:
        {text}
        """

        # Call Groq
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        raw = response.choices[0].message.content.strip()

        # Clean accidental ```json or ``` wrappers
        if raw.startswith("```"):
            try:
                raw = raw.split("```")[1]
            except:
                pass

        raw = raw.replace("```json", "").replace("```", "").strip()

        # Parse JSON
        try:
            quiz_json = json.loads(raw)
        except Exception:
            return {
                "error": "Model returned invalid JSON",
                "raw": raw
            }

        return quiz_json

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ---------------------------------------------------
# GRADING ENDPOINT (IMPROVED)
# ---------------------------------------------------
@app.post("/grade")
async def grade(payload: dict):
    """
    Expected:
    {
      "questions": [ ... ],
      "answers": { "1": "value", "2": "value", ... }
    }
    """
    try:
        questions = payload["questions"]
        user_answers = payload["answers"]

        scores = {}
        correct_count = 0

        # Normalize answers
        def normalize(x: str):
            x = str(x).strip().lower()

            # Normalize common variants
            if x in ["true", "t", "yes", "y"]:
                return "true"
            if x in ["false", "f", "no", "n"]:
                return "false"
            return x

        for q in questions:
            qid = str(q["id"])
            qtype = q.get("type", "short")

            correct = normalize(q.get("answer", ""))
            user = normalize(user_answers.get(qid, ""))

            # ---- Multiple choice grading ----
            if qtype == "mc":
                # User might enter "a", or "a)", or the whole option text
                user_letter = user.replace(")", "").split(" ")[0]
                correct_letter = correct.replace(")", "")
                is_correct = (user_letter == correct_letter)

            # ---- True/false or Yes/No ----
            elif qtype in ["tf", "yn"]:
                is_correct = (user == correct)

            # ---- Short answer ----
            else:
                # Require close match – but allow partial scoring using Groq
                if user == correct:
                    is_correct = True
                else:
                    # Ask Groq to check semantic correctness for short answers
                    try:
                        g = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {"role": "user", "content":
                                    f"""
                                    Compare the correct answer and user's answer.
                                    Respond ONLY with "true" or "false".

                                    Correct: {correct}
                                    User: {user}

                                    Return true if the user answer is semantically correct.
                                    """}
                            ],
                            temperature=0
                        )
                        g_res = g.choices[0].message.content.strip().lower()
                        is_correct = ("true" in g_res)
                    except:
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
