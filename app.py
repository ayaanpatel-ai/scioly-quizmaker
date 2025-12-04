from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from groq import Groq
import os
import traceback

app = FastAPI(title="Quiz Backend")

# Allow all origins (works with Wix)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.get("/")
def root():
    return {"status": "ok", "message": "Quiz backend running"}

# ----------------------------
#   UPLOAD PDF & GENERATE QUIZ
# ----------------------------
@app.post("/upload")
async def upload(pdf: UploadFile = File(...)):
    try:
        if not pdf.filename.lower().endswith(".pdf"):
            raise HTTPException(400, "File must be a PDF")

        # Read raw PDF bytes
        pdf_bytes = await pdf.read()

        # Load PDF with pypdf
        try:
            reader = PdfReader(pdf_bytes)
        except Exception as e:
            raise HTTPException(400, f"Failed to open PDF: {str(e)}")

        # Extract text
        text = ""
        for page in reader.pages:
            try:
                text += page.extract_text() or ""
            except:
                continue

        if not text.strip():
            raise HTTPException(400, "PDF text could not be extracted")

        # Prompt to generate quiz JSON
        prompt = f"""
Turn the following text into a JSON quiz. 
Use this format EXACTLY:

{{
  "questions": [
    {{
      "id": 1,
      "type": "short" | "mc" | "tf" | "yn",
      "question": "...",
      "options": ["a)", "b)", ...]  // only for MC
    }}
  ]
}}

Content to generate quiz from:
{text}
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        quiz_json = response.choices[0].message.content

        return {"quiz": quiz_json}

    except HTTPException as e:
        raise e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Server error: {str(e)}")


# ----------------------------
#   GRADING ROUTE
# ----------------------------
@app.post("/grade")
async def grade(data: dict):
    try:
        questions = data.get("questions")
        user_answers = data.get("answers")

        if not questions or not user_answers:
            raise HTTPException(400, "Missing questions or answers")

        grading_prompt = f"""
Grade the following quiz answers.

Return ONLY valid JSON.

Questions:
{questions}

User answers:
{user_answers}

Respond in this strict JSON format:

{{
  "scores": {{ "1": true/false, ... }},
  "total_correct": 0,
  "total_questions": 0
}}
"""

        result = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": grading_prompt}],
        )

        graded = result.choices[0].message.content
        return {"graded": graded}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Grading error: {str(e)}")
