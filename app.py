from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from groq import Groq
import os

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
    return {"status": "running"}

def extract_pdf_text(pdf_file):
    """
    Extracts text safely — ignores encrypted pages so PyCryptodome is NOT required.
    """
    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        try:
            text += page.extract_text() or ""
        except:
            continue  # ignore encrypted/problematic pages

    return text


@app.post("/generate_quiz")
async def generate_quiz(pdf: UploadFile = File(...)):
    try:
        text = extract_pdf_text(pdf.file)

        if not text.strip():
            return JSONResponse({"error": "PDF text extraction failed"}, 400)

        prompt = f"""
        Create a quiz based on this content:

        {text}

        Format the quiz STRICTLY as JSON:

        {{
          "questions": [
            {{
              "id": 1,
              "type": "mc",
              "question": "...",
              "options": ["A", "B", "C", "D"],
              "answer": "B"
            }},
            {{
              "id": 2,
              "type": "short",
              "question": "...",
              "answer": "the mitochondria is the powerhouse of the cell"
            }}
          ]
        }}

        Do NOT include explanations. Do NOT add extra text.
        Only output valid JSON.
        """

        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        quiz_json = res.choices[0].message.content

        return JSONResponse({"quiz": quiz_json})

    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/grade_quiz")
async def grade_quiz(payload: dict):
    try:
        quiz = payload["quiz"]          # the quiz JSON as dict
        user_answers = payload["answers"]  # { "1": "B", "2": "blah" }

        results = {}
        correct_count = 0

        for q in quiz["questions"]:
            qid = str(q["id"])
            correct_answer = q["answer"]
            user_answer = user_answers.get(qid, "")

            if q["type"] == "mc":
                is_correct = user_answer.strip().upper() == correct_answer.strip().upper()
            else:  # short answer
                is_correct = correct_answer.lower() in user_answer.lower()

            results[qid] = is_correct
            if is_correct:
                correct_count += 1

        return JSONResponse({
            "scores": results,
            "total_correct": correct_count,
            "total_questions": len(quiz["questions"])
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)
