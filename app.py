# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from groq import Groq
import os
import traceback

app = FastAPI(title="Quiz Backend")

# Allow all origins (you can lock this down later to your Wix domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq client (make sure GROQ_API_KEY is set in Render env)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Health / root route so / doesn't 404 ---
@app.get("/")
async def root():
    return {"status": "ok", "message": "Quiz backend is running."}

# --- Upload endpoint (POST /upload) ---
@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "pdf" not in request.files:
            return jsonify({"error": "No PDF uploaded"}), 400

        pdf_file = request.files["pdf"]

        # Extract PDF text
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        if not text.strip():
            return jsonify({"error": "PDF text could not be extracted"}), 400

        prompt = f"Generate a detailed quiz based strictly on the following PDF content:\n\n{text}"

        # Call Groq
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        # ✅ Correct way to read Groq output
        quiz_text = response.choices[0].message.content

        return jsonify({"quiz": quiz_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
