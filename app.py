# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from groq import Groq
import os
import traceback

app = FastAPI(title="Quiz Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock to your Wix domain later if you want
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.get("/")
async def root():
    return {"status": "ok", "message": "Quiz backend is running."}

@app.post("/upload")
async def upload(pdf: UploadFile = File(...)):
    """
    Accepts multipart/form-data with field name 'pdf'.
    Returns JSON: { "quiz": "<generated text>" } or { "error": "..." }
    """
    try:
        # Validate file type quickly
        if not pdf.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Uploaded file is not a PDF (filename check).")

        # Read PDF file contents using PyPDF2
        try:
            reader = PdfReader(pdf.file)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to open PDF: {e}")

        text_chunks = []
        for page in reader.pages:
            ptxt = page.extract_text()
            if ptxt:
                text_chunks.append(ptxt)
        full_text = "\n".join(text_chunks).strip()

        if not full_text:
            raise HTTPException(status_code=400, detail="Could not extract any text from the PDF.")

        # Build prompt
        prompt = f"""Generate a clear, well-formatted quiz based strictly on the text below.
Include a mix of multiple-choice, short answer, and true/false questions. Return plain text only.

Text:
{full_text}
"""

        # Call Groq — use a currently supported model
        # If you get a deprecation error, swap model to another supported one (e.g. "mixtral-8x7b-32768")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        # Robust extraction of generated content:
        # Groq's response.choices[0].message may be an object with .content or a dict-like mapping.
        try:
            first_choice = response.choices[0]
        except Exception:
            raise HTTPException(status_code=500, detail="Model returned unexpected response shape (no choices).")

        message = getattr(first_choice, "message", None) or first_choice.get("message", None)

        # message may be an object with attribute 'content' OR a mapping with key 'content'
        quiz_text = None
        if message is None:
            # Try other fallbacks
            # Some SDKs return: first_choice["message"]["content"] or first_choice["text"]
            try:
                quiz_text = first_choice["message"]["content"]
            except Exception:
                quiz_text = first_choice.get("text") if isinstance(first_choice, dict) else None
        else:
            # attribute-style
            quiz_text = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else None)

        if not quiz_text:
            # Last-ditch: try stringifying the whole response
            quiz_text = str(response)

        return {"quiz": quiz_text}

    except HTTPException as he:
        # FastAPI will convert to proper JSON + status code
        raise he
    except Exception as e:
        tb = traceback.format_exc()
        # Log tb to Render logs; return sanitized error to client
        print(tb)
        # Return 500 with error details for easier debugging (remove stack in production)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

