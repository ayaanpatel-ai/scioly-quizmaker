# Quiz Generator Backend (Groq + FastAPI)

This is a fully working backend that:
- Accepts a PDF upload
- Extracts text
- Generates quiz questions using **Groq LLaMA 3**
- Returns the quiz to the frontend (Wix-compatible)

## 🌐 Deploy on Render
1. Create new Web Service
2. Add environment variable:
   - `GROQ_API_KEY=your_key_here`
3. Build command not needed
4. Start command:
```
uvicorn app:app --host 0.0.0.0 --port 10000
```

## 📦 Install locally
```
pip install -r requirements.txt
uvicorn app:app --reload
```

## 📝 Endpoint
POST `/generate-quiz`
Upload field: `pdf`
