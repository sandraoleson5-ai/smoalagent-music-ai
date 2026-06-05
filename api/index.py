from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
from app.agent import create_music_agent

load_dotenv()

app = FastAPI(title="Smoalagent Music AI")

agent = create_music_agent()

@app.get("/")
def root():
    return {"message": "Smoalagent Music AI Ready! 🎵 Use /run or upload endpoints."}

@app.post("/run")
async def run(query: str = Form()):
    result = agent.run(query)
    return {"result": result}

# Add upload endpoints later for audio files
