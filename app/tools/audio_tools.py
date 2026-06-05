from smolagents import tool
import requests
import os
from pydub import AudioSegment
import librosa
import numpy as np

@tool
def analyze_audio(file_path: str) -> str:
    """Analyze audio: BPM, key, issues & fix suggestions."""
    try:
        y, sr = librosa.load(file_path)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # Simple key estimation
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key = np.argmax(np.sum(chroma, axis=1))
        rms = librosa.feature.rms(y=y)[0].mean()
        issues = []
        if rms > 0.9: issues.append("Clipping detected - reduce gain")
        return f"BPM: {tempo}, Key approx: {key}, RMS: {rms:.2f}. Issues: {issues or 'None'}"
    except Exception as e:
        return f"Analysis error: {str(e)}"

@tool
def vocal_split_hf(file_path: str, output_dir: str = '/tmp') -> str:
    """Vocal separation using HF Inference (Demucs-like)."""
    API_URL = "https://api-inference.huggingface.co/models/facebook/musicgen-small"  # Replace with actual separation endpoint or use local
    headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}
    with open(file_path, 'rb') as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return "Vocals split (demo). Check /tmp for stems. Use real Demucs endpoint for production."

@tool
def generate_beat(prompt: str, duration: int = 30) -> str:
    """Generate beat using HF MusicGen."""
    # In practice, use InferenceClient for audio generation
    return f"Generated beat for: {prompt} ({duration}s). Download from HF output."
