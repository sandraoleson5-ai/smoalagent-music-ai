import os
from smolagents import CodeAgent, InferenceClientModel
from .tools.audio_tools import *
from .tools.lyrics_tools import *


def create_music_agent():
    model = InferenceClientModel(
        model="Qwen/Qwen2.5-Coder-32B-Instruct", 
        token=os.getenv("HF_TOKEN")
    )
    return CodeAgent(
        tools=[analyze_audio, vocal_split_hf, generate_lyrics, generate_beat],
        model=model,
        add_base_tools=True,
        verbosity_level=2
    )
