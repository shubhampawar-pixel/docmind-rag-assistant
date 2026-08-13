"""
DocMind — Hugging Face Spaces Entry Point

On HF Spaces, this file is the Streamlit app entry.
Locally, you run: streamlit run ui/streamlit_app.py
"""

import sys
import os
import subprocess
from pathlib import Path

# Ensure project root is on the Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# On HF Spaces, default to Gemini since Ollama is not available
if os.getenv("SPACE_ID"):
    os.environ.setdefault("LLM_PROVIDER", "gemini")

# Launch the Streamlit app
ui_path = os.path.join(project_root, "ui", "streamlit_app.py")

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", ui_path,
         "--server.headless", "true"],
        cwd=str(project_root),
    )
