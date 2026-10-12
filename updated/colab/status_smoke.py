"""Read-only remote progress query for colab exec."""
from pathlib import Path
import json
import os

state_path = Path('/content/retrieval-heads-smoke-state.json')
if state_path.exists():
    state = json.loads(state_path.read_text())
    try:
        os.kill(state['pid'], 0)
        alive = True
    except ProcessLookupError:
        alive = False
    print('Worker process present:', alive)
else:
    print('Smoke has not been launched')
status = Path('/content/drive/MyDrive/retrieval-heads-colab/qwen3-smoke-status.json')
if status.exists():
    print(status.read_text())
log = Path('/content/drive/MyDrive/retrieval-heads-colab/qwen3-smoke-bootstrap.log')
if log.exists():
    print('\n'.join(log.read_text(errors='replace').splitlines()[-20:]))
