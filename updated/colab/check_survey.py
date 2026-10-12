"""Short read-only runtime and Drive check before launching the survey."""
from pathlib import Path
import torch

print('GPU:', torch.cuda.get_device_name(0))
print('Drive mounted:', Path('/content/drive/MyDrive').is_dir())
parent = Path('/content/drive/MyDrive/retrieval-heads-colab')
status = parent / 'qwen3-8b-no-thinking-qrscore-survey-status.json'
print('Survey status:', status.read_text() if status.exists() else 'not launched')
