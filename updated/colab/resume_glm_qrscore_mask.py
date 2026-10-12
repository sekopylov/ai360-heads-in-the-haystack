"""Resume GLM top-only QRScore masking with frozen ranking and a two-hour limit."""
from zipfile import ZipFile

with ZipFile('/content/retrieval-heads-mask.zip') as bundle:
    source = bundle.read('updated/colab/launch_qwen_qrscore_mask.py').decode()
exec(compile(source, 'launch_qwen_qrscore_mask.py', 'exec'), {
    'MASK_NAME': 'glm4-6v-flash-no-thinking-paul-graham-qrscore-top-mask',
    'RESUME': True,
    'MAX_RUNTIME_SECONDS': 7200,
    '__name__': '__main__',
})
