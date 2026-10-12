"""Run Ministral no-thinking Paul Graham top-only QRScore masking."""
from zipfile import ZipFile

with ZipFile('/content/retrieval-heads-mask.zip') as bundle:
    source = bundle.read('updated/colab/launch_qwen_qrscore_mask.py').decode()
exec(compile(source, 'launch_qwen_qrscore_mask.py', 'exec'), {
    'MASK_NAME': 'ministral3-8b-no-thinking-paul-graham-qrscore-top-mask',
    'MAX_RUNTIME_SECONDS': 7200,
    '__name__': '__main__',
})
