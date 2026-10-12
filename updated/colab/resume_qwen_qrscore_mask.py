"""Continue the original paired mask run without overwriting saved results."""
from zipfile import ZipFile

with ZipFile('/content/retrieval-heads-mask.zip') as bundle:
    source = bundle.read('updated/colab/launch_qwen_qrscore_mask.py').decode()
exec(compile(source, 'launch_qwen_qrscore_mask.py', 'exec'),
     {'RESUME': True, '__name__': '__main__'})
