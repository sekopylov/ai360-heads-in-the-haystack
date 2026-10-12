"""Colab exec entrypoint selecting the Ministral survey, not the Qwen default."""
from zipfile import ZipFile

with ZipFile('/content/retrieval-heads-survey.zip') as bundle:
    source = bundle.read('updated/colab/launch_survey.py').decode('utf-8')
exec(compile(source, 'launch_survey.py', 'exec'), {
    'SURVEY_NAME': 'ministral3-8b-no-thinking-qrscore-survey',
    '__name__': '__main__',
})
