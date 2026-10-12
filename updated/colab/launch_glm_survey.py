"""Colab exec entrypoint selecting the GLM no-thinking QRScore survey."""
from zipfile import ZipFile

with ZipFile('/content/retrieval-heads-survey.zip') as bundle:
    source = bundle.read('updated/colab/launch_survey.py').decode('utf-8')
exec(compile(source, 'launch_survey.py', 'exec'), {
    'SURVEY_NAME': 'glm4-6v-flash-no-thinking-qrscore-survey',
    '__name__': '__main__',
})
