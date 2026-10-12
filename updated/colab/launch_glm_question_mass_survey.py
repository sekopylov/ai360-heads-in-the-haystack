"""Colab exec entrypoint for the separate paired GLM question/needle survey."""
from zipfile import ZipFile

with ZipFile('/content/retrieval-heads-survey.zip') as bundle:
    source = bundle.read('updated/colab/launch_survey.py').decode('utf-8')
exec(compile(source, 'launch_survey.py', 'exec'), {
    'SURVEY_NAME': 'glm4-6v-flash-no-thinking-question-mass-survey',
    '__name__': '__main__',
})
