FROM python:3.8

RUN pip install -U pip
RUN pip install pipenv

COPY templates/ templates/
COPY artifacts/ artifacts/
COPY ["utils.py", "./"]

COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy
RUN pip install --upgrade --user scipy

COPY ["deploy_model.py", "./"]
ENTRYPOINT [ "python", "deploy_model.py" ]