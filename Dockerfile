FROM python:3.8
ADD . /code
WORKDIR /code
RUN pip install -r requirements.txt
RUN [ "python3", "-c", "import nltk; nltk.download('stopwords')" ]
CMD python app.py