FROM python:3.8

ENV HOME=/home/app

COPY . $HOME/SSFER/

WORKDIR $HOME/SSFER

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
