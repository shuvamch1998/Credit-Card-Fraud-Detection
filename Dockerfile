FROM python:3.11.8-slim

WORKDIR /app


COPY ./app /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8501


CMD ["streamlit", "run", "streamlit_app.py"]
