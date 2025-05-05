FROM python:3.11-slim

WORKDIR /usr/src/app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

CMD ["python", "-m", "app.run"]
