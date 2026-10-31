FROM animcogn/face_recognition:cpu

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn face_engine:app --host 0.0.0.0 --port ${PORT:-8000}"]
