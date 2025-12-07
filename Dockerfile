# Dockerfile

FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency file and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into image
COPY . .

# Train model inside the image so model.pkl is available
RUN python ml_pipeline.py

# Expose the FastAPI port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
