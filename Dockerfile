# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose the port Hugging Face uses
ENV PORT 7860

# Run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
