# Use the official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy the app files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install the HuggingFace model on image
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings;\
    HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')"

# Expose the port Streamlit will run on
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
