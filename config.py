PROJECT_ID = "tfm-dev-433217" # ID del proyecto de Google Cloud
GEMINI_API_KEY = "GEMINI_API_KEY" # Secreto con el API Key de Gemini
SECRET_MANAGER_API_KEY_PATH = "projects/{}/secrets/{}/versions/latest".format(PROJECT_ID, GEMINI_API_KEY) # Ruta del secreto con el API Key de Gemini

BUCKET_FAISS_NAME = "faiss-trafico" # Bucket con embeddings
FAISS_FILE_NAMES = ["index.faiss", "index.pkl"] # Archivos a descargar
LOCAL_FAISS_FOLDER = "./faiss_index_google" # Carpeta local donde se guardan los archivos

BUCKET_PROMPTS_NAME = "dynamic-variables" # Bucket con prompts y temperature
PROMPTS_FILE_NAME = "prompts.json"

MODEL_LLM_GOOGLE = "gemini-1.5-pro"

NUMEXPR_MAX_THREADS = "7"

MODEL_TEMPERATURE = 0.4

## SINGLETON FOR STORAGE CLIENT ##
from google.cloud import storage
import os
storage_client = None

def getStorageClient():
    global storage_client
    if(storage_client is None):
        storage_client = storage.Client()
    return storage_client

def is_running_on_gcp():
    return 'GOOGLE_CLOUD_PROJECT' in os.environ or 'GAE_ENV' in os.environ