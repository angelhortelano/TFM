import streamlit as st
import warnings
import os
import random
import string
import time
import logging

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import google.cloud.logging

from google.cloud import secretmanager

import config
import json

CONTEXTUALIZE_Q_SYSTEM_PROMPT = None
PROMPT_TEMPLATE_GOOGLE = None

@st.cache_resource
def create_app():
    logging.warning("Creating resources...")
    if config.is_running_on_gcp():
        logging.warning("Running on Google Cloud...")
        client = google.cloud.logging.Client()
        client.setup_logging()   

        # Load API Key from Secret Manager
        secret_client = secretmanager.SecretManagerServiceClient()
        GOOGLE_API_KEY = secret_client.access_secret_version(name=config.SECRET_MANAGER_API_KEY_PATH).payload.data.decode("utf-8")
        os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

        # Load files from Google Cloud Storage
        # Create local folder for FAISS
        if not os.path.exists(config.LOCAL_FAISS_FOLDER):
            os.makedirs(config.LOCAL_FAISS_FOLDER)

        # Load FAISS files
        bucket_faiss = config.getStorageClient().bucket(config.BUCKET_FAISS_NAME)
        for file_name in config.FAISS_FILE_NAMES:
            blob = bucket_faiss.blob(file_name)
            local_file_path = os.path.join(config.LOCAL_FAISS_FOLDER, file_name)
            blob.download_to_filename(local_file_path)
    else:
        logging.warning("Running locally...")
        with(open("google-api-key.txt", "r")) as f:
            os.environ['GOOGLE_API_KEY'] = f.read()

    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

    db_embeddings = FAISS.load_local(config.LOCAL_FAISS_FOLDER, embeddings, allow_dangerous_deserialization=True)

    model = ChatGoogleGenerativeAI(model = config.MODEL_LLM_GOOGLE, temperature = config.MODEL_TEMPERATURE)

    logging.warning("Resources created.")

    return db_embeddings, model        

def load_prompts():
    if config.is_running_on_gcp():
        global CONTEXTUALIZE_Q_SYSTEM_PROMPT, PROMPT_TEMPLATE_GOOGLE

        bucket_variables = config.getStorageClient().bucket(config.BUCKET_PROMPTS_NAME)
        blob = bucket_variables.blob(config.PROMPTS_FILE_NAME)
        blob.download_to_filename(config.PROMPTS_FILE_NAME)

    with open(config.PROMPTS_FILE_NAME, "r") as f:
        variables = json.load(f)

    CONTEXTUALIZE_Q_SYSTEM_PROMPT = variables["CONTEXTUALIZE_Q_SYSTEM_PROMPT"]
    PROMPT_TEMPLATE_GOOGLE = variables["PROMPT_TEMPLATE_GOOGLE"]

def get_conversational_chain(retriever, session_id, model):    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROMPT_TEMPLATE_GOOGLE),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
 
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    msgs = StreamlitChatMessageHistory(key=session_id)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    return conversational_rag_chain


def user_input(user_question, session_id, db_embeddings, model):
    retriever = db_embeddings.as_retriever(search_kwargs={"k": config.K_VALUES})
    
    chain = get_conversational_chain(retriever, session_id, model)
    response = chain.invoke(
        {"input": user_question},
        config={
            "configurable": {"session_id": session_id}
        },
    )
    return response
    
    
def response_generator(prompt, session_id, db_embeddings, model):
    start_model_exec = time.time()
    response = user_input(prompt, session_id, db_embeddings, model)
    end_model_exec = time.time()

    resp_text = "{0} ({1:.2f} seg.).".format(response["answer"], end_model_exec - start_model_exec)
    for word in resp_text.split(" "):
        yield word + " "
        time.sleep(0.05)


def main():
    try:
        st.set_page_config(page_title="Chat legal v1.0", page_icon="🇪🇸")
        
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        db_embeddings, model = create_app()
        load_prompts()

        # Initialization
        if 'session_id' not in st.session_state:
            st.session_state['session_id'] = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            
        st.header("Chat con el modelo", divider="gray")
    
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Inicializar el contexto de la conversación
        if "context" not in st.session_state:
            st.session_state.context = ""
        
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
        # Accept user input
        if prompt := st.chat_input("Haz una pregunta sobre el Código de Tráfico y Seguridad Vial en España"):
            logging.info(f"User input: {prompt}")
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
                

            with st.spinner("Procesando, por favor espere..."):
                
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    #response = st.write_stream(response_generator(prompt, model_chat, st.session_state.context, chunk_size, k_value))
                    response = st.write_stream(response_generator(prompt, st.session_state['session_id'], db_embeddings, model))

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Agregar la pregunta y la respuesta al contexto
                st.session_state.context += f"Pregunta: {prompt}\n"
                st.session_state.context += f"Respuesta: {response}\n"
            
            logging.info(f"Application response: {response}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        
        
if __name__ == "__main__":
   main()