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

    resp_text = "{0} *({1:.2f} seg.)*.".format(response["answer"], end_model_exec - start_model_exec)
    for word in resp_text.split(" "):
        yield word + " "
        time.sleep(0.05)


def response_generator_bot(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.05)
        
        
def create_new_chat():
    st.session_state["chats_count"] += 1
    st.session_state["history_chats"] = st.session_state["history_chats"] + ["Chat " + str(st.session_state["chats_count"])]
    st.session_state["current_chat_index"] = st.session_state["chats_count"]-1


def main():
    try:
        st.set_page_config(page_title="Semaforín v1.0", page_icon="🤖", layout="wide")
        
        # CSS customization for the app
        st.markdown("""
        <style>
            .st-emotion-cache-1c7y2kd {
                flex-direction: row-reverse;
                text-align: right;
            }
            .rounded-image {
                border-radius: 15px;
            }
            h1 {
                font-size: 2.5rem;
                text-align: center;
            }
            h2 {
                font-size: 1.25rem; 
                text-align: center;
            }
            .center-text {
                text-align: center;
            }
        </style>""", unsafe_allow_html=True)

        warnings.filterwarnings("ignore", category=FutureWarning)

        db_embeddings, model = create_app()
        load_prompts()

        # Initialization
        if "current_chat_index" not in st.session_state:
            st.session_state["current_chat_index"] = 0
            st.session_state["chats_count"] = 1
            st.session_state["history_chats"] = ["Chat 1"]

        session = "session_id" + str(st.session_state["current_chat_index"])
        messages = "messages" + str(st.session_state["current_chat_index"])
        if session not in st.session_state:
            st.session_state[session] = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            st.session_state[messages] = []  

        # Sidebar
        with st.sidebar:
            semaforin_image = './semaforin_image.jpg'  
            st.image(semaforin_image, use_column_width=True)
            st.title("Semaforín v1.0")
            st.subheader("Bienvenida/o a Semaforín v1.0! El primer colega-bot que responde sobre las leyes del Código de Tráfico y Seguridad Vial de España.")
            current_chat = st.radio(
                label="Lista de chats",
                options=st.session_state["history_chats"],
                key="chat_radiobutton"
            )
            if current_chat:
                st.session_state["current_chat_index"] = st.session_state["history_chats"].index(current_chat)

            create_chat_button = st.button("Nuevo chat", use_container_width=True, key="create_chat_button")
            if create_chat_button:
                create_new_chat()
                st.rerun()

        avatar_assistant = 'https://raw.githubusercontent.com/hmdibella/tfm_app/main/perfil_bot.jpg'
        avatar_user = 'https://raw.githubusercontent.com/hmdibella/tfm_app/main/perfil_humano.jpg'

        # Welcoming message
        st.markdown("<h1>Hola! 👋 Soy Semaforín 🤖</h1>", unsafe_allow_html=True)
        st.markdown("<h2>Tu colega-bot que responde preguntas ❓ sobre las leyes de Tráfico y Seguridad Vial 🚗 en España. Hazme las preguntas y yo trataré de responderlas 💪.</h2>", unsafe_allow_html=True)

        # Display chat messages from history on app rerun
        current_state = st.session_state["messages"+str(st.session_state["current_chat_index"])]
        for message in current_state:
            avatar_img = avatar_assistant if message["role"] == "Semaforín" else avatar_user
            with st.chat_message(message["role"], avatar=avatar_img):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Haz la pregunta aquí"):
            logging.info(f"User input: {prompt}")
            # Add user message to chat history
            current_state.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user", avatar=avatar_user):
                st.markdown(prompt)

            session = 'session_id'+str(st.session_state["current_chat_index"])
            with st.spinner("Procesando, por favor espere..."):

                # Display assistant response in chat message container
                with st.chat_message("Semaforín", avatar=avatar_assistant):
                    response = st.write_stream(response_generator(prompt, st.session_state[session], db_embeddings, model))

                # Add assistant response to chat history
                current_state.append({"role": "Semaforín", "content": response})

            logging.info(f"Application response: {response}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()