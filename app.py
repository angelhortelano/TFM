import streamlit as st
import warnings
import os
import random
import string
import time
import logging

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import google.cloud.logging
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from typing import List

from google.cloud import secretmanager

import config
import json

CONTEXTUALIZE_Q_SYSTEM_PROMPT = None
PROMPT_TEMPLATE_GOOGLE = None

@st.cache_resource
def create_app():
    logging.getLogger().setLevel(logging.DEBUG)
    logging.warning("Creating resources...")
    if config.is_running_on_gcp():
        logging.warning("Running on Google Cloud...")
        client = google.cloud.logging.Client()
        client.setup_logging()   

        # Load API Key from Secret Manager
        secret_client = secretmanager.SecretManagerServiceClient()
        GOOGLE_API_KEY = secret_client.access_secret_version(name=config.SECRET_MANAGER_API_KEY_PATH).payload.data.decode("utf-8")
        os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
        
        TAVILY_API_KEY = secret_client.access_secret_version(name=config.SECRET_MANAGER_TAVILY_API_KEY_PATH).payload.data.decode("utf-8")
        os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

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
        with(open("tavily-api-key.txt", "r")) as f:
            os.environ['TAVILY_API_KEY'] = f.read()
    
    

    db_embeddings = config.getDbEmbbedings()
    model = config.getGeminiModel()
    config.getGeminiModelNoTemp()

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

def get_conversational_chain(retriever, session_id):    
    model = config.getGeminiModel()
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


def user_input(user_question, session_id, db_embeddings, model, app):
    '''
    retriever = db_embeddings.as_retriever(search_kwargs={"k": config.K_VALUES})
    
    chain = get_conversational_chain(retriever, session_id, model)
    response = chain.invoke(
        {"input": user_question},
        config={
            "configurable": {"session_id": session_id}
        },
    )
    '''
    inputs = {"question": user_question, "session_id": session_id}
    value = app.invoke(inputs)
    return value["generation"]
    
    
def response_generator(prompt, session_id, db_embeddings, model, app):
    start_model_exec = time.time()
    response = user_input(prompt, session_id, db_embeddings, model, app)
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


class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    web_search: str
    session_id: str
    

def web_search(state):
    logging.debug(f"Doing Web search.")
    question = state["question"]
    documents = state.get("documents")

    # Web search
    web_search_tool = TavilySearchResults(k=3)
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    
    return {"documents": documents, "question": question}


def retrieve(state):
    logging.debug(f"Doing Retrieve.")
    question = state["question"]

    # Retrieval
    db_embeddings = config.getDbEmbbedings()
    retriever = db_embeddings.as_retriever(search_kwargs={"k": config.K_VALUES})
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    logging.debug(f"Doing Generate.")
    question = state["question"]
    documents = state["documents"]
    session_id = state["session_id"]

    new_db = config.getDbEmbbedings()
    retriever = new_db.as_retriever(search_kwargs={"k": config.K_VALUES})
    chain = get_conversational_chain(retriever, session_id)
    generation = chain.invoke(
        {"input": question},
        config={
            "configurable": {"session_id": session_id}
        },
    )    
    
    return {"documents": documents, "question": question, "generation": generation}


def route_question(state):
    logging.debug(f"Starting Route Question.")
    question = state["question"]
    logging.debug(f"Question: " + question)
    
    llm = config.getGeminiModelNoTemp()

    prompt = PromptTemplate(
        template="""You are an expert at routing a user question to a vectorstore or web search. Use the vectorstore for questions related to the laws in the 'Traffic and Road Safety Code in Spain' (in Spanish, 'Código de Tráfico y Seguridad Vial en España'). You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and no premable or explanation.\n\nQuestion to route: {question}""",
        input_variables=["question"],
    )

    question_router = prompt | llm | JsonOutputParser()
    source = question_router.invoke({"question": question})
    
    if source["datasource"] == "web_search":
        logging.debug(f"Route Question to Web Search.")
        return "websearch"
    elif source["datasource"] == "vectorstore":
        logging.debug(f"Route Question to RAG.")
        return "vectorstore"


def grade_generation_v_documents_and_question(state):
    logging.debug(f"Checking Hallucinations.")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    logging.debug(f"Generation: " + generation["answer"])
    
    llm = config.getGeminiModelNoTemp()

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.\n\nHere are the facts:\n\n ------- \n\n{documents}\n\n ------- \n\n\nHere is the answer: {generation}""",
        input_variables=["generation", "documents"],
    )

    hallucination_grader = prompt | llm | JsonOutputParser()

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.\n\nHere is the answer:\n\n ------- \n\n{generation}\n\n ------- \n\n\nHere is the question: {question}""",
        input_variables=["generation", "question"],
    )

    answer_grader = prompt | llm | JsonOutputParser()

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        logging.debug(f"Decision: generation is grounded in documents.")
        # Check question-answering
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            logging.debug(f"Decision: generation addresses question.")
            return "useful"
        else:
            logging.debug(f"Decision: generation does NOT addresses question.")
            return "not useful"
    else:
        logging.debug(f"Decision: generation is not grounded in documents, re-try.")
        return "not supported"


def grade_documents(state):
    logging.debug(f"Check document relevance to question.")
    question = state["question"]
    documents = state["documents"]

    llm = config.getGeminiModelNoTemp()

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question, grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals.\nGive a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.\nProvide the binary score as a JSON with a single key 'score' and no premable or explanation.\n\nHere is the retrieved document: \n\n{document}\n\nHere is the user question: {question}\n""",
        input_variables=["question", "document"],
    )

    retrieval_grader = prompt | llm | JsonOutputParser()

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            logging.debug(f"Grade: document relevant: " + d.page_content)
            filtered_docs.append(d)
        # Document not relevant
        else:
            logging.debug(f"Grade: document NOT relevant: " + d.page_content)
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def decide_to_generate(state):
    logging.debug(f"Assess graded documents.")
    web_search = state["web_search"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        logging.debug(f"Decision: all documents are not relevant to question, include web search.")
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        logging.debug(f"Decision: generate.")
        return "generate"


def main():
    try:
        st.set_page_config(page_title="Semaforín v1.0", page_icon="🤖", layout="wide")
        st.markdown("""
<style>
    .st-emotion-cache-1c7y2kd {
        flex-direction: row-reverse;
        text-align: right;
    }
</style>""",
    unsafe_allow_html=True,
)
        
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        db_embeddings, model = create_app()
        load_prompts()
        
        workflow = StateGraph(GraphState)
        workflow.add_node("websearch", web_search)  # web search
        workflow.add_node("retrieve", retrieve)  # retrieve
        #workflow.add_node("grade_documents", grade_documents)  # grade documents
        workflow.add_node("generate", generate)  # generatae
        
        # Build graph
        workflow.add_conditional_edges(
            START,
            route_question,
            {
                "websearch": "websearch",
                "vectorstore": "retrieve",
            },
        )

        #workflow.add_edge("retrieve", "grade_documents")
        workflow.add_edge("retrieve", "generate")
        # workflow.add_conditional_edges(
            # "grade_documents",
            # decide_to_generate,
            # {
                # "websearch": "websearch",
                # "generate": "generate",
            # },
        # )
        workflow.add_edge("websearch", "generate")
        # workflow.add_conditional_edges(
            # "generate",
            # grade_generation_v_documents_and_question,
            # {
                # "not supported": "generate",
                # "useful": END,
                # "not useful": "websearch",
            # },
        # )
        workflow.add_edge("generate", END)
        app = workflow.compile()

        # Initialization
        avatar_assistant = 'https://raw.githubusercontent.com/hmdibella/tfm_app/main/perfil_bot.jpg'
        avatar_user = 'https://raw.githubusercontent.com/hmdibella/tfm_app/main/perfil_humano.jpg'
        if "current_chat_index" not in st.session_state:
            st.session_state["current_chat_index"] = 0
            st.session_state["chats_count"] = 1
            st.session_state["history_chats"] = ["Chat 1"]
        
        session = "session_id" + str(st.session_state["current_chat_index"])
        messages = "messages" + str(st.session_state["current_chat_index"])
        if session not in st.session_state:
            st.session_state[session] = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            st.session_state[messages] = []
            st.session_state[messages].append({"role": "Semaforín", "content": "Hola! 👋 Soy :blue[Semaforín] 🤖, tu colega-bot que responde preguntas ❓ sobre las leyes de :red[Tráfico y Seguridad Vial] 🚗 en España. Hazme las preguntas y yo trataré de responderlas 💪."})

        with st.sidebar:
            col1, col2 = st.columns([40,25])
            with col1:
                st.title(":blue[Semaforín v1.0]")
            with col2:
                st.image(avatar_assistant, width=40)
       
            st.subheader( "Bienvenida/o a :blue[Semaforín v1.0]! El primer colega-bot 🤖 que responde sobre las leyes del Código de Tráfico y Seguridad Vial de :red[España] 🚗." )
            current_chat = st.radio(
                label="Lista de chats",
                options=st.session_state["history_chats"],
                #index=st.session_state["current_chat_index"],
                key="chat_radiobutton"
            )
            if current_chat:
                st.session_state["current_chat_index"] = st.session_state["history_chats"].index(current_chat)
                    
            create_chat_button = st.button("Nuevo chat", use_container_width=True, key="create_chat_button")
            if create_chat_button:
                create_new_chat()
                st.rerun()
        
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
                    response = st.write_stream(response_generator(prompt, st.session_state[session], db_embeddings, model, app))

                # Add assistant response to chat history
                current_state.append({"role": "Semaforín", "content": response})
            
            logging.info(f"Application response: {response}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        
        
if __name__ == "__main__":
   main()