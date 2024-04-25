import os

import base64
import gc
import random
import tempfile
import time
import uuid
import pandas as pd
import json

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader, set_global_service_context
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.embeddings import HuggingFaceEmbedding

import streamlit as st

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect() 

def display_csv(file):
    # Reading CSV file into a DataFrame
    df = pd.read_csv(file)
    st.markdown("### CSV Preview")
    st.write(df)  # Displaying DataFrame

def json_2_rules(datadict):
    all_judgements=''

    for key in datadict.keys():
        judgement = ''
        conditions = datadict[key]["conditions"]
        if "type" in conditions.keys():
            judgement = conditions["type"].join([
                f"{cond['name']} {(cond['operator'])} {cond['value']}"
                for cond in conditions["sub_condition"]])
        else:
            judgement = [
                f"{cond['name']} {(cond['operator'])} {cond['value']}"
                for cond in conditions["sub_condition"]]
        determines = datadict[key]["determine"]
        determine = [
                f"{cond['name']} {(cond['operator'])} {cond['value']}"
                for cond in conditions["determine"]]
        all_judgements = "If " + judgement + ",then " + determine + ";\n"
    
    return all_judgements



def display_json(data):
    # Reading JSON file
    

    all_judgements_str = json_2_rules(data)

    st.markdown("### Conditions")
    st.write(all_judgements_str)  # Displaying conditions
    

with st.sidebar:
    st.header(f"Add your documents!")

    uploaded_file_csv = st.file_uploader("Choose your `.csv` file", type="csv")
    uploaded_file_json = st.file_uploader("Choose your `.json` file", type="json")

    # Handle JSON file upload
    if uploaded_file_json:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                json_file_path = os.path.join(temp_dir, uploaded_file_json.name)
                with open(json_file_path, "wb") as json_file:
                    json_file.write(uploaded_file_json.getvalue())
                    json_data = json.load(json_file)
                    rules = json_2_rules(json_data)

                

                st.success("rules Ready!")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop() 
    
    # Handle CSV file upload
    if uploaded_file_csv:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file_csv.name)

                csv_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                with open(file_path, "wb") as f:
                    f.write(uploaded_file_csv.getvalue())

                if file_key not in st.session_state.get('file_cache', {}):
                    if os.path.exists(temp_dir):
                            loader = SimpleDirectoryReader(
                                input_dir = temp_dir,
                                required_exts=[".csv"],
                                recursive=True
                            )
                    else:    
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()

                    docs = loader.load_data()

                    # setup llm & embedding model
                    llm=Ollama(model="llama3", request_timeout=120.0)
                    embed_model = HuggingFaceEmbedding()
                    Settings.embed_model = embed_model
                    index = VectorStoreIndex.from_documents(docs, show_progress=True)
                    Settings.llm = llm
                    query_engine = PandasQueryEngine(df=docs, verbose=True, synthesize_response=True)
                
                else:
                    query_engine = st.session_state.file_cache[file_key]

                st.success("csv file Ready!")
                display_csv(file_path)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        st.write("Here are the rules:")
        st.write(rules)
        
        # Simulate stream of response with milliseconds delay
        streaming_response = query_engine.query(prompt)
        
        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        # full_response = query_engine.query(prompt)

        message_placeholder.markdown(full_response)
        # st.session_state.context = ctx

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

