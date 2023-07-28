from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import openai
import streamlit as st
openai.api_key = st.secrets["openai_api_key"]

new_db = Chroma(persist_directory='chroma_db_DT', embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))

def find_match(input):
    result = new_db.similarity_search(input)
    return result[0].page_content+"\n"+result[1].page_content

def query_refiner(conversation, query):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=2,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string