from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
import os
from joblib import load
from supabase import create_client
import openai
from datetime import datetime
from langchain_community.llms import OpenAI
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import sqlite3
import torch
from gensim.models import Word2Vec
import numpy as np  
#from ctransformers import AutoModelForCausalLM
import os

from transformers import BitsAndBytesConfig,AutoTokenizer, GenerationConfig, pipeline, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import SimpleSequentialChain, RetrievalQA, ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain import HuggingFacePipeline
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "secret"
conn = sqlite3.connect('Agrobot.db')
cursor = conn.cursor()

# Create 'Users' table
# Create 'Users' table
cursor.execute('''CREATE TABLE IF NOT EXISTS Users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name VARCHAR,
                    email_id TEXT
                )''')

# Create 'Response' table
create_response_table = """CREATE TABLE IF NOT EXISTS Response (
                            email_id TEXT,
                            query TEXT,
                            response TEXT
                        );"""
cursor.execute(create_response_table)

# Commit changes and close the connection
conn.commit()
conn.close()

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/chat')
def chat():
    try:
        if 'email' not in session:
            return redirect('/login')
        name = request.args.get('name')
        print(name)
        return render_template('chat.html', name=name)
    except Exception as e:
        flash(str(e),'error')
        return redirect('/login')

@app.route('/assist', methods=['POST'])
def assist():
    user_text = request.json.get('userText')
    try:
        response = ask(user_text)  # Your existing code to generate a response
        
        # Assuming you have the user_id available in your session
        
        
        return jsonify({'text': response})
    except Exception as e:
        return jsonify({'text': f"Error: {str(e)}"})
    



template = """
Whatever is asked, give a detailed explanation. If coding is asked, give proper, complete code . Make everything detail and clear. If internet search is done, verify multiple resources. Here you have
{history}
Human: {human_input}
Assistant:"""

prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

chatgpt_chain = LLMChain(
    llm=LLama2(temperature=0.3),
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=2),
)
def call_agent(prompt):
    response = chatgpt_chain.predict(human_input=prompt)
    process_response(response, prompt)
    return response

#Web-Search Starter
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import tools
from langchain.agents import load_tools, Tool

search =DuckDuckGoSearchRun()
tools = [
      Tool(
          name='DuckDuckGo Search',
          func=search.run,
          description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
      ), 
  ]


memory = ConversationBufferMemory(memory_key="chat_history")

prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!"
{chat_history}
Question: {input}
{agent_scratchpad}"""
prompt = ZeroShotAgent.create_prompt(
   tools,
   prefix=prefix,
   suffix=suffix,
   input_variables=["input", "chat_history", "agent_scratchpad"],
)
llm_chain = LLMChain(llm=OpenAI(temperature=0.3), prompt=prompt)
wagent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
wagent_chain = AgentExecutor.from_agent_and_tools(
   agent=wagent, tools=tools, verbose=True, memory=memory
)

def web_search(user_question):
   output = wagent_chain.run(user_question)
   process_response(output, user_question)
   return output

#LLM Starter
# Initialize global variables
conversation_retrieval_chain = None
chat_history = []
llm = None
llm_embeddings = None
db = None

# Function to initialize the language model and its embeddings
def init_llm():
    global llm, llm_embeddings, llm_tokenizer, pipeline
    
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization_config
    )

    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
    generation_config.max_new_tokens = 128000
    generation_config.temperature = 0.3
    generation_config.top_p = 0.95
    generation_config.do_sample = True
    generation_config.repetition_penalty = 1.15

    pipelin = pipeline(
        "text-generation",
        model=model,
        tokenizer=llm_tokenizer,
        return_full_text=True,
        generation_config=generation_config,
    )
    
    llm = HuggingFacePipeline(pipeline=pipelin)
    llm_embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-large",model_kwargs={"device": "cuda"})

def init_document_db(document_path):
    global conversation_retrieval_chain, llm, llm_embeddings, db

    loader = PyPDFLoader(document_path)
    documents = loader.load()
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts_chunks = text_splitter.split_documents(documents)
  
    # Create a vector store from the website content chunks
    db = Chroma.from_documents(texts_chunks, llm_embeddings, persist_directory="db")
    custom_template = """You are an AI Assistant. Given the
    following conversation and a follow up question, rephrase the follow up question
    to be a standalone question. At the end of standalone question add this
    'Answer the question in English language.' If you do not know the answer reply with 'I am sorry, I dont have enough information'.
    Also if the question is like greeting or feedback, reply appropriately.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:
    """

    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        memory=memory,
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
    )
    # Create a conversational retrieval chain from the language model and the retriever
    conversation_retrieval_chain = qa_chain


# Function to process a PDF document
from langchain.docstore.document import Document
def process_response(response,prompt):
    metadata = {"source": "conversation"}
    global conversation_retrieval_chain, llm, llm_embeddings, db

    #prompt_response_pair = {
    #    "prompt": prompt,
    #    "response": response
    #}

    concatenated_text = f"{prompt}\n{response}"

    # Split the concatenated text into chunks (using a fixed chunk size for this example)
    documents = [Document(page_content=concatenated_text, metadata=metadata)]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts_chunks = text_splitter.split_documents(documents)

    # Assuming 'db' is already initialized and represents the vector database
    # Create a vector store from the text chunks
    db = Chroma.from_documents(texts_chunks, llm_embeddings, persist_directory="db")

    global mail
    conn = sqlite3.connect('Aztrabot.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO Response (email_id, query, response) VALUES (?, ?, ?)", (mail,prompt,response))
    conn.commit()
    conn.close()
    
# Function to process a user prompt
def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history
    # Generate a response to the user's prompt
    result = conversation_retrieval_chain({"question": prompt, "chat_history": chat_history})['answer']
    # Update the chat history
    chat_history.append((prompt, result))
    process_response(result, prompt)
    # Return the model's response
    return result

# Initialize the language model
init_llm()
init_document_db("init_doc.pdf")

def llm(prompt):
  response = process_prompt(prompt)
  return response

#Function to classify whether intents are technical or not
tech_nontech_model_path = "./models/tech_nontech/distilbert_classifier"
# Load the saved model
tech_nontech_model = DistilBertForSequenceClassification.from_pretrained(tech_nontech_model_path)
tech_nontech_tokenizer = DistilBertTokenizer.from_pretrained(tech_nontech_model_path)
tech_nontech_encoder = load("./models/tech_nontech/label_encoder.pkl")

def intent_class(prompt):
    global tech_nontech_model, tech_nontech_tokenizer, tech_nontech_encoder
    inputs = tech_nontech_tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)

    # Perform inference
    with torch.no_grad():
        outputs = tech_nontech_model(**inputs)

    # Get predicted label
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Decode the predicted label using the loaded label_encoder
    decoded_label = tech_nontech_encoder.inverse_transform([predicted_class])[0]

    return decoded_label

#Function to classify whether prompt is a question, positive feedback or negative feedback
pos_neg_ques_model_path = "./models/pos_neg_ques/distilbert_classifier"
# Load the saved model
pos_neg_ques_model = DistilBertForSequenceClassification.from_pretrained(pos_neg_ques_model_path)
pos_neg_ques_tokenizer = DistilBertTokenizer.from_pretrained(pos_neg_ques_model_path)
pos_neg_ques_encoder = load("./models/pos_neg_ques/label_encoder.pkl")
def classify_text(prompt):
    global pos_neg_ques_model, pos_neg_ques_tokenizer, pos_neg_ques_encoder
    # Tokenize input text
    inputs = pos_neg_ques_tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)

    # Perform inference
    with torch.no_grad():
        outputs = pos_neg_ques_model(**inputs)

    # Get predicted label
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Decode the predicted label using the loaded label_encoder
    decoded_label = pos_neg_ques_encoder.inverse_transform([predicted_class])[0]

    return decoded_label


#To select to which LLM should the code be given either GPT, or Mistral, or GPT+WebSearch
def model(c, prompt):
    if c=="coding":
        print("GPT")
        return(call_agent(prompt))
    elif c=="llm":
        print("LLM")
        return(llm(prompt))
    else:
        print("WebSearch")
        return(web_search(prompt))
#<<<<<<<<<<<<<<<<<<<<<UTILS END>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#<<<<<<<<<<<<<<<<<<<<<<FLOW>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
pos = [ "llm", "coding", "duckduckgo"]
previous = ""
conversations = []

def ask(prompt):
  global pos
  intent = intent_class(prompt)
  if intent=="technical":
    global previous
    pr = classify_text(prompt)
    print(pr)
    if pr=="ques":
      previous=""
      previous+=prompt
      ans = model(pos[0], prompt)
      conversations.append({prompt:ans})
      return(ans)
    elif pr=="neg":
      previous = prompt = prompt + previous
      pos.append(pos.pop(0))
      print(pos)
      ans = model(pos[0], prompt)
      conversations.append({prompt:ans})
      return(ans)
    else:
      previous=""
      ans = model(pos[0], prompt)
      conversations.append({prompt:ans})
      return(ans)
  else:
    return "Sorry for the inconveniences. Cannot answer non-technical questions or parse your feedback"


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)