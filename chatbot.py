import openai
import json
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document

api_key = 'sk-None-18aYaJ3vJWNUqInUUWxyT3BlbkFJiYUGO3pC6gOGYrdBaWS2'
openai.api_key = api_key

embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vector_store = None

def initialize_data(detailed_summary):
    global vector_store
    summary_description = json.dumps(detailed_summary, indent=2)
    texts = CharacterTextSplitter().split_text(summary_description)

    documents = [Document(page_content=text, metadata={}) for text in texts]

    vector_store = FAISS.from_documents(documents, embeddings)
    print("Vector store initialized with documents")

def get_chatbot_response(user_input, detailed_summary, conversation_history):
    global vector_store
    if vector_store is None:
        print("Initializing vector store with data summary")
        initialize_data(detailed_summary)

    retrieved_docs = vector_store.similarity_search(user_input, k=5)
    qa_chain = load_qa_chain(llm=ChatOpenAI(model_name="gpt-4-turbo", openai_api_key=api_key))
    conversation_history.add_message({"role": "user", "content": user_input})
    print(f"User input: {user_input}")

    try:
        response = qa_chain.invoke({"question": user_input, "input_documents": retrieved_docs})
        print(f"Chatbot response: {response}")
    except Exception as e:
        print(f"Error generating response: {e}")
        response = {"output_text": "There was an error processing your request."}

    conversation_history.add_message({"role": "assistant", "content": response['output_text']})
    return response.get('output_text', 'No response content found.').strip()