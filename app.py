import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

from dotenv import load_dotenv
import os
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

def get_vector_store(chunks):
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def answer_question(question):
    import google.generativeai as genai
    
    embeddings = get_embeddings()
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    docs = vector_store.similarity_search(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""Answer the question using the context below.
If the answer is not in the context, say 'Answer not found in the document.'

Context:
{context}

Question: {question}

Answer:"""
    response = model.generate_content(prompt)
    return response.text

st.set_page_config(page_title="RAG Document Chatbot", page_icon="🤖")
st.title("🤖 RAG Document Q&A Chatbot")
st.markdown("Upload a PDF and ask any question from it!")

with st.sidebar:
    st.header("📄 Upload PDF")
    pdf = st.file_uploader("Choose a PDF file", type="pdf")
    if st.button("Process PDF"):
        if pdf:
            with st.spinner("Reading and processing PDF..."):
                raw_text = get_pdf_text(pdf)
                chunks = get_text_chunks(raw_text)
                get_vector_store(chunks)
                st.success("✅ PDF processed! Ask your question.")
        else:
            st.warning("Please upload a PDF first.")

question = st.text_input("💬 Ask a question from your PDF:")
if st.button("Get Answer"):
    if question:
        with st.spinner("Thinking..."):
            answer = answer_question(question)
            st.write("### Answer:")
            st.write(answer)
    else:
        st.warning("Please enter a question.")
