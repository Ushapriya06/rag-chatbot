import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai
import docx
import os
import time

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def extract_text_from_file(file):
    text = ""
    if file.name.endswith(".pdf"):
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file.name.endswith(".txt"):
        text = file.read().decode("utf-8")
    elif file.name.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(file)
        text = df.to_string()
    return text

def get_all_text(files):
    text = ""
    for file in files:
        text += extract_text_from_file(file)
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
    return len(chunks)

def answer_question(question, language="English"):
    embeddings = get_embeddings()
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    docs = vector_store.similarity_search(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""Answer the question using the context below.
If the answer is not in the context, say 'Answer not found in the document.'
Please answer in {language} language.

Context:
{context}

Question: {question}

Answer:"""
    start_time = time.time()
    response = model.generate_content(prompt)
    end_time = time.time()
    response_time = round(end_time - start_time, 2)
    return response.text, response_time

def summarize_documents(language="English"):
    embeddings = get_embeddings()
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    docs = vector_store.similarity_search("main topics summary", k=10)
    context = "\n\n".join([doc.page_content for doc in docs])
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""Please provide a comprehensive summary of the following document content.
Please respond in {language} language.

{context}

Summary:"""
    response = model.generate_content(prompt)
    return response.text

def export_chat_as_text(chat_history, username):
    content = "DocuMind - Chat History\n"
    content += f"User: {username}\n"
    content += f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    content += "="*50 + "\n\n"
    for i, chat in enumerate(chat_history, 1):
        content += f"Q{i}: {chat['question']}\n"
        content += f"Answer: {chat['answer']}\n"
        if chat.get("time"):
            content += f"Response time: {chat['time']} seconds\n"
        content += "-"*30 + "\n\n"
    return content

# ── Streamlit UI ──────────────────────────────
st.set_page_config(page_title="DocuMind - Intelligent Q&A System", page_icon="🧠")

# ── User Personalization ───────────────────────
if "username" not in st.session_state:
    st.session_state.username = ""

if st.session_state.username == "":
    st.title("🧠 Welcome to DocuMind!")
    st.markdown("### Please enter your name to get started:")
    name_input = st.text_input("Your Name:")
    if st.button("Start"):
        if name_input.strip():
            st.session_state.username = name_input.strip()
            st.rerun()
        else:
            st.warning("Please enter your name!")
    st.stop()

# ── Main App ───────────────────────────────────
st.title("🧠 DocuMind - Intelligent Q&A System")
st.markdown(f"👋 Welcome, **{st.session_state.username}**! Upload documents and chat with your data!")

# ── Chat history init ──────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None

# ── Sidebar ────────────────────────────────────
with st.sidebar:
    st.header(f"👤 {st.session_state.username}")

    st.markdown("---")

    # Language selector
    st.subheader("🌐 Language")
    language = st.selectbox(
        "Answer in:",
        ["English", "Telugu", "Hindi", "Tamil", "Kannada",
         "Malayalam", "Bengali", "Marathi", "Gujarati"]
    )

    st.markdown("---")

    st.header("📄 Upload Documents")
    files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True
    )
    if st.button("Process Documents"):
        if files:
            with st.spinner("Reading and processing documents..."):
                raw_text = get_all_text(files)
                chunks = get_text_chunks(raw_text)
                total_chunks = get_vector_store(chunks)
                total_words = len(raw_text.split())
                total_chars = len(raw_text)
                st.session_state.doc_stats = {
                    "files": len(files),
                    "words": total_words,
                    "characters": total_chars,
                    "chunks": total_chunks
                }
                st.success(f"✅ {len(files)} file(s) processed!")
        else:
            st.warning("Please upload at least one file.")

    st.markdown("---")

    if st.button("📝 Summarize Documents"):
        with st.spinner("Summarizing..."):
            summary = summarize_documents(language)
            st.session_state.chat_history.append({
                "question": "📝 Please summarize the documents",
                "answer": summary,
                "time": "—"
            })
            st.success("Done!")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat cleared!")

    if st.button("🚪 Logout"):
        st.session_state.username = ""
        st.session_state.chat_history = []
        st.session_state.doc_stats = None
        st.rerun()

# ── Document Stats Panel ───────────────────────
if st.session_state.doc_stats:
    st.markdown("### 📊 Document Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 Files", st.session_state.doc_stats["files"])
    col2.metric("📝 Words", f"{st.session_state.doc_stats['words']:,}")
    col3.metric("🔤 Characters", f"{st.session_state.doc_stats['characters']:,}")
    col4.metric("🧩 Chunks", st.session_state.doc_stats["chunks"])
    st.markdown("---")

# ── Export Chat Button ─────────────────────────
if st.session_state.chat_history:
    col1, col2 = st.columns([3, 1])
    with col2:
        chat_text = export_chat_as_text(
            st.session_state.chat_history,
            st.session_state.username
        )
        st.download_button(
            label="📥 Export Chat",
            data=chat_text,
            file_name=f"DocuMind_{st.session_state.username}.txt",
            mime="text/plain"
        )
# ── Display chat history ───────────────────────
st.markdown("### 💬 Chat")
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        st.write(chat["answer"])
        if chat.get("time"):
            st.caption(f"⏱️ Answered in {chat['time']} seconds")

# ── Question input ─────────────────────────────
question = st.chat_input("Ask a question from your documents...")
if question:
    with st.spinner("Thinking..."):
        answer, response_time = answer_question(question, language)
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "time": response_time
        })
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
            st.caption(f"⏱️ Answered in {response_time} seconds")