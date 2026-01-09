import streamlit as st
import os
import gdown
import shutil
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import gc

# --- CONFIGURATION ---
BATCH_SIZE = 50  # Process 50 pages at a time to prevent RAM crashes
VECTOR_DB_PATH = "faiss_index"
DOCS_FOLDER = "downloaded_docs"

st.set_page_config(page_title="Agentic RAG (Zero Cost)", layout="wide")

# --- SENIOR ARCHITECT: MEMORY MANAGEMENT ---
# We use caching to prevent reloading the model on every user interaction
@st.cache_resource
def load_embedding_model():
    # Uses a small, efficient model that runs on Free Tier CPUs
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    # Gemini 1.5 Flash is free and has a large context window
    api_key = st.secrets["GOOGLE_API_KEY"]
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.2)

# --- UTILITY: GOOGLE DRIVE DOWNLOADER ---
def download_folder_from_drive(url):
    """
    Downloads a Google Drive folder using gdown.
    Handles the 'Folder ID' extraction logic.
    """
    if os.path.exists(DOCS_FOLDER):
        shutil.rmtree(DOCS_FOLDER)
    os.makedirs(DOCS_FOLDER)
    
    try:
        # Extract ID from URL
        if "folders/" in url:
            folder_id = url.split("folders/")[1].split("?")[0]
        else:
            folder_id = url # Assume user pasted ID directly
            
        st.info(f"Downloading documents from Drive ID: {folder_id}...")
        
        # gdown download (Folder)
        gdown.download_folder(id=folder_id, output=DOCS_FOLDER, quiet=False, use_cookies=False)
        st.success("Download Complete!")
        return True
    except Exception as e:
        st.error(f"Error downloading: {e}. Make sure the link is 'Anyone with the link' can view.")
        return False

# --- CORE LOGIC: BATCH PROCESSOR ---
def process_documents_in_batches(doc_folder):
    """
    Reads PDFs in small chunks to avoid RAM overflow (The 1GB Constraint).
    """
    embeddings = load_embedding_model()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    files = [f for f in os.listdir(doc_folder) if f.endswith(('.pdf', '.txt'))]
    
    if not files:
        st.warning("No PDF or TXT files found in the folder.")
        return None

    vector_store = None
    total_files = len(files)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, file_name in enumerate(files):
        file_path = os.path.join(doc_folder, file_name)
        status_text.text(f"Processing file {i+1}/{total_files}: {file_name}")
        
        try:
            # 1. Load Document
            if file_name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                # Lazy load pages one by one (generator)
                pages = loader.lazy_load()
            else:
                loader = TextLoader(file_path)
                pages = loader.load()

            # 2. Process in Batches (e.g., 50 pages at a time)
            current_batch = []
            for page in pages:
                current_batch.append(page)
                
                if len(current_batch) >= BATCH_SIZE:
                    # Split and Embed
                    splits = text_splitter.split_documents(current_batch)
                    if vector_store is None:
                        vector_store = FAISS.from_documents(splits, embeddings)
                    else:
                        vector_store.add_documents(splits)
                    
                    # Clear RAM
                    current_batch = []
                    del splits
                    gc.collect() # Force Garbage Collection

            # Process remaining pages in the file
            if current_batch:
                splits = text_splitter.split_documents(current_batch)
                if vector_store is None:
                    vector_store = FAISS.from_documents(splits, embeddings)
                else:
                    vector_store.add_documents(splits)
                del splits
                gc.collect()

        except Exception as e:
            st.error(f"Skipped {file_name} due to error: {e}")
        
        progress_bar.progress((i + 1) / total_files)

    return vector_store

# --- UI LAYOUT ---
st.title("📂 0-Cost Agentic Doc Q&A")
st.markdown("### Powered by Gemini & Streamlit (Strictly Free Tier)")

with st.sidebar:
    st.header("1. Upload Data")
    drive_link = st.text_input("Paste Google Drive Folder Link:")
    
    if st.button("Load Documents"):
        if drive_link:
            if download_folder_from_drive(drive_link):
                with st.spinner("Indexing large files (Batch Mode)..."):
                    vector_store = process_documents_in_batches(DOCS_FOLDER)
                    if vector_store:
                        # Save index locally to avoid re-processing on every reload
                        vector_store.save_local(VECTOR_DB_PATH)
                        st.session_state["vector_db_ready"] = True
                        st.success("Indexing Done! Ready to chat.")
        else:
            st.warning("Please paste a link first.")

    st.markdown("---")
    st.markdown("**System Status:**")
    if st.session_state.get("vector_db_ready"):
        st.write("✅ Knowledge Base Loaded")
    else:
        st.write("❌ Waiting for Data")

# --- CHAT INTERFACE ---
if st.session_state.get("vector_db_ready"):
    # Load the persisted vector store
    embeddings = load_embedding_model()
    new_db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Setup Retriever
    retriever = new_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    # Setup Custom Prompt
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If the answer is not in the context, strictly say "I don't know" and do not make up an answer.
    ALWAYS cite the Source Document and Page Number.

    Context:
    {context}

    Question: {question}
    Answer:"""
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    # Chat UI
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = qa_chain.invoke({"query": prompt})
                answer = response["result"]
                sources = response["source_documents"]
                
                # Format sources
                source_text = "\n\n**Sources:**\n"
                unique_sources = set()
                for doc in sources:
                    src = doc.metadata.get('source', 'Unknown').split('/')[-1]
                    page = doc.metadata.get('page', 'Unknown')
                    unique_sources.add(f"- {src} (Page {page})")
                
                final_response = f"{answer}{source_text}" + "\n".join(unique_sources)
                
                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})

else:
    st.info("👈 Please paste your Drive Link in the sidebar to start.")
