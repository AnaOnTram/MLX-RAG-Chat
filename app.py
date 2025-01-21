import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*__path__._path.*')

import re
import time
import os
import json
from datetime import datetime
from typing import List, Dict

import mlx.core as mx
import streamlit as st
from mlx_lm.utils import load, generate_step
import argparse
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

title = "MLX Chat + RAG"
ver = "0.7.26"
debug = False

# Material Design Theme CSS
material_design_css = """
<style>
    /* Material Design Colors */
    :root {
        /* Light theme variables */
        --md-primary: #1a73e8;
        --md-primary-light: #5095ee;
        --md-primary-dark: #0052b5;
        --md-surface: #ffffff;
        --md-background: #f8f9fa;
        --md-error: #d93025;
        --md-text-primary: rgba(0, 0, 0, 0.87);
        --md-text-secondary: rgba(0, 0, 0, 0.60);
        --md-border: rgba(0, 0, 0, 0.12);
        --md-input-bg: #ffffff;
        --md-chat-user-bg: #e3f2fd;
        --md-chat-assistant-bg: #f5f5f5;
    }

    /* Dark theme variables */
    [data-theme="dark"] {
        --md-primary: #8ab4f8;
        --md-primary-light: #aecbfa;
        --md-primary-dark: #669df6;
        --md-surface: #2d2d2d;
        --md-background: #1a1a1a;
        --md-error: #f28b82;
        --md-text-primary: rgba(255, 255, 255, 0.87);
        --md-text-secondary: rgba(255, 255, 255, 0.60);
        --md-border: rgba(255, 255, 255, 0.12);
        --md-input-bg: #3d3d3d;
        --md-chat-user-bg: #1e3a5f;
        --md-chat-assistant-bg: #2d2d2d;
    }

    /* Base Styles */
    .stApp {
        background-color: var(--md-background) !important;
        color: var(--md-text-primary) !important;
    }

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: var(--md-surface) !important;
    }

    /* Headers */
    h1, h2, h3, .st-emotion-cache-1egp75f {
        color: var(--md-text-primary) !important;
        font-family: 'Google Sans', sans-serif !important;
    }

    /* Chat Messages */
    .st-chat-message {
        border-radius: 8px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24) !important;
        transition: all 0.3s cubic-bezier(.25,.8,.25,1) !important;
        color: var(--md-text-primary) !important;
    }

    .st-chat-message[data-testid="user-message"] {
        background-color: var(--md-chat-user-bg) !important;
    }

    .st-chat-message[data-testid="assistant-message"] {
        background-color: var(--md-chat-assistant-bg) !important;
    }

    .st-chat-message:hover {
        box-shadow: 0 14px 28px rgba(0,0,0,0.25), 0 10px 10px rgba(0,0,0,0.22) !important;
    }

    /* Message Content */
    .st-chat-message p, 
    .st-chat-message span,
    .st-chat-message div {
        color: var(--md-text-primary) !important;
    }

    /* Code blocks in messages */
    .st-chat-message pre {
        background-color: var(--md-surface) !important;
        border: 1px solid var(--md-border) !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 4px !important;
        background-color: var(--md-primary) !important;
        color: var(--md-text-primary) !important;
        text-transform: uppercase !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        border: none !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.24) !important;
        transition: all 0.3s cubic-bezier(.25,.8,.25,1) !important;
    }

    .stButton > button:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.30) !important;
        background-color: var(--md-primary-light) !important;
    }

    /* File Uploader */
    .st-file-uploader {
        border: 2px dashed var(--md-primary-light) !important;
        border-radius: 8px !important;
        padding: 16px !important;
        background-color: var(--md-surface) !important;
        color: var(--md-text-primary) !important;
    }

    /* File Uploader Drop Zone */
    .st-file-uploader > div:first-child {
        background-color: var(--md-surface) !important;
        color: var(--md-text-primary) !important;
        border: 1px dashed var(--md-border) !important;
    }

    /* Progress Bar */
    .stProgress > div > div {
        background-color: var(--md-primary) !important;
    }

    /* Text Input */
    .stTextInput > div > div > input {
        border-radius: 4px !important;
        border: 1px solid var(--md-border) !important;
        background-color: var(--md-input-bg) !important;
        color: var(--md-text-primary) !important;
    }

    /* Select Box */
    .stSelectbox > div > div {
        border-radius: 4px !important;
        border: 1px solid var(--md-border) !important;
        background-color: var(--md-input-bg) !important;
        color: var(--md-text-primary) !important;
    }

    /* Select Box Options */
    .stSelectbox > div > div > div[role="listbox"] {
        background-color: var(--md-surface) !important;
        color: var(--md-text-primary) !important;
    }

    /* Footer Version Number */
    .st-emotion-cache-1egp75f {
        color: var(--md-text-secondary) !important;
    }

    /* Divider */
    hr {
        border-color: var(--md-border) !important;
    }
</style>
"""
class ChatLogger:
    def __init__(self, log_dir="chat_logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    def generate_conversation_title(self, messages):
        """Generate a title based on conversation content"""
        if not messages:
            return "Empty Conversation"
            
        # Find the first substantive user message
        user_messages = [msg['content'] for msg in messages if msg['role'] == 'user']
        if not user_messages:
            return "New Conversation"
            
        # Use the first user message to create a title
        first_msg = user_messages[0]
        # Truncate and clean the message
        title = first_msg[:50].strip()
        # Remove special characters and make it filesystem-friendly
        title = re.sub(r'[^\w\s-]', '', title)
        title = re.sub(r'\s+', '_', title)
        
        return f"{title[:30]}"
    
    def save_conversation(self, messages, session_id=None):
        if not session_id:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate a title for the conversation
        conversation_title = self.generate_conversation_title(messages)
        
        # Ensure messages is a list of dictionaries with 'role' and 'content' keys
        validated_messages = []
        for msg in messages:
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                validated_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        conversation = {
            "session_id": session_id,
            "title": conversation_title,
            "timestamp": datetime.now().isoformat(),
            "messages": validated_messages
        }
        
        filename = os.path.join(self.log_dir, f"chat_{session_id}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
        
        return filename
    
    def list_conversations(self):
        conversations = []
        try:
            for filename in os.listdir(self.log_dir):
                if filename.startswith("chat_") and filename.endswith(".json"):
                    session_id = filename[5:-5]  # Remove 'chat_' and '.json'
                    try:
                        with open(os.path.join(self.log_dir, filename), 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            conversations.append({
                                "session_id": session_id,
                                "title": data.get("title", "Untitled"),
                                "timestamp": data["timestamp"],
                                "message_count": len(data["messages"])
                            })
                    except Exception as e:
                        st.warning(f"Skipping corrupted chat log {filename}: {str(e)}")
            return sorted(conversations, key=lambda x: x["timestamp"], reverse=True)
        except Exception as e:
            st.error(f"Error listing conversations: {str(e)}")
            return []

# Cache the model loading
@st.cache_resource(show_spinner=True)
def load_model_and_cache(ref):
    return load(ref, {"trust_remote_code": True})

# Initialize embeddings and vector store
@st.cache_resource
def init_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def init_vectorstore():
    embeddings = init_embeddings()
    if os.path.exists("vectorstore"):
        try:
            return FAISS.load_local(
                "vectorstore", 
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.warning("Could not load existing vectorstore. Creating new one.")
            if os.path.exists("vectorstore"):
                import shutil
                shutil.rmtree("vectorstore")
    return FAISS.from_texts([""], embeddings)

def save_vectorstore(vectorstore):
    """Safely save the vectorstore to disk"""
    try:
        vectorstore.save_local("vectorstore")
    except Exception as e:
        st.error(f"Error saving vectorstore: {str(e)}")

# Document processing
class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def process_pdf(self, pdf_file) -> List[str]:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return self.text_splitter.split_text(text)

    def process_image(self, image_file) -> List[str]:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return self.text_splitter.split_text(text)

def build_memory():
    if len(st.session_state.messages) > 2:
        return st.session_state.messages[1:-1]
    return []

def generate(the_prompt, the_model, vectorstore=None):
    if vectorstore is not None:
        relevant_docs = vectorstore.similarity_search(the_prompt, k=3)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        enhanced_prompt = f"""Context information:
{context}

User question: {the_prompt}

Based on the context above, please provide a relevant answer:"""
    else:
        enhanced_prompt = the_prompt

    tokens = []
    skip = 0

    inputs = mx.array(tokenizer.encode(enhanced_prompt))
    
    for token_data in generate_step(inputs, the_model):
        token = token_data[0]
        
        if token == tokenizer.eos_token_id:
            break

        tokens.append(token)
        text = tokenizer.decode(tokens)
        trim = None

        for sw in stop_words:
            if text[-len(sw):].lower() == sw:
                return
            else:
                for i, _ in enumerate(sw, start=1):
                    if text[-i:].lower() == sw[:i]:
                        trim = -i

        yield text[skip:trim]
        skip = len(text)

def show_chat(the_prompt, previous=""):
    if debug:
        print(the_prompt)
        print("-" * 80)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        response = previous

        for chunk in generate(the_prompt, model, st.session_state.vectorstore):
            response = response + chunk
            if not previous:
                response = re.sub(r"^/\*+/", "", response)
                response = re.sub(r"^:+", "", response)
            response = response.replace('�', '')
            message_placeholder.markdown(response + "▌")

        message_placeholder.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Save the conversation after each message
    st.session_state.chat_logger.save_conversation(
        st.session_state.messages,
        st.session_state.current_session_id
    )


def main():
    # Setup Streamlit page
    st.set_page_config(
        page_title=title,
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject Material Design CSS
    st.markdown(material_design_css, unsafe_allow_html=True)

    # Initialize components
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = init_vectorstore()
    if 'chat_logger' not in st.session_state:
        st.session_state.chat_logger = ChatLogger()
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Parse command line arguments
    if 'args' not in st.session_state:
        parser = argparse.ArgumentParser(description="mlx-ui")
        parser.add_argument("--models", type=str, help="models list file", default="models.txt")
        st.session_state.args = parser.parse_args()

    # Sidebar UI
    with st.sidebar:
        st.title("📚 Document Upload")
        st.markdown("---")
        
        uploaded_files = st.file_uploader(
            "Upload documents (PDF/Images)",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            progress_bar = st.progress(0)
            for i, file in enumerate(uploaded_files):
                try:
                    if file.name.lower().endswith('.pdf'):
                        texts = st.session_state.processor.process_pdf(file)
                    else:
                        texts = st.session_state.processor.process_image(file)
                    
                    if texts:
                        st.session_state.vectorstore.add_texts(texts)
                        save_vectorstore(st.session_state.vectorstore)
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        st.success(f"✅ Processed {file.name}")
                    else:
                        st.warning(f"⚠️ No text extracted from {file.name}")
                except Exception as e:
                    st.error(f"❌ Error processing {file.name}: {str(e)}")

        st.divider()
    
     # Chat History Section
    with st.sidebar:
        st.subheader("💾 Chat History")
        conversations = st.session_state.chat_logger.list_conversations()
        
        if conversations:
            selected_chat = st.selectbox(
                "Select a conversation to load",
                options=conversations,
                format_func=lambda x: f"{x['title']} - {datetime.fromisoformat(x['timestamp']).strftime('%Y-%m-%d %H:%M')} ({x['message_count']} msgs)",
                key="chat_history_select"
            )
            
            if st.button("Load Selected Conversation", key="load_chat_btn"):
                if selected_chat:
                    loaded_chat = st.session_state.chat_logger.load_conversation(selected_chat["session_id"])
                    if loaded_chat:
                        st.session_state.messages = loaded_chat["messages"]
                        st.session_state.current_session_id = selected_chat["session_id"]
                        st.rerun()
                    else:
                        st.error("Failed to load the selected conversation")
        else:
            st.info("No previous conversations found")

        st.divider()

    # Load model references
    try:
        with open(st.session_state.args.models, 'r') as file:
            model_refs = [line.strip() for line in file.readlines() if not line.startswith('#')]
        model_refs = {k.strip(): v.strip() for k, v in [line.split("|") for line in model_refs]}
    except Exception as e:
        st.error(f"Error loading models file: {str(e)}")
        model_refs = {"none": "No models available"}

    # Model selection and parameters UI
    model_ref = st.sidebar.selectbox(
        "🤖 Select Model", 
        model_refs.keys(), 
        format_func=lambda value: model_refs[value]
    )

    if model_ref.strip() != "-" and model_ref != "none":
        global model, tokenizer, stop_words
        model, tokenizer = load_model_and_cache(model_ref)
        
        chat_template = tokenizer.chat_template or (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
        
        supports_system_role = "system role not supported" not in chat_template.lower()
        
        system_prompt = st.sidebar.text_area(
            "🔧 System Prompt",
            "You are a helpful AI assistant that can access and reference uploaded documents to provide accurate information.",
            disabled=not supports_system_role
        )
        
        st.sidebar.markdown("### ⚙️ Generation Parameters")
        
        context_length = st.sidebar.number_input(
            'Context Length', 
            value=400, 
            min_value=100,
            step=100, 
            max_value=32000
        )
        
        temperature = st.sidebar.slider(
            'Temperature',
            min_value=0.,
            max_value=1.,
            step=.10,
            value=.5
        )

        stop_words = ["<|im_start|>", "<|im_end|>", "<s>", "</s>"]

        # Initialize chat if needed
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "👋 Hello! How may I help you today?"}]

        # Main chat container with custom styling
        st.markdown("<h1 style='text-align: center; color: var(--md-primary);'>💬 Chat Interface</h1>", unsafe_allow_html=True)
        
        chat_container = st.container()
        with chat_container:
            # Display all messages
            for msg in st.session_state.messages:
                avatar = "🤖" if msg["role"] == "assistant" else "👤"
                st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Display user message immediately
            st.chat_message("user", avatar="👤").write(prompt)
            
            # Add to session state
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            messages = []
            if supports_system_role:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(build_memory())
            messages.append({"role": "user", "content": prompt})

            full_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=chat_template
            ).rstrip("\n")

            show_chat(full_prompt)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<p style='text-align: center; color: var(--md-text-secondary);'>v{ver}</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()