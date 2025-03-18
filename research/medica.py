import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

HF_TOKEN = "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"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Page Config
st.set_page_config(page_title="Medica", page_icon="🩺", layout="wide")

# Sidebar - Header
st.sidebar.title("🩺 Medica - Wellness At Your finger tips")
st.sidebar.markdown("Feel free to ask about medical information or symptoms!")

# Sidebar - Disease Selection
selected_disease = st.sidebar.selectbox(
    "Choose a Disease:",
    ["Hypertension", "Diabetes", "Chronic Kidney Disease"]
)
st.sidebar.write(f"You selected: {selected_disease}")

# Sidebar - Contact Info
st.sidebar.markdown("## 📞 Contact Us")
st.sidebar.write("- Email: support@medibot.com")
st.sidebar.write("- Phone: +91-1234567890")

# Custom CSS for Styling
st.markdown(
    """
    <style>
    .chat-bubble {
        background-color: #2B2B2B;
        color: #FFFFFF;
        border-radius: 20px;
        padding: 10px 15px;
        margin: 5px;
        display: inline-block;
        max-width: 70%;
    }
    .chat-bubble.user {
        background-color: #1E88E5;
        color: #FFFFFF;
    }
    .chat-bubble.assistant {
        background-color: #424242;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load Vector Store
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

# Custom Prompt Template
def set_custom_prompt():
    template = """
    Use the information provided to answer the user's question.
    If the answer is not available, simply say 'I don't know'.
    Avoid generating unrelated information.

    Context: {context}
    Question: {question}

    Provide a clear and concise answer.
    """
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Load LLM
def load_llm(repo_id, token):
    try:
        return HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=0.5,
            model_kwargs={"token": token, "max_length": 512}
        )
    except Exception as e:
        st.error(f"Error loading LLM: {e}")
        return None

def main():
    st.title("🩺 Ask Medica - Your Medical Chat Assistant")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        role, content = message['role'], message['content']
        chat_class = f'chat-bubble {role}'
        st.markdown(f"<div class='{chat_class}'>{content}</div>", unsafe_allow_html=True)

    # User Input
    prompt = st.chat_input("Type your medical question here...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Configuration
        repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
        vectorstore = get_vectorstore()
        if not vectorstore:
            return

        llm = load_llm(repo_id, HF_TOKEN)
        if not llm:
            return

        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt()}
            )
            response = qa_chain.invoke({'query': prompt})
            result = response.get("result", "I couldn't fetch a response.")
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()