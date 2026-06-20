import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from pydantic import Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List, Optional, Any

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)  # pass the file object directly
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            print(f"Could not read PDF: {e}")
            continue
    # Sanitize: replace non-ASCII characters to prevent encoding errors
    text = text.encode("ascii", "ignore").decode("ascii")
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    api_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=api_key
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


class HuggingFaceChatModel(BaseChatModel):
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    api_key: str = Field(default="")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        hf_messages = []
        for msg in messages:
            if msg.type == "human" or isinstance(msg, HumanMessage):
                hf_messages.append({"role": "user", "content": msg.content})
            elif msg.type == "ai" or isinstance(msg, AIMessage):
                hf_messages.append({"role": "assistant", "content": msg.content})
            else:
                hf_messages.append({"role": "system", "content": msg.content})

        from huggingface_hub import InferenceClient
        client = InferenceClient(api_key=self.api_key)

        models_to_try = [
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "HuggingFaceH4/zephyr-7b-beta",
            "microsoft/Phi-3-mini-4k-instruct"
        ]

        last_error = None
        for model in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=hf_messages,
                    max_tokens=512,
                    temperature=0.7
                )
                text = response.choices[0].message.content
                generation = ChatGeneration(message=AIMessage(content=text))
                return ChatResult(generations=[generation])
            except Exception as e:
                if "401" in str(e) or "unauthorized" in str(e).lower():
                    raise e
                last_error = e
                print(f"Model {model} failed: {e}. Trying fallback...")
                continue

        raise last_error if last_error else Exception("All models failed to generate response.")

    @property
    def _llm_type(self) -> str:
        return "custom-huggingface-chat"


def get_conversation_chain(vectorstore):
    api_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    llm = HuggingFaceChatModel(api_key=api_key)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    if not st.session_state.conversation or isinstance(st.session_state.conversation, type(lambda: None)):
        st.warning("Please upload and process your PDFs first to start the conversation!")
        return

    # Ensure API key is set before querying
    if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        st.error("Please provide your Hugging Face API Token in the sidebar to ask questions.")
        return

    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.markdown(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.markdown(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs (Free HF Edition)",
                       page_icon=":books:",
                       layout="wide")
    st.markdown(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs (Hugging Face) :books:")
    
    # Custom instructions container
    with st.expander("ℹ️ How to use", expanded=True):
        st.write("""
        1. Enter your **Hugging Face API Token** (starts with `hf_...`) in the sidebar.
        2. Upload your PDF documents in the sidebar.
        3. Click **Process** and wait for the documents to be analyzed.
        4. Ask questions about your documents in the chat input below!
        """)

    user_question = st.text_input("Ask a question about your documents:", placeholder="e.g., What are the key takeaways from the uploaded documents?")
    
    # Clear / Reset chat button
    if st.session_state.chat_history:
        if st.button("Clear Chat"):
            st.session_state.chat_history = None
            st.rerun()

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Configuration")
        
        # Check environment first
        env_key = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
        
        # User input for API Key (masked)
        api_key_raw = st.text_input("Hugging Face API Token", type="password", value=env_key, placeholder="hf_...")
        # Clean the token: strip whitespace and remove any non-ASCII characters
        api_key = api_key_raw.strip().encode("ascii", "ignore").decode("ascii") if api_key_raw else ""
        if api_key:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
            st.session_state.api_key_set = True
        else:
            st.session_state.api_key_set = False

        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            if not api_key:
                st.error("⚠️ Please enter your Hugging Face API token first.")
            elif not pdf_docs:
                st.error("⚠️ Please upload at least one PDF file.")
            else:
                with st.spinner("Analyzing and embedding PDFs..."):
                    try:
                        # get pdf text
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text.strip():
                            st.error("No text could be extracted from the uploaded PDF files. Please ensure they contain selectable text.")
                            return

                        # get the text chunks
                        text_chunks = get_text_chunks(raw_text)

                        # create vector store
                        vectorstore = get_vectorstore(text_chunks)

                        # create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.success("🎉 Documents processed successfully! You can now start chatting.")
                    except Exception as e:
                        st.error(f"Error processing PDFs: {e}")


if __name__ == '__main__':
    main()
