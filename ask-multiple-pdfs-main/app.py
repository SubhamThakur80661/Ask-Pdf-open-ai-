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
        st.warning("Upload and process your PDFs first, then ask away.")
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
    st.set_page_config(page_title="Ask your PDFs",
                       page_icon=":books:",
                       layout="wide")
    st.markdown(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Status pill: honest state, no invented progress
    if st.session_state.conversation:
        status = '<span class="status-pill"><span class="dot"></span>Documents ready. Ask away.</span>'
    elif st.session_state.get("api_key_set"):
        status = '<span class="status-pill"><span class="dot amber"></span>Token added. Upload your PDFs next.</span>'
    else:
        status = '<span class="status-pill"><span class="dot amber"></span>Add your token to start.</span>'

    st.markdown(
        '<div class="hero">'
        '<div class="hero-eyebrow"><span class="dot"></span>Chat with your PDFs</div>'
        '<h1 class="hero-title">Ask your PDFs<br><span class="grad">anything.</span></h1>'
        '<p class="hero-sub">Upload your documents, then ask questions in plain words. '
        'Every answer comes only from the pages you upload.</p>'
        + status +
        '</div>',
        unsafe_allow_html=True)

    # Step flow
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="step-card"><div class="step-num">01</div><h3>Add your token</h3><p>Paste a free Hugging Face token in the sidebar. It is the key that lets the app read and answer.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="step-card"><div class="step-num">02</div><h3>Upload and process</h3><p>Drop in one or more PDFs and press Process. The app reads every page and gets it ready to search.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="step-card"><div class="step-num">03</div><h3>Ask away</h3><p>Type a question below. The answer comes from your documents, and the chat stays on screen so you can follow along.</p></div>', unsafe_allow_html=True)

    user_question = st.text_input("Ask a question about your documents:", placeholder="e.g., What are the key takeaways from the uploaded documents?")

    # Clear / Reset chat button
    if st.session_state.chat_history:
        if st.button("Clear Chat"):
            st.session_state.chat_history = None
            st.rerun()

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.markdown('<div class="side-head">Set up</div>', unsafe_allow_html=True)
        st.markdown('<p class="side-note">You need a free token from Hugging Face to get answers. <a href="https://huggingface.co/settings/tokens" target="_blank">Get one here</a>, then paste it below.</p>', unsafe_allow_html=True)

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

        st.markdown('<div class="side-head">Your documents</div>', unsafe_allow_html=True)
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process", type="primary"):
            if not api_key:
                st.error("⚠️ Please enter your Hugging Face API token first.")
            elif not pdf_docs:
                st.error("⚠️ Please upload at least one PDF file.")
            else:
                with st.spinner("Reading your PDFs..."):
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
                        st.success("Done. Your documents are ready. Ask a question below.")
                    except Exception as e:
                        st.error(f"Error processing PDFs: {e}")

        st.markdown('<p class="side-foot">Answers come from the PDFs you upload, nothing else.</p>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
