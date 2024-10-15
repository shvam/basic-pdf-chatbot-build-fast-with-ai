import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the extracted text into smaller chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from the text chunks using Chroma
def get_vectorstore(text_chunks):
    try:
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPEN_API_KEY"])
        if not text_chunks:
            st.error("No text chunks to process. Please check your PDF content.")
            return None
        # Create and return the Chroma vector store
        vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

# Function to create a custom prompt template
def get_custom_prompt_template():
    template = """You are an AI assistant specialized in answering questions based on the content of PDF documents. 
    Use the following pieces of context to answer the human's question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Always try to provide a source or reference from the context if possible.

    Context: {context}
    Human: {question}
    AI Assistant: Let me analyze the provided information to answer your question.
    """
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

# Function to set up the conversational chain
def get_conversation_chain(vectorstore):
    if vectorstore is None:
        return None
    
    # Initialize the language model
    llm = ChatOpenAI(openai_api_key=st.secrets["OPEN_API_KEY"], temperature=0.2)
    # Set up memory for conversation history
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    # Get the custom prompt template
    prompt = get_custom_prompt_template()
    # Create and return the conversational chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return conversation_chain

# Function to handle user input and generate responses
def handle_user_input(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process PDFs before asking questions.")
        return

    # Generate response using the conversation chain
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Display the conversation history
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            user = st.chat_message(name="User", avatar="👤")
            user.write(message.content)
        else:
            assistant = st.chat_message(name="PDF BOT", avatar="🤖")
            assistant.write(message.content)

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚")
    
    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat with multiple PDFs 📚")

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click `Process`", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing files ..."):
                    # Extract text from PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No text could be extracted from the PDFs. Please check your files.")
                        return
                    # Create text chunks
                    text_chunks = get_text_chunks(raw_text)
                    # Create vector store
                    vectorstore = get_vectorstore(text_chunks)
                    if vectorstore:
                        # Set up conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.success("PDFs processed successfully!")
                    else:
                        st.error("Failed to process PDFs. Please try again.")
            else:
                st.warning("Please upload PDF files before processing.")

    # User input for questions
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)

if __name__ == "__main__":
    main()