import os
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Page configuration
st.set_page_config(page_title="Murder Mystery Assistant", page_icon="🔍")
st.title("🔍 Murder Mystery Assistant")

# Initialize the embedding model and vectorstore
@st.cache_resource
def initialize_chain():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Check if vector database exists, if not create it
    if not os.path.exists("./mystery_chroma_db"):
        # Code to create the vector db from your source files
        # Load the document
        loader = TextLoader("./murder_mystery_template.txt")
        documents = loader.load()
        
        # Split the document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        # Create the vector store
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embedding_model,
            persist_directory="./mystery_chroma_db"
        )
        vectorstore.persist()
        print("Created new vector database")
    else:
        # Load existing vector store
        vectorstore = Chroma(
            persist_directory="./mystery_chroma_db",
            embedding_function=embedding_model
        )
    
    llm = ChatAnthropic(
        model="claude-3-7-sonnet-20250219",
        temperature=0.7,
        anthropic_api_key=st.secrets["ANTHROPIC_API_KEY"]
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        verbose=False
    )
    
    return qa_chain

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your Murder Mystery Assistant. How can I help you craft the perfect mystery today?"}]

# Initialize the chain
qa_chain = initialize_chain()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if prompt := st.chat_input("Ask about creating your murder mystery..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get the assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Get the answer from the chain
        result = qa_chain({"question": prompt})
        response = result["answer"]
        
        # Display the response
        message_placeholder.write(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})