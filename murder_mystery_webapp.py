import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Page configuration
st.set_page_config(page_title="Murder Mystery Assistant", page_icon="🔍")
st.title("🔍 Murder Mystery Assistant")

# Initialize the chain
@st.cache_resource
def initialize_chain():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load murder mystery template directly
    try:
        with open("murder_mystery_template.txt", "r") as file:
            template_content = file.read()
    except FileNotFoundError:
        template_content = """
        Murder Mystery Writing Guide:
        A good murder mystery has an intriguing detective, red herrings, plot twists,
        and a satisfying resolution. Create memorable characters and an atmospheric setting.
        """
    
    # Create a document
    documents = [Document(page_content=template_content, metadata={"source": "murder_mystery_template.txt"})]
    
    # Split document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Create a vector store using FAISS (in-memory)
    vectorstore = FAISS.from_documents(texts, embedding_model)
    
    # Initialize the LLM
    llm = ChatAnthropic(
        model="claude-3-7-sonnet-20250219",
        temperature=0.7,
        anthropic_api_key=st.secrets["ANTHROPIC_API_KEY"]
    )
    
    # Set up conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Create the conversational retrieval chain
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
