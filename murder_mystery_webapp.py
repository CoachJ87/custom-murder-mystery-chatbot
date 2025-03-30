import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# Page configuration
st.set_page_config(page_title="Murder Mystery Assistant", page_icon="🔍")
st.title("🔍 Murder Mystery Assistant")

@st.cache_resource
def initialize_chain():
    # Try to load the template content
    try:
        with open("murder_mystery_template.txt", "r") as file:
            template_content = file.read()
    except FileNotFoundError:
        template_content = """
        Murder Mystery Writing Guide:
        A good murder mystery has an intriguing detective, red herrings, plot twists,
        and a satisfying resolution. Create memorable characters and an atmospheric setting.
        """
    
    # Create a template with the murder mystery guide embedded
    template = f"""You are a Murder Mystery Writing Assistant. Use the following guidelines to help users craft engaging murder mysteries:

    {template_content}

    Current conversation:
    {{chat_history}}
    Human: {{input}}
    AI Assistant:"""
    
    prompt = PromptTemplate(
        input_variables=["chat_history", "input"], 
        template=template
    )
    
    # Initialize the LLM
    llm = ChatAnthropic(
        model="claude-3-7-sonnet-20250219",
        temperature=0.7,
        anthropic_api_key=st.secrets["ANTHROPIC_API_KEY"]
    )
    
    # Set up conversation memory
    memory = ConversationBufferMemory(return_messages=True)
    
    # Create the chain
    conversation = ConversationChain(
        llm=llm, 
        verbose=False, 
        memory=memory,
        prompt=prompt
    )
    
    return conversation

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your Murder Mystery Assistant. How can I help you craft the perfect mystery today?"}]

# Initialize the chain
conversation = initialize_chain()

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
        response = conversation.predict(input=prompt)
        
        # Display the response
        message_placeholder.write(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
