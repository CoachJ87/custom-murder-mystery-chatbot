import streamlit as st
import os
from anthropic import Anthropic

# Page configuration
st.set_page_config(page_title="Murder Mystery Assistant", page_icon="🔍")
st.title("🔍 Murder Mystery Assistant")

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

# Initialize Anthropic client
def get_anthropic_client():
    return Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Hello! I'm your Murder Mystery Assistant. How can I help you craft the perfect mystery today?"
    }]

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = f"""You are a Murder Mystery Writing Assistant. Use the following guidelines to help users craft engaging murder mysteries:

{template_content}

Be creative, helpful, and provide detailed suggestions when asked."""

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
        
        # Format messages for Anthropic
        messages_for_api = [
            {"role": "system", "content": st.session_state.system_prompt}
        ]
        
        # Add conversation history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                messages_for_api.append({"role": "user", "content": msg["content"]})
            else:
                messages_for_api.append({"role": "assistant", "content": msg["content"]})
        
        # Remove the last user message since we already added it to the history
        if messages_for_api[-1]["role"] == "user":
            messages_for_api.pop()
        
        # Add the current user's message
        messages_for_api.append({"role": "user", "content": prompt})
        
        try:
            # Get response from Anthropic
            client = get_anthropic_client()
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                messages=messages_for_api,
                temperature=0.7,
                max_tokens=1000
            )
            
            result = response.content[0].text
            
            # Display the response
            message_placeholder.write(result)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": result})
            
        except Exception as e:
            message_placeholder.write(f"An error occurred: {str(e)}")
