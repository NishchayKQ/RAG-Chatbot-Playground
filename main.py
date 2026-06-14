import streamlit as st

from core import vector_store
from core.model_config import get_agent_response_for_query

# docs - https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps
st.title("Rag chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?", accept_file=True, file_type=["pdf"]):

    if prompt.files:
        vector_store.load_pdf(prompt.files)


    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt.text})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt.text)

        if prompt.files:
            for file in prompt.files:
                st.write(file.name)

    if prompt.text: # if user sent a text message process it
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # noinspection PyTypeChecker
            response = st.write_stream(get_agent_response_for_query(prompt.text))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
