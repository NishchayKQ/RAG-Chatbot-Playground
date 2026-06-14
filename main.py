import streamlit as st

from core import vector_store
from core.model_config import get_agent_response_for_query

# docs - https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps
st.title("Rag chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Redraw all previous messages on the screen after a rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?", accept_file=True, file_type=["pdf"]):

    # process any files user uploads
    if prompt.files:
        # Show a UI spinner while ChromaDB processes the PDF
        with st.status(f"Indexing {len(prompt.files)} file(s)...", expanded=True) as status:
            for file in prompt.files:
                st.write(f"Chunking and embedding {file.name}...")
                vector_store.load_pdf(file=file, file_name=file.name)

            # Change spinner to a green checkmark when done
            status.update(label="Files successfully added to database!", state="complete", expanded=False)

        # Add a visual note to the chat history so the user knows the file was processed
        file_msg = '\n'.join([f"📁{file.name}" for file in prompt.files])
        st.session_state.messages.append({"role": "user", "content": file_msg})
        with st.chat_message("user"):
            st.markdown(file_msg)

    # We check if text exists, because they might have ONLY uploaded a file
    if prompt.text:
        # Display and save the user's question
        with st.chat_message("user"):
            st.markdown(prompt.text)
        st.session_state.messages.append({"role": "user", "content": prompt.text})

        # Display the streaming assistant response
        with st.chat_message("assistant"):
            # noinspection PyTypeChecker
            response = st.write_stream(get_agent_response_for_query(prompt.text))

        # Save the final assistant string to the history
        st.session_state.messages.append({"role": "assistant", "content": response})
