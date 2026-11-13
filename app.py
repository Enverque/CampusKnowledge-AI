import streamlit as st

from rag_backend import (
    load_vectorstore,
    load_chat_model,
    generate_answer
)


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="College RAG Assistant",
    page_icon="🎓",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def get_vectorstore():
    return load_vectorstore()


@st.cache_resource
def get_chat_model():
    return load_chat_model()


vectorstore = get_vectorstore()
chat_model = get_chat_model()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🎓 College RAG Assistant")

st.caption(
    "Ask questions about placements, admissions, "
    "fees, rankings, hostel rules and more."
)

st.success(
    f"📚 Knowledge base loaded • "
    f"{vectorstore.index.ntotal} document chunks"
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("🎓 College Assistant")

    st.write(
        "Ask me anything about the information "
        "available in the college documents."
    )

    st.divider()

    st.subheader("💡 Try asking")

    st.write("• What is the placement percentage?")
    st.write("• What are the admission requirements?")
    st.write("• What are the hostel rules?")
    st.write("• What is the NIRF ranking?")
    st.write("• What are the college fees?")

    st.divider()

    if st.button("🗑️ Clear conversation"):

        st.session_state.messages = []

        st.rerun()


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# --------------------------------------------------
# WELCOME
# --------------------------------------------------

if len(st.session_state.messages) == 0:

    st.info(
        "👋 Hello! I'm your College Assistant.\n\n"
        "Ask me a question about your college documents."
    )


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "💬 Ask your question..."
)


# --------------------------------------------------
# PROCESS QUESTION
# --------------------------------------------------

if question:

    # Show user question
    with st.chat_message("user"):

        st.markdown(question)


    # Save user question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("🔎 Searching college documents..."):

            answer, documents = generate_answer(
                question,
                vectorstore,
                chat_model
            )


        st.markdown(answer)


        # --------------------------------------------------
        # SOURCES
        # --------------------------------------------------

        st.markdown("### 📚 Sources")

        shown_sources = set()

        for doc in documents:

            source = doc.metadata.get(
                "source_file",
                "Unknown document"
            )

            page = doc.metadata.get(
                "page_number",
                "Unknown"
            )

            source_key = (source, page)

            if source_key not in shown_sources:

                st.caption(
                    f"📄 {source} — Page {page}"
                )

                shown_sources.add(source_key)


    # Save assistant answer
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )