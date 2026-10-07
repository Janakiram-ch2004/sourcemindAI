

import os
import uuid

import streamlit as st

from new_backend import workflow

from rag import (
    add_documents,
    get_document_count
)

from langchain_core.messages import (
    HumanMessage
)


# ==========================================
# 1. PAGE CONFIG
# ==========================================

st.set_page_config(

    page_title="SourceMindAI",

    page_icon="💬",

    layout="wide"

)


# ==========================================
# 2. GENERATE THREAD ID
# ==========================================

def generate_thread_id():

    return str(uuid.uuid4())


# ==========================================
# 3. EXTRACT MESSAGE TEXT
# ==========================================

def extract_text(content):

    if isinstance(
        content,
        str
    ):

        return content


    if isinstance(
        content,
        list
    ):

        text = ""

        for item in content:

            if isinstance(
                item,
                dict
            ):

                text += item.get(
                    "text",
                    ""
                )

            else:

                text += str(item)


        return text


    return str(content)


# ==========================================
# 4. ADD THREAD
# ==========================================

def add_thread(thread_id):

    if thread_id not in st.session_state[
        "chat_threads"
    ]:

        st.session_state[
            "chat_threads"
        ].append(thread_id)


# ==========================================
# 5. RESET CHAT
# ==========================================

def reset_chat():

    thread_id = generate_thread_id()


    st.session_state[
        "thread_id"
    ] = thread_id


    add_thread(thread_id)


    st.session_state[
        "message_history"
    ] = []


# ==========================================
# 6. LOAD CONVERSATION
# ==========================================

def load_conversation(thread_id):

    state = workflow.get_state(

        config={
            "configurable": {
                "thread_id": thread_id
            }
        }

    )


    return state.values.get(

        "messages",

        []

    )

st.title("🧠 SourceMindAI")
st.caption("Answers with sources from documents and the web")
# ==========================================
# 7. DISPLAY SOURCES
# ==========================================


def display_sources(sources):
   
    if not sources or len(sources) == 0:
        return

    
    st.markdown("---")
    st.markdown("### 📚 Source Citations:")

    for index, source in enumerate(sources, start=1):
        source_type = source.get("type", "")

        # Web search nunchi vasthe clickable link istham
        if source_type == "web":
            title = source.get("title", "Web source")
            url = source.get("url", "")
            
            if url:
                
                st.markdown(f"{index}. 🌐 [{title}]({url})")

        
        elif source_type == "document":
            file_name = source.get("file_name", "Unknown document")
            page = source.get("page")
            
            if page:
                st.markdown(f"{index}. 📄 **{file_name}** — Page {page}")
            else:
                st.markdown(f"{index}. 📄 **{file_name}**")


        # ==================================
        # WEB SOURCE
        # ==================================

        if source_type == "web":

            title = source.get(

                "title",

                "Web source"

            )


            url = source.get(

                "url",

                ""

            )


            snippet = source.get(

                "snippet",

                ""

            )


            if url:

                st.markdown(

                    f"{index}. "
                    f"[{title}]({url})"

                )


            if snippet:

                st.caption(

                    snippet

                )


        # ==================================
        # DOCUMENT SOURCE
        # ==================================

        elif source_type == "document":

            file_name = source.get(

                "file_name",

                "Unknown document"

            )


            page = source.get(

                "page"

            )


            if page:

                st.markdown(

                    f"{index}. "
                    f"📄 `{file_name}` "
                    f"— Page {page}"

                )

            else:

                st.markdown(

                    f"{index}. "
                    f"📄 `{file_name}`"

                )


# ==========================================
# 8. SESSION STATE
# ==========================================

if "message_history" not in st.session_state:

    st.session_state[
        "message_history"
    ] = []


if "thread_id" not in st.session_state:

    st.session_state[
        "thread_id"
    ] = generate_thread_id()


if "chat_threads" not in st.session_state:

    st.session_state[
        "chat_threads"
    ] = []


if "chat_titles" not in st.session_state:

    st.session_state[
        "chat_titles"
    ] = {}


# ==========================================
# 9. ADD CURRENT THREAD
# ==========================================

add_thread(

    st.session_state[
        "thread_id"
    ]

)


# ==========================================
# 10. SIDEBAR
# ==========================================

st.sidebar.title(

    "SourceMindAI"

)


# ==========================================
# NEW CHAT
# ==========================================

if st.sidebar.button(

    "New Chat",

    use_container_width=True

):

    reset_chat()

    st.rerun()


st.sidebar.divider()


# ==========================================
# 11. KNOWLEDGE BASE
# ==========================================

st.sidebar.header(

    "Knowledge Base"

)


uploaded_files = st.sidebar.file_uploader(

    "Upload documents",

    type=[
        "pdf",
        "docx",
        "txt"
    ],

    accept_multiple_files=True

)


# ==========================================
# PROCESS DOCUMENTS
# ==========================================

if st.sidebar.button(

    "Process Documents",

    use_container_width=True

):

    if not uploaded_files:

        st.sidebar.warning(

            "Please upload at least one document."

        )

    else:

        os.makedirs(

            "uploads",

            exist_ok=True

        )


        file_paths = []


        for uploaded_file in uploaded_files:

            file_path = os.path.join(

                "uploads",

                uploaded_file.name

            )


            with open(

                file_path,

                "wb"

            ) as file:

                file.write(

                    uploaded_file.getbuffer()

                )


            file_paths.append(

                file_path

            )


        # ------------------------------
        # Process
        # ------------------------------

        with st.spinner(

            "Processing documents..."

        ):

            results = add_documents(

                file_paths

            )


        # ------------------------------
        # Show results
        # ------------------------------

        for result in results:

            status = result.get(

                "status"

            )


            file_name = result.get(

                "file"

            )


            if status == "added":

                st.sidebar.success(

                    f"{file_name} added "
                    f"({result['chunks']} chunks)"

                )


            elif status == "already_exists":

                st.sidebar.info(

                    f"{file_name} already exists. "
                    f"Skipped."

                )


            elif status == "error":

                st.sidebar.error(

                    f"{file_name}: "
                    f"{result.get('error')}"

                )


        st.sidebar.success(

            f"Total stored chunks: "
            f"{get_document_count()}"

        )


st.sidebar.divider()


# ==========================================
# 12. RECENT CHATS
# ==========================================

st.sidebar.header(

    "Recent Chats"

)


for thread_id in st.session_state[
    "chat_threads"
]:

    title = st.session_state[
        "chat_titles"
    ].get(

        thread_id,

        "New Chat"

    )


    if st.sidebar.button(

        title,

        key=f"thread_{thread_id}",

        use_container_width=True

    ):

        st.session_state[
            "thread_id"
        ] = thread_id


        messages = load_conversation(

            thread_id

        )


        temp_messages = []


        for msg in messages:

            if isinstance(
                msg,
                HumanMessage
            ):

                role = "user"

            else:

                role = "assistant"


            content = extract_text(

                msg.content

            )


            if content:

                temp_messages.append({

                    "role": role,

                    "content": content,

                    "sources": []

                })


        st.session_state[
            "message_history"
        ] = temp_messages


        st.rerun()


# ==========================================
# 13. DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state[
    "message_history"
]:

    with st.chat_message(

        message["role"]

    ):

        st.markdown(

            message["content"]

        )


        # Display old sources
        display_sources(

            message.get(
                "sources",
                []
            )

        )


# ==========================================
# 14. CHAT INPUT
# ==========================================

user_input = st.chat_input(

    "Type here..."

)


# ==========================================
# 15. CURRENT THREAD
# ==========================================

thread_id = st.session_state[
    "thread_id"
]


# ==========================================
# 16. LANGGRAPH CONFIG
# ==========================================

config = {

    "configurable": {

        "thread_id": thread_id

    },

    "metadata": {

        "thread_id": thread_id

    },

    "tags": [

        "streamlit-chatbot"

    ]

}


# ==========================================
# 17. USER SENT MESSAGE
# ==========================================

if user_input:

    # --------------------------------------
    # Create chat title
    # --------------------------------------

    if thread_id not in st.session_state[
        "chat_titles"
    ]:

        title = user_input.strip()


        if len(title) > 30:

            title = title[:30] + "..."


        st.session_state[
            "chat_titles"
        ][thread_id] = title


    # --------------------------------------
    # Store user message
    # --------------------------------------

    st.session_state[
        "message_history"
    ].append({

        "role": "user",

        "content": user_input,

        "sources": []

    })


    # --------------------------------------
    # Display user message
    # --------------------------------------

    with st.chat_message(

        "user"

    ):

        st.markdown(

            user_input

        )


    # ======================================
    # ASSISTANT
    # ======================================

    with st.chat_message(

        "assistant"

    ):

        # ----------------------------------
        # Streaming generator
        # ----------------------------------

        def stream_generator():

            for (

                msg_chunk,

                metadata

            ) in workflow.stream(

                {

                    "messages": [

                        HumanMessage(

                            content=user_input

                        )

                    ]

                },

                config=config,

                stream_mode="messages"

            ):

                node_name = metadata.get(

                    "langgraph_node"

                )


                # --------------------------
                # Only final answer nodes
                # --------------------------

                allowed_nodes = {

                    "tool_agent",

                    "direct",

                    "generate",

                    "web_fallback"

                }


                if node_name not in allowed_nodes:

                    continue


                # --------------------------
                # Don't display tool calls
                # --------------------------

                if getattr(

                    msg_chunk,

                    "tool_calls",

                    None

                ):

                    continue


                content = msg_chunk.content


                # --------------------------
                # String content
                # --------------------------

                if isinstance(

                    content,

                    str

                ):

                    if content:

                        yield content


                # --------------------------
                # List content
                # --------------------------

                elif isinstance(

                    content,

                    list

                ):

                    for item in content:

                        if isinstance(

                            item,

                            dict

                        ):

                            text = item.get(

                                "text",

                                ""

                            )


                            if text:

                                yield text


        # ----------------------------------
        # Stream answer
        # ----------------------------------

        AI_Message = st.write_stream(

            stream_generator()

        )


        # ==================================
        # GET FINAL LANGGRAPH STATE
        # ==================================

        final_state = workflow.get_state(

            config

        )


        # ----------------------------------
        # Get sources
        # ----------------------------------

        sources = final_state.values.get(

            "sources",

            []

        )

        sources = final_state.values.get(
    "sources",
    []
)


        # ==================================
        # DISPLAY SOURCES
        # ==================================

        display_sources(

            sources

        )


    # ======================================
    # SAVE ASSISTANT MESSAGE
    # ======================================

    st.session_state[
        "message_history"
    ].append({

        "role": "assistant",

        "content": AI_Message,

        "sources": sources

    })