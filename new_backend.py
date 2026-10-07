

import os
import json
import requests
import numexpr as ne
from typing import (
    Annotated,
    TypedDict
)

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    ToolMessage,
    HumanMessage
)
from langchain_core.tools import tool

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langgraph.graph.message import add_messages

from langgraph.checkpoint.memory import MemorySaver

from langgraph.prebuilt import ToolNode

from ddgs import DDGS

from rag import get_retriever


# ==========================================
# 1. ENVIRONMENT
# ==========================================

load_dotenv()


# ==========================================
# 2. LLM
# ==========================================

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0.3,
    max_tokens=700,
    reasoning_effort="none"
)


# Small LLM for routing
router_llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    max_tokens=10,
    reasoning_effort="none"
)


# Small LLM for document grading
grader_llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    max_tokens=10,
    reasoning_effort="none"
)


# ==========================================
# 3. RETRIEVER
# ==========================================

retriever = get_retriever()


# ==========================================
# 4. WEATHER TOOL
# ==========================================

@tool
def weather_tool(location: str) -> str:
    """
    Returns current weather and temperature
    for a given city.
    """

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return "Weather API key is not configured."

    try:

        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "q": location,
                "appid": api_key,
                "units": "metric"
            },
            timeout=10
        )

        if response.status_code == 200:

            data = response.json()

            temp = data["main"]["temp"]

            description = data["weather"][0]["description"]

            return (
                f"The temperature in {location} "
                f"is {temp}°C with {description}."
            )

        return f"Could not find weather for {location}."

    except Exception as e:

        return f"Weather service error: {str(e)}"


# ==========================================
# 5. CALCULATOR TOOL
# ==========================================

@tool
def calculate_math(expression: str) -> str:
    """
    Evaluates mathematical expressions.
    """

    try:

        result = ne.evaluate(expression)

        return f"The calculated result is: {result}"

    except Exception:

        return (
            "Could not calculate. "
            "Please provide a valid expression."
        )


# ==========================================
# 6. WEB SEARCH TOOL
# ==========================================
# ==========================================

@tool
def web_search_tool(query: str) -> str:
    """
    Search the internet for latest news,
    current events, real-time information,
    or information unavailable in documents.

    Returns structured results containing
    title, URL and snippet.
    """

    try:

        results = DDGS().text(
            query,
            max_results=5
        )

        web_results = []

        for result in results:

            web_results.append({

                "title": result.get(
                    "title",
                    ""
                ),

                "url": result.get(
                    "href",
                    ""
                ),

                "snippet": result.get(
                    "body",
                    ""
                )
            })

        return json.dumps(
            web_results,
            ensure_ascii=False
        )

    except Exception as e:

        return json.dumps({

            "error": str(e)

        })


# ==========================================
# 7. ALL TOOLS
# ==========================================

tools = [

    weather_tool,

    calculate_math,

    web_search_tool

]


# ==========================================
# 8. LLM WITH TOOLS
# ==========================================

llm_with_tools = llm.bind_tools(tools)


# ==========================================
# 9. STATE
# ==========================================

class State(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]

    route: str

    retrieved_docs: list

    rag_context: str

    retrieval_relevant: bool

    sources: list


# ==========================================
# 10. ROUTER
# ==========================================


def route_question(state: State):

    question = state["messages"][-1].content

    
    
    prompt = f"""
You are the routing system of an AI assistant.

User question:
{question}

Choose exactly ONE route.

TOOL:
Use TOOL for ANY factual information, including:
- current weather, calculations
- latest news, current events
- internet search
- factual questions about real-world topics (e.g., "What is Ganesh Chaturthi?")
- people, history, festivals, companies, countries, places, science, technology, general knowledge

RAG:
Use RAG ONLY for:
- uploaded documents, PDFs, DOCX files, TXT files
- questions specifically asking about uploaded files

DIRECT:
Use DIRECT ONLY for:
- Greetings (Hi, Hello, Good morning)
- Casual conversation (How are you?)
- Creative writing (Write a poem)

Do NOT use DIRECT for any questions about festivals, history, or real-world facts.

Return ONLY one word:
tool
or
rag
or
direct
"""

    response = router_llm.invoke(prompt)
    route = response.content.strip().lower()

    if "tool" in route:
        final_route = "tool"
    elif "rag" in route:
        final_route = "rag"
    else:
        final_route = "direct"

    return {"route": final_route, "sources": []}

# ==========================================
# 11. ROUTE AFTER ROUTER
# ==========================================

def route_after_router(state: State):

    route = state["route"]

    if route == "tool":

        return "tool"

    if route == "rag":

        return "rag"

    return "direct"


# ==========================================
# 12. TOOL AGENT
# ==========================================


def tool_agent(state: State):

    messages = state["messages"]

    
    system_message = SystemMessage(
        content="""
You are a strictly evidence-based AI assistant.

CRITICAL RULES:
1. FACTUAL QUESTIONS: If the user asks a factual question (e.g., festivals, history, news, concepts, general knowledge), you MUST use the web_search_tool BEFORE answering, EVEN IF YOU ALREADY KNOW THE ANSWER. Do NOT answer from internal memory.
2. GREETINGS & CASUAL CHAT (SAFETY VALVE): If the user is just greeting you (e.g., "Hi", "Hello") or making casual conversation, DO NOT use any tools. Just respond politely and naturally.

Tool rules:
- Use weather_tool for current weather.
- Use calculate_math for calculations.
- Use web_search_tool for factual web information.

When web_search_tool is used:
- Use the returned search results as evidence.
- Answer the user's question using those results.
- Do not invent facts or URLs.
- Do not write markdown source links in your response. The application will handle the display of URLs.

After using a tool, provide only the natural-language answer.
"""
    )

    
    response = llm_with_tools.invoke(
        [system_message] + messages
    )

    return {
        "messages": [response]
    }


# ==========================================
# 13. ROUTE AFTER TOOL AGENT
# ==========================================

def route_after_tool_agent(state: State):

    last_message = state["messages"][-1]


    if getattr(
        last_message,
        "tool_calls",
        None
    ):

        return "tools"


    return "end"


# ==========================================
# 14. DIRECT ANSWER
# ==========================================

def direct_answer(state: State):

    question = state["messages"][-1].content

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question clearly and naturally.

Rules:

1. Answer conversational questions directly.
2. Do not invent sources.
3. Do not create citations.
4. Do not create URLs.
5. Do not create localhost links.
6. Do not create markdown source links.
7. Do not output HTML or SVG.

USER QUESTION:

{question}
"""

    response = llm.invoke(prompt)

    return {
        "messages": [response],
        "sources": []
    }


# ==========================================
# 15. RETRIEVE DOCUMENTS
# ==========================================

def retrieve_documents(state: State):

    question = state["messages"][-1].content


    documents = retriever.invoke(

        question

    )


    if not documents:

        return {

            "retrieved_docs": [],

            "rag_context": "",

            "sources": []

        }


    context_parts = []

    sources = []

    seen_sources = set()


    for document in documents:

        file_name = document.metadata.get(

            "file_name",

            "Unknown document"

        )


        page = document.metadata.get(

            "page"

        )


        # ------------------------------
        # Source text for LLM
        # ------------------------------

        if page is not None:

            source_text = (

                f"Source: {file_name}, "

                f"Page: {page + 1}"

            )

        else:

            source_text = (

                f"Source: {file_name}"

            )


        context_parts.append(

            f"{source_text}\n"
            f"{document.page_content}"

        )


        # ------------------------------
        # Unique source
        # ------------------------------

        if page is not None:

            source_key = (

                file_name,

                page + 1

            )

        else:

            source_key = (

                file_name,

                None

            )


        if source_key not in seen_sources:

            seen_sources.add(

                source_key

            )


            source_data = {

                "type": "document",

                "file_name": file_name

            }


            if page is not None:

                source_data["page"] = page + 1


            sources.append(

                source_data

            )


    context = "\n\n---\n\n".join(

        context_parts

    )


    return {

        "retrieved_docs": documents,

        "rag_context": context,

        "sources": sources

    }


# ==========================================
# 16. GRADE DOCUMENTS
# ==========================================

def grade_documents(state: State):

    question = state["messages"][-1].content


    context = state.get(

        "rag_context",

        ""

    )


    if not context.strip():

        return {

            "retrieval_relevant": False

        }


    prompt = f"""
You are a document relevance grader.

User question:
{question}

Retrieved document context:
{context}

Does the retrieved context contain useful
information for answering the user question?

Return ONLY:

yes

or

no

Do not explain.
"""


    response = grader_llm.invoke(

        prompt

    )


    result = response.content.strip().lower()


    if "yes" in result:

        relevant = True

    else:

        relevant = False


    return {

        "retrieval_relevant": relevant

    }


# ==========================================
# 17. ROUTE AFTER GRADING
# ==========================================

def route_after_grading(state: State):

    if state["retrieval_relevant"]:

        return "generate"

    return "web_fallback"


# ==========================================
# 18. GENERATE ANSWER FROM RAG
# ==========================================

def generate_from_rag(state: State):

    question = state["messages"][-1].content

    context = state["rag_context"]


    prompt = f"""
You are an AI assistant answering
questions from uploaded documents.

Answer using the document context.

Rules:

1. Do not invent information.

2. Use only the provided document context.

3. If the answer is not present,
   say that it is not available
   in the uploaded documents.

4. Give a clear and useful answer.

5. Do not create or invent sources.

6. Do not create URLs.

7. Do not output HTML or SVG.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
"""


    response = llm.invoke(

        prompt

    )


    return {

        "messages": [response]

    }


# ==========================================
# 19. WEB FALLBACK
# ==========================================

def web_fallback(state: State):

    question = state["messages"][-1].content


    # --------------------------------------
    # Run our custom web search tool
    # --------------------------------------

    search_result = web_search_tool.invoke(

        question

    )


    # --------------------------------------
    # Parse JSON results
    # --------------------------------------

    web_sources = []

    try:

        parsed_results = json.loads(

            search_result

        )

    except Exception:

        parsed_results = []


    if isinstance(
        parsed_results,
        list
    ):

        for result in parsed_results:

            if not isinstance(
                result,
                dict
            ):

                continue


            title = result.get(

                "title",

                "Web source"

            )


            url = result.get(

                "url",

                ""

            )


            snippet = result.get(

                "snippet",

                ""

            )


            if url:

                web_sources.append({

                    "type": "web",

                    "title": title,

                    "url": url,

                    "snippet": snippet

                })


    # --------------------------------------
    # LLM prompt
    # --------------------------------------

    prompt = f"""
You are an AI assistant.

The uploaded documents did not contain
sufficient information.

Answer the user's question using
the web search results below.

WEB SEARCH RESULTS:

{search_result}

USER QUESTION:

{question}

Rules:

1. Use the search results as evidence.

2. Do not invent facts.

3. If the search results do not provide
   enough information, clearly say so.

4. Give a concise and useful answer.

5. Do not create or invent URLs.

6. Do not create fake citations.

7. Do not output HTML or SVG.

8. Do not output localhost links.

9. Do not write markdown source links.
   The application will display the
   real source URLs separately.
"""


    response = llm.invoke(

        prompt

    )


    return {

        "messages": [response],

        "sources": web_sources

    }


# ==========================================
# 20. COLLECT TOOL SOURCES
# ==========================================
# This is the IMPORTANT part.
#
# After the tool executes, LangGraph creates
# a ToolMessage.
#
# We read that ToolMessage and extract:
#
# title
# url
# snippet
#
# Then save them into State["sources"].
# ==========================================

def collect_tool_sources(state: State):

    sources = []
    seen_urls = set()

    msgs = state["messages"]

    # Only messages after the latest user question
    last_human = max(
        i for i, m in enumerate(msgs)
        if isinstance(m, HumanMessage)
    )

    for message in msgs[last_human + 1:]:

        if not isinstance(message, ToolMessage):
            continue

        if getattr(message, "name", "") != "web_search_tool":
            continue

        content = message.content
        results = []

        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    results = parsed
            except Exception:
                results = []

        elif isinstance(content, list):
            results = content

        for result in results:

            if not isinstance(result, dict):
                continue

            title = result.get("title", "Web source")
            url = result.get("url", result.get("href", ""))
            snippet = result.get("snippet", result.get("body", ""))

            if not url or url in seen_urls:
                continue

            seen_urls.add(url)

            sources.append({
                "type": "web",
                "title": title,
                "url": url,
                "snippet": snippet
            })

    return {"sources": sources}


# ==========================================
# 21. CREATE GRAPH
# ==========================================

graph = StateGraph(State)


# ==========================================
# 22. ADD NODES
# ==========================================

graph.add_node(

    "router",

    route_question

)


graph.add_node(

    "tool_agent",

    tool_agent

)


graph.add_node(

    "tools",

    ToolNode(tools)

)


graph.add_node(

    "collect_tool_sources",

    collect_tool_sources

)


graph.add_node(

    "direct",

    direct_answer

)


graph.add_node(

    "retrieve",

    retrieve_documents

)


graph.add_node(

    "grade_documents",

    grade_documents

)


graph.add_node(

    "generate",

    generate_from_rag

)


graph.add_node(

    "web_fallback",

    web_fallback

)


# ==========================================
# 23. START → ROUTER
# ==========================================

graph.add_edge(

    START,

    "router"

)


# ==========================================
# 24. ROUTER → NEXT NODE
# ==========================================

graph.add_conditional_edges(

    "router",

    route_after_router,

    {

        "tool": "tool_agent",

        "rag": "retrieve",

        "direct": "direct"

    }

)


# ==========================================
# 25. TOOL AGENT → TOOLS / END
# ==========================================

graph.add_conditional_edges(

    "tool_agent",

    route_after_tool_agent,

    {

        "tools": "tools",

        "end": END

    }

)


# ==========================================
# 26. TOOLS → SOURCE COLLECTOR
# ==========================================

graph.add_edge(

    "tools",

    "collect_tool_sources"

)


# ==========================================
# 27. SOURCE COLLECTOR → TOOL AGENT
# ==========================================

graph.add_edge(

    "collect_tool_sources",

    "tool_agent"

)


# ==========================================
# 28. RAG RETRIEVE → GRADER
# ==========================================

graph.add_edge(

    "retrieve",

    "grade_documents"

)


# ==========================================
# 29. GRADER → GENERATE / WEB
# ==========================================

graph.add_conditional_edges(

    "grade_documents",

    route_after_grading,

    {

        "generate": "generate",

        "web_fallback": "web_fallback"

    }

)


# ==========================================
# 30. FINAL NODES
# ==========================================

graph.add_edge(

    "generate",

    END

)


graph.add_edge(

    "web_fallback",

    END

)


graph.add_edge(

    "direct",

    END

)


# ==========================================
# 31. MEMORY
# ==========================================

checkpointer = MemorySaver()


# ==========================================
# 32. COMPILE
# ==========================================

workflow = graph.compile(

    checkpointer=checkpointer

)