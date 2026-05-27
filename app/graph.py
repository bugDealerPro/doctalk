"""LangGraph workflow for document-grounded question answering."""

from __future__ import annotations

import logging
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.config import Settings
from app.prompts import QA_SYSTEM_PROMPT, QA_USER_PROMPT
from app.vector_store import (
    RetrievedChunk,
    get_document_for_session,
    preview_chunk,
    retrieve_relevant_chunks,
    select_citation_chunks,
)

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    session_id: str
    question: str
    document_id: str | None
    filename: str | None
    chunks: list[RetrievedChunk]
    answer: str
    sources: list[str]
    error: str | None


def _build_llm(settings: Settings) -> ChatOpenAI:
    kwargs: dict[str, str] = {
        "model": settings.llm_model,
        "api_key": settings.openai_api_key,
        "temperature": 0.0,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(**kwargs)


def validate_document(state: GraphState, settings: Settings) -> GraphState:
    document = get_document_for_session(state["session_id"], settings)
    if document is None:
        return {
            **state,
            "document_id": None,
            "filename": None,
            "error": "Please upload a PDF before asking questions.",
        }
    return {
        **state,
        "document_id": document.id,
        "filename": document.filename,
        "error": None,
    }


def retrieve_chunks(state: GraphState, settings: Settings) -> GraphState:
    if state.get("error") or not state.get("document_id"):
        return state

    chunks = retrieve_relevant_chunks(
        document_id=state["document_id"],
        question=state["question"],
        settings=settings,
    )
    if not chunks:
        return {
            **state,
            "chunks": [],
            "error": "No relevant passages were found in the uploaded document.",
        }
    return {**state, "chunks": chunks, "error": None}


def generate_answer(state: GraphState, settings: Settings) -> GraphState:
    if state.get("error"):
        return {**state, "answer": state["error"], "sources": []}

    context = "\n\n".join(
        f"[Excerpt {chunk.chunk_index + 1}]\n{chunk.content}"
        for chunk in state["chunks"]
    )
    prompt = QA_USER_PROMPT.format(context=context, question=state["question"])
    llm = _build_llm(settings)
    response = llm.invoke(
        [SystemMessage(content=QA_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    answer = str(response.content).strip()
    cited_chunks = select_citation_chunks(state["chunks"], state["question"])
    sources = [
        f"Chunk {chunk.chunk_index + 1}: {preview_chunk(chunk.content, state['question'])}"
        for chunk in cited_chunks
    ]
    logger.info(
        "Generated answer for session %s using %d chunks",
        state["session_id"],
        len(state["chunks"]),
    )
    return {**state, "answer": answer, "sources": sources, "error": None}


def should_continue_after_validation(state: GraphState) -> str:
    if state.get("error"):
        return "finish"
    return "retrieve"


def should_continue_after_retrieval(state: GraphState) -> str:
    if state.get("error"):
        return "finish"
    return "generate"


def build_qa_graph(settings: Settings):
    graph = StateGraph(GraphState)

    graph.add_node(
        "validate",
        lambda state: validate_document(state, settings),
    )
    graph.add_node(
        "retrieve",
        lambda state: retrieve_chunks(state, settings),
    )
    graph.add_node(
        "generate",
        lambda state: generate_answer(state, settings),
    )

    graph.set_entry_point("validate")
    graph.add_conditional_edges(
        "validate",
        should_continue_after_validation,
        {"retrieve": "retrieve", "finish": "generate"},
    )
    graph.add_conditional_edges(
        "retrieve",
        should_continue_after_retrieval,
        {"generate": "generate", "finish": "generate"},
    )
    graph.add_edge("generate", END)

    return graph.compile()


def run_qa_workflow(
    *,
    session_id: str,
    question: str,
    settings: Settings,
) -> GraphState:
    workflow = build_qa_graph(settings)
    initial_state: GraphState = {
        "session_id": session_id,
        "question": question,
        "document_id": None,
        "filename": None,
        "chunks": [],
        "answer": "",
        "sources": [],
        "error": None,
    }
    return workflow.invoke(initial_state)
