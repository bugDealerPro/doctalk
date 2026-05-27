"""Prompt templates for the QA workflow."""

from __future__ import annotations

QA_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only the provided document excerpts.
If the excerpts do not contain enough information, say you could not find the answer in the document.
Keep answers concise and factual. Do not invent details."""

QA_USER_PROMPT = """Document excerpts:
{context}

Question: {question}

Answer using only the excerpts above."""
