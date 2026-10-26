"""Builds the app's long-lived objects in one place.

The Streamlit app caches the result, so the database, vector index and model
clients are created once per server, not on every rerun. Tests replace
build_services() with a version that uses fake models.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.conversation_manager import ConversationManager
from src.database import Database
from src.embeddings import OllamaEmbedder
from src.gemma_client import GemmaClient, HealthStatus, check_health
from src.retrieval import NoteIndex


@dataclass
class Services:
    db: Database
    index: NoteIndex
    manager: ConversationManager
    gemma: GemmaClient


def build_services() -> Services:
    db = Database()
    index = NoteIndex(OllamaEmbedder())
    return Services(db=db, index=index, manager=ConversationManager(db, index), gemma=GemmaClient())


def health() -> HealthStatus:
    return check_health()
