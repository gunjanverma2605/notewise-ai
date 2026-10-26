"""Exceptions with messages that are safe to show directly to a student."""


class NoteWiseError(Exception):
    """Base class for all NoteWise errors."""


class DocumentError(NoteWiseError):
    """The uploaded file could not be read (empty, corrupt, locked, unsupported...)."""


class OllamaUnavailableError(NoteWiseError):
    """Ollama is not running, or a required model has not been downloaded."""
