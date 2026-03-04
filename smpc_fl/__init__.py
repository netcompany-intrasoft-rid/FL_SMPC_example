"""SMPC Federated Learning Package."""

from .client_app import app as client_app
from .server_app import app as server_app

__version__ = "1.0.0"
__all__ = ["client_app", "server_app"]
