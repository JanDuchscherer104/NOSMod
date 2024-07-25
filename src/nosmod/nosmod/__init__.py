from rich.console import Console

from .sqrt_filter.raised_cosine_filter import RaisedCosineParams

CONSOLE = Console(width=120)

__all__ = ["RaisedCosineParams", "CONSOLE"]
