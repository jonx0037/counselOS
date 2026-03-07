"""
BaseAgent: contract that all CounselOS agents must implement.

Each agent:
  - Receives the current MatterState
  - Does one bounded job
  - Returns the updated MatterState
  - Never communicates with other agents directly
"""
from abc import ABC, abstractmethod

from core.llm.base import BaseLLMProvider
from orchestrator.state import MatterState


class BaseAgent(ABC):
    def __init__(self, llm: BaseLLMProvider) -> None:
        self.llm = llm

    @abstractmethod
    def run(self, state: MatterState) -> MatterState:
        """Execute this agent's task and return updated state."""
        ...
