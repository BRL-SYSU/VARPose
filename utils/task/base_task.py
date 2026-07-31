"""
Base Task class for unified task execution.
Similar to BaseTrainer, but for standalone tasks that don't require training.
"""

import argparse
from abc import ABC, abstractmethod
from typing import Any


class BaseTask(ABC):
    """
    Base task class that provides a common interface for task execution.

    Subclasses need to implement:
    - add_parser_args(): Add command line arguments specific to the task
    - run(): Execute the task logic
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize task with arguments.

        Args:
            args: Parsed command line arguments
        """
        self.args = args
        self.task_name = self.__class__.__name__

    @staticmethod
    @abstractmethod
    def add_parser_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add task-specific command line arguments to the argument parser.

        This method should add all arguments that are specific to this task.
        It can call the parent class's add_parser_args method first to get
        common arguments.

        Args:
            parser: ArgumentParser object to add arguments to

        Returns:
            The same ArgumentParser object with added arguments
        """
        pass

    @abstractmethod
    def run(self) -> None:
        """
        Execute the task logic.

        This is the main entry point for the task. All task logic should
        be implemented here.
        """
        pass
