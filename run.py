#!/usr/bin/env python
"""
Unified training entry script.
Dynamically selects and configures different Trainers or Tasks based on command-line arguments.
"""
import argparse
import sys
import os
import importlib
import inspect


def discover_trainers():
    """
    Automatically discover all Trainer classes under the utils/train folder.

    Returns:
        dict: A dictionary of the form {trainer_name: (TrainerClass, pretty_name)}.
    """
    trainers = {}
    trainer_dir = os.path.join(os.path.dirname(__file__), 'utils', 'train')

    if not os.path.exists(trainer_dir):
        return trainers

    for filename in os.listdir(trainer_dir):
        if filename.endswith('.py') and not filename.startswith('__') and filename != 'base_trainer.py' and filename != '__init__.py':
            module_name = filename[:-3]
            try:
                module = importlib.import_module(f'utils.train.{module_name}')
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Filter out BaseTrainer; only find concrete Trainer subclasses
                    if (obj.__module__ == f'utils.train.{module_name}' and
                        hasattr(obj, 'add_parser_args') and
                        name != 'BaseTrainer' and
                        not name.endswith('BaseTrainer')):
                        # Extract trainer name: DensePoseVQVAETrainer -> vqvae
                        class_name = obj.__name__
                        # Remove common suffixes
                        for suffix in ['Trainer', 'DensePose']:
                            class_name = class_name.replace(suffix, '')
                        # Convert to lowercase
                        trainer_name = class_name.lower()
                        # Avoid duplicates (a single module may contain multiple classes)
                        if trainer_name and trainer_name not in trainers:
                            trainers[trainer_name] = (obj, obj.__name__)
            except Exception as e:
                print(f"Warning: Failed to import trainer from {filename}: {e}")

    return trainers


def discover_tasks():
    """
    Automatically discover all Task classes under the utils/task folder.

    Returns:
        dict: A dictionary of the form {task_name: (TaskClass, pretty_name)}.
    """
    tasks = {}
    task_dir = os.path.join(os.path.dirname(__file__), 'utils', 'task')

    if not os.path.exists(task_dir):
        return tasks

    for filename in os.listdir(task_dir):
        if filename.endswith('.py') and not filename.startswith('__') and filename != 'base_task.py':
            module_name = filename[:-3]
            try:
                module = importlib.import_module(f'utils.task.{module_name}')
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if obj.__module__ == f'utils.task.{module_name}' and hasattr(obj, 'add_parser_args') and hasattr(obj, 'run'):
                        task_name = module_name
                        tasks[task_name] = (obj, obj.__name__)
            except Exception as e:
                print(f"Warning: Failed to import task from {filename}: {e}")

    return tasks


def get_parser():
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Unified training script for DensePose models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        allow_abbrev=False
    )

    # Discover all available trainers and tasks
    available_trainers = discover_trainers()
    available_tasks = discover_tasks()
    trainer_choices = list(available_trainers.keys())
    task_choices = list(available_tasks.keys())

    # Mutually exclusive group: choose either a trainer or a task
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--trainer',
        type=str,
        choices=trainer_choices,
        help=f'Trainer type to use: {", ".join(trainer_choices)}'
    )
    mode_group.add_argument(
        '--task',
        type=str,
        choices=task_choices,
        help=f'Task type to use: {", ".join(task_choices)}'
    )

    parser.add_argument('-h', '--help', action='store_true', help='Show this help message and exit')

    return parser


def main():
    """Main function."""
    parser = get_parser()
    args, _ = parser.parse_known_args()

    show_help = args.help
    is_trainer_mode = args.trainer is not None

    if is_trainer_mode:
        # Trainer mode
        trainer_name = args.trainer
        available_trainers = discover_trainers()

        if show_help and not trainer_name:
            print("Error: --trainer is required to show specific help")
            print(f"Available trainers: {', '.join(available_trainers.keys())}")
            print("\nExample: python run.py --trainer vqvae --help")
            sys.exit(1)

        if trainer_name not in available_trainers:
            raise ValueError(f"Unknown trainer: {trainer_name}. Available trainers: {list(available_trainers.keys())}")

        TrainerClass, trainer_name_pretty = available_trainers[trainer_name]
        parser = TrainerClass.add_parser_args(parser)

        if show_help:
            parser.print_help()
            sys.exit(0)

        args = parser.parse_args()

        print("=" * 80)
        print(f"Training with: {trainer_name_pretty}")
        print("=" * 80)
        print(f"Command: {' '.join(sys.argv)}")
        print("=" * 80)

        trainer = TrainerClass(args)
        trainer.run()

    else:
        # Task mode
        task_name = args.task
        available_tasks = discover_tasks()

        if show_help and not task_name:
            print("Error: --task is required to show specific help")
            print(f"Available tasks: {', '.join(available_tasks.keys())}")
            print("\nExample: python run.py --task example_task --help")
            sys.exit(1)

        if task_name not in available_tasks:
            raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(available_tasks.keys())}")

        TaskClass, task_name_pretty = available_tasks[task_name]
        parser = TaskClass.add_parser_args(parser)

        if show_help:
            parser.print_help()
            sys.exit(0)

        args = parser.parse_args()

        print("=" * 80)
        print(f"Running task: {task_name_pretty}")
        print("=" * 80)
        print(f"Command: {' '.join(sys.argv)}")
        print("=" * 80)

        task = TaskClass(args)
        task.run()

if __name__ == '__main__':
    main()