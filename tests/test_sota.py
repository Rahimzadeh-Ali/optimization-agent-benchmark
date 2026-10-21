import importlib

from optbench.registry import PROBLEMS
from optbench.sota import SOTA_METHODS, list_sota_methods
from optbench.tasks.common import TASK_NUMBERS


def test_every_task_has_exactly_one_sota_method() -> None:
    assert set(SOTA_METHODS) == set(PROBLEMS)
    assert len(list_sota_methods()) == 30


def test_all_numbered_task_modules_exist() -> None:
    assert set(TASK_NUMBERS) == set(PROBLEMS)
    for number in range(1, 31):
        importlib.import_module(f"optbench.tasks.task{number:03d}")
