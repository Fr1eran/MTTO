from __future__ import annotations

import pytest

from mtto.domain.scenario import Scenario, Task
from paper.figures import load_paper_scenario, load_paper_task


@pytest.fixture(scope="session")
def paper_scenario() -> Scenario:
    return load_paper_scenario()


@pytest.fixture(scope="session")
def paper_task() -> Task:
    return load_paper_task()
