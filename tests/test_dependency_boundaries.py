import ast
import sys
from pathlib import Path

import pytest

DOMAIN_MODULES = [
    "src/mtto/domain/_numerics.py",
    "src/mtto/domain/line.py",
    "src/mtto/domain/dynamics.py",
    "src/mtto/domain/kinematics.py",
    "src/mtto/domain/energy.py",
    "src/mtto/domain/srtsp.py",
    "src/mtto/domain/scenario.py",
    "src/mtto/domain/speed_profile.py",
    "src/mtto/domain/safeguard/__init__.py",
    "src/mtto/domain/safeguard/curves.py",
    "src/mtto/domain/safeguard/dynamic_limits.py",
    "src/mtto/domain/safeguard/static_region.py",
    "src/mtto/domain/safeguard/geometry.py",
]

EVALUATION_MODULES = ["src/mtto/evaluation/quality.py"]


def _get_imported_modules(file_path: Path) -> list[str]:
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module)
    return modules


def test_library_does_not_import_paper_only_dependencies() -> None:
    forbidden = {"matplotlib", "pandas", "openpyxl"}
    for file_path in Path("src/mtto").rglob("*.py"):
        for module in _get_imported_modules(file_path):
            assert module.split(".")[0] not in forbidden, (
                f"{file_path} imports paper-only dependency {module}"
            )


@pytest.mark.parametrize("module_path", [*DOMAIN_MODULES, *EVALUATION_MODULES])
def test_domain_module_dependency_boundaries(module_path: str) -> None:
    file_path = Path(module_path)
    imported_modules = _get_imported_modules(file_path)

    for mod in imported_modules:
        top_level = mod.split(".")[0]
        if top_level == "numpy" or (
            top_level == "numba" and module_path in DOMAIN_MODULES
        ):
            continue
        if top_level in sys.stdlib_module_names or top_level == "__future__":
            continue
        if mod.startswith("mtto.domain"):
            if (
                "scenario" in mod
                and module_path in DOMAIN_MODULES
                and file_path != Path("src/mtto/domain/scenario.py")
            ):
                pytest.fail(
                    f"Domain module {module_path} imports mtto.domain.scenario, "
                    "which is only allowed in scenario.py."
                )
            continue
        pytest.fail(
            f"Module {module_path} imports disallowed dependency: {mod}. "
            "Domain modules may only import numpy, numba, stdlib and "
            "mtto.domain; evaluation modules may only import numpy, stdlib and "
            "mtto.domain."
        )


def test_rl_state_dependency_boundary() -> None:
    file_path = Path("src/mtto/rl/state.py")
    for module in _get_imported_modules(file_path):
        top_level = module.split(".")[0]
        assert top_level in sys.stdlib_module_names or module.startswith(
            "mtto.domain."
        ), f"{file_path} imports disallowed dependency: {module}"


def test_rl_and_dp_do_not_import_io() -> None:
    for base_dir in ("src/mtto/rl", "src/mtto/dp"):
        for file_path in Path(base_dir).rglob("*.py"):
            for mod in _get_imported_modules(file_path):
                assert not mod.startswith("mtto.io"), f"{file_path} imports {mod}"


def test_rl_and_dp_do_not_import_each_other() -> None:
    for file_path in Path("src/mtto/rl").rglob("*.py"):
        for mod in _get_imported_modules(file_path):
            assert not mod.startswith("mtto.dp"), f"{file_path} imports {mod}"
    for file_path in Path("src/mtto/dp").rglob("*.py"):
        for mod in _get_imported_modules(file_path):
            assert not mod.startswith("mtto.rl"), f"{file_path} imports {mod}"


def test_io_does_not_import_workflows() -> None:
    for file_path in Path("src/mtto/io").rglob("*.py"):
        for mod in _get_imported_modules(file_path):
            assert not mod.startswith("mtto.workflows"), f"{file_path} imports {mod}"


def test_artifacts_module_dependency_boundaries() -> None:
    file_path = Path("src/mtto/io/artifacts.py")
    disallowed = ("gymnasium", "torch", "stable_baselines3")
    for mod in _get_imported_modules(file_path):
        for prefix in disallowed:
            assert not mod.startswith(prefix), (
                f"{file_path} imports disallowed dependency {mod}"
            )


def test_ppo_and_diagnostics_dependency_and_io_boundaries() -> None:
    import re

    ppo_path = Path("src/mtto/rl/ppo.py")
    diag_path = Path("src/mtto/rl/diagnostics.py")
    disallowed_common = ("mtto.io",)

    for file_path in (ppo_path, diag_path):
        for mod in _get_imported_modules(file_path):
            for prefix in disallowed_common:
                assert not mod.startswith(prefix), f"{file_path} imports {mod}"

    for mod in _get_imported_modules(diag_path):
        assert not mod.startswith("stable_baselines3"), f"{diag_path} imports {mod}"

    io_pattern = re.compile(r"savez|open\(|os\.(replace|makedirs)|mkdir|\.save\(")
    for file_path in (ppo_path, diag_path):
        content = file_path.read_text(encoding="utf-8")
        matches = io_pattern.findall(content)
        assert not matches, f"{file_path} contains forbidden I/O operations: {matches}"


def test_workflows_dependency_and_source_boundaries() -> None:
    disallowed = (
        "argparse",
        "matplotlib",
        "paper",
        "app",
    )
    for file_path in Path("src/mtto/workflows").rglob("*.py"):
        content = file_path.read_text(encoding="utf-8")
        assert "print(" not in content, f"{file_path} contains forbidden 'print('"
        for mod in _get_imported_modules(file_path):
            for prefix in disallowed:
                assert not mod.startswith(prefix), (
                    f"{file_path} imports disallowed dependency {mod}"
                )


def test_rl_does_not_import_workflows() -> None:
    for file_path in Path("src/mtto/rl").rglob("*.py"):
        for mod in _get_imported_modules(file_path):
            assert not mod.startswith("mtto.workflows"), f"{file_path} imports {mod}"


def test_dp_modules_dependency_boundaries() -> None:
    disallowed = ("mtto.io", "mtto.rl", "mtto.workflows")
    for module_name in ("graph.py", "cache.py", "solver.py"):
        file_path = Path("src/mtto/dp") / module_name
        for mod in _get_imported_modules(file_path):
            for prefix in disallowed:
                assert not mod.startswith(prefix), f"{file_path} imports {mod}"


def test_training_analysis_dependency_and_io_boundaries() -> None:
    import re

    base_dir = Path("src/mtto/rl/training_analysis")
    disallowed_modules = ("tensorboard", "mtto.io", "mtto.workflows")
    io_pattern = re.compile(
        r"np\.load|np\.savez|open\(|read_text|write_text|mkdir|csv\.writer|json\.dump"
    )

    for file_path in base_dir.rglob("*.py"):
        for mod in _get_imported_modules(file_path):
            for prefix in disallowed_modules:
                assert not mod.startswith(prefix), (
                    f"{file_path} imports disallowed dependency: {mod}"
                )
        content = file_path.read_text(encoding="utf-8")
        matches = io_pattern.findall(content)
        assert not matches, f"{file_path} contains forbidden I/O operations: {matches}"


def test_tensorboard_io_does_not_import_workflows() -> None:
    file_path = Path("src/mtto/io/tensorboard.py")
    for mod in _get_imported_modules(file_path):
        assert not mod.startswith("mtto.workflows"), (
            f"{file_path} imports disallowed dependency: {mod}"
        )


def test_cli_dependency_boundaries() -> None:
    for file_path in Path("src/mtto").rglob("*.py"):
        if file_path == Path("src/mtto/cli.py"):
            continue
        for mod in _get_imported_modules(file_path):
            assert not (mod == "mtto.cli" or mod.startswith("mtto.cli.")), (
                f"{file_path} imports mtto.cli"
            )

    cli_path = Path("src/mtto/cli.py")
    disallowed = (
        "matplotlib",
        "pandas",
        "paper",
        "app",
    )
    for mod in _get_imported_modules(cli_path):
        for prefix in disallowed:
            assert not mod.startswith(prefix), (
                f"{cli_path} imports disallowed dependency: {mod}"
            )


@pytest.mark.parametrize("forbidden_root", ["paper", "app"])
def test_mtto_does_not_import_paper_or_app(forbidden_root: str) -> None:
    for file_path in Path("src/mtto").rglob("*.py"):
        for module in _get_imported_modules(file_path):
            assert module != forbidden_root and not module.startswith(
                f"{forbidden_root}."
            ), f"{file_path} imports {module}"
