"""Checks for package star imports."""

import ast
import importlib
from pathlib import Path

import pytest


def _packages_with_all():
    package_root = Path(__file__).resolve().parents[1] / "src" / "genjax"
    for init_path in sorted(package_root.rglob("__init__.py")):
        tree = ast.parse(init_path.read_text())
        if any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            )
            for node in tree.body
        ):
            parts = init_path.parent.relative_to(package_root).parts
            yield ".".join(("genjax", *parts))


@pytest.mark.parametrize("module_name", list(_packages_with_all()))
def test_star_import_resolves_every_export(module_name):
    module = importlib.import_module(module_name)
    namespace = {}

    exec(f"from {module_name} import *", namespace)

    for name in module.__all__:
        assert namespace[name] is getattr(module, name)


def test_root_star_import_excludes_lazy_plotting():
    import genjax

    assert {"viz", "raincloud", "horizontal_raincloud"}.isdisjoint(genjax.__all__)
