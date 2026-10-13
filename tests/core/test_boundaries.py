"""Import-boundary checks and a cold-import check for optional dependencies."""

import ast
import subprocess
import sys
from pathlib import Path


def test_cold_import():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import robopolicy; assert not ({'torch','cv2','torchcodec','h5py','wandb','transformers','torchvision'} & sys.modules.keys())",
        ],
        check=True,
    )


def test_dependency_boundaries():
    root = Path(__file__).parents[2] / "src/robopolicy"
    forbidden = {
        "contracts": {
            "datasets",
            "modules",
            "policies",
            "training",
            "checkpoints",
            "config",
            "processing",
        },
        "algorithms": {"datasets", "training", "policies", "checkpoints", "config"},
        "modules": {"datasets", "training", "policies", "checkpoints", "config"},
        "datasets": {"policies", "training"},
        "processing": {"datasets", "training", "policies"},
        "policies": {"datasets", "training"},
    }
    for path in root.rglob("*.py"):
        area = path.relative_to(root).parts[0]
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            imports = []
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            if isinstance(node, ast.ImportFrom):
                imports = [node.module or ""]
            for module in imports:
                assert not module.startswith(
                    (
                        "projects",
                        "corl2026_cg",
                        "policies.",
                        "robomimic",
                        "robot_envs",
                        "robodecision",
                        "isaaclab",
                        "isaacsim",
                        "mujoco",
                    )
                ), (path, module)
                parts = module.split(".")
                if len(parts) > 1 and parts[0] == "robopolicy":
                    assert parts[1] not in forbidden.get(area, set()), (path, module)
