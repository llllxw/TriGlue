import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_root_files_contain_implementations():
    expected = {"data_process.py": {"prepare_input", "compound_graph", "protein_graph", "esm2_embeddings"},
                "Dataset.py": {"PreparedFeatureDataset"},
                "model.py": {"TriComplexClassifier", "AblationConfig"}}
    for filename, names in expected.items():
        tree = ast.parse((ROOT / filename).read_text())
        actual = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        assert names <= actual
    assert not (ROOT / "triglue").exists()


def test_export_excludes_runtime_outputs(tmp_path):
    spec = importlib.util.spec_from_file_location("export_release", ROOT / "scripts/export_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("model.py", "results/scores.csv", "logs/run.log", "checkpoints/model.pth",
                 "tests/fixtures/mock.csv", "tests/__pycache__/test.pyc", "docs/MODEL_CARD.md"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    selected = {str(path.relative_to(tmp_path)) for path in module.source_files(tmp_path)}
    assert selected == {"model.py", "tests/fixtures/mock.csv", "docs/MODEL_CARD.md"}
