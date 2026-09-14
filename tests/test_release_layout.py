import ast
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
