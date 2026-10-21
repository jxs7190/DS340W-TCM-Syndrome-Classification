import joblib
import numpy as np
import pandas as pd
import pytest

from tcm.data import make_splits, normalize, load_cases
from tcm.model import SyndromeModel, evaluate


def test_group_and_duplicate_split():
    rows = pd.DataFrame({"case_id": [str(i) for i in range(100)], "symptoms": [f"症状{i//2}" for i in range(100)],
                         "group": [str(i//2) for i in range(100)], "syndromes": [["A"]]*100})
    parts = make_splits(rows)
    assert sum(len(p) for p in parts.values()) == 100
    assert len(parts["train"]) == 70
    for a, b in [("train", "test"), ("train", "validation"), ("test", "validation")]:
        assert not set(parts[a].group) & set(parts[b].group)
        assert not set(parts[a].symptoms) & set(parts[b].symptoms)


@pytest.mark.parametrize("name", ["dummy", "logistic", "rf_all", "rf_selected"])
def test_constant_labels_train_only_vocabulary_and_serialization(name, tmp_path):
    model = SyndromeModel(name, trees=5, top_k=2).fit(
        ["腹痛、口干", "腹痛、恶心", "发热、口干", "发热、恶心"], [["A", "ALWAYS"], ["A", "ALWAYS"], ["B", "ALWAYS"], ["B", "ALWAYS"]])
    score = model.scores(["全新症状"])
    assert "全新症状" not in model.vectorizer.vocabulary_
    assert np.isfinite(score).all()
    assert score.shape == (1, 3)
    assert score[0, model.classes.index("ALWAYS")] == 1
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(score, joblib.load(path).scores(["全新症状"]))


def test_unseen_test_label_is_false_negative():
    metrics, rows = evaluate([["A", "UNSEEN"]], np.array([[.9]]), ["A"], .5)
    assert metrics["micro_recall"] == .5
    assert metrics["exact_match_accuracy"] == 0
    assert metrics["unseen_positive_labels"] == 1
    assert rows[1]["fn"] == 1


def test_normalization_preserves_negation():
    assert normalize("无发热$腹痛、腹痛") == "无发热、腹痛"


def test_import_connects_same_source_and_duplicate_patterns(tmp_path):
    rows = [{"case_id": str(i), "source_group": str(i), "symptoms": f"症状{i}", "syndromes": "A|B"} for i in range(30)]
    rows[1]["source_group"] = "0"
    rows[2]["symptoms"] = rows[1]["symptoms"]
    file = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(file, index=False)
    cases, _ = load_cases("chen", file, "authorized source")
    first = cases[cases.case_id.isin(["0", "1", "2"])]
    assert first.group.nunique() == 1
    assert first.syndromes.iloc[0] == ["A", "B"]
