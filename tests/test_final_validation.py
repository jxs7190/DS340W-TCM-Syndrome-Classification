"""Exercise finalization using artificial unit-test fixtures, not clinical holdouts."""
from argparse import Namespace
import json

import pandas as pd
import pytest

from tcm.__main__ import train, validate


def test_freeze_and_one_time_validation_on_fixture(tmp_path):
    source = tmp_path / "fixture.csv"
    pd.DataFrame([{"case_id": str(i), "source_group": str(i),
                   "symptoms": f"共同症状、词{i}、组{i % 2}", "syndromes": "A" if i % 2 else "B"}
                  for i in range(40)]).to_csv(source, index=False)
    args = Namespace(dataset="chen", input=source, citation="artificial unit-test fixture only",
                     output=tmp_path / "runs", seed=0, top_k=4, trees=5)
    train(args)
    out = args.output / "chen"
    assert json.loads((out / "run.json").read_text())["validation_status"] == "UNUSED"
    with pytest.raises(ValueError, match="requires --final"):
        validate(Namespace(run=out, final=False))
    model = out / "model.joblib"
    original = model.read_bytes()
    model.write_bytes(original + b"modified")
    with pytest.raises(ValueError, match="changed"):
        validate(Namespace(run=out, final=True))
    model.write_bytes(original)
    validate(Namespace(run=out, final=True))
    assert json.loads((out / "run.json").read_text())["validation_status"] == "EVALUATED_ONCE"
    with pytest.raises(ValueError, match="already evaluated"):
        validate(Namespace(run=out, final=True))
