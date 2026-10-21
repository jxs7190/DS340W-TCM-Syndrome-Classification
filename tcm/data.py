"""Dataset provenance, explicit adapters, and leakage-resistant partitions."""
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

REVISION = "b3ebee62d3adf2ee15a93c2af85fcb85963e6146"
FILES = {"stomach": "胃部疾病数据.xlsx", "kidney": "肾衰数据(1).xls"}
CHECKSUMS = {
    "stomach": "ed5b61802a325dab6a7f67c069edecf435c33667ad825e82bf148e5e2487ff60",
    "kidney": "03365a85b3226f447700406d25afcd7523993ee515be83b2825ee91d7e8c2a58",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tokens(text):
    """Split symptom delimiters; preserve negation and compound symptom phrases."""
    return sorted(set(p.strip() for p in re.split(r"[、，,;；$\n|]+", str(text)) if p.strip()))


def normalize(text):
    return "、".join(tokens(text))


def download(name, root=Path("data/raw")):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / FILES[name]
    url = f"https://raw.githubusercontent.com/web333panda/TCM-Dataset/{REVISION}/" + urllib.parse.quote(FILES[name])
    if not path.exists():
        temp = path.with_suffix(path.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                temp.write_bytes(response.read())
            pd.read_excel(temp, engine="xlrd" if name == "kidney" else "openpyxl")
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
    if sha256(path) != CHECKSUMS[name]:
        raise ValueError(f"{path} does not match the pinned source file. Move it aside before downloading again.")
    return path, {"source_url": url, "source_commit": REVISION, "sha256": sha256(path)}


def load_cases(name, input_path=None, citation=None):
    if input_path:
        if not citation:
            raise ValueError("Imported data require --citation with the actual source/permission reference.")
        path = Path(input_path)
        raw = pd.read_csv(path, dtype=str) if path.suffix.lower() == ".csv" else pd.read_excel(path, dtype=str)
        needed = {"case_id", "source_group", "symptoms", "syndromes"}
        if not needed <= set(raw):
            raise ValueError(f"Import needs columns {sorted(needed)}; syndromes are separated by |.")
        provenance = {"citation": citation, "sha256": sha256(path), "adapter": "canonical-v1"}
    else:
        path, provenance = download(name)
        raw = pd.read_excel(path, dtype=str)
        if name == "stomach":
            raw = raw.rename(columns={"编号": "case_id", "症状": "symptoms", "症候": "syndromes"})
            raw["source_group"] = raw["case_id"]
        else:
            # A paper/source can describe several symptom patterns. ID is a row ID,
            # NOT evidence of a unique patient. Group patterns within source article.
            raw = raw.rename(columns={"YW_ID": "source_group", "ZZMC": "symptoms", "ZHMC": "syndromes"})
            raw["case_id"] = raw["source_group"].astype(str) + ":" + raw["symptoms"].fillna("").map(normalize)
    raw_rows = len(raw)
    raw = raw.dropna(subset=["case_id", "source_group", "symptoms", "syndromes"]).copy()
    for col in ["case_id", "source_group", "symptoms", "syndromes"]:
        raw[col] = raw[col].astype(str).str.strip()
    raw["symptoms"] = raw["symptoms"].map(normalize)
    raw = raw[(raw[["case_id", "source_group", "symptoms", "syndromes"]] != "").all(axis=1)]
    if raw.groupby("case_id")["source_group"].nunique().max() > 1:
        raise ValueError("A case_id maps to multiple source_group values; fix the import.")
    cases = raw.groupby("case_id", sort=True).agg(
        symptoms=("symptoms", lambda s: normalize("、".join(s))),
        syndromes=("syndromes", lambda s: sorted(set(label.strip() for value in s for label in value.split("|") if label.strip()))),
        source_group=("source_group", "first"),
    ).reset_index()
    if cases.syndromes.map(len).eq(0).any():
        raise ValueError("Every case must contain at least one nonempty syndrome label.")
    if len(cases) < 20:
        raise ValueError("At least 20 usable cases are required for this workflow.")
    # Connected components: same source OR identical normalized symptom set.
    # This avoids duplicates crossing splits even when IDs differ.
    parent = list(range(len(cases)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for column in ["source_group", "symptoms"]:
        seen = {}
        for i, value in enumerate(cases[column]):
            if value in seen:
                parent[find(i)] = find(seen[value])
            else:
                seen[value] = i
    cases["group"] = [str(find(i)) for i in range(len(cases))]
    provenance.update(raw_rows=raw_rows, usable_rows=len(raw), cases=len(cases), groups=cases.group.nunique(),
                      label_count=len(set(x for row in cases.syndromes for x in row)))
    return cases, provenance


def make_splits(cases, seed=0):
    if cases.group.nunique() < 10:
        raise ValueError("Too few independent source/duplicate groups for a 70/20/10 experiment.")
    development, final = next(GroupShuffleSplit(n_splits=1, test_size=.1, random_state=seed).split(cases, groups=cases.group))
    dev = cases.iloc[development]
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=2/9, random_state=seed).split(dev, groups=dev.group))
    parts = {"train": dev.iloc[tr].copy(), "test": dev.iloc[te].copy(), "validation": cases.iloc[final].copy()}
    for a, b in [("train", "test"), ("train", "validation"), ("test", "validation")]:
        assert not set(parts[a].group) & set(parts[b].group)
        assert not set(parts[a].symptoms) & set(parts[b].symptoms)
    return parts


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
