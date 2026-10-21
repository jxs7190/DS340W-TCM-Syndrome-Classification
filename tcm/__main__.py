"""python -m tcm: train, predict, or explicitly unlock final validation."""
import argparse
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from .data import load_cases, make_splits, sha256, write_json
from .model import MODEL_NAMES, SyndromeModel, encode, evaluate


def build_report(out, comparison, metadata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    comparison.set_index("model")[["micro_f1", "macro_f1", "exact_match_accuracy"]].plot.bar(ax=ax)
    ax.set(ylim=(0, 1), ylabel="Development test score", xlabel="")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)
    html = """<!doctype html><html><meta charset='utf-8'><title>TCM experiment</title>
    <style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px}td,th{padding:8px;text-align:left}table{border-collapse:collapse}tr{border-bottom:1px solid #ddd}img{max-width:100%}</style>
    <h1>TCM syndrome classification</h1><p>Research predictions from symptom text.</p>
    <p>The winner and thresholds were selected using grouped cross-validation inside training only.
    This table is the development test evaluation.</p>"""
    html += f"<p>Selected model: <b>{metadata['winner']}</b>. Dataset: {metadata['dataset']}.</p>"
    html += f"<p>Final validation status: <b>{metadata['validation_status']}</b>.</p>"
    html += comparison.round(4).to_html(index=False, escape=True)
    html += "<img src='comparison.png' alt='Test metrics by model'><p>Accuracy is exact label-set match. Hamming loss can look low when labels are sparse; read F1 and recall too. Macro F1 includes training labels plus any unseen evaluation labels. Source/duplicate groups remain within one partition. Cross-paper scores are not directly comparable.</p></html>"
    (out / "report.html").write_text(html, encoding="utf-8")


def train(args):
    if args.top_k < 1 or args.trees < 1:
        raise ValueError("--top-k and --trees must be positive.")
    cases, provenance = load_cases(args.dataset, args.input, args.citation)
    parts = make_splits(cases, args.seed)
    out = Path(args.output) / args.dataset
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} already contains a run. Choose another --output to preserve the frozen experiment.")
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in parts.items():
        frame.to_json(out / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
    tr, te = parts["train"], parts["test"]
    # Deliberately do not transform, evaluate, or tune using final validation.
    del cases, parts
    labels = sorted(set(label for row in tr.syndromes for label in row))
    y = encode(tr.syndromes.tolist(), labels)
    folds = min(3, tr.group.nunique())
    if folds < 2:
        raise ValueError("Need at least two training groups.")
    comparisons, cv, models = [], {}, {}
    for name in MODEL_NAMES:
        print(f"{args.dataset}: grouped training CV for {name}", flush=True)
        oof = np.zeros(y.shape, dtype=float)
        for fit, held in GroupKFold(n_splits=folds).split(tr, groups=tr.group):
            model = SyndromeModel(name, args.top_k, args.trees, args.seed).fit(
                tr.iloc[fit].symptoms, tr.iloc[fit].syndromes)
            score = model.scores(tr.iloc[held].symptoms)
            positions = [labels.index(c) for c in model.classes]
            oof[np.ix_(held, positions)] = score
        grid = np.round(np.arange(.1, .81, .05), 2)
        values = [float(f1_score(y, oof >= t, average="micro", zero_division=0)) for t in grid]
        best = int(np.argmax(values))
        threshold = float(grid[best])
        cv[name] = {"threshold": threshold, "oof_micro_f1": values[best],
                    "threshold_search": [{"threshold": float(t), "micro_f1": v} for t, v in zip(grid, values)]}
        model = SyndromeModel(name, args.top_k, args.trees, args.seed).fit(tr.symptoms, tr.syndromes)
        models[name] = model
        score = model.scores(te.symptoms)
        metrics, per_label = evaluate(te.syndromes.tolist(), score, model.classes, threshold)
        write_json(out / f"{name}_test_metrics.json", metrics)
        pd.DataFrame(per_label).to_csv(out / f"{name}_per_label.csv", index=False)
        comparisons.append({"model": name, "threshold": threshold, "training_cv_micro_f1": values[best],
                            **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}})
    # Winner is chosen entirely from TRAIN CV, never from test scores.
    winner = max(MODEL_NAMES, key=lambda name: cv[name]["oof_micro_f1"])
    model = models[winner]
    bundle = {"model": model, "threshold": cv[winner]["threshold"], "dataset": args.dataset}
    joblib.dump(bundle, out / "model.joblib")
    score = model.scores(te.symptoms)
    preview = pd.DataFrame({"case_id": te.case_id.tolist(),
        "true_syndromes": [" | ".join(x) for x in te.syndromes],
        "predicted_syndromes": [" | ".join(c for c, s in zip(model.classes, row) if s >= bundle["threshold"]) for row in score]})
    preview.to_csv(out / "test_predictions.csv", index=False)
    features = model.vectorizer.get_feature_names_out()
    pd.DataFrame({"feature": features[model.selected],
                  "ranking_weight": model.weights[model.selected] if model.weights is not None else np.nan}).to_csv(out / "selected_features.csv", index=False)
    metadata = {"dataset": args.dataset, "provenance": provenance, "seed": args.seed,
        "research_question": "How accurately can machine learning classify TCM syndromes from symptom data, and does feature selection improve a multi-output Random Forest?",
        "parameters": {"top_k": args.top_k, "trees": args.trees, "cv_folds": folds},
        "split_counts": {k: len(pd.read_json(out / f"{k}.jsonl", lines=True)) for k in ["train", "test"]},
        "split_policy": "70/20/10 approximate by independent source/duplicate group; final validation unused",
        "winner": winner, "selection_rule": "highest training grouped-OOF micro-F1 after threshold search",
        "validation_status": "UNUSED", "model_sha256": sha256(out / "model.joblib"),
        "validation_sha256": sha256(out / "validation.jsonl"),
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__}}
    # Count only, without reloading held-out records.
    metadata["split_counts"]["validation"] = sum(1 for _ in (out / "validation.jsonl").open(encoding="utf-8"))
    write_json(out / "run.json", metadata)
    write_json(out / "training_cv.json", cv)
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(out / "comparison.csv", index=False)
    build_report(out, comparison, metadata)
    print(comparison[["model", "micro_f1", "macro_f1", "exact_match_accuracy"]].to_string(index=False))
    print(f"Selected from training CV: {winner}. Final validation UNUSED. Report: {out / 'report.html'}")


def validate(args):
    out = Path(args.run)
    if not args.final:
        raise ValueError("Final validation requires --final after the approach is frozen.")
    if (out / "final_validation.json").exists():
        raise ValueError("Final validation was already evaluated for this run.")
    metadata = json.loads((out / "run.json").read_text(encoding="utf-8"))
    for filename, key in [("model.joblib", "model_sha256"), ("validation.jsonl", "validation_sha256")]:
        if sha256(out / filename) != metadata[key]:
            raise ValueError(f"Frozen {filename} changed; validation refused.")
    bundle = joblib.load(out / "model.joblib")
    frame = pd.read_json(out / "validation.jsonl", lines=True)
    model = bundle["model"]
    metrics, per_label = evaluate(frame.syndromes.tolist(), model.scores(frame.symptoms), model.classes, bundle["threshold"])
    write_json(out / "final_validation.json", {"metrics": metrics, "model_sha256": metadata["model_sha256"]})
    pd.DataFrame(per_label).to_csv(out / "final_validation_per_label.csv", index=False)
    metadata["validation_status"] = "EVALUATED_ONCE"
    write_json(out / "run.json", metadata)
    build_report(out, pd.read_csv(out / "comparison.csv"), metadata)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def predict(args):
    bundle = joblib.load(args.model)  # Load only artifacts created by this project.
    scores = bundle["model"].scores([args.symptoms])[0]
    rows = [{"syndrome": c, "score": float(s), "predicted": bool(s >= bundle["threshold"])}
            for c, s in zip(bundle["model"].classes, scores)]
    rows.sort(key=lambda row: -row["score"])
    print(json.dumps({"dataset": bundle["dataset"], "threshold": bundle["threshold"], "top_scores": rows[:args.top]}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train")
    p.add_argument("--dataset", choices=["stomach", "kidney", "huang", "chen"], default="stomach")
    p.add_argument("--input", help="Authorized canonical CSV/XLSX for imported paper data")
    p.add_argument("--citation")
    p.add_argument("--output", default="runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--trees", type=int, default=200)
    p.set_defaults(action=train)
    p = commands.add_parser("validate")
    p.add_argument("--run", required=True)
    p.add_argument("--final", action="store_true")
    p.set_defaults(action=validate)
    p = commands.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--symptoms", required=True)
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(action=predict)
    args = parser.parse_args()
    if args.command == "train" and args.dataset in {"huang", "chen"} and not args.input:
        parser.error("This paper's original records are not bundled/publicly downloaded. Supply authorized --input and --citation; see docs/DATA_SOURCES.md.")
    try:
        args.action(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
