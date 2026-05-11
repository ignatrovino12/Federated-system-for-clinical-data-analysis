import sys
import json
from typing import List

from django.core.management.base import BaseCommand

from clinical_ai import services
import numpy as np


def _param_stats(model) -> dict:
    stats = {}
    for name, param in model.named_parameters():
        arr = param.detach().cpu().numpy()
        stats[name] = {
            "shape": arr.shape,
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return stats


def _sample_predictions(artifacts, n=5):
    cols = artifacts["feature_columns"]
    scaler = artifacts["scaler"]
    model = artifacts["model"]
    device = artifacts["device"]

    # create sample rows: zeros, ones, random, min, max
    sample_base = np.zeros((1, len(cols)), dtype=np.float32)
    sample_one = np.ones((1, len(cols)), dtype=np.float32)
    sample_rand = np.random.RandomState(0).randn(n, len(cols)).astype(np.float32)
    samples = np.vstack([sample_base, sample_one, sample_rand[: n - 2]])

    try:
        scaled = scaler.transform(samples)
    except Exception as e:
        return {"error": f"Scaler transform failed: {e}"}

    import torch
    with torch.no_grad():
        out = model(torch.tensor(scaled, dtype=torch.float32).to(device))
        out_np = out.cpu().numpy().ravel()
        # if outputs are logits (not probabilities), show sigmoid too
        probs = 1 / (1 + np.exp(-out_np))

    rows = []
    for i in range(len(out_np)):
        rows.append({"raw": float(out_np[i]), "sigmoid": float(probs[i])})

    return {"samples": rows}


class Command(BaseCommand):
    help = "Inspect federated model weights and sample predictions for debugging."

    def add_arguments(self, parser):
        parser.add_argument("--models", nargs="*", choices=["alex5050", "mustafa"], default=["alex5050", "mustafa"])

    def handle(self, *args, **options):
        models: List[str] = options["models"]

        out = {}
        if "alex5050" in models:
            try:
                art = services._load_alex_artifacts()
                out_alex = {}
                out_alex["outputs_probability"] = art.get("outputs_probability")
                out_alex["param_stats"] = _param_stats(art["model"]) if art.get("model") is not None else {}
                out_alex["sample_predictions"] = _sample_predictions(art, n=5)
                out["alex5050"] = out_alex
            except Exception as exc:
                out["alex5050"] = {"error": str(exc)}

        if "mustafa" in models:
            try:
                art = services._load_mustafa_artifacts()
                out_m = {}
                out_m["outputs_probability"] = art.get("outputs_probability")
                out_m["param_stats"] = _param_stats(art["model"]) if art.get("model") is not None else {}
                out_m["sample_predictions"] = _sample_predictions(art, n=5)
                out["mustafa"] = out_m
            except Exception as exc:
                out["mustafa"] = {"error": str(exc)}

        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
