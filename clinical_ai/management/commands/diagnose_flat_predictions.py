from django.core.management.base import BaseCommand
import json
import numpy as np
import pandas as pd
import torch

from clinical_ai import services
from clinical_ai.models import PatientClinicalRecord
from django.db.models import Count


class Command(BaseCommand):
    help = "Diagnose flat predictions: scaler names, scaled std, label counts, logits, biases"

    def add_arguments(self, parser):
        parser.add_argument("--sample-size", type=int, default=200)
        parser.add_argument("--patients", type=int, default=10)
        parser.add_argument("--zero-bias", action="store_true", help="Temporarily zero final-layer biases for diagnostic")

    def handle(self, *args, **options):
        sample_size = options["sample_size"]
        patients_n = options["patients"]
        zero_bias = options["zero_bias"]

        report = {"alex": {}, "mustafa": {}, "label_counts": None}

        # label counts
        qs = list(PatientClinicalRecord.objects.values("diabetes_status").annotate(c=Count("id")))
        report["label_counts"] = qs

        for key, loader in (("alex", services._load_alex_artifacts), ("mustafa", services._load_mustafa_artifacts)):
            try:
                art = loader()
            except Exception as e:
                report[key]["error"] = str(e)
                continue

            sfn = art.get("scaler_feature_names")
            if sfn is not None:
                try:
                    cols = list(sfn)
                except Exception:
                    cols = art["feature_columns"]
            else:
                cols = art["feature_columns"]
            report[key]["feature_columns"] = art["feature_columns"]
            # ensure scaler feature names are JSON-serializable
            if sfn is None:
                sfn_list = None
            else:
                try:
                    sfn_list = list(sfn)
                except Exception:
                    sfn_list = [str(x) for x in sfn]
            report[key]["scaler_feature_names"] = sfn_list
            report[key]["outputs_probability"] = art.get("outputs_probability")

            # sample data - build feature dicts from model helper methods to avoid DB field name mismatches
            rec_objs = list(PatientClinicalRecord.objects.all()[:sample_size])
            feats = []
            for r in rec_objs:
                try:
                    if key == "alex":
                        feats.append(r.alex5050_features())
                    else:
                        feats.append(r.mustafa_features())
                except Exception:
                    feats.append({c: None for c in cols})
            if feats:
                df = pd.DataFrame(feats, columns=cols).astype("float32")
                arr = df.values
            else:
                arr = np.zeros((0, len(cols)), dtype=np.float32)
            try:
                if arr.shape[0] > 0:
                    if sfn is None:
                        scaled = art["scaler"].transform(arr)
                    else:
                        scaled = art["scaler"].transform(df[list(sfn)])
                else:
                    scaled = np.zeros_like(arr)
                report[key]["scaled_std"] = np.std(scaled, axis=0).tolist() if arr.shape[0] > 0 else []
                report[key]["scaled_overall_std"] = float(np.std(scaled)) if arr.shape[0] > 0 else 0.0
            except Exception as e:
                report[key]["scaled_error"] = str(e)

            # logits for first patients_n using helper methods
            patient_objs = list(PatientClinicalRecord.objects.all()[:patients_n])
            logits_list = []
            try:
                for r in patient_objs:
                    try:
                        feat = r.alex5050_features() if key == "alex" else r.mustafa_features()
                    except Exception:
                        feat = {c: None for c in cols}
                    row = pd.DataFrame([feat], columns=cols).astype("float32")
                    scaled_row = art["scaler"].transform(row[cols])
                    x = torch.tensor(scaled_row, dtype=torch.float32).to(art["device"])
                    with torch.no_grad():
                        logits = art["model"](x).cpu().numpy().ravel().tolist()
                        logits_list.append(logits)
                report[key]["sample_logits"] = [list(map(float, l)) for l in logits_list]
            except Exception as e:
                report[key]["sample_logits_error"] = str(e)

            # biases
            biases = {}
            try:
                sd = art["model"].state_dict()
                for k, v in sd.items():
                    if "bias" in k and v.ndim == 1:
                        biases[k] = {"mean": float(v.mean().item()), "first5": [float(x) for x in v[:5].tolist()]}
                # biases already converted to floats
                report[key]["biases"] = biases
            except Exception as e:
                report[key]["biases_error"] = str(e)

            # zero-bias diagnostic
            if zero_bias:
                try:
                    sd = art["model"].state_dict()
                    for k in list(sd.keys()):
                        if "bias" in k and sd[k].ndim == 1:
                            sd[k] = torch.zeros_like(sd[k])
                    art["model"].load_state_dict(sd)
                    logits_z = []
                    for r in patient_objs:
                        try:
                            feat = r.alex5050_features() if key == "alex" else r.mustafa_features()
                        except Exception:
                            feat = {c: None for c in cols}
                        row = pd.DataFrame([feat], columns=cols).astype("float32")
                        scaled_row = art["scaler"].transform(row[cols])
                        x = torch.tensor(scaled_row, dtype=torch.float32).to(art["device"])
                        with torch.no_grad():
                            logits = art["model"](x).cpu().numpy().ravel().tolist()
                            logits_z.append(logits)
                    report[key]["sample_logits_zero_bias"] = [list(map(float, l)) for l in logits_z]
                except Exception as e:
                    report[key]["sample_logits_zero_bias_error"] = str(e)

        self.stdout.write(json.dumps(report, indent=2))
