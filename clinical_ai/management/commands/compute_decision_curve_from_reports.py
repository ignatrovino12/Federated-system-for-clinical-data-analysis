#!/usr/bin/env python3
"""Compute decision-curve (net benefit) from a report JSON that contains per-sample predictions.

Usage (inside container):
  python scripts/compute_decision_curve_from_reports.py --report reports/model_quality/base_core_quality_mustafa.json --out-dir reports/figures --bootstrap 1000

The script expects the JSON to contain either `y_true` and `y_pred` (probabilities) fields or `labels` and `scores`.
If such per-sample fields are missing the script will exit with an explanatory message.
"""
import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError


def load_report(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_examples(report):
    # Try several common keys
    for k_true, k_score in [('y_true', 'y_pred'), ('labels', 'scores'), ('y', 'p'), ('y_true', 'predictions')]:
        if k_true in report and k_score in report:
            y = np.array(report[k_true])
            p = np.array(report[k_score])
            return y, p
    # Some reports embed arrays under 'examples' or similar
    if 'examples' in report and isinstance(report['examples'], list) and len(report['examples'])>0:
        ex0 = report['examples'][0]
        if 'y' in ex0 and 'p' in ex0:
            y = np.array([e['y'] for e in report['examples']])
            p = np.array([e['p'] for e in report['examples']])
            return y, p
    # Newer reports may include a `per_sample` block with predictions
    if 'per_sample' in report and isinstance(report['per_sample'], dict):
        per = report['per_sample']
        # prefer calibrated if available
        if 'y_true' in per and 'y_pred_calibrated' in per:
            return np.array(per['y_true']), np.array(per['y_pred_calibrated'])
        if 'y_true' in per and 'y_pred_raw' in per:
            return np.array(per['y_true']), np.array(per['y_pred_raw'])
    return None, None


def net_benefit(y, p, thresholds):
    N = len(y)
    prevalence = y.mean()
    nbs = []
    for t in thresholds:
        preds = (p >= t).astype(int)
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        w = t / (1 - t)
        nb = tp / N - w * (fp / N)
        nbs.append(nb)
    return np.array(nbs)


def bootstrap_nbs(y, p, thresholds, B=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y)
    all_nbs = np.zeros((B, len(thresholds)))
    for i in range(B):
        idx = rng.randint(0, n, size=n)
        yi = y[idx]
        pi = p[idx]
        all_nbs[i] = net_benefit(yi, pi, thresholds)
    lower = np.percentile(all_nbs, 2.5, axis=0)
    upper = np.percentile(all_nbs, 97.5, axis=0)
    median = np.percentile(all_nbs, 50, axis=0)
    return median, lower, upper


def plot_nb(thresholds, nb, lower, upper, nb_all, outpath, title=None):
    plt.figure(figsize=(8,5))
    plt.plot(thresholds, nb, label='Model', color='C0')
    plt.fill_between(thresholds, lower, upper, color='C0', alpha=0.25)
    plt.plot(thresholds, nb_all, '--', color='C1', label='Treat-all')
    plt.plot(thresholds, np.zeros_like(thresholds), ':', color='k', label='Treat-none')
    plt.xlabel('Decision threshold')
    plt.ylabel('Net benefit')
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True, lw=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--report', required=True, help='Path to report JSON containing per-sample predictions')
    p.add_argument('--out-dir', default='reports/figures', help='Directory to write figures and CSVs')
    p.add_argument('--bootstrap', type=int, default=1000, help='Bootstrap samples for CI')
    p.add_argument('--thresholds', default=None, help='Comma-separated thresholds or leave empty for linspace(0.01,0.99,99)')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(argv)

    report_path = Path(args.report)
    # If the user passed a basename, try reports/model_quality
    if not report_path.exists() and Path('reports', 'model_quality', args.report).exists():
        report_path = Path('reports', 'model_quality', args.report)

    if not report_path.exists():
        raise FileNotFoundError(f'Report file not found: {report_path}')

    report = load_report(report_path)
    y, p_scores = extract_examples(report)
    if y is None:
        raise ValueError('Could not find per-sample labels and scores in the JSON. Expected keys like (y_true,y_pred) or (labels,scores) or report["per_sample"].')

    if args.thresholds:
        thresholds = np.array([float(x) for x in args.thresholds.split(',')])
    else:
        thresholds = np.linspace(0.01, 0.99, 99)

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    nb = net_benefit(y, p_scores, thresholds)
    prevalence = float(y.mean())
    nb_all = prevalence - (thresholds / (1 - thresholds)) * (1 - prevalence)

    print('Computing bootstrap CIs (B=%d)...' % args.bootstrap)
    median, lower, upper = bootstrap_nbs(y, p_scores, thresholds, B=args.bootstrap, seed=args.seed)

    base = report_path.stem
    title = f'Decision curve: {base}'
    png_out = outdir / f'{base}_decision_curve.png'
    csv_out = outdir / f'{base}_decision_curve.csv'

    plot_nb(thresholds, nb, lower, upper, nb_all, png_out, title=title)

    # Save numeric results
    import csv
    with open(csv_out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['threshold','nb','ci_low','ci_high','nb_all'])
        for t, nbi, lo, hi, nball in zip(thresholds, nb, lower, upper, nb_all):
            writer.writerow([t, nbi, lo, hi, nball])

    print('Wrote:', png_out)
    print('Wrote:', csv_out)



class Command(BaseCommand):
    help = 'Compute decision-curve (net benefit) from a report JSON that contains per-sample predictions.'

    def add_arguments(self, parser):
        parser.add_argument('--report', required=True, help='Path to report JSON containing per-sample predictions')
        parser.add_argument('--out-dir', default='reports/figures', help='Directory to write figures and CSVs')
        parser.add_argument('--bootstrap', type=int, default=1000, help='Bootstrap samples for CI')
        parser.add_argument('--thresholds', default=None, help='Comma-separated thresholds or leave empty for linspace(0.01,0.99,99)')
        parser.add_argument('--seed', type=int, default=42)

    def handle(self, *args, **options):
        argv = []
        argv.extend(['--report', options['report']])
        argv.extend(['--out-dir', options.get('out_dir') or options.get('out-dir') or 'reports/figures'])
        argv.extend(['--bootstrap', str(options.get('bootstrap', 1000))])
        if options.get('thresholds'):
            argv.extend(['--thresholds', options.get('thresholds')])
        argv.extend(['--seed', str(options.get('seed', 42))])

        try:
            main(argv)
        except FileNotFoundError as e:
            raise CommandError(str(e))
        except ValueError as e:
            raise CommandError(str(e))
