"""Command-line entry point.

    python -m dkex.cli calibrate --config configs/quick.yaml
    python -m dkex.cli simulate  --config configs/quick.yaml --jobs 4
    python -m dkex.cli analyse   --config configs/quick.yaml
    python -m dkex.cli plot      --config configs/quick.yaml
    python -m dkex.cli all       --config configs/quick.yaml
    python -m dkex.cli ood       --config configs/quick.yaml --wmd path/to/00036-xxxx.wmd
    python -m dkex.cli trace     --config configs/quick.yaml --baseline dyn_nsga3_ucb1_myopic
"""

from __future__ import annotations

import argparse
import json

from dkex import pipeline
from dkex.config import Config


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="dkex", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["calibrate", "simulate", "analyse", "plot", "all", "ood", "trace"])
    ap.add_argument("--config", default=None, help="YAML/JSON config (defaults = concept note)")
    ap.add_argument("--jobs", type=int, default=None, help="parallel worker processes")
    ap.add_argument("--runs", type=int, default=None, help="override experiment.runs")
    ap.add_argument("--wmd", default=None, help="PrefLib .wmd file for the ood stage")
    ap.add_argument("--baseline", default="dyn_nsga3_ucb1_myopic", help="baseline for trace")
    ap.add_argument("--tau", type=float, default=12.0, help="tau for trace")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config) if args.config else Config()
    if args.jobs is not None:
        cfg.experiment.n_jobs = args.jobs
    if args.runs is not None:
        cfg.experiment.runs = args.runs

    if args.stage in ("calibrate", "all"):
        rep = pipeline.calibrate(cfg)
        print(json.dumps({k: rep[k] for k in ("mean", "patient_abo", "cit_infeasible_centre_pairs")},
                         indent=2, default=float))
        print(rep["logistics"])
        for w in rep["warnings"]:
            print("WARNING:", w)
    if args.stage in ("simulate", "all"):
        pipeline.simulate_all(cfg)
    if args.stage in ("analyse", "all"):
        out = pipeline.analyse(cfg)
        print((pipeline.out_dir(cfg) / "summary.md").read_text())
        del out
    if args.stage in ("plot", "all"):
        for p in pipeline.plot(cfg):
            print("figure:", p)
    if args.stage == "ood":
        if not args.wmd:
            ap.error("--wmd is required for the ood stage")
        print(pipeline.ood(cfg, args.wmd).groupby(["config", "tau"]).mean(numeric_only=True))
    if args.stage == "trace":
        from dkex.experiments import simulate

        simulate(cfg, args.baseline, args.tau, 0, verbose=True)


if __name__ == "__main__":
    main()
