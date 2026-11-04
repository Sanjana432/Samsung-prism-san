"""One checked entry point for the modified pipeline; each GPU stage runs in its own process."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prism.common import P, load_config, read_jsonl
from prism.runtime import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--cpu-only", action="store_true", help="generate, validate and self-test without loading models")
    parser.add_argument("--stages", nargs="+", choices=["data", "check", "baseline", "train", "infer", "evaluate"],
                        default=["data", "check", "baseline", "train", "infer", "evaluate"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    stages = ["data", "check"] if args.cpu_only else args.stages
    config_args = ["--config", str(Path(args.config).resolve())] if args.config else []
    history = {"model": cfg["model"], "started_utc": datetime.now(timezone.utc).isoformat(),
               "status": "running", "stages": [], "accuracy_measured": False}
    status_path = P(cfg, "reports_dir", "pipeline_status.json")

    def run(script, *options):
        command = [sys.executable, str(ROOT / "scripts" / script), *config_args, *options]
        print("[pipeline] " + " ".join(command), flush=True)
        started = time.monotonic()
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as error:
            history.update(status="failed", failed_script=script, exit_code=error.returncode)
            atomic_json(status_path, history)
            raise
        history["stages"].append({"script": script, "seconds": round(time.monotonic() - started, 2)})
        atomic_json(status_path, history)

    gpu_checked = False
    for stage in stages:
        if stage in ("baseline", "train", "infer") and not gpu_checked:
            run("validate_dataset.py")
            run("gpu_smoke_test.py")
            gpu_checked = True
        if stage == "data":
            if not Path(P(cfg, "metadata_dir", "ladder.jsonl")).exists():
                run("01_generate_dataset.py")
            else:
                count = len(read_jsonl(P(cfg, "metadata_dir", "ladder.jsonl")))
                if count != cfg["ladder"]["n_originals"]:
                    raise ValueError(f"Existing shared data has {count} Originals; configuration requests {cfg['ladder']['n_originals']}. "
                                     "Use a matching configuration or a new dataset directory; existing images were preserved.")
            run("02_build_splits.py")
            run("validate_dataset.py")
        elif stage == "check":
            run("validate_dataset.py")
            run("check_ladder_scores.py")
            run("00_selftest.py")
        elif stage == "baseline":
            run("03_prompt_study_and_baseline.py")
        elif stage == "train":
            run("04_train_qlora.py")
        elif stage == "infer":
            run("05_infer.py", "--adapter", P(cfg, "outputs_dir", "best_adapter"), "--split", "test")
        elif stage == "evaluate":
            run("06_evaluate.py", "--models", "baseline_zero_shot", "finetuned_qlora", "--split", "test")
    history["status"] = "selected_stages_completed"
    atomic_json(status_path, history)
    print("Selected stages completed. Model accuracy is defined by the evaluation report, not by a software check.")


if __name__ == "__main__":
    main()
