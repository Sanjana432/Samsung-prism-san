"""Real CLI failures must leave a useful status and preserve the supplied dataset."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

from prism.common import ROOT, load_config


class PipelineFailureReporting(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.cfg = load_config()
        self.cfg.pop("_root", None)
        for key in self.cfg["paths"]:
            self.cfg["paths"][key] = str(self.directory / key)
        self.config = self.directory / "config.yaml"
        self.cfg["ladder"]["n_originals"] = 2
        self.config.write_text(yaml.safe_dump(self.cfg), encoding="utf-8")

    def run_stage(self, stage):
        result = subprocess.run([sys.executable, str(Path(ROOT) / "scripts/run_pipeline.py"),
                                 "--config", str(self.config), "--stages", stage],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        path = Path(self.cfg["paths"]["reports_dir"]) / "pipeline_status.json"
        status = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["failed_stage"], stage)
        self.assertFalse(status["accuracy_measured"])
        self.assertIn("finished_utc", status)
        return status

    def test_existing_dataset_count_mismatch_is_reported_before_any_subprocess(self):
        metadata = Path(self.cfg["paths"]["metadata_dir"]) / "ladder.jsonl"
        metadata.parent.mkdir(parents=True)
        metadata.write_text('{"sample_id":"preserve"}\n', encoding="utf-8")
        original = metadata.read_bytes()
        status = self.run_stage("data")
        self.assertEqual(status["error_type"], "ValueError")
        self.assertEqual(status["stages"], [])
        self.assertNotIn("failed_script", status)
        self.assertEqual(metadata.read_bytes(), original)
        self.assertFalse(Path(self.cfg["paths"]["images_dir"]).exists())

    def test_invalid_dataset_stops_before_gpu_loading_and_records_failed_script(self):
        status = self.run_stage("baseline")
        self.assertEqual(status["failed_script"], "validate_dataset.py")
        self.assertEqual(status["error_type"], "CalledProcessError")
        self.assertNotEqual(status["exit_code"], 0)
        self.assertEqual(status["stages"], [])
        self.assertFalse(Path(self.cfg["paths"]["predictions_dir"]).exists())


if __name__ == "__main__":
    unittest.main()
