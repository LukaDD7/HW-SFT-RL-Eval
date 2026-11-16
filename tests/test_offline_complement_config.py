from pathlib import Path

from dual_track_opd.eval.benchmark_suite import load_suite, select_benchmarks


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/eval/project_vision_opd_v1_offline_complement.yaml"


def test_offline_complement_config_loads_nine_tasks() -> None:
    suite = load_suite(CONFIG)
    selected = select_benchmarks(suite, "project15_complement_offline")
    assert [spec.benchmark_id for spec in selected] == [
        "mindcube",
        "vqav2",
        "scienceqa",
        "mathverse",
        "mathvista",
        "mmsi_bench",
        "blink",
        "mmvet",
        "mv_math",
    ]
    assert suite.defaults["include_task_path"] == "eval_tasks/opd_v1_offline"
