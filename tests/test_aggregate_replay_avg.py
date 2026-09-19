import json

from dual_track_opd.eval.aggregate_replay_avg import aggregate


def test_aggregate_replay_avg_reports_mean(tmp_path):
    paths = []
    for index, value in enumerate((0.2, 0.4, 0.6, 0.8)):
        path = tmp_path / f"repeat_{index}.json"
        path.write_text(json.dumps({"strict_weighted_accuracy": value}), encoding="utf-8")
        paths.append(path)

    result = aggregate(paths, "strict_weighted_accuracy")

    assert result["repeat_count"] == 4
    assert result["values"] == [0.2, 0.4, 0.6, 0.8]
    assert result["mean"] == 0.5
