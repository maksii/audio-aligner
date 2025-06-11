import json
import os
from audio_aligner.reports import JsonReporter, CsvReporter


def test_json_reporter(tmp_path):
    reporter = JsonReporter(os.path.join(tmp_path, "report.json"))
    data = [{"a": 1, "b": 2}]
    reporter.write(data)

    with open(reporter.filename, "r") as f:
        read_data = json.load(f)
    assert read_data["results"] == data


def test_csv_reporter(tmp_path):
    reporter = CsvReporter(os.path.join(tmp_path, "report.csv"))
    data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    reporter.write(data)

    with open(reporter.filename, "r") as f:
        lines = f.readlines()
    assert len(lines) == 3
    assert lines[0].strip() == "a,b"
    assert lines[1].strip() == "1,2" 