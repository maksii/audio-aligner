import csv
import json
from typing import Any


class Reporter:
    def __init__(self, filename: str) -> None:
        self.filename = filename

    def write(self, data: list[dict[str, Any]]) -> None:
        raise NotImplementedError


class JsonReporter(Reporter):
    def write(self, data: list[dict[str, Any]]) -> None:
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump({'results': data}, f, indent=2)


class CsvReporter(Reporter):
    def write(self, data: list[dict[str, Any]]) -> None:
        if not data:
            return

        with open(self.filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)


def get_reporter(filename: str, file_format: str) -> Reporter:
    if file_format == 'json':
        return JsonReporter(filename)
    if file_format == 'csv':
        return CsvReporter(filename)
    raise ValueError(f'Unsupported file format: {file_format}') 