"""Regenerate ground-truth CSVs for local tasks from their gold SQL.

    python scripts/make_local_gt.py [--tasks tasks]

For each task: load the flat-file sources with the reference loader, run every
``grader/gold/<model>.sql`` and write ``grader/gt/<model>.csv``. Row counts in
``grader/table.json`` are checked against the loaded tables.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import duckdb

from elt_rl.destinations.duckdb_dest import build_flat_file_template
from elt_rl.models import render
from elt_rl.task import ELTTask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="tasks")
    args = parser.parse_args()

    for task_dir in sorted(p for p in Path(args.tasks).iterdir() if p.is_dir()):
        grader = task_dir / "grader"
        expected = json.loads((grader / "table.json").read_text())
        task = ELTTask(task_dir.name, task_dir / "workspace", expected, grader / "gt")
        with tempfile.TemporaryDirectory() as tmp:
            db = build_flat_file_template(task, Path(tmp) / "ref.duckdb")
            conn = duckdb.connect(str(db))
            try:
                for table, n in expected.items():
                    got = conn.execute(f"SELECT COUNT(*) FROM elt.{table}").fetchone()[0]
                    assert got == n, f"{task.name}.{table}: {got} rows, table.json says {n}"
                (grader / "gt").mkdir(exist_ok=True)
                for gold in sorted((grader / "gold").glob("*.sql")):
                    sql = render(gold.read_text(), lambda t: f"elt.{t}")
                    df = conn.execute(sql).df()
                    df.to_csv(grader / "gt" / f"{gold.stem}.csv", index=False)
                    print(f"{task.name}/{gold.stem}: {len(df)} rows")
            finally:
                conn.close()


if __name__ == "__main__":
    main()
