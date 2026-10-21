"""Put an existing predictions.parquet on the zoo leaderboard under NAME.

python -m zoo.score NAME predictions.parquet --truth test.parquet [--out zoo_runs]
"""
from __future__ import annotations

import argparse
import json
import os

from zoo.run import ROOT, rebuild_leaderboard


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('name')
    p.add_argument('predictions')
    p.add_argument('--truth', required=True)
    p.add_argument('--out', default=os.path.join(ROOT, 'zoo_runs'))
    args = p.parse_args()
    import pyarrow.parquet as pq

    from score_submission import score_table

    table = pq.read_table(args.predictions)
    scores = score_table(table, args.truth, by_source=True)
    by_source = scores.pop('by_source')
    run_dir = os.path.join(args.out, args.name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'scores.json'), 'w') as f:
        json.dump({'overall': {str(m): v for m, v in scores.items()},
                   'by_source': {s: {str(m): v for m, v in r.items()}
                                 for s, r in by_source.items()},
                   'rows': table.num_rows, 'source': os.path.abspath(args.predictions)},
                  f, indent=1)
    rebuild_leaderboard(args.out)


if __name__ == '__main__':
    main()
