"""Precompute the 12 example sentence trajectories.

Each sentence is tokenised, run through GPT-2 with `output_hidden_states=True`,
projected via the fitted basis to 3D, and written as [num_layers+1, seq, 3]
inside a single JSON payload under `web/public/assets/sentences.json`
(target size < 100 KB across all 12).

Sentence list is authored by hand to sidestep corpus licensing (spec §Risks).
Good picks show something: an ambiguous word in two contexts, a garden-path
sentence, a line of Python, a counting sequence, months / days.

Stub only for Phase 0.
"""
from __future__ import annotations

import argparse

from common import Config, load_config

# Hand-written sentences. Kept here (not corpus-derived) so we can ship them.
SENTENCES: list[str] = [
    "The bank raised interest rates today.",                 # bank = finance
    "We walked along the river bank at sunset.",             # bank = riverbank
    "The old man the boats.",                                # garden path
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "One, two, three, four, five, six, seven.",              # counting
    "January, February, March, April, May, June.",           # months
    "Monday, Tuesday, Wednesday, Thursday, Friday.",         # days
    "Time flies like an arrow; fruit flies like a banana.",  # dual parse
    "She lifted the heavy box onto the shelf.",              # concrete
    "Democracy requires patience and compromise.",           # abstract
    "The cat sat on the mat.",                               # tiny baseline
    "The mitochondria is the powerhouse of the cell.",       # famous fact
]


def precompute(cfg: Config) -> None:
    raise NotImplementedError("sentences: Phase 3")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    precompute(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
