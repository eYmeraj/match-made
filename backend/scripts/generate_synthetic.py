"""Generate a pair of messy company datasets with a known ground truth.

    uv run python scripts/generate_synthetic.py --rows 1000000 --out ../.scratch/synthetic

Writes a.csv, b.csv and truth.csv (row_a, row_b pairs, 0-based data rows). B holds a
noisy copy of `--overlap` of A's rows (typos, suffix and case changes, abbreviations,
day-first dates, "$1,234" amounts) plus unrelated rows, shuffled.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

ADJECTIVES = [
    "acme",
    "global",
    "united",
    "pacific",
    "atlantic",
    "northern",
    "southern",
    "eastern",
    "western",
    "central",
    "royal",
    "golden",
    "silver",
    "blue",
    "green",
    "red",
    "black",
    "white",
    "prime",
    "first",
    "premier",
    "apex",
    "summit",
    "pinnacle",
    "evergreen",
    "redwood",
    "oak",
    "maple",
    "cedar",
    "pine",
    "willow",
    "aspen",
    "birch",
    "elm",
    "iron",
    "steel",
    "copper",
    "crystal",
    "diamond",
    "emerald",
    "sapphire",
    "ruby",
    "onyx",
    "granite",
    "marble",
    "harbor",
    "river",
    "lake",
    "ocean",
    "mountain",
    "valley",
    "canyon",
    "prairie",
    "desert",
    "forest",
    "meadow",
    "sunrise",
    "sunset",
    "horizon",
    "star",
    "lunar",
    "solar",
    "polar",
    "arctic",
    "tropical",
    "coastal",
    "urban",
    "metro",
    "capital",
    "liberty",
    "pioneer",
]
NOUNS = [
    "widgets",
    "systems",
    "solutions",
    "dynamics",
    "industries",
    "technologies",
    "networks",
    "logistics",
    "partners",
    "ventures",
    "holdings",
    "labs",
    "works",
    "foods",
    "energy",
    "motors",
    "textiles",
    "pharma",
    "media",
    "analytics",
    "robotics",
    "devices",
    "materials",
    "chemicals",
    "foundry",
    "studios",
    "outfitters",
    "supply",
    "trading",
    "imports",
    "exports",
    "consulting",
    "capital",
    "finance",
    "insurance",
    "realty",
    "builders",
    "construction",
    "engineering",
    "design",
    "apparel",
    "beverages",
    "brewing",
    "bakery",
    "dairy",
    "farms",
]
SUFFIXES = ["Inc", "Inc.", "LLC", "Ltd", "Corp", "Corporation", "Co", "Group", "GmbH", ""]
ALT_SUFFIX = {"Inc": "Incorporated", "Inc.": "Inc", "Ltd": "Limited", "Corp": "Corporation",
              "Corporation": "Corp.", "Co": "Company", "LLC": "L.L.C.", "Group": "Grp",
              "GmbH": "GmbH", "": ""}  # fmt: skip
CITIES = [
    "Springfield",
    "Riverton",
    "Fairview",
    "Franklin",
    "Greenville",
    "Clinton",
    "Madison",
    "Georgetown",
    "Salem",
    "Arlington",
    "Ashland",
    "Bristol",
    "Dover",
    "Milton",
    "Newport",
    "Oxford",
    "Burlington",
    "Manchester",
    "Kingston",
]


_ONSETS = [
    "b",
    "c",
    "d",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "r",
    "s",
    "t",
    "v",
    "w",
    "z",
    "br",
    "cr",
    "dr",
    "fr",
    "gr",
    "kr",
    "pr",
    "tr",
    "st",
    "sp",
    "sk",
    "bl",
    "cl",
    "fl",
    "gl",
    "pl",
    "sl",
    "ch",
    "sh",
    "th",
]
_VOWELS = ["a", "e", "i", "o", "u", "ai", "ea", "io", "ou", "y"]
_CODAS = ["", "", "n", "r", "s", "l", "x", "m", "nd", "rt", "st", "ck", "ng"]


def make_vocabulary(rng: np.random.Generator, size: int) -> list[str]:
    """Pronounceable pseudo-words, so name tokens have a realistic long tail."""
    words: set[str] = set()
    while len(words) < size:
        syllables = int(rng.integers(2, 4))
        word = "".join(
            str(rng.choice(_ONSETS)) + str(rng.choice(_VOWELS)) for _ in range(syllables)
        ) + str(rng.choice(_CODAS))
        words.add(word)
    return [str(w) for w in rng.permutation(sorted(words))]


def make_names(rng: np.random.Generator, n: int) -> list[str]:
    """Company-like names: 1-2 Zipf-distributed brand words plus common industry words."""
    vocabulary = make_vocabulary(rng, 60_000)
    ranks = np.minimum(rng.zipf(1.15, size=4 * n), len(vocabulary)) - 1
    seen: set[str] = set()
    names: list[str] = []
    i = 0
    while len(names) < n:
        if i + 2 >= len(ranks):
            ranks = np.minimum(rng.zipf(1.15, size=4 * n), len(vocabulary)) - 1
            i = 0
        words = [vocabulary[ranks[i]].title()]
        if rng.random() < 0.5:
            words.append(vocabulary[ranks[i + 1]].title())
        i += 2
        if rng.random() < 0.6:
            words.append(str(rng.choice(ADJECTIVES)).title())
        if rng.random() < 0.8:
            words.append(str(rng.choice(NOUNS)).title())
        name = " ".join(words)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def typo(rng: np.random.Generator, s: str) -> str:
    if len(s) < 4:
        return s
    i = int(rng.integers(1, len(s) - 2))
    match int(rng.integers(0, 3)):
        case 0:
            return s[:i] + s[i + 1 :]  # drop
        case 1:
            return s[:i] + s[i + 1] + s[i] + s[i + 2 :]  # swap
        case _:
            return s[:i] + s[i] + s[i:]  # duplicate


def noisy_name(rng: np.random.Generator, base: str, suffix: str) -> str:
    name = base
    if rng.random() < 0.3:
        name = typo(rng, name)
    suffix = ALT_SUFFIX[suffix] if rng.random() < 0.5 else suffix
    if rng.random() < 0.2:
        suffix = ""
    full = f"{name} {suffix}".strip()
    r = rng.random()
    if r < 0.2:
        full = full.upper()
    elif r < 0.3:
        full = full.lower()
    if rng.random() < 0.1:
        full = full.replace(" And ", " & ")
    return full


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--overlap", type=float, default=0.7)
    parser.add_argument("--extra", type=float, default=0.3, help="unrelated B rows, as share of A")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    n = args.rows
    n_extra = int(n * args.extra)

    names = make_names(rng, n + n_extra)
    suffixes = rng.choice(SUFFIXES, n + n_extra)
    founded = pl.date_range(pl.date(1950, 1, 1), pl.date(2023, 12, 31), "1d", eager=True).sample(
        n + n_extra, with_replacement=True, seed=args.seed
    )
    revenue = np.round(np.exp(rng.normal(14, 2, n + n_extra)), -2)
    cities = rng.choice(CITIES, n + n_extra)

    a = pl.DataFrame(
        {
            "company_id": [f"A{i:07d}" for i in range(n)],
            "Company Name": [
                f"{nm} {sx}".strip() for nm, sx in zip(names[:n], suffixes[:n], strict=True)
            ],
            "Founded": founded[:n].dt.strftime("%Y-%m-%d"),
            "Revenue": revenue[:n],
            "City": cities[:n],
        }
    )

    matched = np.sort(rng.choice(n, int(n * args.overlap), replace=False))
    b_names = [noisy_name(rng, names[i], str(suffixes[i])) for i in matched]
    b_names += [f"{nm} {sx}".strip() for nm, sx in zip(names[n:], suffixes[n:], strict=True)]
    shift = rng.choice([0, 0, 0, 0, 1, -1, 30], len(matched))
    b_dates = (
        pl.DataFrame({"d": founded.gather(matched), "s": shift})
        .select(pl.col("d") + pl.duration(days=pl.col("s")))
        .to_series()
        .to_list()
    )
    b_dates += founded[n:].to_list()
    noise = 1 + rng.normal(0, 0.01, len(matched)) * (rng.random(len(matched)) < 0.3)
    b_revenue = np.concatenate([revenue[matched] * noise, revenue[n:]])
    b_cities = np.concatenate([cities[matched], cities[n:]])

    b = pl.DataFrame(
        {
            "vendor": b_names,
            "incorporated": pl.Series(b_dates, dtype=pl.Date).dt.strftime("%d/%m/%Y"),
            "annual_revenue": [f"${v:,.0f}" for v in b_revenue],
            "city": [c.upper() for c in b_cities],
            "__truth": np.concatenate([matched, np.full(n_extra, -1)]),
        }
    )
    order = rng.permutation(b.height)
    b = b[order]

    args.out.mkdir(parents=True, exist_ok=True)
    truth = (
        b.with_row_index("row_b")
        .filter(pl.col("__truth") >= 0)
        .select(pl.col("__truth").alias("row_a"), "row_b")
        .sort("row_a")
    )
    a.write_csv(args.out / "a.csv")
    b.drop("__truth").write_csv(args.out / "b.csv")
    truth.write_csv(args.out / "truth.csv")
    print(f"A: {a.height:,} rows, B: {b.height:,} rows, true pairs: {truth.height:,}")


if __name__ == "__main__":
    main()
