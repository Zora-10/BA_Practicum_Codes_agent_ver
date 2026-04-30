"""
Step 1: Keyword Generation
Generates YouTube search queries for hard case / protective case demand signals.
Fidelity: mirrors 01_keyword_generate.ipynb exactly.
"""

import re
import pandas as pd
from pathlib import Path
from datetime import datetime

from .config import (
    OBJECT_CATEGORIES,
    PROTECTION_QUERIES,
    STORAGE_QUERIES,
    USAGE_QUERIES,
    PROBLEM_QUERIES,
    LEADER_DISCOVERY_QUERIES,
    NEGATIVE_TERMS,
    BASE_DIR,
)


def _filter_queries(df: pd.DataFrame, negative_terms: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["query_lower"] = df["query"].str.lower()

    def _contains_negative(text: str) -> bool:
        for term in negative_terms:
            if " " in term:
                if term in text:
                    return True
            else:
                pattern = r"\b" + re.escape(term) + r"\b"
                if re.search(pattern, text):
                    return True
        return False

    mask = ~df["query_lower"].apply(_contains_negative)
    return df[mask].drop(columns=["query_lower"]).drop_duplicates()


def generate_member_queries() -> pd.DataFrame:
    rows = []
    query_types = {
        "protection": PROTECTION_QUERIES,
        "storage": STORAGE_QUERIES,
        "usage": USAGE_QUERIES,
        "problem": PROBLEM_QUERIES,
    }

    for cat_name, objects in OBJECT_CATEGORIES.items():
        for obj in objects:
            for query_type, patterns in query_types.items():
                for pattern in patterns:
                    rows.append({
                        "category": cat_name,
                        "object": obj,
                        "query": pattern.format(obj=obj),
                        "type": query_type,
                    })

    df = pd.DataFrame(rows)
    return _filter_queries(df, NEGATIVE_TERMS)


def generate_leader_queries() -> pd.DataFrame:
    rows = []
    for q in LEADER_DISCOVERY_QUERIES:
        rows.append({"query": q, "type": "discovery"})
    df = pd.DataFrame(rows)
    return _filter_queries(df, NEGATIVE_TERMS)


def generate_all(output_dir: Path | None = None) -> pd.DataFrame:
    """
    Run full keyword generation pipeline — single unified pool, no member assignment.
    Returns a single DataFrame with all keywords.
    """
    if output_dir is None:
        output_dir = BASE_DIR / "data" / "query_assignments"
    output_dir.mkdir(parents=True, exist_ok=True)

    member_df = generate_member_queries()
    leader_df = generate_leader_queries()
    leader_df["category"] = "discovery"
    leader_df["object"] = "discovery"

    all_df = pd.concat([member_df, leader_df], ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["query"])

    # Save as single unified CSV
    output_path = output_dir / "all_keywords.csv"
    all_df.to_csv(output_path, index=False)

    summary = {
        "total_queries": len(all_df),
        "by_type": all_df["type"].value_counts().to_dict(),
        "by_category": all_df["category"].value_counts().to_dict(),
    }

    return all_df, summary


def get_query_list_for_categories(categories: list[str]) -> list[str]:
    """Return deduplicated query strings for a set of categories."""
    df = generate_member_queries()
    filtered = df[df["category"].isin(categories)]
    queries = filtered["query"].dropna().unique().tolist()
    leader = generate_leader_queries()["query"].tolist()
    combined = list(dict.fromkeys(queries + leader))
    return combined


def get_all_queries_flat() -> list[str]:
    """Return all queries as a single flat list (for single-user dashboard)."""
    all_q = generate_member_queries()["query"].tolist()
    leader_q = generate_leader_queries()["query"].tolist()
    return list(dict.fromkeys(all_q + leader_q))
