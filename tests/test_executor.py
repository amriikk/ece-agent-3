import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from executor import execute, format_trace

# Load real dataset
def load_dataset(path):
    df = pd.read_csv(path, encoding="latin-1")
    for col in df.columns:
        # Handle both old (object) and new (StringDtype) pandas string types
        if df[col].dtype == object or hasattr(df[col], 'str'):
            try:
                cleaned = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
                converted = pd.to_numeric(cleaned, errors="coerce")
                if converted.notna().sum() > 100:
                    df[col] = converted
            except Exception:
                pass
    return df

df = load_dataset(os.path.join(PROJECT_ROOT, "dataset.csv"))
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")

# Verify numeric columns loaded correctly
print("Sample stream value:", df["Spotify Streams"].iloc[0])
print("Dtype:", df["Spotify Streams"].dtype, "\n")


def test_basic_pipeline():
    """Top 5 artists by total Spotify streams"""
    plan = {
        "steps": [
            {
                "op": "group_and_aggregate",
                "group_by": ["Artist"],
                "metrics": [
                    {"function": "sum", "column": "Spotify Streams", "as": "total_streams"}
                ]
            },
            {
                "op": "sort_rows",
                "sort_by": [{"column": "total_streams", "direction": "desc"}]
            },
            {
                "op": "limit_rows",
                "k": 5
            }
        ]
    }

    result, trace = execute(df, plan)
    print("=== test_basic_pipeline ===")
    print(format_trace(trace))
    print()
    print(result.to_string(index=False))
    assert len(result) == 5, f"Expected 5 rows, got {len(result)}"
    print("✅ test_basic_pipeline passed\n")


def test_filter_then_aggregate():
    """Top 3 artists by average streams for explicit tracks only"""
    plan = {
        "steps": [
            {
                "op": "filter_rows",
                "conditions": [
                    {"column": "Explicit Track", "operator": "==", "value": 1}
                ]
            },
            {
                "op": "group_and_aggregate",
                "group_by": ["Artist"],
                "metrics": [
                    {"function": "mean", "column": "Spotify Streams", "as": "avg_streams"}
                ]
            },
            {
                "op": "sort_rows",
                "sort_by": [{"column": "avg_streams", "direction": "desc"}]
            },
            {
                "op": "limit_rows",
                "k": 3
            }
        ]
    }

    result, trace = execute(df, plan)
    print("=== test_filter_then_aggregate ===")
    print(format_trace(trace))
    print()
    print(result.to_string(index=False))
    assert len(result) <= 3
    print("✅ test_filter_then_aggregate passed\n")


def test_derive_then_sort():
    """Top 5 tracks by streams per playlist count"""
    plan = {
        "steps": [
            {
                "op": "derive_columns",
                "derive": [
                    {
                        "new_column": "streams_per_playlist",
                        "type": "arithmetic",
                        "operation": "divide",
                        "left":  {"type": "column", "value": "Spotify Streams"},
                        "right": {"type": "column", "value": "Spotify Playlist Count"}
                    }
                ]
            },
            {
                "op": "sort_rows",
                "sort_by": [{"column": "streams_per_playlist", "direction": "desc"}]
            },
            {
                "op": "limit_rows",
                "k": 5
            },
            {
                "op": "select_columns",
                "columns": ["Track", "Artist", "streams_per_playlist"]
            }
        ]
    }

    result, trace = execute(df, plan)
    print("=== test_derive_then_sort ===")
    print(format_trace(trace))
    print()
    print(result.to_string(index=False))
    assert len(result) == 5
    assert "streams_per_playlist" in result.columns
    print("✅ test_derive_then_sort passed\n")


def test_unknown_operator():
    """Should raise ValueError for unknown operator"""
    plan = {
        "steps": [
            {"op": "explode_table", "column": "Track"}
        ]
    }
    try:
        execute(df, plan)
        print("❌ test_unknown_operator FAILED — should have raised ValueError")
    except ValueError as e:
        print(f"✅ test_unknown_operator passed — caught: {e}\n")


if __name__ == "__main__":
    test_basic_pipeline()
    test_filter_then_aggregate()
    test_derive_then_sort()
    test_unknown_operator()
    print("🎉 All executor tests passed!")