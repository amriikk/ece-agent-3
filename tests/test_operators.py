import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from operators import (
    derive_columns, filter_rows, group_and_aggregate,
    sort_rows, limit_rows, select_columns, distinct_rows
)

# Small fake dataset that mimics Spotify columns
sample_data = {
    "Track":            ["Song A", "Song B", "Song C", "Song D", "Song E"],
    "Artist":           ["Artist 1", "Artist 1", "Artist 2", "Artist 3", "Artist 2"],
    "Spotify Streams":  [1000000, 500000, 2000000, 300000, 1500000],
    "Release Year":     [2020, 2021, 2020, 2022, 2021],
    "Explicit Track":   [0, 1, 0, 1, 0],
}
df = pd.DataFrame(sample_data)


def test_filter_rows():
    result = filter_rows(df, [
        {"column": "Spotify Streams", "operator": ">", "value": 600000}
    ])
    assert len(result) == 3, f"Expected 3 rows, got {len(result)}"
    print("✅ test_filter_rows passed")


def test_filter_rows_multiple_conditions():
    result = filter_rows(df, [
        {"column": "Spotify Streams", "operator": ">", "value": 400000},
        {"column": "Explicit Track", "operator": "==", "value": 0}
    ])
    assert len(result) == 3, f"Expected 3 rows, got {len(result)}"
    print("✅ test_filter_rows_multiple_conditions passed")


def test_group_and_aggregate():
    result = group_and_aggregate(df, group_by=["Artist"], metrics=[
        {"function": "sum", "column": "Spotify Streams", "as": "total_streams"},
        {"function": "count", "column": "Track", "as": "track_count"}
    ])
    assert len(result) == 3, f"Expected 3 artists, got {len(result)}"
    artist1_streams = result[result["Artist"] == "Artist 1"]["total_streams"].values[0]
    assert artist1_streams == 1500000, f"Expected 1500000, got {artist1_streams}"
    print("✅ test_group_and_aggregate passed")


def test_sort_rows():
    result = sort_rows(df, [{"column": "Spotify Streams", "direction": "desc"}])
    # Song C has 2,000,000 streams — should be first
    top_streams = result.iloc[0]["Spotify Streams"]
    assert top_streams == 2000000, f"Expected 2000000 first, got {top_streams}"
    print("✅ test_sort_rows passed")


def test_limit_rows():
    result = limit_rows(df, 3)
    assert len(result) == 3, f"Expected 3 rows, got {len(result)}"
    print("✅ test_limit_rows passed")


def test_select_columns():
    result = select_columns(df, ["Track", "Artist"])
    assert list(result.columns) == ["Track", "Artist"], f"Unexpected columns: {result.columns}"
    print("✅ test_select_columns passed")


def test_distinct_rows():
    result = distinct_rows(df, columns=["Artist"])
    assert len(result) == 3, f"Expected 3 unique artists, got {len(result)}"
    print("✅ test_distinct_rows passed")


def test_derive_columns_arithmetic():
    result = derive_columns(df, [
        {
            "new_column": "streams_millions",
            "type": "arithmetic",
            "operation": "divide",
            "left":  {"type": "column",  "value": "Spotify Streams"},
            "right": {"type": "literal", "value": 1000000}
        }
    ])
    assert "streams_millions" in result.columns
    assert round(result.iloc[2]["streams_millions"], 1) == 2.0
    print("✅ test_derive_columns_arithmetic passed")


if __name__ == "__main__":
    test_filter_rows()
    test_filter_rows_multiple_conditions()
    test_group_and_aggregate()
    test_sort_rows()
    test_limit_rows()
    test_select_columns()
    test_distinct_rows()
    test_derive_columns_arithmetic()
    print("\n🎉 All operator tests passed!")