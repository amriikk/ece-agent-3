import anthropic
import pandas as pd
import json
import os
import re

from planner import generate_plan
from executor import execute, format_trace


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    for col in df.columns:
        try:
            cleaned = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
            converted = pd.to_numeric(cleaned, errors="coerce")
            if converted.notna().sum() > 100:
                df[col] = converted
        except Exception:
            pass
    return df


def generate_answer(question: str, result_df: pd.DataFrame) -> str:
    """Send the result table back to Claude for a natural language answer."""
    client = anthropic.Anthropic()

    # Limit to 20 rows so we don't blow the context window
    preview = result_df.head(20).to_string(index=False)

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": (
                    f"I ran a data analysis on a Spotify dataset to answer this question:\n"
                    f"'{question}'\n\n"
                    f"Here are the results:\n{preview}\n\n"
                    f"Please give a clear, concise answer to the question based only. "
                    f"on this data. Be specific and include the actual numbers."
                )
            }
        ]
    )

    return response.content[0].text.strip()


def run(question: str, dataset_path: str = "dataset.csv", verbose: bool = True) -> dict:
    """
    Full pipeline: question → plan → execute → answer.

    Returns a dict with: question, plan, trace, result, answer
    """
    # 1. Load data
    df = load_dataset(dataset_path)

    # 2. Generate plan
    if verbose:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        print("\n[1] Generating plan...")

    plan = generate_plan(question)

    if verbose:
        print(json.dumps(plan, indent=2))

    # 3. Execute plan
    if verbose:
        print("\n[2] Executing plan...")

    result_df, trace = execute(df, plan)

    if verbose:
        print(format_trace(trace))
        print()
        print("Result table:")
        print(result_df.head(20).to_string(index=False))

    # 4. Generate natural language answer
    if verbose:
        print("\n[3] Generating answer...")

    answer = generate_answer(question, result_df)

    if verbose:
        print(f"\nAnswer: {answer}")

    return {
        "question": question,
        "plan":     plan,
        "trace":    trace,
        "result":   result_df,
        "answer":   answer
    }


if __name__ == "__main__":
    test_questions = [
        "What are the top 3 artists by total TikTok views?",
        "Which explicit tracks have more YouTube views than Spotify streams?",
    ]

    for q in test_questions:
        output = run(q)
        print(f"\nFinal Answer: {output['answer']}")
        print("=" * 60)