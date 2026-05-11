import csv
import json
import os
from agent import run

QUESTIONS = [
    # Basic
    "What are the top 5 most streamed tracks on Spotify?",
    "Which 10 tracks have the highest YouTube views?",
    "What is the total number of explicit vs non-explicit tracks in the dataset?",
    "Which artist appears most frequently in the dataset?",
    "What are the top 5 tracks by Shazam count?",

    # Intermediate
    "Which artists have an average Spotify popularity score above 80, and how many tracks do they have?",
    "What are the top 5 tracks by YouTube likes-to-views ratio?",
    "Which non-explicit tracks have more TikTok views than YouTube views?",
    "What is the average Spotify streams per artist for artists with more than 5 tracks?",
    "Which 5 artists have the highest total streams across Spotify and YouTube combined?",

    # Advanced
    "Among tracks released after 2020, which artist has the highest average streams-per-playlist on Spotify?",
    "What are the top 3 artists by total cross-platform reach combining Spotify streams, YouTube views, and TikTok views?",
    "Which explicit tracks released after 2022 have a Spotify popularity above 75, sorted by TikTok views descending?",
    "For each release year, what is the average Spotify streams, and which year performs best?",
    "Among artists with more than 3 tracks, which has the best ratio of Shazam counts to Spotify streams?",
]


def run_all(dataset_path: str = "dataset.csv", output_path: str = "results.csv"):
    results = []

    for i, question in enumerate(QUESTIONS, 1):
        level = "Basic" if i <= 5 else "Intermediate" if i <= 10 else "Advanced"
        print(f"\n[{i}/15] ({level}) {question}")
        print("-" * 60)

        try:
            output = run(question, dataset_path=dataset_path, verbose=True)
            results.append({
                "question":     output["question"],
                "level":        level,
                "plan":         json.dumps(output["plan"]),
                "final_answer": output["answer"],
                "status":       "success"
            })
            print(f"✅ Done")

        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({
                "question":     question,
                "level":        level,
                "plan":         "",
                "final_answer": f"ERROR: {e}",
                "status":       "failed"
            })

    # Write results.csv
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "level", "plan", "final_answer", "status"]
        )
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    success = sum(1 for r in results if r["status"] == "success")
    print(f"\n{'='*60}")
    print(f"✅ {success}/15 questions succeeded")
    print(f"📄 Results written to {output_path}")

    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        print(f"\n❌ Failed questions:")
        for r in failed:
            print(f"  - {r['question']}")
            print(f"    {r['final_answer']}")


if __name__ == "__main__":
    run_all()