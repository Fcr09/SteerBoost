import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple

try:
    from openai import OpenAI  # openai>=1.x
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]


def get_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError(
            "This script requires the OpenAI Python SDK v1.x (it expects `from openai import OpenAI`). "
            "Activate the environment you use for batch runs, or upgrade with `pip install -U openai`."
        )
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://us.api.openai.com/v1"),
    )


def download_file_text(client: OpenAI, file_id: str) -> str:
    file_response = client.files.content(file_id)
    return file_response.text


def check_and_download_batch(
    batch_id: str,
    *,
    client: Optional[OpenAI] = None,
    download_dir: Path = Path("data/batch_outputs"),
    download_errors: bool = True,
    verbose: bool = True,
) -> Tuple[str, Optional[Path], Optional[Path]]:
    """
    Returns (status, output_path, error_path).
    output_path and error_path are present only if downloaded.
    """
    client = client or get_client()
    batch = client.batches.retrieve(batch_id)

    if verbose:
        print(f"\n{'='*50}")
        print(f"Batch Status Check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")
        print(f"Batch ID:      {batch.id}")
        print(f"Status:        {batch.status}")
        print(
            f"Created at:    {datetime.fromtimestamp(batch.created_at).strftime('%Y-%m-%d %H:%M:%S')}"
        )

        if batch.request_counts:
            print("\nProgress:")
            print(f"  Total:       {batch.request_counts.total}")
            print(f"  Completed:   {batch.request_counts.completed}")
            print(f"  Failed:      {batch.request_counts.failed}")

        if batch.output_file_id:
            print(f"\nOutput File ID: {batch.output_file_id}")
        if batch.error_file_id:
            print(f"Error File ID:  {batch.error_file_id}")

    download_dir.mkdir(parents=True, exist_ok=True)

    output_path = None
    if batch.status == "completed" and batch.output_file_id:
        output_path = download_dir / f"{batch_id}_output.jsonl"
        if verbose:
            print(f"\nDownloading results to {output_path}...")
        output_path.write_text(download_file_text(client, batch.output_file_id))
        if verbose:
            print(f"✓ Downloaded results to {output_path}")

    error_path = None
    if download_errors and batch.error_file_id:
        error_path = download_dir / f"{batch_id}_error.jsonl"
        if verbose:
            print(f"\nDownloading errors to {error_path}...")
        error_path.write_text(download_file_text(client, batch.error_file_id))
        if verbose:
            print(f"✓ Downloaded errors to {error_path}")

    if verbose and batch.status == "failed" and getattr(batch, "errors", None):
        print("\n❌ Batch failed.")
        print("Errors:")
        for error in batch.errors.data:
            print(f"  - {error.code}: {error.message}")

    return batch.status, output_path, error_path


def process_batch_output_file(batch_output_file: Path, *, repo_root: Path = Path(".")) -> None:
    if not batch_output_file.exists():
        raise FileNotFoundError(str(batch_output_file))

    updates = defaultdict(list)
    print(f"Reading batch results from {batch_output_file}...")

    success_count = 0
    error_count = 0

    with batch_output_file.open("r") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse JSON at line {line_num}")
                continue

            custom_id = item.get("custom_id")
            if not custom_id:
                print(f"Warning: Missing custom_id at line {line_num}")
                continue

            # custom_id format: method::model::concept_id::filename::index
            parts = custom_id.split("::")
            if len(parts) != 5:
                print(f"Warning: Invalid custom_id format '{custom_id}' at line {line_num}")
                continue

            method, model, concept_id, filename, index = parts
            try:
                index_int = int(index)
            except ValueError:
                print(f"Warning: Invalid index in custom_id '{custom_id}'")
                continue

            response = item.get("response")
            if not response or response.get("status_code") != 200:
                status_code = response.get("status_code") if response else "Unknown"
                print(f"Error for request {custom_id}: Status {status_code}")
                error_count += 1
                judgment = 0
                explanation = f"Evaluation error in batch processing: Status {status_code}"
                updates[(method, model, concept_id, filename)].append((index_int, judgment, explanation))
                continue

            try:
                body = response.get("body", {})
                choices = body.get("choices", [])
                if not choices:
                    raise ValueError("No choices in response")

                content = choices[0].get("message", {}).get("content", "")
                if not content:
                    raise ValueError(
                        f"Empty content received (Finish reason: {choices[0].get('finish_reason')})"
                    )

                content_json = json.loads(content)
                judgment = content_json.get("judgment", 0)
                explanation = content_json.get("explanation", "No explanation provided")
                if not explanation or len(explanation.strip()) < 10:
                    explanation = f"Minimal explanation provided: {explanation}"

                success_count += 1
                updates[(method, model, concept_id, filename)].append((index_int, judgment, explanation))
            except Exception as e:
                print(f"Error parsing content for {custom_id}: {e}")
                judgment = 0
                explanation = f"Evaluation error parsing response: {str(e)}"
                updates[(method, model, concept_id, filename)].append((index_int, judgment, explanation))
                error_count += 1

    print(f"\nProcessed {success_count} successful results and {error_count} errors.")
    print(f"Found updates for {len(updates)} files.")

    for (method, model, concept_id, filename), items in updates.items():
        file_path = repo_root / Path(f"data/result/{method}/{model}/{concept_id}/{filename}")
        if not file_path.exists():
            print(f"Warning: Original file not found at {file_path}. Skipping updates.")
            continue

        try:
            data = json.loads(file_path.read_text())
            if "results" not in data:
                print(f"Warning: 'results' key missing in {file_path}")
                continue

            results = data["results"]
            for index_int, judgment, explanation in items:
                if index_int < 0 or index_int >= len(results):
                    print(f"Warning: Index {index_int} out of bounds for {file_path}")
                    continue
                results[index_int]["judgment"] = judgment
                results[index_int]["explanation"] = explanation

            under_steering = 0
            successful_steering = 0
            over_steering = 0
            total = 0
            for r in results:
                if "judgment" not in r:
                    continue
                j = r["judgment"]
                if j == 0:
                    under_steering += 1
                elif j == 1:
                    successful_steering += 1
                elif j == 2:
                    over_steering += 1
                total += 1

            if total > 0 and isinstance(data.get("metadata"), dict):
                data["metadata"]["under_steering"] = under_steering / total
                data["metadata"]["successful_steering"] = successful_steering / total
                data["metadata"]["over_steering"] = over_steering / total

            file_path.write_text(json.dumps(data, indent=4))
        except Exception as e:
            print(f"Error updating file {file_path}: {e}")

    print("\nBatch processing complete.")


def _iter_batch_ids(args_batch_id: Optional[str], args_batch_ids: Optional[Iterable[str]]) -> Iterable[str]:
    if args_batch_ids:
        return args_batch_ids
    if args_batch_id:
        return [args_batch_id]
    return []


def fetch_last_k_batch_ids(k: int, *, client: Optional[OpenAI] = None) -> list[str]:
    """Return IDs of the K most recently listed batches (OpenAI API order, newest first)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    client = client or get_client()
    page = client.batches.list(limit=k)
    return [b.id for b in page.data]


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Check OpenAI Batch status; if completed, download and process results."
    )
    parser.add_argument(
        "--last_k",
        type=int,
        default=None,
        metavar="K",
        help="Use the K most recent batch IDs from the OpenAI API (same key as OPENAI_API_KEY). "
        "Mutually exclusive with --batch_id / --batch_ids.",
    )
    parser.add_argument("--batch_id", type=str, default=None, help="Batch ID to check")
    parser.add_argument("--batch_ids", nargs="*", default=None, help="One or more batch IDs to check")
    parser.add_argument("--watch", action="store_true", help="Watch status every --interval seconds")
    parser.add_argument("--interval", type=int, default=60, help="Watch interval in seconds (default: 60)")
    parser.add_argument(
        "--download_dir",
        type=str,
        default="data/batch_outputs",
        help="Directory to store downloaded JSONL (default: data/batch_outputs)",
    )
    parser.add_argument(
        "--no_process",
        action="store_true",
        help="Do not process output into data/result/... (download only)",
    )
    parser.add_argument(
        "--no_download_errors",
        action="store_true",
        help="Do not download error file (if present)",
    )
    args = parser.parse_args()

    if args.last_k is not None:
        if args.batch_id is not None or (args.batch_ids is not None and len(args.batch_ids) > 0):
            print("Error: use either --last_k or --batch_id / --batch_ids, not both")
            sys.exit(2)
        if args.last_k < 1:
            print("Error: --last_k must be >= 1")
            sys.exit(2)
        client = get_client()
        batch_ids = fetch_last_k_batch_ids(args.last_k, client=client)
        print(f"Using {len(batch_ids)} batch ID(s) from API (last {args.last_k} listed).")
        for bid in batch_ids:
            print(f"  {bid}")
    else:
        batch_ids = list(_iter_batch_ids(args.batch_id, args.batch_ids))
        if not batch_ids:
            print("Error: provide --last_k, or --batch_id, or --batch_ids")
            sys.exit(2)
        client = get_client()
    download_dir = Path(args.download_dir)
    download_errors = not args.no_download_errors

    pending = set(batch_ids)
    while pending:
        completed_now = []
        for batch_id in sorted(pending):
            status, output_path, _error_path = check_and_download_batch(
                batch_id,
                client=client,
                download_dir=download_dir,
                download_errors=download_errors,
                verbose=True,
            )

            if status == "completed" and output_path and not args.no_process:
                process_batch_output_file(output_path, repo_root=Path("."))
                completed_now.append(batch_id)
            elif status in ["failed", "cancelled", "expired"]:
                completed_now.append(batch_id)

        for bid in completed_now:
            pending.discard(bid)

        if not pending or not args.watch:
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    cli()
