import os
import argparse
from collections import defaultdict
import re


def clean_duplicates(model_name):
    # Determine the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # The model folder is expected to be a sibling or child of the script directory,
    # but based on workspace structure, it's in the same folder as this script.
    base_path = os.path.join(script_dir, model_name)

    if not os.path.exists(base_path):
        print(f"Error: Directory '{base_path}' does not exist.")
        return

    print(f"Scanning directory: {base_path}")


def count_lines(filepath):
    """Count variables lines in a file."""
    try:
        with open(filepath, "r") as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"    Error reading {filepath}: {e}")
        return -1


def get_desta_reference_count(script_dir, rel_path, shot_num):
    """Find the corresponding file in desta2_5 and return its line count."""
    parts = rel_path.split(os.sep)
    if len(parts) > 0 and parts[0] in ["SER", "GR"]:
        if "keywords_existence" in parts or "keywords_forbidden_words" in parts:
            return 0
    desta_base = os.path.join(script_dir, "desta2_5")
    desta_dir = os.path.join(desta_base, rel_path)

    if not os.path.exists(desta_dir):
        return None

    # Look for files matching the shot number
    candidates = []
    for f in os.listdir(desta_dir):
        if f.endswith(".jsonl") and f.startswith(f"output_{shot_num}-shot_"):
            candidates.append(os.path.join(desta_dir, f))

    if not candidates:
        return None

    # If multiple, prefer newest
    candidates.sort(reverse=True)
    return count_lines(candidates[0])


def clean_duplicates(model_name):
    # Determine the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # The model folder is expected to be a sibling or child of the script directory,
    # but based on workspace structure, it's in the same folder as this script.
    base_path = os.path.join(script_dir, model_name)

    if not os.path.exists(base_path):
        print(f"Error: Directory '{base_path}' does not exist.")
        return

    print(f"Scanning directory: {base_path}")

    # Walk through the directory tree
    for root, dirs, files in os.walk(base_path):
        # We are looking for files starting with 'output_' and ending with '.jsonl'
        jsonl_files = [
            f for f in files if f.endswith(".jsonl") and f.startswith("output_")
        ]

        # Only process directories containing these files
        if not jsonl_files:
            continue

        print(f"\nProcessing directory: {root}")

        # Calculate relative path to look up in desta2_5
        rel_path = os.path.relpath(root, base_path)

        # Group files by shot count
        # Pattern: output_{i}-shot_{timestamp}.jsonl or similar
        shot_files = defaultdict(list)

        for filename in jsonl_files:
            # Extract the shot part. Expecting output_X-shot_...
            # Using regex to be safer
            match = re.match(r"output_(\d+)-shot_.*\.jsonl", filename)
            if match:
                shot_num = match.group(1)
                key = f"{shot_num}-shot"
                shot_files[key].append(filename)
            else:
                # Handle cases specifically mentioned: output_{i}-shot_....jsonl
                # If format doesn't match exactly, maybe warn or skip?
                # Let's try simple split if regex fails, or just skip
                pass

        # Check for duplicates and clean
        for shot, file_list in shot_files.items():
            shot_num = shot.split("-")[0]

            # Get expected line count from desta2_5 if not processing desta2_5 itself
            expected_count = None
            if model_name != "desta2_5":
                expected_count = get_desta_reference_count(
                    script_dir, rel_path, shot_num
                )
                if expected_count is None:
                    raise ValueError(
                        f"Could not find reference file in desta2_5 for {rel_path}, shot {shot_num}"
                    )

                print(
                    f"  Reference line count from desta2_5 for {shot}: {expected_count}"
                )

            # Sort by filename descending (newest timestamp first)
            file_list.sort(reverse=True)

            candidates = []
            for f in file_list:
                full_path = os.path.join(root, f)
                lc = count_lines(full_path)
                candidates.append({"name": f, "path": full_path, "lines": lc})

            # If we have an expected count, try to filter candidates
            final_choice = None

            if expected_count is not None:
                valid_candidates = [
                    c for c in candidates if c["lines"] == expected_count
                ]
                if valid_candidates:
                    # Pick newest valid
                    final_choice = valid_candidates[0]
                    if len(valid_candidates) < len(candidates):
                        print(
                            f"  Filtered out {len(candidates) - len(valid_candidates)} candidates with incorrect line counts."
                        )
                else:
                    print(
                        f"  [Warning] No candidates for {shot} match reference line count {expected_count}."
                    )
                    # Fallback to newest regardless of count? Or keep newest (as per orig behavior)
                    # The user said "Make sure...", but if none match, we can't invent one.
                    # We will keep the newest one available and warn.
                    final_choice = candidates[0]
            else:
                # No reference or we are desta2_5, just pick newest
                final_choice = candidates[0]

            if final_choice:
                keep_file = final_choice["name"]

                # Identify files to remove (everything else in the original list)
                remove_files = [f for f in file_list if f != keep_file]

                if remove_files:
                    print(f"  Found duplicates for {shot}:")
                    print(f"    Keeping: {keep_file} (lines: {final_choice['lines']})")
                    for rm_file in remove_files:
                        path_to_remove = os.path.join(root, rm_file)
                        print(f"    Removing: {rm_file}")
                        try:
                            os.remove(path_to_remove)
                        except OSError as e:
                            print(f"    Error removing {rm_file}: {e}")
                else:
                    # Even if no duplicates, we might want to warn if line count mismatches single file
                    if (
                        expected_count is not None
                        and final_choice["lines"] != expected_count
                    ):
                        print(
                            f"  [Warning] Kept file {keep_file} (lines: {final_choice['lines']}) does not match reference count {expected_count}."
                        )

        # Verification step: Check if we have exactly 9 files for 0-8 shots
        # Re-scan or just use the keys from shot_files (since we only removed extras)
        present_shots = sorted([int(k.split("-")[0]) for k in shot_files.keys()])
        expected_shots = list(range(9))

        if present_shots == expected_shots:
            # We have exactly one file for each of 0, 1, ..., 8
            # And since we cleaned duplicates, this directory is good.
            print(
                f"  [OK] Directory contains exactly 9 unique files (0-shot to 8-shot)."
            )
        else:
            print(
                f"  [Warning] Directory does not contain exactly the 9 expected files (0-8 shots)."
            )
            print(f"  Found shots: {present_shots}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean duplicate model response files."
    )
    parser.add_argument(
        "model_name", help="Name of the model folder under model_responses"
    )

    args = parser.parse_args()

    clean_duplicates(args.model_name)
