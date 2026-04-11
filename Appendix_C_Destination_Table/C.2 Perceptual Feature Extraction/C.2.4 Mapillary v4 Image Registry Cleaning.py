#!/usr/bin/env python3
"""
C.2.4 - Mapillary v4 Image Registry Cleaning Script (Concurrent + Autosave)

- Validate image_id via Mapillary Graph API v4
- Keep only images accessible via v4 (HTTP 200 + thumbnail available)
- Concurrent requests with global rate limiting
- Autosave every N images
- Retry non-400 errors up to MAX_RETRIES

Author: Zhang Wenyu
Date: 2026-01-27
"""

import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# CONFIGURATION
# =========================

INPUT_CSV = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery/Image_registry.csv"
OUTPUT_DIR = os.path.dirname(INPUT_CSV)

CLEAN_CSV = os.path.join(OUTPUT_DIR, "Image_registry_v4_clean.csv")
CHECKPOINT_CSV = os.path.join(OUTPUT_DIR, "Image_registry_v4_checkpoint.csv")
ERROR_CSV = os.path.join(OUTPUT_DIR, "Image_registry_v4_errors.csv")

ACCESS_TOKEN = "YOUR_TOKEN"

API_URL = "https://graph.mapillary.com/{}"
FIELDS = "id,thumb_1024_url"

MAX_WORKERS = 8
MAX_RETRIES = 3
REQUEST_TIMEOUT = 10

# Global rate limit (~6–7 requests per second total)
MIN_INTERVAL = 0.15

AUTOSAVE_EVERY = 50

# =========================
# GLOBAL STATE
# =========================

rate_lock = Lock()
last_request_time = 0.0

save_lock = Lock()
processed_counter = 0

stats = {
    "processed": 0,
    "v4_valid": 0,
    "bad_400": 0,
    "other_errors": 0
}

clean_rows = []
error_rows = []

HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}"
}

# =========================
# HELPERS
# =========================

def rate_limited_request(url, params):
    """Ensure global rate limiting across threads."""
    global last_request_time

    with rate_lock:
        elapsed = time.time() - last_request_time
        if elapsed < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - elapsed)
        last_request_time = time.time()

    return requests.get(
        url,
        headers=HEADERS,
        params=params,
        timeout=REQUEST_TIMEOUT
    )


def validate_image_v4(row):
    """
    Validate a single image_id using Mapillary v4 API.
    Returns a tuple: (status, payload)
    """
    image_id = str(row["image_id"])

    url = API_URL.format(image_id)
    params = {"fields": FIELDS}

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = rate_limited_request(url, params)

            if response.status_code == 200:
                data = response.json()
                thumb = data.get("thumb_1024_url")

                if thumb:
                    merged = row.to_dict()
                    merged["thumb_1024_url"] = thumb
                    return ("ok", merged)
                else:
                    return ("bad_400", image_id)

            elif response.status_code == 400:
                return ("bad_400", image_id)

            else:
                last_error = f"HTTP {response.status_code}"
                time.sleep(1)

        except Exception as e:
            last_error = str(e)
            time.sleep(1)

    return ("error", {
        "image_id": image_id,
        "error": last_error
    })


def autosave_checkpoint():
    """Save checkpoint CSV and print summary."""
    with save_lock:
        if clean_rows:
            pd.DataFrame(clean_rows).to_csv(CHECKPOINT_CSV, index=False)

        if error_rows:
            pd.DataFrame(error_rows).to_csv(ERROR_CSV, index=False)

        print("\n================ CHECKPOINT =================")
        print(f"Processed      : {stats['processed']}")
        print(f"v4 valid       : {stats['v4_valid']}")
        print(f"Invalid (400)  : {stats['bad_400']}")
        print(f"Other errors   : {stats['other_errors']}")
        print("Checkpoint saved.")
        print("============================================\n")


# =========================
# MAIN
# =========================

def main():
    global processed_counter

    print("=" * 70)
    print("Mapillary v4 Image Registry Cleaning (Concurrent)")
    print("=" * 70)

    print("\n[1] Loading registry...")
    df = pd.read_csv(INPUT_CSV, dtype={"image_id": str})
    print(f"Total rows: {len(df)}")

    # Resume from checkpoint if exists
    if os.path.exists(CHECKPOINT_CSV):
        done_ids = set(pd.read_csv(CHECKPOINT_CSV)["image_id"].astype(str))
        df = df[~df["image_id"].astype(str).isin(done_ids)]
        print(f"Resuming from checkpoint, remaining: {len(df)}")

    print("\n[2] Start cleaning...\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(validate_image_v4, row): idx
            for idx, row in df.iterrows()
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            status, payload = future.result()

            stats["processed"] += 1
            processed_counter += 1

            if status == "ok":
                clean_rows.append(payload)
                stats["v4_valid"] += 1

            elif status == "bad_400":
                stats["bad_400"] += 1

            else:
                error_rows.append(payload)
                stats["other_errors"] += 1

            if processed_counter % AUTOSAVE_EVERY == 0:
                autosave_checkpoint()

    # Final save
    print("\n[3] Final save...")
    if clean_rows:
        pd.DataFrame(clean_rows).to_csv(CLEAN_CSV, index=False)
    if error_rows:
        pd.DataFrame(error_rows).to_csv(ERROR_CSV, index=False)

    print("\n================ FINAL SUMMARY ================")
    print(f"Total processed : {stats['processed']}")
    print(f"v4 valid images : {stats['v4_valid']}")
    print(f"Invalid (400)   : {stats['bad_400']}")
    print(f"Other errors   : {stats['other_errors']}")
    print(f"Saved to       : {CLEAN_CSV}")
    print("================================================")


if __name__ == "__main__":
    main()
