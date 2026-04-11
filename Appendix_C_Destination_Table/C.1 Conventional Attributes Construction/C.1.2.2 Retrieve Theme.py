"""
C.1.2.2 – Retreive Theme

This script:
Retreive all required themes from OneMap Singapore.

Author: Zhang Wenyu
Date: 2025-12-15
"""

import requests
import json
import time
import os

# ========== Config ==========
BASE_URL = "https://www.onemap.gov.sg/api/public/themesvc/retrieveTheme"
OUTPUT_DIR = "/Users/zhangwenyu/Desktop/"

headers = {"Authorization": YOUR_HEADER}

QUERY_NAMES = [
    "ssot_hawkercentres",
    "hotels",
    "museum",
    "monuments",
    "historicsites",
    "tourism",
    "theatre",
    "libraries",
    "communityclubs",
    "nationalparks",
    "hdb_active_blk_p",
    "customs_tabacco_product",
    "customs_zero_gst_goods",
    "ura_popspoints_pt"
]

# Close Proxies
PROXIES = {"http": None, "https": None}

# ========== Loop ==========
for q in QUERY_NAMES:
    print(f"\nRetrieving theme: {q}")

    params = {
        "queryName": q,
        "fjson": "Y"
    }

    try:
        response = requests.get(
            BASE_URL,
            params=params,
            headers=headers,
            proxies=PROXIES,
            timeout=120
        )

        response.raise_for_status()
        data = response.json()

        output_path = os.path.join(OUTPUT_DIR, f"onemap_{q}.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved: {output_path}")

        # Feature Count（GeoJSON）
        if "features" in data:
            print(f"Total features: {len(data['features'])}")
        elif "SrchResults" in data:
            print(f"Total records: {len(data['SrchResults'])}")

    except Exception as e:
        print(f"Failed for {q}: {e}")

    # ========== Rate limit ==========
    print("Sleeping 30 seconds...")
    time.sleep(30)

print("\nAll themes retrieved.")
