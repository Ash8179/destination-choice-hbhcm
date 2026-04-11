import pandas as pd
import uuid
import ast

# =========================
# File paths
# =========================
MALL_XLSX_PATH = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.1 Define Origins and Destinations/Destinations/mall raw.xlsx"
DEST_XLSX_PATH = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.1 Define Origins and Destinations/Destinations/Destination.xlsx"

# =========================
# Load data
# =========================
df_dest = pd.read_excel(DEST_XLSX_PATH)
df_mall = pd.read_excel(MALL_XLSX_PATH)

records = []

for idx, row in df_mall.iterrows():
    # Parse lat,lon tuple from "(lat, lon)"
    latlon = ast.literal_eval(row['(lat,lon)'])
    lat = float(latlon[0])
    lon = float(latlon[1])
    
    record = {
        # ---------- Group A ----------
        "poi_id": f"ML_{uuid.uuid4().hex[:8]}",
        "name": row['name'],
        "category": "retail",
        "subtype": "mall",
        
        # ---------- Group B ----------
        "centroid_x": lat,
        "centroid_y": lon,
        
        # ---------- Group C ----------
        "storey_count": row.get('storey_count'),
        "fnb_count": row.get('fnb_count'),
        "retail_count": row.get('retail_count'),
        "entertainment_count": row.get('entertainment_count'),
        "service_count": row.get('service_count'),
        "total_count": row.get('total_count'),
        "diversity_index": None,  # 先留空
        
        "has_supermarket": row.get('has_supermarket'),
        "has_cinema": row.get('has_cinema'),
        "has_department_store": row.get('has_department_store'),
        "has_foodcourt": row.get('has_foodcourt'),
        "has_event_space": row.get('has_event_space'),
        
        # ---------- Group G ----------
        "experiential_flag": 1,
        "heritage_flag": 0,
        "night_only_flag": 0
    }
    
    records.append(record)

# =========================
# Append to destination table
# =========================
df_new = pd.DataFrame(records)
df_updated = pd.concat([df_dest, df_new], ignore_index=True)

# =========================
# Save
# =========================
df_updated.to_excel(DEST_XLSX_PATH, index=False)

print(f"Successfully added {len(df_new)} malls to destination table.")
