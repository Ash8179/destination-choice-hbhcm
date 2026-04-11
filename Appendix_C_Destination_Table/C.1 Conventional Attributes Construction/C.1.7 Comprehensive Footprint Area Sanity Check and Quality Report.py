"""
C.1.7 - Comprehensive Footprint Area Sanity Check and Quality Report

This script performs quality checks on all destination types:
- Statistical analysis by subtype
- Outlier detection using IQR and domain knowledge
- Visual distribution plots
- Detailed flagging of suspicious matches

Categories covered:
- Retail: mall, lifestyle_street, hawker_centre
- Leisure: museum, theatre, historic_site, monument, park

Author: Zhang Wenyu
Date: 2025-12-24
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

INPUT_PATH = (
    "/Users/zhangwenyu/Desktop/Destination.xlsx"
)

OUTPUT_PATH = (
    "/Users/zhangwenyu/Desktop/NUSFYP/Footprint_Quality_Report.xlsx"
)

PLOT_PATH = (
    "/Users/zhangwenyu/Desktop/NUSFYP/Footprint_Distribution.png"
)

# Area thresholds by subtype (sqm)
# Based on Singapore's actual building characteristics
AREA_THRESHOLDS = {
    # Retail category
    "mall": {
        "min": 500,           # Small retail with parking
        "typical_min": 2000,  # Small neighbourhood mall
        "typical_max": 100000, # Large shopping centre
        "max": 200000,        # Mega mall (VivoCity, Jewel Changi)
        "description": "Shopping malls and centres"
    },
    "lifestyle_street": {
        "min": 200,           # Small shophouse block
        "typical_min": 1000,  # Typical lifestyle street
        "typical_max": 50000, # Large lifestyle precinct
        "max": 100000,        # Major lifestyle district
        "description": "Lifestyle streets, shophouse areas, dining precincts"
    },
    "hawker_centre": {
        "min": 300,           # Very small hawker
        "typical_min": 800,   # Small hawker centre
        "typical_max": 5000,  # Typical hawker
        "max": 15000,         # Maxwell, Old Airport Road
        "description": "Hawker centres and food courts"
    },
    
    # Leisure category
    "museum": {
        "min": 100,           # Small gallery/museum
        "typical_min": 500,  # Small to medium museum
        "typical_max": 20000, # Large museum
        "max": 50000,        # National Museum, ArtScience Museum
        "description": "Museums, galleries, exhibition spaces"
    },
    "theatre": {
        "min": 300,           # Small black box theatre
        "typical_min": 1500,  # Medium theatre
        "typical_max": 15000, # Large theatre complex
        "max": 25000,         # Esplanade complex
        "description": "Theatres, performance venues, concert halls"
    },
    "historic_site": {
        "min": 100,           # Small historic building
        "typical_min": 500,   # Typical heritage building
        "typical_max": 10000, # Large historic complex
        "max": 50000,         # Fort Canning complex
        "description": "Historic buildings and heritage sites"
    },
    "monument": {
        "min": 50,            # Small monument/statue
        "typical_min": 100,   # Typical monument
        "typical_max": 3000,  # Large monument complex
        "max": 10000,         # Major memorial complex
        "description": "Monuments, memorials, statues"
    },
    "park": {
        "min": 500,           # Pocket park
        "typical_min": 2000,  # Small neighbourhood park
        "typical_max": 500000, # Large regional park
        "max": 1500000,       # Gardens by the Bay, East Coast Park
        "description": "Parks, gardens, green spaces"
    }
}

# Category grouping
CATEGORY_MAP = {
    "mall": "Retail",
    "lifestyle_street": "Retail",
    "hawker_centre": "Retail",
    "museum": "Leisure",
    "theatre": "Leisure",
    "historic_site": "Leisure",
    "monument": "Leisure",
    "park": "Leisure"
}

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------

print("="*80)
print("COMPREHENSIVE FOOTPRINT QUALITY ANALYSIS")
print("="*80)

print("\nLoading footprint data...")
df = pd.read_excel(INPUT_PATH)

# Add category column
df['category'] = df['subtype'].map(CATEGORY_MAP)

# Filter only matched POIs
matched_df = df[df['match_type'] != 'no_match'].copy()
no_match_df = df[df['match_type'] == 'no_match'].copy()

print(f"\nDataset Overview:")
print(f"  Total POIs:           {len(df):>6,}")
print(f"  Successfully matched: {len(matched_df):>6,} ({len(matched_df)/len(df):>6.1%})")
print(f"  No match:            {len(no_match_df):>6,} ({len(no_match_df)/len(df):>6.1%})")

# ------------------------------------------------------------------
# MATCH RATE BY SUBTYPE
# ------------------------------------------------------------------

print("\n" + "="*80)
print("MATCH RATE BY SUBTYPE")
print("="*80)

match_stats = []
for subtype in sorted(df['subtype'].dropna().unique()):
    subtype_df = df[df['subtype'] == subtype]
    matched = subtype_df[subtype_df['match_type'] != 'no_match']
    
    match_stats.append({
        'Subtype': subtype,
        'Category': CATEGORY_MAP.get(subtype, 'Unknown'),
        'Total': len(subtype_df),
        'Matched': len(matched),
        'No Match': len(subtype_df) - len(matched),
        'Match Rate': len(matched) / len(subtype_df) if len(subtype_df) > 0 else 0
    })

match_stats_df = pd.DataFrame(match_stats)
match_stats_df = match_stats_df.sort_values(['Category', 'Subtype'])

for category in ['Retail', 'Leisure']:
    cat_data = match_stats_df[match_stats_df['Category'] == category]
    if not cat_data.empty:
        print(f"\n{category.upper()}:")
        for _, row in cat_data.iterrows():
            print(f"  {row['Subtype']:<20s}: {row['Matched']:>4}/{row['Total']:<4} "
                  f"({row['Match Rate']:>6.1%})  [{row['No Match']:>3} unmatched]")

# ------------------------------------------------------------------
# STATISTICAL ANALYSIS BY SUBTYPE
# ------------------------------------------------------------------

def analyze_subtype(subtype_df, subtype_name, thresholds):
    """Perform statistical analysis for a subtype"""
    
    areas = subtype_df['footprint_area'].dropna()
    
    if len(areas) == 0:
        return None
    
    # Basic statistics
    stats = {
        'subtype': subtype_name,
        'category': CATEGORY_MAP.get(subtype_name.lower(), 'Unknown'),
        'count': len(areas),
        'mean': areas.mean(),
        'median': areas.median(),
        'std': areas.std(),
        'min': areas.min(),
        'max': areas.max(),
        'q25': areas.quantile(0.25),
        'q75': areas.quantile(0.75)
    }
    
    # IQR outlier detection (using 3*IQR for more lenient bounds)
    q1 = stats['q25']
    q3 = stats['q75']
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    
    stats['iqr_lower'] = max(lower_bound, 0)
    stats['iqr_upper'] = upper_bound
    
    # Threshold-based checks
    stats['below_min'] = (areas < thresholds['min']).sum()
    stats['above_max'] = (areas > thresholds['max']).sum()
    stats['below_typical'] = (areas < thresholds['typical_min']).sum()
    stats['above_typical'] = (areas > thresholds['typical_max']).sum()
    
    # IQR outliers
    stats['iqr_outliers_low'] = (areas < stats['iqr_lower']).sum()
    stats['iqr_outliers_high'] = (areas > stats['iqr_upper']).sum()
    
    return stats


print("\n" + "="*80)
print("STATISTICAL ANALYSIS BY SUBTYPE")
print("="*80)

subtype_stats = []

for category in ['Retail', 'Leisure']:
    print(f"\n{category.upper()} CATEGORY")
    print("-"*80)
    
    cat_subtypes = [st for st, cat in CATEGORY_MAP.items() if cat == category]
    
    for subtype in sorted(cat_subtypes):
        subtype_df = matched_df[matched_df['subtype'] == subtype]
        
        if len(subtype_df) == 0:
            continue
        
        thresholds = AREA_THRESHOLDS.get(subtype.lower(), {
            'min': 100, 'typical_min': 500, 'typical_max': 50000, 'max': 200000
        })
        
        stats = analyze_subtype(subtype_df, subtype, thresholds)
        
        if stats:
            subtype_stats.append(stats)
            
            print(f"\n  {subtype.upper()} ({thresholds.get('description', '')})")
            print(f"    Sample size:        {stats['count']:>6,}")
            print(f"    Mean:               {stats['mean']:>10,.1f} sqm")
            print(f"    Median:             {stats['median']:>10,.1f} sqm")
            print(f"    Std Dev:            {stats['std']:>10,.1f} sqm")
            print(f"    Range:              [{stats['min']:>10,.1f}, {stats['max']:>10,.1f}] sqm")
            print(f"    IQR (Q25-Q75):      [{stats['q25']:>10,.1f}, {stats['q75']:>10,.1f}]")
            print(f"    IQR Bounds (3x):    [{stats['iqr_lower']:>10,.1f}, {stats['iqr_upper']:>10,.1f}]")
            
            print(f"\n    Quality Flags:")
            print(f"      Below absolute min:     {stats['below_min']:>4} ({stats['below_min']/stats['count']:>5.1%})")
            print(f"      Above absolute max:     {stats['above_max']:>4} ({stats['above_max']/stats['count']:>5.1%})")
            print(f"      Below typical range:    {stats['below_typical']:>4} ({stats['below_typical']/stats['count']:>5.1%})")
            print(f"      Above typical range:    {stats['above_typical']:>4} ({stats['above_typical']/stats['count']:>5.1%})")
            print(f"      IQR outliers (low):     {stats['iqr_outliers_low']:>4} ({stats['iqr_outliers_low']/stats['count']:>5.1%})")
            print(f"      IQR outliers (high):    {stats['iqr_outliers_high']:>4} ({stats['iqr_outliers_high']/stats['count']:>5.1%})")

# ------------------------------------------------------------------
# FLAG SUSPICIOUS ENTRIES
# ------------------------------------------------------------------

print("\n" + "="*80)
print("FLAGGING SUSPICIOUS ENTRIES")
print("="*80)

flagged_entries = []

for subtype in matched_df['subtype'].dropna().unique():
    subtype_df = matched_df[matched_df['subtype'] == subtype].copy()
    thresholds = AREA_THRESHOLDS.get(subtype.lower(), {
        'min': 100, 'typical_min': 500, 'typical_max': 50000, 'max': 200000
    })
    
    # Calculate IQR bounds
    q1 = subtype_df['footprint_area'].quantile(0.25)
    q3 = subtype_df['footprint_area'].quantile(0.75)
    iqr = q3 - q1
    iqr_lower = max(q1 - 3 * iqr, 0)
    iqr_upper = q3 + 3 * iqr
    
    for idx, row in subtype_df.iterrows():
        area = row['footprint_area']
        flags = []
        severity = []
        
        # Check against absolute thresholds
        if area < thresholds['min']:
            flags.append(f"Below absolute minimum ({thresholds['min']:,} sqm)")
            severity.append(3)  # Critical
        elif area < thresholds['typical_min']:
            flags.append(f"Below typical minimum ({thresholds['typical_min']:,} sqm)")
            severity.append(2)  # Warning
        
        if area > thresholds['max']:
            flags.append(f"Above absolute maximum ({thresholds['max']:,} sqm)")
            severity.append(3)  # Critical
        elif area > thresholds['typical_max']:
            flags.append(f"Above typical maximum ({thresholds['typical_max']:,} sqm)")
            severity.append(2)  # Warning
        
        # Check IQR outliers
        if area < iqr_lower:
            flags.append(f"Statistical outlier (low, < {iqr_lower:,.0f} sqm)")
            severity.append(1)  # Info
        if area > iqr_upper:
            flags.append(f"Statistical outlier (high, > {iqr_upper:,.0f} sqm)")
            severity.append(1)  # Info
        
        if flags:
            flagged_entries.append({
                'poi_id': row['poi_id'],
                'name': row['name'],
                'subtype': row['subtype'],
                'category': CATEGORY_MAP.get(row['subtype'], 'Unknown'),
                'footprint_area': area,
                'match_type': row['match_type'],
                'flags': ' | '.join(flags),
                'max_severity': max(severity),
                'flag_count': len(flags)
            })

flagged_df = pd.DataFrame(flagged_entries)

if not flagged_df.empty:
    # Categorize by severity
    critical = flagged_df[flagged_df['max_severity'] == 3]
    warning = flagged_df[flagged_df['max_severity'] == 2]
    info = flagged_df[flagged_df['max_severity'] == 1]
    
    print(f"\nTotal flagged entries: {len(flagged_df)}")
    print(f"  Critical (beyond absolute limits): {len(critical)}")
    print(f"  Warning (outside typical range):   {len(warning)}")
    print(f"  Info (statistical outliers):       {len(info)}")
    
    # Show critical cases
    if not critical.empty:
        print(f"\n{'='*80}")
        print("CRITICAL CASES (REQUIRE IMMEDIATE REVIEW)")
        print('='*80)
        
        for category in ['Retail', 'Leisure']:
            cat_critical = critical[critical['category'] == category]
            if not cat_critical.empty:
                print(f"\n{category}:")
                for _, row in cat_critical.iterrows():
                    print(f"  • {row['name'][:50]:<50s}")
                    print(f"    Type: {row['subtype']:<20s} | Area: {row['footprint_area']:>12,.1f} sqm")
                    print(f"    Match: {row['match_type']:<20s}")
                    print(f"    Issue: {row['flags']}")
                    print()
    
    # Show top warnings by category
    if not warning.empty:
        print(f"\n{'='*80}")
        print("WARNING CASES (RECOMMEND REVIEW)")
        print('='*80)
        
        for category in ['Retail', 'Leisure']:
            cat_warning = warning[warning['category'] == category]
            if not cat_warning.empty:
                print(f"\n{category} ({len(cat_warning)} cases):")
                for _, row in cat_warning.head(5).iterrows():
                    print(f"  • {row['name'][:50]:<50s} | {row['subtype']:<15s} | "
                          f"{row['footprint_area']:>10,.1f} sqm")
                if len(cat_warning) > 5:
                    print(f"  ... and {len(cat_warning) - 5} more")
else:
    print("\nNo suspicious entries found! All footprints within expected ranges.")

# ------------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------------

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Create figure with subplots
n_subtypes = len(matched_df['subtype'].dropna().unique())
n_cols = 3
n_rows = (n_subtypes + n_cols - 1) // n_cols

fig = plt.figure(figsize=(20, 5 * n_rows))
gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)

# Color scheme
colors = {
    'Retail': '#FF6B6B',
    'Leisure': '#4ECDC4'
}

plot_idx = 0
for category in ['Retail', 'Leisure']:
    cat_subtypes = sorted([st for st, cat in CATEGORY_MAP.items() if cat == category])
    
    for subtype in cat_subtypes:
        subtype_data = matched_df[matched_df['subtype'] == subtype]['footprint_area']
        
        if len(subtype_data) == 0:
            continue
        
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        thresholds = AREA_THRESHOLDS.get(subtype.lower(), {})
        
        # Histogram
        n, bins, patches = ax.hist(subtype_data, bins=40, alpha=0.7,
                                   color=colors[category], edgecolor='black', linewidth=0.5)
        
        # Add threshold lines
        if 'min' in thresholds:
            ax.axvline(thresholds['min'], color='red', linestyle='--',
                      linewidth=2, label=f"Absolute min", alpha=0.7)
        if 'max' in thresholds:
            ax.axvline(thresholds['max'], color='red', linestyle='--',
                      linewidth=2, label=f"Absolute max", alpha=0.7)
        if 'typical_min' in thresholds:
            ax.axvline(thresholds['typical_min'], color='orange', linestyle=':',
                      linewidth=1.5, label=f"Typical range", alpha=0.7)
        if 'typical_max' in thresholds:
            ax.axvline(thresholds['typical_max'], color='orange', linestyle=':',
                      linewidth=1.5, alpha=0.7)
        
        # Add median and mean
        median = subtype_data.median()
        mean = subtype_data.mean()
        ax.axvline(median, color='green', linestyle='-',
                  linewidth=2.5, label=f"Median ({median:,.0f})", alpha=0.8)
        ax.axvline(mean, color='blue', linestyle='--',
                  linewidth=2, label=f"Mean ({mean:,.0f})", alpha=0.7)
        
        # Formatting
        ax.set_title(f'{subtype.upper().replace("_", " ")}\n(n={len(subtype_data)})',
                    fontweight='bold', fontsize=11)
        ax.set_xlabel('Footprint Area (sqm)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
        ax.tick_params(axis='both', labelsize=8)
        
        # Format x-axis
        ax.ticklabel_format(style='plain', axis='x')
        
        plot_idx += 1

# Add overall title
fig.suptitle('Footprint Area Distribution Analysis by Destination Type',
            fontsize=18, fontweight='bold', y=0.995)

plt.savefig(PLOT_PATH, dpi=300, bbox_inches='tight')
print(f"Distribution plot saved to: {PLOT_PATH}")

# ------------------------------------------------------------------
# CREATE BOX PLOT SUMMARY
# ------------------------------------------------------------------

BOXPLOT_PATH = PLOT_PATH.replace('.png', '_boxplot.png')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Retail category
retail_subtypes = [st for st, cat in CATEGORY_MAP.items() if cat == 'Retail']
retail_data = []
retail_labels = []
for st in sorted(retail_subtypes):
    data = matched_df[matched_df['subtype'] == st]['footprint_area'].dropna()
    if len(data) > 0:
        retail_data.append(data)
        retail_labels.append(f"{st}\n(n={len(data)})")

if retail_data:
    bp1 = ax1.boxplot(retail_data, labels=retail_labels, patch_artist=True,
                     showmeans=True, meanline=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('#FF6B6B')
        patch.set_alpha(0.6)
    ax1.set_title('RETAIL CATEGORY', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Footprint Area (sqm)', fontsize=12)
    ax1.grid(axis='y', alpha=0.3, linestyle=':')
    ax1.tick_params(axis='x', rotation=0, labelsize=10)

# Leisure category
leisure_subtypes = [st for st, cat in CATEGORY_MAP.items() if cat == 'Leisure']
leisure_data = []
leisure_labels = []
for st in sorted(leisure_subtypes):
    data = matched_df[matched_df['subtype'] == st]['footprint_area'].dropna()
    if len(data) > 0:
        leisure_data.append(data)
        leisure_labels.append(f"{st}\n(n={len(data)})")

if leisure_data:
    bp2 = ax2.boxplot(leisure_data, labels=leisure_labels, patch_artist=True,
                     showmeans=True, meanline=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('#4ECDC4')
        patch.set_alpha(0.6)
    ax2.set_title('LEISURE CATEGORY', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Footprint Area (sqm)', fontsize=12)
    ax2.grid(axis='y', alpha=0.3, linestyle=':')
    ax2.tick_params(axis='x', rotation=0, labelsize=10)

plt.suptitle('Footprint Area Distribution: Box Plot Summary',
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(BOXPLOT_PATH, dpi=300, bbox_inches='tight')
print(f"Box plot saved to: {BOXPLOT_PATH}")

# ==================================================================
# ADDITIONAL CRITICAL SANITY CHECKS (MUST-HAVE)
# ==================================================================

print("\n" + "="*80)
print("ADDITIONAL SANITY CHECKS")
print("="*80)

# --------------------------------------------------
# [1] CRS / UNIT SANITY CHECK
# --------------------------------------------------
# Extremely small positive areas often indicate CRS or unit errors
unit_sanity_issues = matched_df[
    (matched_df['footprint_area'] > 0) &
    (matched_df['footprint_area'] < 20)
]

if not unit_sanity_issues.empty:
    print("\nUNIT / CRS SANITY WARNING")
    print(f"Detected {len(unit_sanity_issues)} footprints < 20 sqm.")
    print("Likely causes:")
    print("  - Area computed in degrees instead of meters")
    print("  - Invalid or degenerate polygon")
    print("  - Incorrect CRS during area calculation")
    
    print("\nExamples:")
    print(
        unit_sanity_issues[['poi_id', 'name', 'subtype', 'footprint_area']]
        .head(5)
        .to_string(index=False)
    )
else:
    print("No CRS/unit anomalies detected (<20 sqm).")

# --------------------------------------------------
# [2] BUFFER MATCH vs AREA CONFLICT CHECK
# --------------------------------------------------
# Large areas derived from buffer-based matches are suspicious
buffer_area_conflict = matched_df[
    matched_df['match_type'].str.contains('buffer', case=False, na=False) &
    (matched_df['footprint_area'] > 20000)
]

if not buffer_area_conflict.empty:
    print("\nBUFFER MATCH AREA CONFLICT")
    print(f"{len(buffer_area_conflict)} large-area footprints derived from buffer matching.")
    print("These may represent buffered points rather than true polygons.")
    
    print("\nExamples:")
    print(
        buffer_area_conflict[['poi_id', 'name', 'subtype', 'footprint_area', 'match_type']]
        .head(5)
        .to_string(index=False)
    )
else:
    print("No buffer-area conflicts detected.")

# --------------------------------------------------
# [3] CBD PARK CONTEXTUAL CHECK
# --------------------------------------------------
# CBD parks are expected to be systematically smaller
cbd_parks = matched_df[
    (matched_df['subtype'] == 'park') &
    (matched_df.get('cbd_flag', 0) == 1)
]

if not cbd_parks.empty:
    median_cbd_park = cbd_parks['footprint_area'].median()
    median_all_park = matched_df[matched_df['subtype'] == 'park']['footprint_area'].median()
    
    print("\nCBD PARK CONTEXT CHECK")
    print(f"Median park area (CBD): {median_cbd_park:,.0f} sqm")
    print(f"Median park area (All): {median_all_park:,.0f} sqm")
    
    if median_cbd_park < 0.5 * median_all_park:
        print("CBD parks are significantly smaller (expected pattern).")
    else:
        print("CBD park sizes comparable to non-CBD parks. Verify matches.")
else:
    print("No CBD parks found or CBD flag unavailable.")

# --------------------------------------------------
# [4] EXTREME VALUE MANUAL REVIEW LIST
# --------------------------------------------------
print("\n" + "="*80)
print("EXTREME VALUE REVIEW LIST (TOP / BOTTOM 5)")
print("="*80)

for subtype in matched_df['subtype'].dropna().unique():
    sub_df = matched_df[matched_df['subtype'] == subtype]
    
    if len(sub_df) < 10:
        continue
    
    print(f"\n{subtype.upper()}")
    
    print("  TOP 5 LARGEST FOOTPRINTS:")
    print(
        sub_df.nlargest(5, 'footprint_area')[
            ['poi_id', 'name', 'footprint_area', 'match_type']
        ].to_string(index=False)
    )
    
    print("\n  TOP 5 SMALLEST FOOTPRINTS:")
    print(
        sub_df.nsmallest(5, 'footprint_area')[
            ['poi_id', 'name', 'footprint_area', 'match_type']
        ].to_string(index=False)
    )

print("\nAdditional sanity checks completed.")

# ------------------------------------------------------------------
# SAVE QUALITY REPORT
# ------------------------------------------------------------------

print("\n" + "="*80)
print("GENERATING EXCEL QUALITY REPORT")
print("="*80)

with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
    # Sheet 1: Match rate summary
    match_stats_df.to_excel(writer, sheet_name='Match_Rate_Summary', index=False)
    
    # Sheet 2: Statistical summary
    stats_df = pd.DataFrame(subtype_stats)
    stats_df = stats_df.round(1)
    stats_df.to_excel(writer, sheet_name='Statistical_Analysis', index=False)
    
    # Sheet 3: Flagged entries (all)
    if not flagged_df.empty:
        flagged_sorted = flagged_df.sort_values(['max_severity', 'footprint_area'],
                                                ascending=[False, False])
        flagged_sorted.to_excel(writer, sheet_name='Flagged_Entries_All', index=False)
        
        # Sheet 4: Critical cases only
        if not critical.empty:
            critical_sorted = critical.sort_values('footprint_area', ascending=False)
            critical_sorted.to_excel(writer, sheet_name='Critical_Cases', index=False)
    
    # Sheet 5: No match summary
    no_match_summary = []
    for subtype in sorted(df['subtype'].dropna().unique()):
        no_match_sub = no_match_df[no_match_df['subtype'] == subtype]
        if len(no_match_sub) > 0:
            no_match_summary.append({
                'Subtype': subtype,
                'Category': CATEGORY_MAP.get(subtype, 'Unknown'),
                'No Match Count': len(no_match_sub),
                'Percentage of Total': len(no_match_sub) / len(df[df['subtype'] == subtype])
            })
    
    no_match_summary_df = pd.DataFrame(no_match_summary)
    no_match_summary_df = no_match_summary_df.sort_values(['Category', 'Subtype'])
    no_match_summary_df.to_excel(writer, sheet_name='No_Match_Summary', index=False)
    
    # Sheet 6: All matched data with quality flags
    matched_output = matched_df.copy()
    matched_output['quality_flag'] = ''
    matched_output['severity'] = 0
    
    for idx, row in flagged_df.iterrows():
        mask = matched_output['poi_id'] == row['poi_id']
        matched_output.loc[mask, 'quality_flag'] = row['flags']
        matched_output.loc[mask, 'severity'] = row['max_severity']
    
    matched_output = matched_output.sort_values(['severity', 'footprint_area'],
                                                ascending=[False, False])
    matched_output.to_excel(writer, sheet_name='All_Matched_Data', index=False)

print(f"Quality report saved to: {OUTPUT_PATH}")

# ------------------------------------------------------------------
# FINAL RECOMMENDATIONS
# ------------------------------------------------------------------

print("\n" + "="*80)
print("RECOMMENDATIONS & ACTION ITEMS")
print("="*80)

print("\n1. MATCH RATE ASSESSMENT:")
for category in ['Retail', 'Leisure']:
    cat_data = match_stats_df[match_stats_df['Category'] == category]
    if not cat_data.empty:
        avg_match_rate = cat_data['Match Rate'].mean()
        print(f"   {category}: Average match rate = {avg_match_rate:.1%}")
        
        low_match = cat_data[cat_data['Match Rate'] < 0.7]
        if not low_match.empty:
            print(f"   ⚠ Low match rate subtypes:")
            for _, row in low_match.iterrows():
                print(f"      - {row['Subtype']}: {row['Match Rate']:.1%} "
                      f"({row['No Match']} unmatched)")

print("\n2. DATA QUALITY ISSUES:")
if not flagged_df.empty:
    print(f"   Total flagged entries: {len(flagged_df)}")
    print(f"   - Critical (MUST REVIEW): {len(critical)}")
    print(f"   - Warning (SHOULD REVIEW): {len(warning)}")
    print(f"   - Info (OPTIONAL REVIEW): {len(info)}")
    
    if not critical.empty:
        print(f"\n CRITICAL: {len(critical)} entries beyond absolute thresholds")
        print(f"   Action: Manual verification required for all critical cases")
        print(f"   Check: Wrong polygon matches, merged buildings, OSM errors")
else:
    print("No critical quality issues detected")

print("\n3. NO MATCH HANDLING:")
total_no_match = len(no_match_df)
print(f"   Total unmatched: {total_no_match} ({total_no_match/len(df):.1%})")

for subtype in ['park', 'monument', 'historic_site']:
    sub_no_match = no_match_df[no_match_df['subtype'] == subtype]
    if len(sub_no_match) > 0:
        pct = len(sub_no_match) / len(df[df['subtype'] == subtype])
        print(f"   - {subtype}: {len(sub_no_match)} unmatched ({pct:.1%})")
        
        if subtype == 'park' and pct > 0.15:
            print(f"     → Acceptable for parks (OSM coverage limitation)")
        elif subtype == 'monument' and pct > 0.2:
            print(f"     → Acceptable for monuments (often have no defined footprint)")
        elif pct > 0.3:
            print(f"     → HIGH: Consider manual review or alternative data source")

print("\n4. SUGGESTED ACTIONS:")
print("   Priority 1: Review all CRITICAL flagged entries")
print("   Priority 2: Verify low match rate subtypes")
print("   Priority 3: Spot-check WARNING cases (random sample of 10-20)")
print("   Priority 4: Accept no_match for parks and monuments if < 20%")

print("\n5. DATA VALIDATION:")
print("   Run manual spot-checks on 5-10 random entries per subtype")
print("   Cross-reference large footprints (> typical_max) with Google Maps")
print("   Verify small footprints (< typical_min) are legitimate")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
