"""
C.2.5 - Image Registry Refinement for Destination Perception Modeling

This script implements a multi-dimensional scoring algorithm to refine street-view
image registries for computer vision-based perception analysis in destination choice models.

Author: Zhang Wenyu
Date: 2026-01-27
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class ImageRegistryRefiner:
    """
    Refines raw image registry by selecting representative images for each POI
    based on recency, diversity, spatial balance, and panorama value.
    """
    
    def __init__(
        self,
        w_R: float = 0.40,  # Weight for recency
        w_D: float = 0.30,  # Weight for diversity
        w_S: float = 0.20,  # Weight for spatial balance
        w_P: float = 0.10,  # Weight for panorama value
        max_panorama_ratio: float = 0.30  # Maximum proportion of panoramic images
    ):
        """
        Initialize the refiner with scoring weights.
        
        Parameters:
        -----------
        w_R : float
            Weight for recency score (temporal freshness)
        w_D : float
            Weight for diversity score (visual/geometric uniqueness)
        w_S : float
            Weight for spatial balance score (anti-clustering)
        w_P : float
            Weight for panorama value (contextual information bonus)
        max_panorama_ratio : float
            Maximum allowed proportion of panoramic images per POI
        """
        self.w_R = w_R
        self.w_D = w_D
        self.w_S = w_S
        self.w_P = w_P
        self.max_panorama_ratio = max_panorama_ratio
        
        # Minimum image thresholds by POI subtype (based on POI_ID prefix)
        self.N_min_baseline = {
            '1': 20,   # Mall - retail complex requiring multiple perspectives
            '2': 18,   # Lifestyle street - moderate coverage
            '3': 15,   # Hawker centre - compact with moderate variance
            '5': 10,   # Museum - cultural landmark with limited viewpoints
            '6': 10,   # Monument - cultural landmark with limited viewpoints
            '7': 12,   # Theatre - cultural venue
            '8': 12,   # Historic site - heritage location
            '9': 25,   # Park - open space with high spatial variance
            'default': 15  # Default threshold for unrecognized types
        }
        
        # Capacity constraints by POI subtype (based on POI_ID prefix)
        self.C_p_baseline = {
            '1': 50,   # Mall - large retail complex
            '2': 45,   # Lifestyle street
            '3': 40,   # Hawker centre
            '5': 30,   # Museum
            '6': 30,   # Monument
            '7': 35,   # Theatre
            '8': 35,   # Historic site
            '9': 60,   # Park - open space needs more coverage
            'default': 40  # Default capacity
        }
    
    def _extract_subtype_from_poi_id(self, poi_id: str) -> str:
        """
        Extract subtype from POI_ID format: X_XX_XXXX
        
        Parameters:
        -----------
        poi_id : str
            POI identifier in format like "1_ML_0001"
            
        Returns:
        --------
        str
            First digit indicating subtype (1-9)
        """
        try:
            # Extract first character before underscore
            subtype = poi_id.split('_')[0]
            return subtype
        except:
            return 'default'
    
    def refine_registry(
        self,
        df: pd.DataFrame,
        poi_subtype_map: Dict[str, str] = None
    ) -> pd.DataFrame:
        """
        Main pipeline to refine the image registry.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw image registry with columns: POI_ID, image_id, lon, lat, 
            image_year, camera_heading, is_panorama
        poi_subtype_map : Dict[str, str], optional
            Mapping from POI_ID to subtype. If None, extracts from POI_ID prefix
            
        Returns:
        --------
        pd.DataFrame
            Refined image registry
        """
        print("=" * 80)
        print("IMAGE REGISTRY REFINEMENT")
        print("=" * 80)
        
        # Step 1: Global Pre-Filtering
        df_filtered = self._global_prefiltering(df)
        
        # Step 2-6: POI-Level Processing
        refined_dfs = []
        
        poi_ids = df_filtered['POI_ID'].unique()
        print(f"\nProcessing {len(poi_ids)} POIs...")
        
        for idx, poi_id in enumerate(poi_ids, 1):
            if idx % 50 == 0:
                print(f"  Processed {idx}/{len(poi_ids)} POIs...")
            
            # Get POI subtype from POI_ID prefix
            subtype = self._extract_subtype_from_poi_id(poi_id)
            
            # Process single POI
            poi_images = self._process_poi(
                df_filtered[df_filtered['POI_ID'] == poi_id].copy(),
                poi_id,
                subtype
            )
            
            refined_dfs.append(poi_images)
        
        # Step 6: Registry Aggregation
        df_refined = pd.concat(refined_dfs, ignore_index=True)
        
        print("\n" + "=" * 80)
        print("REFINEMENT SUMMARY")
        print("=" * 80)
        print(f"Original images: {len(df):,}")
        print(f"After global filtering: {len(df_filtered):,}")
        print(f"Final refined registry: {len(df_refined):,}")
        print(f"Reduction rate: {(1 - len(df_refined)/len(df))*100:.1f}%")
        print(f"Average images per POI: {len(df_refined)/len(poi_ids):.1f}")
        
        # Print distribution by POI type
        print("\n" + "-" * 80)
        print("DISTRIBUTION BY POI TYPE")
        print("-" * 80)
        type_names = {
            '1': 'Mall',
            '2': 'Lifestyle Street',
            '3': 'Hawker Centre',
            '5': 'Museum',
            '6': 'Monument',
            '7': 'Theatre',
            '8': 'Historic Site',
            '9': 'Park'
        }
        
        df_refined['subtype'] = df_refined['POI_ID'].apply(self._extract_subtype_from_poi_id)
        type_counts = df_refined['subtype'].value_counts().sort_index()
        
        for subtype_code, count in type_counts.items():
            type_name = type_names.get(subtype_code, f"Type {subtype_code}")
            n_pois = len(df_refined[df_refined['subtype'] == subtype_code]['POI_ID'].unique())
            avg_per_poi = count / n_pois if n_pois > 0 else 0
            print(f"  {type_name:20s}: {count:5d} images ({n_pois:3d} POIs, {avg_per_poi:.1f} avg/POI)")
        
        print("=" * 80)
        
        return df_refined
    
    def _global_prefiltering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 1: Remove duplicate images and filter by year.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw image registry
            
        Returns:
        --------
        pd.DataFrame
            Filtered registry
        """
        print("\n[Step 1] Global Pre-Filtering")
        print("-" * 80)
        
        original_count = len(df)
        
        # Remove exact duplicates based on image_id
        df = df.drop_duplicates(subset=['image_id'])
        duplicates_removed = original_count - len(df)
        print(f"  • Removed {duplicates_removed:,} duplicate image_ids")
        
        # Remove images before 2019
        df = df[df['image_year'] >= 2019].copy()
        old_images_removed = original_count - duplicates_removed - len(df)
        print(f"  • Removed {old_images_removed:,} images before 2019")
        print(f"  • Retained {len(df):,} images for POI-level processing")
        
        return df
    
    def _process_poi(
        self,
        df_poi: pd.DataFrame,
        poi_id: str,
        subtype: str
    ) -> pd.DataFrame:
        """
        Steps 2-5: Process images for a single POI.
        
        Parameters:
        -----------
        df_poi : pd.DataFrame
            Images belonging to one POI
        poi_id : str
            POI identifier
        subtype : str
            POI subtype extracted from POI_ID prefix
            
        Returns:
        --------
        pd.DataFrame
            Selected images for this POI
        """
        n_images = len(df_poi)
        
        # Step 2: Check minimum threshold
        N_min = self.N_min_baseline.get(subtype, self.N_min_baseline['default'])
        
        if n_images < N_min:
            # Retain all images if below threshold
            return df_poi
        
        # Step 3: Determine capacity constraint
        C_p = self.C_p_baseline.get(subtype, self.C_p_baseline['default'])
        C_p = min(C_p, n_images)  # Cannot exceed available images
        
        # Step 4: Compute multi-dimensional scores
        df_poi = self._compute_scores(df_poi)
        
        # Step 5: Score-based selection
        selected_df = self._select_images(df_poi, C_p)
        
        return selected_df
    
    def _compute_scores(self, df_poi: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Compute composite score for each image.
        
        Score(r) = w_R * R(r) + w_D * D(r) + w_S * S(r) + w_P * P(r)
        
        Parameters:
        -----------
        df_poi : pd.DataFrame
            Images for one POI
            
        Returns:
        --------
        pd.DataFrame
            Input dataframe with added score columns
        """
        n = len(df_poi)
        
        # R(r): Recency Score
        # Normalize image_year to [0, 1], with more recent years getting higher scores
        df_poi['score_R'] = self._compute_recency(df_poi['image_year'].values)
        
        # D(r): Diversity Score
        # Measures uniqueness in spatial location and camera heading
        df_poi['score_D'] = self._compute_diversity(
            df_poi[['lon', 'lat', 'camera_heading']].values
        )
        
        # S(r): Spatial Balance Score
        # Penalizes images in densely sampled spatial regions
        df_poi['score_S'] = self._compute_spatial_balance(
            df_poi[['lon', 'lat']].values
        )
        
        # P(r): Panorama Value
        # Fixed bonus for panoramic images
        df_poi['score_P'] = df_poi['is_panorama'].map({True: 1.0, False: 0.0})
        
        # --------------------------------------------------
        # Composite Score (robust to NaN)
        # --------------------------------------------------
        score_cols = ['score_R', 'score_D', 'score_S', 'score_P']

        # Defensive: replace NaN sub-scores with 0
        df_poi[score_cols] = df_poi[score_cols].fillna(0)

        df_poi['composite_score'] = (
            self.w_R * df_poi['score_R'] +
            self.w_D * df_poi['score_D'] +
            self.w_S * df_poi['score_S'] +
            self.w_P * df_poi['score_P']
        )

        # Final safety net
        df_poi['composite_score'] = df_poi['composite_score'].fillna(0)

        return df_poi
    
    def _compute_recency(self, years: np.ndarray) -> np.ndarray:
        """
        Compute recency score: R(r) = (year - min_year) / (max_year - min_year)
        
        More recent images receive higher scores, reflecting temporal freshness.
        
        Parameters:
        -----------
        years : np.ndarray
            Array of image capture years
            
        Returns:
        --------
        np.ndarray
            Normalized recency scores in [0, 1]
        """
        min_year = years.min()
        max_year = years.max()
        
        if max_year == min_year:
            # All images from same year
            return np.ones(len(years))
        
        return (years - min_year) / (max_year - min_year)
    
    def _compute_diversity(self, features: np.ndarray) -> np.ndarray:
        """
        Compute diversity score: D(r) based on spatial-angular distance.
        
        Diversity rewards images that are geometrically unique relative to others.
        Combines:
        - Spatial distance (lon, lat)
        - Angular difference (camera_heading)
        
        Parameters:
        -----------
        features : np.ndarray
            Array of shape (n, 3) with columns [lon, lat, camera_heading]
            
        Returns:
        --------
        np.ndarray
            Diversity scores in [0, 1]
        """
        n = len(features)
        
        if n == 1:
            return np.array([1.0])
        
        # Extract spatial and angular components
        spatial_coords = features[:, :2]  # lon, lat
        angles = features[:, 2].astype(float)  # camera_heading

        # Treat invalid headings (-1 or NaN) as missing
        valid_angle_mask = np.isfinite(angles) & (angles >= 0)

        # Compute pairwise spatial distances (always needed)
        spatial_dist_matrix = cdist(spatial_coords, spatial_coords, metric='euclidean')

        # If all angles invalid, fall back to spatial-only diversity
        if not valid_angle_mask.any():
            if spatial_dist_matrix.max() > 0:
                spatial_dist_matrix = spatial_dist_matrix / spatial_dist_matrix.max()
            diversity_scores = spatial_dist_matrix.sum(axis=1) / (n - 1)
            if diversity_scores.max() > 0:
                return diversity_scores / diversity_scores.max()
            else:
                return np.ones(n)
        
        # Compute pairwise angular differences
        # Handle circular nature of angles (0° and 360° are same)
        angle_diff_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if valid_angle_mask[i] and valid_angle_mask[j]:
                    diff = abs(angles[i] - angles[j])
                    angle_diff_matrix[i, j] = min(diff, 360 - diff)
                else:
                    angle_diff_matrix[i, j] = 0  # no angular contribution
        
        # Normalize distance matrices to [0, 1]
        if spatial_dist_matrix.max() > 0:
            spatial_dist_matrix = spatial_dist_matrix / spatial_dist_matrix.max()
        if angle_diff_matrix.max() > 0:
            angle_diff_matrix = angle_diff_matrix / angle_diff_matrix.max()
        
        # Combined distance: 60% spatial + 40% angular
        combined_dist = 0.6 * spatial_dist_matrix + 0.4 * angle_diff_matrix
        
        # Diversity score: average distance to all other images
        # Images far from others (high uniqueness) get high scores
        diversity_scores = combined_dist.sum(axis=1) / (n - 1)
        
        # Normalize to [0, 1]
        if diversity_scores.max() > 0:
            diversity_scores = diversity_scores / diversity_scores.max()
        
        return diversity_scores
    
    def _compute_spatial_balance(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute spatial balance score: S(r) based on kernel density estimation.
        
        Penalizes images in spatially clustered regions to ensure even coverage.
        Uses inverse of local density as the balance score.
        
        Parameters:
        -----------
        coords : np.ndarray
            Array of shape (n, 2) with columns [lon, lat]
            
        Returns:
        --------
        np.ndarray
            Spatial balance scores in [0, 1]
        """
        n = len(coords)
        
        if n == 1:
            return np.array([1.0])
        
        if n < 3:
            # KDE requires at least 3 points; use uniform scores
            return np.ones(n)
        
        try:
            # Transpose for KDE: expects (2, n) shape
            kde = gaussian_kde(coords.T)
            
            # Evaluate density at each point
            densities = kde(coords.T)
            
            # Spatial balance: inverse of density (high density → low balance)
            # Add 1 to avoid division by zero, then take log for smoothing
            balance_scores = 1 / (1 + np.log(densities + 1))
            
            # Normalize to [0, 1]
            if balance_scores.max() > balance_scores.min():
                balance_scores = (balance_scores - balance_scores.min()) / \
                                 (balance_scores.max() - balance_scores.min())
            else:
                balance_scores = np.ones(n)
            
            return balance_scores
        
        except:
            # Fallback if KDE fails (e.g., singular covariance)
            # Use simple distance-based metric
            dist_matrix = cdist(coords, coords, metric='euclidean')
            avg_dist = dist_matrix.sum(axis=1) / (n - 1)
            
            if avg_dist.max() > 0:
                return avg_dist / avg_dist.max()
            else:
                return np.ones(n)
    
    def _select_images(self, df_poi: pd.DataFrame, C_p: int) -> pd.DataFrame:
        """
        Step 5: Select top C_p images based on composite scores.
        
        Applies panorama ratio constraint to avoid dominance by single image type.
        
        Parameters:
        -----------
        df_poi : pd.DataFrame
            POI images with computed scores
        C_p : int
            Target number of images to select
            
        Returns:
        --------
        pd.DataFrame
            Selected images
        """
        # Filter out rows with NaN composite scores
        df_poi = df_poi[df_poi['composite_score'].notna()].copy()
        
        if len(df_poi) == 0:
            return df_poi  # Return empty if all scores are NaN
        
        # Rank by composite score (descending)
        df_ranked = df_poi.sort_values('composite_score', ascending=False)
        
        # Apply panorama ratio constraint
        max_panorama_count = int(C_p * self.max_panorama_ratio)
        
        selected_indices = []
        panorama_count = 0
        
        for idx, row in df_ranked.iterrows():
            if len(selected_indices) >= C_p:
                break
            
            if row['is_panorama']:
                if panorama_count < max_panorama_count:
                    selected_indices.append(idx)
                    panorama_count += 1
                # Skip if panorama quota exceeded
            else:
                selected_indices.append(idx)
        
        # If we haven't reached C_p and there are more panoramas available,
        # add them until we reach C_p (relaxing the constraint if needed)
        if len(selected_indices) < C_p:
            remaining = df_ranked.loc[~df_ranked.index.isin(selected_indices)]
            additional_needed = C_p - len(selected_indices)
            selected_indices.extend(remaining.head(additional_needed).index.tolist())
        
        return df_poi.loc[selected_indices].copy()


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    
    # Load the image registry
    csv_path = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery/Image_registry_v4_clean.csv"
    
    print("Loading image registry...")
    print(f"Path: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    print(f"\nLoaded {len(df):,} images")
    print(f"Columns: {', '.join(df.columns.tolist())}")
    print(f"\nNumber of unique POIs: {df['POI_ID'].nunique()}")
    print(f"Year range: {df['image_year'].min()} - {df['image_year'].max()}")
    print(f"Panorama images: {df['is_panorama'].sum():,} ({df['is_panorama'].sum()/len(df)*100:.1f}%)")
    
    # Initialize the refiner with research-specific weights
    # Heavy emphasis on recency (w_R=0.40) as required by research design
    refiner = ImageRegistryRefiner(
        w_R=0.40,  # Strong emphasis on temporal freshness
        w_D=0.30,  # Visual and geometric uniqueness
        w_S=0.20,  # Spatial balance to avoid clustering
        w_P=0.10,  # Panorama contextual value
        max_panorama_ratio=0.30  # Maximum 30% panoramic images per POI
    )
    
    # Run refinement algorithm
    df_refined = refiner.refine_registry(df)
    
    # Save refined registry
    output_path = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery/refined_image_registry.csv"
    df_refined.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("REFINEMENT COMPLETE")
    print("="*80)
    print(f"Refined registry saved to:")
    print(f"  {output_path}")
    print("="*80)
