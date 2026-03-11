import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.cluster import KMeans

def create_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    method: str = 'stratified',
    seed: int = 42
) -> pd.DataFrame:
    """
    Create cross-validation folds
    
    Args:
        df: DataFrame with annotations
        n_folds: Number of folds
        method: 'stratified', 'group', 'cluster'
        seed: Random seed
    
    Returns:
        DataFrame with fold column
    """
    df = df.copy()
    
    # Get unique images
    image_df = df[['image_filename']].drop_duplicates().reset_index(drop=True)
    
    if method == 'stratified':
        # Stratify by cell count
        cell_counts = df.groupby('image_filename').size()
        image_df = image_df.merge(
            cell_counts.rename('cell_count'),
            left_on='image_filename',
            right_index=True
        )
        
        # Create bins for stratification
        image_df['strata'] = pd.qcut(
            image_df['cell_count'],
            q=min(10, len(image_df) // n_folds),
            labels=False,
            duplicates='drop'
        )
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(image_df['image_filename'], image_df['strata'])
        ):
            image_df.loc[val_idx, 'fold'] = fold
    
    elif method == 'group':
        # Group by some criteria (e.g., patient/slide)
        # For now, use image name prefix as group
        image_df['group'] = image_df['image_filename'].apply(
            lambda x: x.split('_')[0] if '_' in x else x
        )
        
        # Count groups
        group_counts = image_df['group'].value_counts()
        
        # Create folds ensuring groups don't split
        gkf = GroupKFold(n_splits=n_folds)
        
        for fold, (train_idx, val_idx) in enumerate(
            gkf.split(image_df['image_filename'], groups=image_df['group'])
        ):
            image_df.loc[val_idx, 'fold'] = fold
    
    elif method == 'cluster':
        # Cluster based on image features
        # For simplicity, use cell count and average cell size
        image_stats = df.groupby('image_filename').agg({
            'width': ['mean', 'std'],
            'height': ['mean', 'std'],
            'x': 'count'
        }).reset_index()
        
        image_stats.columns = [
            'image_filename', 'width_mean', 'width_std',
            'height_mean', 'height_std', 'cell_count'
        ]
        
        # Normalize features
        features = image_stats[['width_mean', 'width_std', 
                                'height_mean', 'height_std', 'cell_count']].values
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        # KMeans clustering
        kmeans = KMeans(n_clusters=n_folds, random_state=seed)
        clusters = kmeans.fit_predict(features)
        
        image_stats['fold'] = clusters
        image_df = image_df.merge(
            image_stats[['image_filename', 'fold']],
            on='image_filename'
        )
    
    else:
        raise ValueError(f"Unknown fold method: {method}")
    
    # Merge back to original dataframe
    df = df.merge(image_df[['image_filename', 'fold']], on='image_filename')
    
    return df


def validate_folds(df: pd.DataFrame) -> Dict[str, float]:
    """
    Validate fold distribution
    """
    stats = {}
    
    # Check fold sizes
    fold_sizes = df.groupby('fold')['image_filename'].nunique()
    stats['fold_sizes'] = fold_sizes.to_dict()
    stats['fold_size_std'] = fold_sizes.std()
    
    # Check cell count distribution
    cell_counts = df.groupby(['fold', 'image_filename']).size().groupby('fold')
    stats['mean_cells_per_fold'] = cell_counts.mean().to_dict()
    stats['std_cells_per_fold'] = cell_counts.std().to_dict()
    
    # Check cell size distribution
    cell_sizes = df.groupby('fold').agg({
        'width': 'mean',
        'height': 'mean'
    }).to_dict()
    stats['mean_width_per_fold'] = cell_sizes['width']
    stats['mean_height_per_fold'] = cell_sizes['height']
    
    return stats


class StratifiedGroupKFold:
    """
    Stratified Group K-Fold cross-validator
    """
    
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X, y, groups):
        """
        Generate indices to split data into training and test set
        """
        unique_groups = np.unique(groups)
        group_y = []
        
        for group in unique_groups:
            group_indices = np.where(groups == group)[0]
            group_y.append(y[group_indices[0]])  # Use first label in group
        
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        
        for train_group_idx, test_group_idx in skf.split(unique_groups, group_y):
            train_indices = []
            test_indices = []
            
            for idx in train_group_idx:
                train_indices.extend(np.where(groups == unique_groups[idx])[0])
            
            for idx in test_group_idx:
                test_indices.extend(np.where(groups == unique_groups[idx])[0])
            
            yield np.array(train_indices), np.array(test_indices)