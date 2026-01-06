import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import GROUP_COLUMN, TARGET_COLUMN, NUM_PLANTS


def show_basic_info(df, name="Dataset"):
    print("="*70)
    print(f"BASIC INFO: {name}")
    print("="*70)
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}")
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def show_missing_summary(df):
    print("\n" + "="*70)
    print("MISSING VALUES SUMMARY")
    print("="*70)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Missing_Percent': missing_pct.values
    })
    
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
        'Missing_Count', ascending=False
    )
    
    if len(missing_df) == 0:
        print("\n✓ No missing values found!")
    else:
        print(f"\nColumns with missing values ({len(missing_df)}):\n")
        print(missing_df.to_string(index=False))


def show_demand_statistics(df):
    print("\n" + "="*70)
    print("DEMAND ID STATISTICS")
    print("="*70)
    
    n_demands = df[GROUP_COLUMN].nunique()
    demand_counts = df[GROUP_COLUMN].value_counts()
    
    print(f"\nUnique Demand IDs: {n_demands}")
    print(f"Total rows: {len(df)}")
    print(f"Plants per Demand (expected): {NUM_PLANTS}")
    print(f"\nRows per Demand ID:")
    print(f"  Min: {demand_counts.min()}")
    print(f"  Max: {demand_counts.max()}")
    print(f"  Mean: {demand_counts.mean():.1f}")
    print(f"  Median: {demand_counts.median():.1f}")
    
    # Check if all demands have correct number of plants
    wrong_counts = demand_counts[demand_counts != NUM_PLANTS]
    if len(wrong_counts) > 0:
        print(f"\n WARNING: {len(wrong_counts)} Demand IDs don't have {NUM_PLANTS} plants!")
        print(wrong_counts.head())
    else:
        print(f"\n All Demand IDs have exactly {NUM_PLANTS} plants")


def show_target_statistics(df):
    print("\n" + "="*70)
    print("TARGET VARIABLE STATISTICS (Cost)")
    print("="*70)
    
    costs = df[TARGET_COLUMN]
    
    print(f"\nDescriptive Statistics:")
    print(f"  Count: {costs.count()}")
    print(f"  Mean: ${costs.mean():.2f}/MWh")
    print(f"  Std: ${costs.std():.2f}/MWh")
    print(f"  Min: ${costs.min():.2f}/MWh")
    print(f"  25%: ${costs.quantile(0.25):.2f}/MWh")
    print(f"  50%: ${costs.quantile(0.50):.2f}/MWh")
    print(f"  75%: ${costs.quantile(0.75):.2f}/MWh")
    print(f"  Max: ${costs.max():.2f}/MWh")
    
    # Check for negative values
    negative_count = (costs < 0).sum()
    if negative_count > 0:
        print(f"\n WARNING: {negative_count} negative cost values!")
    else:
        print(f"\n No negative cost values")


def plot_cost_distribution(df, save_path=None):
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(df[TARGET_COLUMN], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Cost (USD/MWh)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Cost Distribution', fontsize=13, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(df[TARGET_COLUMN], vert=True)
    plt.ylabel('Cost (USD/MWh)', fontsize=11)
    plt.title('Cost Box Plot', fontsize=13, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Plot saved: {save_path}")
    
    plt.show()


def plot_demand_distribution(df, save_path=None):
    demand_counts = df[GROUP_COLUMN].value_counts().sort_index()
    
    plt.figure(figsize=(14, 6))
    plt.bar(range(len(demand_counts)), demand_counts.values, 
            color='coral', edgecolor='black', alpha=0.7)
    plt.axhline(y=NUM_PLANTS, color='red', linestyle='--', 
                label=f'Expected: {NUM_PLANTS} plants')
    plt.xlabel('Demand ID (sorted)', fontsize=11)
    plt.ylabel('Number of Plants', fontsize=11)
    plt.title('Plants per Demand ID', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Plot saved: {save_path}")
    
    plt.show()


def generate_eda_report(df, name="Dataset"):
    print("\n" + "🔍"*35)
    print(f"EDA REPORT: {name}")
    print("🔍"*35 + "\n")
    
    show_basic_info(df, name)
    show_missing_summary(df)
    
    if GROUP_COLUMN in df.columns:
        show_demand_statistics(df)
    
    if TARGET_COLUMN in df.columns:
        show_target_statistics(df)
        plot_cost_distribution(df)
    
    if GROUP_COLUMN in df.columns:
        plot_demand_distribution(df)
    
    print("\n✓ EDA Report Complete!\n")

# TESTING

if __name__ == "__main__":
    """Test EDA utilities."""
    print("\n" + "="*70)
    print("TESTING EDA UTILITIES MODULE")
    print("="*70 + "\n")
    
    try:
        from src.data_ingestion import create_train_test
        
        # Load data
        print("Loading data...")
        train_df, test_df = create_train_test(verbose=False)
        
        # Generate EDA report for training data
        generate_eda_report(train_df, "Training Data")
        
        print("\n EDA Utilities Module Test PASSED!\n")
        
    except Exception as e:
        print(f"\n Error: {str(e)}\n")
        raise