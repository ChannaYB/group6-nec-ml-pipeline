"""
Visualization Module for NEC ML Pipeline
Creates performance plots and charts for evaluation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.config import (
    PLOTS_DIR,
    PLOT_FIGURE_SIZE,
    PLOT_DPI,
    VERBOSE
)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_logo_cv_folds(cv_results, model_name="Model", save=True, show=False):
    """
    Plot LOGO CV performance across folds.
    
    Creates a figure with 3 subplots:
    - RMSE per fold
    - R² per fold
    - Selection Error per fold
    
    Parameters:
    -----------
    cv_results : dict
        Cross-validation results
    model_name : str
        Model name for title
    save : bool
        Save plot to file
    show : bool
        Display plot
    
    Returns:
    --------
    str : Path to saved plot (if save=True)
    
    Reference: Assessment Brief - "Fold-level outputs"
    """
    if cv_results is None:
        print("No CV results to plot")
        return None
    
    fold_results = cv_results['fold_results']
    n_folds = len(fold_results)
    
    # Extract metrics
    folds = list(range(1, n_folds + 1))
    rmse_values = [f['rmse'] for f in fold_results]
    r2_values = [f['r2'] for f in fold_results]
    error_values = [f['selection_error'] * 100 for f in fold_results]  # Convert to %
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: RMSE
    axes[0].bar(folds, rmse_values, color='steelblue', alpha=0.7)
    axes[0].axhline(y=cv_results['avg_rmse'], color='red', linestyle='--', 
                    label=f'Average: ${cv_results["avg_rmse"]:.2f}')
    axes[0].set_xlabel('Fold', fontsize=12)
    axes[0].set_ylabel('RMSE ($/MWh)', fontsize=12)
    axes[0].set_title('RMSE per Fold', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: R²
    axes[1].bar(folds, r2_values, color='seagreen', alpha=0.7)
    axes[1].axhline(y=cv_results['avg_r2'], color='red', linestyle='--',
                    label=f'Average: {cv_results["avg_r2"]:.4f}')
    axes[1].set_xlabel('Fold', fontsize=12)
    axes[1].set_ylabel('R² Score', fontsize=12)
    axes[1].set_title('R² per Fold', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Selection Error
    axes[2].bar(folds, error_values, color='coral', alpha=0.7)
    axes[2].axhline(y=cv_results['avg_selection_error'] * 100, color='red', linestyle='--',
                    label=f'Average: {cv_results["avg_selection_error"]*100:.2f}%')
    axes[2].set_xlabel('Fold', fontsize=12)
    axes[2].set_ylabel('Selection Error (%)', fontsize=12)
    axes[2].set_title('Selection Error per Fold', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'LOGO CV Performance: {model_name}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    filepath = None
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logo_cv_{model_name.replace(' ', '_')}_{timestamp}.png"
        filepath = Path(PLOTS_DIR) / filename
        plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
        print(f" Plot saved: {filepath}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return str(filepath) if filepath else None


def plot_model_comparison(comparison_df, save=True, show=False):
    """
    Plot comparison of multiple models.
    
    Creates bar plots comparing:
    - Test RMSE
    - Test Selection Error
    - CV Selection Error (if available)
    
    Parameters:
    -----------
    comparison_df : pandas.DataFrame
        Comparison table from ModelEvaluator.compare_models()
    save : bool
        Save plot to file
    show : bool
        Display plot
    
    Returns:
    --------
    str : Path to saved plot (if save=True)
    
    Reference: Assessment Brief - "Comparison with untuned baseline"
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = comparison_df['Model'].values
    x = np.arange(len(models))
    width = 0.6
    
    # Plot 1: Test RMSE
    if 'Test_RMSE' in comparison_df.columns:
        test_rmse = comparison_df['Test_RMSE'].values
        bars = axes[0].bar(x, test_rmse, width, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Model', fontsize=12)
        axes[0].set_ylabel('RMSE ($/MWh)', fontsize=12)
        axes[0].set_title('Test Set RMSE', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'${height:.2f}',
                        ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Test Selection Error
    if 'Test_Selection_Error' in comparison_df.columns:
        test_error = comparison_df['Test_Selection_Error'].values * 100  # Convert to %
        bars = axes[1].bar(x, test_error, width, color='coral', alpha=0.7)
        axes[1].set_xlabel('Model', fontsize=12)
        axes[1].set_ylabel('Selection Error (%)', fontsize=12)
        axes[1].set_title('Test Set Selection Error', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=10)
    
    # Plot 3: CV Selection Error (with error bars if available)
    if 'CV_Selection_Error' in comparison_df.columns:
        cv_error = comparison_df['CV_Selection_Error'].values * 100
        cv_error_std = comparison_df.get('CV_Selection_Error_Std', pd.Series([0]*len(models))).values * 100
        
        bars = axes[2].bar(x, cv_error, width, yerr=cv_error_std, 
                          color='seagreen', alpha=0.7, capsize=5)
        axes[2].set_xlabel('Model', fontsize=12)
        axes[2].set_ylabel('Selection Error (%)', fontsize=12)
        axes[2].set_title('CV Average Selection Error', fontsize=14, fontweight='bold')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(models, rotation=45, ha='right')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Model Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    filepath = None
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_comparison_{timestamp}.png"
        filepath = Path(PLOTS_DIR) / filename
        plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
        print(f" Plot saved: {filepath}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return str(filepath) if filepath else None


def plot_prediction_scatter(y_true, y_pred, dataset_name="Test", model_name="Model", 
                            save=True, show=False):
    """
    Plot actual vs predicted costs (scatter plot).
    
    Parameters:
    -----------
    y_true : array-like
        True costs
    y_pred : array-like
        Predicted costs
    dataset_name : str
        Dataset name (Train/Test)
    model_name : str
        Model name
    save : bool
        Save plot
    show : bool
        Display plot
    
    Returns:
    --------
    str : Path to saved plot
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color='steelblue')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Labels
    ax.set_xlabel('Actual Cost ($/MWh)', fontsize=12)
    ax.set_ylabel('Predicted Cost ($/MWh)', fontsize=12)
    ax.set_title(f'Actual vs Predicted: {model_name} ({dataset_name})', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add R² to plot
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    filepath = None
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prediction_scatter_{model_name.replace(' ', '_')}_{dataset_name}_{timestamp}.png"
        filepath = Path(PLOTS_DIR) / filename
        plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
        print(f" Plot saved: {filepath}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return str(filepath) if filepath else None


def plot_selection_error_distribution(selection_table, model_name="Model", save=True, show=False):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Cost Difference Histogram
    cost_diff = selection_table['Cost Difference'].values
    axes[0].hist(cost_diff, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', lw=2, label='Perfect Selection')
    axes[0].set_xlabel('Cost Difference ($/MWh)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Cost Differences', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    mean_diff = cost_diff.mean()
    axes[0].text(0.95, 0.95, f'Mean: ${mean_diff:.2f}\nMedian: ${np.median(cost_diff):.2f}',
                transform=axes[0].transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Correct vs Incorrect Pie Chart
    correct_count = selection_table['Selection Correct'].sum()
    incorrect_count = len(selection_table) - correct_count
    
    sizes = [correct_count, incorrect_count]
    labels = [f'Correct\n({correct_count})', f'Incorrect\n({incorrect_count})']
    colors = ['#90EE90', '#FFB6C1']
    explode = (0.05, 0.05)
    
    axes[1].pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12})
    axes[1].set_title('Plant Selection Accuracy', fontsize=14, fontweight='bold')
    
    plt.suptitle(f'Selection Error Analysis: {model_name}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    filepath = None
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"selection_analysis_{model_name.replace(' ', '_')}_{timestamp}.png"
        filepath = Path(PLOTS_DIR) / filename
        plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
        print(f" Plot saved: {filepath}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return str(filepath) if filepath else None


def create_evaluation_dashboard(results, model_name="Model", save=True, show=False):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Extract data
    cv_results = results.get('logo_cv')
    train_results = results.get('train')
    test_results = results.get('test')
    test_selection_table = results.get('test_selection_table')
    
    # Plot 1: CV Fold RMSE (top left)
    if cv_results:
        ax1 = fig.add_subplot(gs[0, 0])
        fold_results = cv_results['fold_results']
        folds = list(range(1, len(fold_results) + 1))
        rmse_values = [f['rmse'] for f in fold_results]
        ax1.bar(folds, rmse_values, color='steelblue', alpha=0.7)
        ax1.axhline(y=cv_results['avg_rmse'], color='red', linestyle='--', lw=2)
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('RMSE ($/MWh)')
        ax1.set_title('CV Fold RMSE', fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: CV Fold Selection Error (top middle)
    if cv_results:
        ax2 = fig.add_subplot(gs[0, 1])
        error_values = [f['selection_error'] * 100 for f in fold_results]
        ax2.bar(folds, error_values, color='coral', alpha=0.7)
        ax2.axhline(y=cv_results['avg_selection_error'] * 100, color='red', linestyle='--', lw=2)
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('Selection Error (%)')
        ax2.set_title('CV Fold Selection Error', fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Train vs Test Metrics (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    metrics = ['RMSE', 'Selection\nError']
    train_vals = [train_results['rmse'], train_results['selection_error'] * 100]
    test_vals = [test_results['rmse'], test_results['selection_error'] * 100]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax3.bar(x - width/2, train_vals, width, label='Train', color='seagreen', alpha=0.7)
    ax3.bar(x + width/2, test_vals, width, label='Test', color='coral', alpha=0.7)
    ax3.set_ylabel('Value')
    ax3.set_title('Train vs Test Performance', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Selection Accuracy Pie Chart (middle left)
    if test_selection_table is not None:
        ax4 = fig.add_subplot(gs[1, 0])
        correct = test_selection_table['Selection Correct'].sum()
        incorrect = len(test_selection_table) - correct
        ax4.pie([correct, incorrect], labels=['Correct', 'Incorrect'],
                colors=['#90EE90', '#FFB6C1'], autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 10})
        ax4.set_title('Selection Accuracy', fontweight='bold')
    
    # Plot 5: Cost Difference Distribution (middle span)
    if test_selection_table is not None:
        ax5 = fig.add_subplot(gs[1, 1:])
        cost_diff = test_selection_table['Cost Difference'].values
        ax5.hist(cost_diff, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax5.axvline(x=0, color='red', linestyle='--', lw=2, label='Perfect')
        ax5.set_xlabel('Cost Difference ($/MWh)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Cost Difference Distribution', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Metrics Summary Table (bottom span)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    summary_data = []
    if cv_results:
        summary_data.append(['CV Average', f'${cv_results["avg_rmse"]:.2f}', 
                           f'{cv_results["avg_r2"]:.4f}', 
                           f'{cv_results["avg_selection_error"]*100:.2f}%'])
    if train_results:
        summary_data.append(['Train', f'${train_results["rmse"]:.2f}', 
                           f'{train_results["r2"]:.4f}', 
                           f'{train_results["selection_error"]*100:.2f}%'])
    if test_results:
        summary_data.append(['Test', f'${test_results["rmse"]:.2f}', 
                           f'{test_results["r2"]:.4f}', 
                           f'{test_results["selection_error"]*100:.2f}%'])
    
    table = ax6.table(cellText=summary_data,
                     colLabels=['Dataset', 'RMSE', 'R²', 'Selection Error'],
                     cellLoc='center', loc='center',
                     colWidths=[0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle(f'Evaluation Dashboard: {model_name}', fontsize=18, fontweight='bold', y=0.98)
    
    # Save
    filepath = None
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dashboard_{model_name.replace(' ', '_')}_{timestamp}.png"
        filepath = Path(PLOTS_DIR) / filename
        plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
        print(f" Dashboard saved: {filepath}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return str(filepath) if filepath else None


# TESTING

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING VISUALIZATION MODULE")
    print("="*70)
    
    try:
        # Load data and evaluate baseline
        from src.data_ingestion import create_train_test
        from src.evaluation import evaluate_baseline_model
        
        print("\n[1] Loading data and evaluating baseline...")
        train_df, test_df = create_train_test(verbose=False)
        results = evaluate_baseline_model(train_df, test_df, verbose=False)
        print(" Evaluation complete")
        
        # Create visualizations
        print("\n[2] Creating visualizations...")
        
        # LOGO CV plot
        print("\n  Creating LOGO CV plot...")
        plot_logo_cv_folds(results['logo_cv'], "Baseline_RF", save=True, show=False)
        
        # Selection error analysis
        print("  Creating selection error analysis...")
        plot_selection_error_distribution(
            results['test_selection_table'], 
            "Baseline_RF", 
            save=True, 
            show=False
        )
        
        # Prediction scatter (test set)
        print("  Creating prediction scatter plot...")
        from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
        feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
        X_test = test_df[feature_cols]
        y_test = test_df['Cost_USD_per_MWh']
        y_pred = results['fitted_pipeline'].predict(X_test)
        plot_prediction_scatter(y_test, y_pred, "Test", "Baseline_RF", save=True, show=False)
        
        # Evaluation dashboard
        print("  Creating evaluation dashboard...")
        create_evaluation_dashboard(results, "Baseline_RF", save=True, show=False)
        
        print("\n" + "="*70)
        print(" VISUALIZATION MODULE TEST PASSED")
        print(f" All plots saved to: {PLOTS_DIR}")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n Error: {e}\n")
        import traceback
        traceback.print_exc()