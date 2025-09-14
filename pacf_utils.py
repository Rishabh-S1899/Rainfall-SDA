import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_pacf_directory_structure():
    """Create the PACF plots directory structure"""
    base_dir = Path('plots')
    subdirs = [
        'pacf_state_wise/hourly',
        'pacf_state_wise/daily', 
        'pacf_state_wise/monthly',
        'pacf_state_wise/yearly',
        'pacf_regional/hourly',
        'pacf_regional/daily',
        'pacf_regional/monthly', 
        'pacf_regional/yearly',
        'pacf_combined/state_wise',
        'pacf_combined/regional'
    ]
    
    for subdir in subdirs:
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print("PACF Directory structure created successfully!")

def prepare_data_pacf(df):
    """Prepare and validate the rainfall dataframe for PACF analysis"""
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Convert datetime to proper datetime type
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # Sort by datetime
    data = data.sort_values('datetime')
    
    # Extract additional time components
    data['year'] = data['datetime'].dt.year
    data['month'] = data['datetime'].dt.month
    data['date'] = data['datetime'].dt.date
    
    print(f"Data prepared successfully for PACF analysis!")
    print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
    print(f"States: {data['state'].unique()}")
    print(f"Years: {sorted(data['year'].unique())}")
    
    return data

def aggregate_data_pacf(data, level='hourly'):
    """Aggregate rainfall data at different temporal levels for PACF"""
    
    if level == 'hourly':
        # For hourly, group by datetime 
        agg_data = data.groupby('datetime')['rainfall'].sum().reset_index()
        agg_data['datetime'] = pd.to_datetime(agg_data['datetime'])
        
    elif level == 'daily':
        # For daily aggregation, use the existing date column directly
        daily_data = data.groupby('date')['rainfall'].sum().reset_index()
        daily_data['datetime'] = pd.to_datetime(daily_data['date'])
        agg_data = daily_data
        
    elif level == 'monthly':
        # For monthly aggregation, use year and month columns
        monthly_data = data.groupby(['year', 'month'])['rainfall'].sum().reset_index()
        monthly_data['datetime'] = pd.to_datetime(monthly_data[['year', 'month']].assign(day=1))
        agg_data = monthly_data
        
    return agg_data.sort_values('datetime')

def plot_single_pacf(series, title, filename, lags=50, figsize=(12, 6)):
    """Plot PACF for a single time series"""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle cases where series might have insufficient data
    max_lags = min(lags, len(series) // 4)  # PACF typically uses n/4 as max lags
    if max_lags < 10:
        max_lags = min(10, len(series) - 1)
    
    try:
        plot_pacf(series, ax=ax, lags=max_lags, alpha=0.05)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Partial Autocorrelation')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")
        
    except Exception as e:
        print(f"Error plotting {title}: {str(e)}")
        plt.close()

def plot_pacf_state_wise(df, level='hourly', save_dir='plots/PACF/pacf_state_wise'):
    """Generate PACF plots for each state separately"""
    
    print(f"\n=== Generating State-wise PACF plots for {level} data ===")
    
    # Prepare data
    data = prepare_data_pacf(df)
    
    # Get states
    states = data['state'].unique()
    years = sorted(data['year'].unique())
    
    # Create save directory
    save_path = Path(save_dir) / level
    save_path.mkdir(parents=True, exist_ok=True)
    
    for state in states:
        state_data = data[data['state'] == state]
        
        # 1. All years combined
        agg_data = aggregate_data_pacf(state_data, level)
            
        # Set appropriate lags based on level
        if level == 'hourly':
            lags = 168  # 7 days worth of hours
        elif level == 'daily':
            lags = 90   # ~3 months
        elif level == 'monthly':
            lags = 24   # 2 years worth of months
            
        # Plot all years combined
        if len(agg_data) > 10:  # Ensure sufficient data
            title = f'PACF - {state} ({level.title()}) - All Years'
            filename = save_path / f'{state.replace(" ", "_").lower()}_{level}_all_years.png'
            plot_single_pacf(agg_data['rainfall'], title, filename, lags)
        
        # 2. Year by year analysis
        for year in years:
            year_data = state_data[state_data['year'] == year]
            if len(year_data) == 0:
                continue
                
            year_agg = aggregate_data_pacf(year_data, level)
            
            if len(year_agg) > 5:  # Ensure sufficient data
                title = f'PACF - {state} ({level.title()}) - {year}'
                filename = save_path / f'{state.replace(" ", "_").lower()}_{level}_{year}.png'
                plot_single_pacf(year_agg['rainfall'], title, filename, 
                                lags=min(lags, len(year_agg)//2))

def plot_pacf_regional(df, level='hourly', save_dir='plots/PACF/pacf_regional'):
    """Generate PACF plots for overall regional average"""
    
    print(f"\n=== Generating Regional PACF plots for {level} data ===")
    
    # Prepare data
    data = prepare_data_pacf(df)
    years = sorted(data['year'].unique())
    
    # Create save directory
    save_path = Path(save_dir) / level
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 1. All years combined - Regional average
    regional_data = aggregate_data_pacf(data, level)
    
    # Set appropriate lags
    if level == 'hourly':
        lags = 168  # 7 days
    elif level == 'daily':
        lags = 90   # ~3 months  
    elif level == 'monthly':
        lags = 24   # 2 years
    
    # Plot all years combined
    if len(regional_data) > 10:
        title = f'PACF - Regional Average ({level.title()}) - All Years'
        filename = save_path / f'regional_{level}_all_years.png'
        plot_single_pacf(regional_data['rainfall'], title, filename, lags)
    
    # 2. Year by year analysis
    for year in years:
        year_data = data[data['year'] == year]
        if len(year_data) == 0:
            continue
            
        year_regional = aggregate_data_pacf(year_data, level)
        
        if len(year_regional) > 5:
            title = f'PACF - Regional Average ({level.title()}) - {year}'
            filename = save_path / f'regional_{level}_{year}.png'
            plot_single_pacf(year_regional['rainfall'], title, filename,
                            lags=min(lags, len(year_regional)//2))

def plot_pacf_combined_comparison(df, save_dir='plots/PACF/pacf_combined'):
    """Generate combined PACF comparison plots"""
    
    print(f"\n=== Generating Combined PACF Comparison plots ===")
    
    data = prepare_data_pacf(df)
    states = data['state'].unique()
    
    # Create save directories
    state_save_path = Path(save_dir) / 'state_wise'
    regional_save_path = Path(save_dir) / 'regional'
    state_save_path.mkdir(parents=True, exist_ok=True)
    regional_save_path.mkdir(parents=True, exist_ok=True)
    
    levels = ['hourly', 'daily', 'monthly']
    
    # Combined state comparison for each level
    for level in levels:
        fig, axes = plt.subplots(len(states), 1, figsize=(15, 6*len(states)))
        if len(states) == 1:
            axes = [axes]
            
        for i, state in enumerate(states):
            state_data = data[data['state'] == state]
            
            agg_data = aggregate_data_pacf(state_data, level)
            
            if level == 'hourly':
                lags = 168
            elif level == 'daily':
                lags = 90
            elif level == 'monthly':
                lags = 24
            
            if len(agg_data) > 10:
                max_lags = min(lags, len(agg_data) // 4)
                try:
                    plot_pacf(agg_data['rainfall'], ax=axes[i], lags=max_lags, alpha=0.05)
                    axes[i].set_title(f'PACF - {state} ({level.title()})', fontsize=12, fontweight='bold')
                    axes[i].grid(True, alpha=0.3)
                except:
                    axes[i].text(0.5, 0.5, f'Insufficient data for {state}', 
                               ha='center', va='center', transform=axes[i].transAxes)
        
        plt.tight_layout()
        filename = state_save_path / f'states_comparison_{level}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    
    # Combined level comparison for regional data
    fig, axes = plt.subplots(len(levels), 1, figsize=(15, 6*len(levels)))
    
    for i, level in enumerate(levels):
        regional_data = aggregate_data_pacf(data, level)
        
        if level == 'hourly':
            lags = 168
        elif level == 'daily':
            lags = 90
        elif level == 'monthly':
            lags = 24
        
        if len(regional_data) > 10:
            max_lags = min(lags, len(regional_data) // 4)
            try:
                plot_pacf(regional_data['rainfall'], ax=axes[i], lags=max_lags, alpha=0.05)
                axes[i].set_title(f'PACF - Regional Average ({level.title()})', fontsize=12, fontweight='bold')
                axes[i].grid(True, alpha=0.3)
            except:
                axes[i].text(0.5, 0.5, f'Insufficient data for {level}', 
                           ha='center', va='center', transform=axes[i].transAxes)
    
    plt.tight_layout()
    filename = regional_save_path / 'regional_levels_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_acf_pacf_side_by_side(df, save_dir='plots/PACF/acf_pacf_comparison'):
    """Generate side-by-side ACF and PACF comparison plots"""
    
    print(f"\n=== Generating ACF vs PACF Comparison plots ===")
    
    data = prepare_data_pacf(df)
    states = data['state'].unique()
    levels = ['hourly', 'daily', 'monthly']
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 1. State-wise ACF vs PACF comparison
    for state in states:
        state_data = data[data['state'] == state]
        
        for level in levels:
            fig, axes = plt.subplots(1, 2, figsize=(20, 6))
            
            agg_data = aggregate_data_pacf(state_data, level)
            
            if level == 'hourly':
                lags = 168
            elif level == 'daily':
                lags = 90
            elif level == 'monthly':
                lags = 24
            
            if len(agg_data) > 10:
                max_lags = min(lags, len(agg_data) // 4)
                
                try:
                    # ACF plot
                    from statsmodels.graphics.tsaplots import plot_acf
                    plot_acf(agg_data['rainfall'], ax=axes[0], lags=max_lags, alpha=0.05)
                    axes[0].set_title(f'ACF - {state} ({level.title()})', fontsize=14, fontweight='bold')
                    axes[0].grid(True, alpha=0.3)
                    
                    # PACF plot
                    plot_pacf(agg_data['rainfall'], ax=axes[1], lags=max_lags, alpha=0.05)
                    axes[1].set_title(f'PACF - {state} ({level.title()})', fontsize=14, fontweight='bold')
                    axes[1].grid(True, alpha=0.3)
                    
                except:
                    axes[0].text(0.5, 0.5, f'Insufficient data for ACF', 
                               ha='center', va='center', transform=axes[0].transAxes)
                    axes[1].text(0.5, 0.5, f'Insufficient data for PACF', 
                               ha='center', va='center', transform=axes[1].transAxes)
            
            plt.tight_layout()
            filename = save_path / f'{state.replace(" ", "_").lower()}_{level}_acf_pacf_comparison.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {filename}")
    
    # 2. Regional ACF vs PACF comparison
    for level in levels:
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        
        regional_data = aggregate_data_pacf(data, level)
        
        if level == 'hourly':
            lags = 168
        elif level == 'daily':
            lags = 90
        elif level == 'monthly':
            lags = 24
        
        if len(regional_data) > 10:
            max_lags = min(lags, len(regional_data) // 4)
            
            try:
                # ACF plot
                from statsmodels.graphics.tsaplots import plot_acf
                plot_acf(regional_data['rainfall'], ax=axes[0], lags=max_lags, alpha=0.05)
                axes[0].set_title(f'ACF - Regional Average ({level.title()})', fontsize=14, fontweight='bold')
                axes[0].grid(True, alpha=0.3)
                
                # PACF plot
                plot_pacf(regional_data['rainfall'], ax=axes[1], lags=max_lags, alpha=0.05)
                axes[1].set_title(f'PACF - Regional Average ({level.title()})', fontsize=14, fontweight='bold')
                axes[1].grid(True, alpha=0.3)
                
            except:
                axes[0].text(0.5, 0.5, f'Insufficient data for ACF', 
                           ha='center', va='center', transform=axes[0].transAxes)
                axes[1].text(0.5, 0.5, f'Insufficient data for PACF', 
                           ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        filename = save_path / f'regional_{level}_acf_pacf_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

def generate_all_pacf_plots(df):
    """
    Main function to generate all PACF plots as requested
    
    Parameters:
    df: pandas DataFrame with columns ['latitude', 'longitude', 'datetime', 'rainfall', 'state']
    """
    
    print("Starting comprehensive PACF analysis...")
    print("=" * 60)
    
    # Create directory structure
    create_pacf_directory_structure()
    
    # Generate all plots
    levels = ['hourly', 'daily', 'monthly']
    
    # 1. State-wise PACF plots for all levels
    for level in levels:
        plot_pacf_state_wise(df, level)
    
    # 2. Regional PACF plots for all levels  
    for level in levels:
        plot_pacf_regional(df, level)
    
    # 3. Combined PACF comparison plots
    plot_pacf_combined_comparison(df)
    
    # 4. ACF vs PACF side-by-side comparison plots
    plot_acf_pacf_side_by_side(df)
    
    print("\n" + "=" * 60)
    print("All PACF plots generated successfully!")
    print("Check the 'plots/PACF/' directory for all generated plots.")
    print("\nPACF Directory structure:")
    print("plots/PACF/")
    print("├── pacf_state_wise/")
    print("│   ├── hourly/")
    print("│   ├── daily/") 
    print("│   ├── monthly/")
    print("│   └── yearly/")
    print("├── pacf_regional/")
    print("│   ├── hourly/")
    print("│   ├── daily/")
    print("│   ├── monthly/")
    print("│   └── yearly/")
    print("├── pacf_combined/")
    print("│   ├── state_wise/")
    print("│   └── regional/")
    print("└── acf_pacf_comparison/")
    print("    ├── State-wise ACF vs PACF plots")
    print("    └── Regional ACF vs PACF plots")

def generate_comprehensive_analysis(df):
    """
    Generate both ACF and PACF plots in one go
    
    Parameters:
    df: pandas DataFrame with columns ['latitude', 'longitude', 'datetime', 'rainfall', 'state']
    """
    
    print("Starting COMPREHENSIVE ACF & PACF analysis...")
    print("=" * 80)
    
    # Import ACF functions (assuming acf_utils.py is available)
    try:
        from acf_utils import generate_all_acf_plots
        print("ACF functions imported successfully!")
        generate_all_acf_plots(df)
    except ImportError:
        print("ACF functions not available. Run ACF analysis separately.")
    
    # Generate PACF plots
    generate_all_pacf_plots(df)
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS COMPLETE!")
    print("Both ACF and PACF plots have been generated successfully!")

# Example usage:
# For PACF plots only:
# generate_all_pacf_plots(your_dataframe)

# For both ACF and PACF plots:
# generate_comprehensive_analysis(your_dataframe)