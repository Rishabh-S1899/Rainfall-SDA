import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_directory_structure():
    """Create the plots directory structure"""
    base_dir = Path('plots')
    subdirs = [
        'state_wise/hourly',
        'state_wise/daily', 
        'state_wise/monthly',
        'state_wise/yearly',
        'regional/hourly',
        'regional/daily',
        'regional/monthly', 
        'regional/yearly',
        'combined/state_wise',
        'combined/regional'
    ]
    
    for subdir in subdirs:
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print("Directory structure created successfully!")

def prepare_data(df):
    """Prepare and validate the rainfall dataframe"""
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
    
    print(f"Data prepared successfully!")
    print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
    print(f"States: {data['state'].unique()}")
    print(f"Years: {sorted(data['year'].unique())}")
    
    return data

def aggregate_data(data, level='hourly'):
    """Aggregate rainfall data at different temporal levels"""
    
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

def plot_single_acf(series, title, filename, lags=50, figsize=(12, 6)):
    """Plot ACF for a single time series"""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle cases where series might have insufficient data
    max_lags = min(lags, len(series) // 4)  # ACF typically uses n/4 as max lags
    if max_lags < 10:
        max_lags = min(10, len(series) - 1)
    
    try:
        plot_acf(series, ax=ax, lags=max_lags, alpha=0.05)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")
        
    except Exception as e:
        print(f"Error plotting {title}: {str(e)}")
        plt.close()

def plot_acf_state_wise(df, level='hourly', save_dir='plots/ACF/state_wise'):
    """Generate ACF plots for each state separately"""
    
    print(f"\n=== Generating State-wise ACF plots for {level} data ===")
    
    # Prepare data
    data = prepare_data(df)
    
    # Get states
    states = data['state'].unique()
    years = sorted(data['year'].unique())
    
    # Create save directory
    save_path = Path(save_dir) / level
    save_path.mkdir(parents=True, exist_ok=True)
    
    for state in states:
        state_data = data[data['state'] == state]
        
        # 1. All years combined
        agg_data = aggregate_data(state_data, level)
            
        # Set appropriate lags based on level
        if level == 'hourly':
            lags = 168  # 7 days worth of hours
        elif level == 'daily':
            lags = 90   # ~3 months
        elif level == 'monthly':
            lags = 24   # 2 years worth of months
            
        # Plot all years combined
        if len(agg_data) > 10:  # Ensure sufficient data
            title = f'ACF - {state} ({level.title()}) - All Years'
            filename = save_path / f'{state.replace(" ", "_").lower()}_{level}_all_years.png'
            plot_single_acf(agg_data['rainfall'], title, filename, lags)
        
        # 2. Year by year analysis
        for year in years:
            year_data = state_data[state_data['year'] == year]
            if len(year_data) == 0:
                continue
                
            year_agg = aggregate_data(year_data, level)
            
            if len(year_agg) > 5:  # Ensure sufficient data
                title = f'ACF - {state} ({level.title()}) - {year}'
                filename = save_path / f'{state.replace(" ", "_").lower()}_{level}_{year}.png'
                plot_single_acf(year_agg['rainfall'], title, filename, 
                               lags=min(lags, len(year_agg)//2))

def plot_acf_regional(df, level='hourly', save_dir='plots/ACF/regional'):
    """Generate ACF plots for overall regional average"""
    
    print(f"\n=== Generating Regional ACF plots for {level} data ===")
    
    # Prepare data
    data = prepare_data(df)
    years = sorted(data['year'].unique())
    
    # Create save directory
    save_path = Path(save_dir) / level
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 1. All years combined - Regional average
    regional_data = aggregate_data(data, level)
    
    # Set appropriate lags
    if level == 'hourly':
        lags = 168  # 7 days
    elif level == 'daily':
        lags = 90   # ~3 months  
    elif level == 'monthly':
        lags = 24   # 2 years
    
    # Plot all years combined
    if len(regional_data) > 10:
        title = f'ACF - Regional Average ({level.title()}) - All Years'
        filename = save_path / f'regional_{level}_all_years.png'
        plot_single_acf(regional_data['rainfall'], title, filename, lags)
    
    # 2. Year by year analysis
    for year in years:
        year_data = data[data['year'] == year]
        if len(year_data) == 0:
            continue
            
        year_regional = aggregate_data(year_data, level)
        
        if len(year_regional) > 5:
            title = f'ACF - Regional Average ({level.title()}) - {year}'
            filename = save_path / f'regional_{level}_{year}.png'
            plot_single_acf(year_regional['rainfall'], title, filename,
                           lags=min(lags, len(year_regional)//2))

def plot_acf_combined_comparison(df, save_dir='plots/ACF/combined'):
    """Generate combined comparison plots"""
    
    print(f"\n=== Generating Combined Comparison plots ===")
    
    data = prepare_data(df)
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
            
            agg_data = aggregate_data(state_data, level)
            
            if level == 'hourly':
                lags = 168
            elif level == 'daily':
                lags = 90
            elif level == 'monthly':
                lags = 24
            
            if len(agg_data) > 10:
                max_lags = min(lags, len(agg_data) // 4)
                try:
                    plot_acf(agg_data['rainfall'], ax=axes[i], lags=max_lags, alpha=0.05)
                    axes[i].set_title(f'ACF - {state} ({level.title()})', fontsize=12, fontweight='bold')
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
        regional_data = aggregate_data(data, level)
        
        if level == 'hourly':
            lags = 168
        elif level == 'daily':
            lags = 90
        elif level == 'monthly':
            lags = 24
        
        if len(regional_data) > 10:
            max_lags = min(lags, len(regional_data) // 4)
            try:
                plot_acf(regional_data['rainfall'], ax=axes[i], lags=max_lags, alpha=0.05)
                axes[i].set_title(f'ACF - Regional Average ({level.title()})', fontsize=12, fontweight='bold')
                axes[i].grid(True, alpha=0.3)
            except:
                axes[i].text(0.5, 0.5, f'Insufficient data for {level}', 
                           ha='center', va='center', transform=axes[i].transAxes)
    
    plt.tight_layout()
    filename = regional_save_path / 'regional_levels_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def generate_all_acf_plots(df):
    """
    Main function to generate all ACF plots as requested
    
    Parameters:
    df: pandas DataFrame with columns ['latitude', 'longitude', 'datetime', 'rainfall', 'state']
    """
    
    print("Starting comprehensive ACF analysis...")
    print("=" * 60)
    
    # Create directory structure
    create_directory_structure()
    
    # Generate all plots
    levels = ['hourly', 'daily', 'monthly']
    
    # 1. State-wise plots for all levels
    for level in levels:
        plot_acf_state_wise(df, level)
    
    # 2. Regional plots for all levels  
    for level in levels:
        plot_acf_regional(df, level)
    
    # 3. Combined comparison plots
    plot_acf_combined_comparison(df)
    
    print("\n" + "=" * 60)
    print("All ACF plots generated successfully!")
    print("Check the 'plots/ACF/' directory for all generated plots.")
    print("\nDirectory structure:")
    print("plots/ACF/")
    print("├── state_wise/")
    print("│   ├── hourly/")
    print("│   ├── daily/") 
    print("│   ├── monthly/")
    print("│   └── yearly/")
    print("├── regional/")
    print("│   ├── hourly/")
    print("│   ├── daily/")
    print("│   ├── monthly/")
    print("│   └── yearly/")
    print("└── combined/")
    print("    ├── state_wise/")
    print("    └── regional/")

# Example usage:
# generate_all_acf_plots(your_dataframe)