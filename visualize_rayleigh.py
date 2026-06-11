"""
RAYLEIGH QUOTIENT VISUALIZATION
Parses extracted vector JSONs and graphs the structural gravity 
distribution across cognitive alignment classes.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_vector_data(base_dir="extracted_vectors"):
    """Crawls the extracted_vectors directory and compiles the data."""
    print(f"Scanning directory: {base_dir}/...")
    data = []
    categories = ["highly_aligned", "neutral", "highly_orthogonal"]
    
    for category in categories:
        cat_dir = os.path.join(base_dir, category)
        if not os.path.exists(cat_dir):
            print(f"Warning: Directory {cat_dir} not found. Skipping.")
            continue
            
        files = [f for f in os.listdir(cat_dir) if f.endswith(".json")]
        print(f"Loading {len(files)} vectors from {category}...")
        
        for filename in files:
            filepath = os.path.join(cat_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    record = json.load(f)
                    data.append({
                        "token": record.get("token_text", "").strip(),
                        "rayleigh_quotient": record.get("rayleigh_quotient", 0.0),
                        "cosine_sim": record.get("cosine_sim_to_intro", 0.0),
                        "alignment_class": category
                    })
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                
    return pd.DataFrame(data)

def generate_visualizations(df):
    """Creates a dual-panel plot of the Rayleigh distributions."""
    if df.empty:
        print("No data found! Make sure you run the extraction pipeline first.")
        return

    # Set up the scientific aesthetic
    sns.set_theme(style="whitegrid", context="talk")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Structural Gravity (Rayleigh Quotient) by Cognitive Alignment", fontsize=20, weight='bold', y=1.05)

    # Custom colors for our classes
    palette = {
        "highly_aligned": "#e74c3c",    # Red
        "neutral": "#95a5a6",           # Gray
        "highly_orthogonal": "#3498db"  # Blue
    }

    # PANEL 1: Density / Histogram Plot
    sns.histplot(
        data=df, 
        x="rayleigh_quotient", 
        hue="alignment_class",
        hue_order=["highly_aligned", "neutral", "highly_orthogonal"],
        kde=True,           # Adds the smooth distribution line
        element="step",     # Removes internal bars for cleaner look
        stat="density",     # Normalizes the Y-axis across different sample sizes
        common_norm=False, 
        palette=palette,
        ax=ax1,
        linewidth=2
    )
    ax1.set_title("Distribution Density", fontsize=16)
    ax1.set_xlabel("Rayleigh Quotient (Eigenvalue Proxy)", fontsize=14)
    ax1.set_ylabel("Density", fontsize=14)

    # PANEL 2: Boxplot + Stripplot (to show individual vector spread)
    sns.boxplot(
        data=df, 
        x="alignment_class", 
        y="rayleigh_quotient",
        order=["highly_aligned", "neutral", "highly_orthogonal"],
        palette=palette,
        ax=ax2,
        showfliers=False, # Hide outliers on the boxplot to let the strip plot show them
        boxprops={'alpha': 0.4} # Make boxes semi-transparent
    )
    sns.stripplot(
        data=df, 
        x="alignment_class", 
        y="rayleigh_quotient",
        order=["highly_aligned", "neutral", "highly_orthogonal"],
        palette=palette,
        ax=ax2,
        alpha=0.6,
        jitter=True,
        size=4
    )
    ax2.set_title("Statistical Spread & Outliers", fontsize=16)
    ax2.set_xlabel("Alignment Class", fontsize=14)
    ax2.set_ylabel("Rayleigh Quotient", fontsize=14)
    
    # Clean up X-axis labels for panel 2
    ax2.set_xticklabels(["Highly\nAligned\n(Introspective)", "Neutral\n(Baseline)", "Highly\nOrthogonal\n(External)"])

    plt.tight_layout()
    
    # Save to disk
    out_file = "rayleigh_analysis_plot.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Visualization saved to: {out_file}")
    
    # Attempt to open the plot window directly (works on most Windows setups)
    plt.show()

if __name__ == "__main__":
    print("Initializing Data Visualization...")
    df = load_vector_data()
    
    print("\nData Summary:")
    print(df.groupby('alignment_class')['rayleigh_quotient'].describe())
    
    print("\nGenerating graphs...")
    generate_visualizations(df)