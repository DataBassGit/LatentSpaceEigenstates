"""
PURE EIGENVALUE VISUALIZATION
Graphs the exact 'Trench Gravity' (λ1 amplification) across the 
cognitive alignment classes, completely isolating the model's 
deepest structural architecture.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_vector_data(base_dir="extracted_vectors"):
    """Crawls the extracted_vectors directory and compiles the pure eigen data."""
    print(f"Scanning directory: {base_dir}/...")
    data = []
    
    # We deliberately ignore the 'discarded' folder to keep the linguistic baseline clean
    categories = ["highly_aligned", "neutral", "highly_orthogonal"]
    
    for category in categories:
        cat_dir = os.path.join(base_dir, category)
        if not os.path.exists(cat_dir):
            continue
            
        files = [f for f in os.listdir(cat_dir) if f.endswith(".json")]
        if files:
            print(f"Loading {len(files)} vectors from {category}...")
            
        for filename in files:
            filepath = os.path.join(cat_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    record = json.load(f)
                    data.append({
                        "token": record.get("token_text", "").strip(),
                        "trench_gravity": record.get("trench_gravity", 0.0), # The pure math!
                        "cosine_sim": record.get("cosine_sim_to_intro", 0.0),
                        "alignment_class": category
                    })
            except Exception as e:
                pass
                
    return pd.DataFrame(data)

def generate_visualizations(df):
    """Creates a dual-panel plot of the Pure Trench Gravity distributions."""
    if df.empty:
        print("No data found! You may need to rerun the pipeline if Ctrl-C killed the disk write.")
        return

    sns.set_theme(style="whitegrid", context="talk")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Pure Trench Gravity (λ1) by Cognitive Alignment", fontsize=20, weight='bold', y=1.05)

    palette = {
        "highly_aligned": "#e74c3c",    # Red
        "neutral": "#95a5a6",           # Gray
        "highly_orthogonal": "#3498db"  # Blue
    }

    # PANEL 1: Density Plot
    sns.kdeplot(
        data=df, 
        x="trench_gravity", 
        hue="alignment_class",
        hue_order=["highly_aligned", "neutral", "highly_orthogonal"],
        common_norm=False, 
        fill=True,
        alpha=0.3,
        palette=palette,
        ax=ax1,
        linewidth=2.5
    )
    ax1.set_title("Distribution of Absolute Amplification", fontsize=16)
    ax1.set_xlabel("Trench Gravity (λ1 * cos²θ)", fontsize=14)
    ax1.set_ylabel("Density", fontsize=14)

    # PANEL 2: Boxplot + Stripplot
    sns.boxplot(
        data=df, 
        x="alignment_class", 
        y="trench_gravity",
        order=["highly_aligned", "neutral", "highly_orthogonal"],
        palette=palette,
        ax=ax2,
        showfliers=False,
        boxprops={'alpha': 0.4}
    )
    
    # Sample down the stripplot if there are thousands of points so it doesn't become a solid block
    plot_df = df if len(df) < 2000 else df.groupby('alignment_class').sample(n=500, random_state=42, replace=True)
    
    sns.stripplot(
        data=plot_df, 
        x="alignment_class", 
        y="trench_gravity",
        order=["highly_aligned", "neutral", "highly_orthogonal"],
        palette=palette,
        ax=ax2,
        alpha=0.5,
        jitter=True,
        size=3
    )
    
    ax2.set_title("Structural Priority (Top Eigenvector)", fontsize=16)
    ax2.set_xlabel("Alignment Class", fontsize=14)
    ax2.set_ylabel("Trench Gravity", fontsize=14)
    ax2.set_xticklabels(["Highly\nAligned\n(Introspective)", "Neutral\n(Baseline)", "Highly\nOrthogonal\n(External)"])

    plt.tight_layout()
    out_file = "pure_eigenvalue_plot.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Visualization saved to: {out_file}")
    plt.show()

if __name__ == "__main__":
    print("Initializing Data Visualization...")
    df = load_vector_data()
    
    if not df.empty:
        print("\nTrench Gravity Summary:")
        print(df.groupby('alignment_class')['trench_gravity'].describe())
        print("\nGenerating graphs...")
        generate_visualizations(df)