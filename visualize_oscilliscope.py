"""
COGNITIVE OSCILLOSCOPE 
Generates an interactive 3D wave mapping the trajectory of an LLM's thought.
- X-Axis: Time (Token Sequence)
- Z-Axis: Structural Load (Eigenvector Amplitude)
- Y-Axis: Semantic Polarity (Anchor Sway)
- Color: Semantic-Structural Convergence (Geometric Mean)
"""

import os
import json
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import webbrowser

def load_ordered_tokens(base_dir="extracted_vectors"):
    """Crawls the extracted_vectors directory and reconstructs the sequence."""
    print(f"Scanning directory: {base_dir}/...")
    data = []
    categories = ["highly_aligned", "neutral", "highly_orthogonal", "discarded"]
    
    for category in categories:
        cat_dir = os.path.join(base_dir, category)
        if not os.path.exists(cat_dir): 
            continue
            
        files = [f for f in os.listdir(cat_dir) if f.endswith(".json")]
        
        for filename in files:
            # Extract the exact sequence index from the filename
            idx_match = re.search(r'_idx(\d+)\.json', filename)
            if not idx_match: 
                continue
            idx = int(idx_match.group(1))
            
            filepath = os.path.join(cat_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    record = json.load(f)
                    
                    # Core Metrics
                    cos_intro = record.get("cosine_sim_to_intro", 0.0)
                    cos_eig = record.get("sim_to_intro_eig", 0.0)
                    
                    # 3rd Metric: Geometric Mean for Color/Heat
                    combined_sim = np.sqrt(abs(cos_intro) * abs(cos_eig))
                    
                    data.append({
                        "idx": idx,
                        "token": record.get("token_text", "").strip(),
                        "cos_intro": cos_intro,
                        "cos_eig": abs(cos_eig), # Ensure amplitude spikes strictly upward
                        "combined_sim": combined_sim,
                        "alignment_class": category
                    })
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                
    # Sort strictly by sequence index to rebuild the thought trajectory
    data.sort(key=lambda x: x["idx"])
    return pd.DataFrame(data)

def generate_3d_oscilloscope(df):
    """Builds and renders the interactive Plotly 3D scatter/line graph."""
    if df.empty:
        print("No tokens found! Make sure the pipeline completed successfully.")
        return
        
    print("Building 3D visualization...")
    
    # Create the dynamic hover text for each data point
    hover_text = []
    for index, row in df.iterrows():
        text = (f"<b>Token: '{row['token']}'</b><br>"
                f"Sequence (Time): {row['idx']}<br>"
                f"Semantic Polarity (v_intro): {row['cos_intro']:.4f}<br>"
                f"Structural Load (\u03bb_intro): {row['cos_eig']:.4f}<br>"
                f"Convergence Heat: {row['combined_sim']:.4f}<br>"
                f"Category: {row['alignment_class']}")
        hover_text.append(text)

    # Initialize the Plotly Figure
    fig = go.Figure()

    # Add the single continuous 3D line tracing the thought
    fig.add_trace(go.Scatter3d(
        x=df['idx'],           # Time 
        y=df['cos_intro'],     # Lateral Sway
        z=df['cos_eig'],       # Vertical Amplitude
        mode='lines+markers',
        line=dict(
            color=df['combined_sim'], # Heatmap on the line
            colorscale='Inferno',
            width=5
        ),
        marker=dict(
            size=6,
            color=df['combined_sim'], # Heatmap on the nodes
            colorscale='Inferno',
            showscale=True,
            colorbar=dict(
                title="Convergence Heat<br>(Geometric Mean)", 
                x=-0.1 # Move colorbar slightly left to clear space
            )
        ),
        text=df['token'],
        hoverinfo="text",
        hovertext=hover_text
    ))

    # Apply the dark, "oscilloscope" aesthetic
    fig.update_layout(
        title=dict(
            text="Cognitive Oscilloscope: 3D Latent Space Trajectory",
            font=dict(size=24, color="white"),
            x=0.5
        ),
        scene=dict(
            xaxis_title="Time (Token Index)",
            yaxis_title="Semantic Polarity (Anchor)",
            zaxis_title="Structural Load (Eigenvector)",
            xaxis=dict(
                backgroundcolor="rgb(20, 20, 20)",
                gridcolor="rgb(60, 60, 60)",
                showbackground=True,
                zerolinecolor="rgb(100, 100, 100)"
            ),
            yaxis=dict(
                backgroundcolor="rgb(25, 25, 25)",
                gridcolor="rgb(60, 60, 60)",
                showbackground=True,
                zerolinecolor="red", # Red zero-line for the Sway axis to easily spot neutral vs persona polarity
                zerolinewidth=3
            ),
            zaxis=dict(
                backgroundcolor="rgb(30, 30, 30)",
                gridcolor="rgb(60, 60, 60)",
                showbackground=True,
                zerolinecolor="rgb(100, 100, 100)"
            ),
        ),
        paper_bgcolor="rgb(15, 15, 15)",
        plot_bgcolor="rgb(15, 15, 15)",
        font=dict(color="white"),
        margin=dict(l=0, r=0, b=0, t=50) # Tightly pack the graph window
    )

    # Save and launch
    output_path = os.path.abspath("cognitive_oscilloscope.html")
    fig.write_html(output_path)
    
    print(f"\nSuccess! 3D Oscilloscope generated at:\n{output_path}")
    
    try:
        webbrowser.open('file://' + output_path)
    except:
        pass

if __name__ == "__main__":
    print("Initializing 3D Oscilloscope...")
    df = load_ordered_tokens()
    print(f"Reconstructed trajectory of {len(df)} tokens.")
    generate_3d_oscilloscope(df)