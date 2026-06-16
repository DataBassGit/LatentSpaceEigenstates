"""
SMOOTHED COGNITIVE OSCILLOSCOPE 
Generates an interactive 3D wave mapping the trajectory of an LLM's thought.
Applies a rolling average to filter out grammatical "strobe" noise and reveal 
the macro-trajectory of the cognitive state.

- X-Axis: Time (Token Sequence)
- Y-Axis: Semantic Polarity (Anchor Sway)
- Z-Axis: Structural Load (Trench Gravity)
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
                    trench_grav = record.get("trench_gravity", 0.0)
                    
                    # 3rd Metric: Geometric Mean for Color/Heat
                    combined_sim = np.sqrt(abs(cos_intro) * abs(trench_grav))
                    
                    data.append({
                        "idx": idx,
                        "token": record.get("token_text", "").replace('\n', '\\n'),
                        "cos_intro": cos_intro,
                        "trench_gravity": trench_grav,
                        "combined_sim": combined_sim,
                        "alignment_class": category
                    })
            except Exception as e:
                pass
                
    # Sort strictly by sequence index
    data.sort(key=lambda x: x["idx"])
    df = pd.DataFrame(data)
    
    # Calculate Rolling Averages (Window of 5 tokens smooths over most grammar gaps)
    # min_periods=1 ensures the ends of the sequence don't get truncated
    df['smooth_y'] = df['cos_intro'].rolling(window=5, center=True, min_periods=1).mean()
    df['smooth_z'] = df['trench_gravity'].rolling(window=5, center=True, min_periods=1).mean()
    df['smooth_color'] = df['combined_sim'].rolling(window=5, center=True, min_periods=1).mean()
    
    return df

def generate_3d_oscilloscope(df):
    """Builds and renders the interactive Plotly 3D graph."""
    if df.empty:
        print("No tokens found! Make sure the pipeline completed successfully.")
        return
        
    print("Building Smoothed 3D visualization...")
    
    # Create the dynamic hover text for the raw points
    raw_hover_text = []
    for index, row in df.iterrows():
        text = (f"<b>Token: '{row['token']}'</b><br>"
                f"Time: {row['idx']}<br>"
                f"Polarity (Anchor): {row['cos_intro']:.4f}<br>"
                f"Gravity (λ1): {row['trench_gravity']:.4f}<br>"
                f"Convergence Heat: {row['combined_sim']:.4f}")
        raw_hover_text.append(text)

    # Initialize the Plotly Figure
    fig = go.Figure()

    # TRACE 1: The Raw Tokens (Faint, disconnected dots)
    fig.add_trace(go.Scatter3d(
        x=df['idx'],           
        y=df['cos_intro'],     
        z=df['trench_gravity'],       
        mode='markers',
        marker=dict(
            size=3,
            color='rgba(200, 200, 200, 0.3)' # Faint grey/white
        ),
        text=df['token'],
        hoverinfo="text",
        hovertext=raw_hover_text,
        name="Raw Token Fire"
    ))

    # TRACE 2: The Smoothed Cognitive Wave (Thick, continuous, glowing line)
    fig.add_trace(go.Scatter3d(
        x=df['idx'],           
        y=df['smooth_y'],     
        z=df['smooth_z'],       
        mode='lines',
        line=dict(
            color=df['smooth_color'], # Heatmap on the line
            colorscale='Inferno',
            width=8,
            colorbar=dict(
                title=dict(
                    text="Convergence Heat<br>(Geom Mean)",
                    font=dict(color="white", size=14)
                ),
                tickfont=dict(color="white"),
                x=0.9, 
                len=0.6,
                thickness=15
            )
        ),
        hoverinfo="skip", # We skip hover on the line so the user's mouse snaps to the raw tokens
        name="Cognitive Trajectory"
    ))

    # Apply the dark, "oscilloscope" aesthetic
    fig.update_layout(
        title=dict(
            text="Cognitive Oscilloscope: Smoothed 3D Trajectory",
            font=dict(size=24, color="white"),
            x=0.5
        ),
        scene=dict(
            xaxis_title="Time (Token Sequence)",
            yaxis_title="Semantic Polarity (Anchor)",
            zaxis_title="Structural Load (Trench Gravity)",
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
                zerolinecolor="rgba(255, 0, 0, 0.5)", # Red zero-line for Polarity
                zerolinewidth=2
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
        margin=dict(l=0, r=0, b=0, t=50),
        showlegend=False
    )

    # Save and launch
    output_path = os.path.abspath("smoothed_oscilloscope.html")
    fig.write_html(output_path)
    
    print(f"\nSuccess! 3D Oscilloscope generated at:\n{output_path}")
    
    try:
        webbrowser.open('file://' + output_path)
    except:
        pass

if __name__ == "__main__":
    print("Initializing 3D Oscilloscope...")
    df = load_ordered_tokens()
    generate_3d_oscilloscope(df)