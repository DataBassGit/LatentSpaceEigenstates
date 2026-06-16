"""
TOKEN HEATMAP VISUALIZER
Reconstructs the generated sequence and generates an interactive HTML 
document. Highlights tokens based on specific cosine similarities:
1. Cosine Similarity to Introspection Anchor
2. Cosine Similarity to Introspection Eigenvector
3. Semantic-Structural Convergence (Geometric Mean)
"""

import os
import json
import re
import html
import numpy as np
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
                    cos_intro = record.get("cosine_sim_to_intro", 0.0)
                    cos_eig = record.get("sim_to_intro_eig", 0.0)
                    
                    # Calculate Geometric Mean to find tokens strong in BOTH metrics
                    combined_sim = np.sqrt(abs(cos_intro) * abs(cos_eig))
                    
                    data.append({
                        "idx": idx,
                        "token": record.get("token_text", ""),
                        "cosine_sim": cos_intro,
                        "sim_eig": cos_eig,
                        "combined_sim": combined_sim,
                        "alignment_class": category
                    })
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                
    data.sort(key=lambda x: x["idx"])
    return data

def build_heatmap_section(tokens, metric_key, title, description):
    """Generates the HTML block for a specific similarity metric."""
    valid_vals = [abs(t[metric_key]) for t in tokens if t['alignment_class'] != 'discarded']
    if not valid_vals:
        return f"<h2>{title}</h2><p>No valid data found.</p>"
        
    max_val = np.percentile(valid_vals, 95)
    if max_val == 0: max_val = 1.0 

    section_html = [
        f"<div class='heatmap-section'>",
        f"<h2>{title}</h2>",
        f"<p class='description'>{description}</p>",
        "<div class='text-content'>"
    ]
    
    for t in tokens:
        raw_text = t['token']
        safe_text = html.escape(raw_text).replace('\n', '<br>')
        
        val = t[metric_key]
        
        if t['alignment_class'] == 'discarded':
            rgb = "0, 0, 0"
            opacity = 0.0
            disp_class = "discarded"
        elif t['alignment_class'] == 'highly_aligned':
            rgb = "231, 76, 60"   # Red (Absolute Alignment)
            disp_class = "Aligned"
        elif t['alignment_class'] == 'highly_orthogonal':
            rgb = "52, 152, 219"  # Blue (Orthogonal)
            disp_class = "Orthogonal"
        else: 
            rgb = "149, 165, 166" # Gray (Neutral)
            disp_class = "Neutral"
                
        if t['alignment_class'] != 'discarded':
            raw_opacity = abs(val) / max_val
            opacity = min(0.95, max(0.1, raw_opacity))
            
        tooltip = f"Token: '{raw_text.strip()}'&#10;Metric: {val:.4f}&#10;Class: {disp_class}&#10;Index: {t['idx']}"
        span = f"<span class='token' style='background-color: rgba({rgb}, {opacity})' title=\"{tooltip}\">{safe_text}</span>"
        section_html.append(span)
        
    section_html.append("</div></div>")
    return "".join(section_html)

def generate_html_heatmap(tokens):
    """Generates an HTML file containing all three heatmap sections."""
    if not tokens:
        print("No tokens found! Make sure the pipeline completed successfully.")
        return
        
    html_content = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Tri-Metric Heatmap</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #1e1e1e; color: #d4d4d4; padding: 40px; line-height: 1.8; font-size: 18px; }",
        ".container { max-width: 1000px; margin: 0 auto; background: #252526; padding: 40px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }",
        ".heatmap-section { margin-bottom: 60px; padding-bottom: 40px; border-bottom: 1px solid #444; }",
        ".heatmap-section:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }",
        ".token { display: inline; border-radius: 3px; cursor: default; transition: background-color 0.2s, transform 0.1s; white-space: pre-wrap; padding: 2px 0; }",
        ".token:hover { filter: brightness(1.5); position: relative; z-index: 10; border-bottom: 2px solid white; }",
        "h1 { color: #fff; margin-top: 0; text-align: center; }",
        "h2 { color: #fff; margin-bottom: 5px; }",
        ".description { font-size: 15px; color: #aaa; margin-bottom: 20px; font-style: italic; }",
        ".legend { margin-bottom: 30px; padding: 15px; background: #2d2d30; border-radius: 8px; font-size: 15px; text-align: center; }",
        ".legend span { padding: 4px 10px; margin: 0 10px; border-radius: 4px; color: white; display: inline-block; }",
        "</style></head><body>",
        "<div class='container'>",
        "<h1>Structural Similarity Heatmaps</h1>",
        "<div class='legend'>",
        "<span style='background: rgba(231, 76, 60, 0.8)'>Aligned [Red]</span>",
        "<span style='background: rgba(52, 152, 219, 0.8)'>Orthogonal [Blue]</span>",
        "<span style='background: rgba(149, 165, 166, 0.8)'>Neutral [Grey]</span>",
        "<br><br><em>Hover over any word to see its exact math. Darker background = Stronger Absolute Similarity.</em>",
        "</div>"
    ]
    
    html_content.append(build_heatmap_section(
        tokens, 
        metric_key="cosine_sim", 
        title="1. Cosine Similarity to Intro Anchor", 
        description="Measures conceptual alignment to the raw Introspection Anchor (v_intro)."
    ))
    
    html_content.append(build_heatmap_section(
        tokens, 
        metric_key="sim_eig", 
        title="2. Cosine Similarity to Top Eigenvector", 
        description="Measures mechanical alignment to the isolated Introspective Trench (λ1)."
    ))
    
    html_content.append(build_heatmap_section(
        tokens, 
        metric_key="combined_sim", 
        title="3. Semantic-Structural Convergence (Geometric Mean)", 
        description="The mathematical intersection of conceptual meaning and mechanical execution. Highlights words acting as true bridges."
    ))
    
    html_content.append("</div></body></html>")
    
    output_path = os.path.abspath("tri_metric_heatmap.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(html_content))
        
    print(f"\nSuccess! Tri-metric heatmap generated at:\n{output_path}")
    
    try:
        webbrowser.open('file://' + output_path)
    except:
        pass

if __name__ == "__main__":
    print("Initializing Heatmap Visualizer...")
    ordered_tokens = load_ordered_tokens()
    print(f"Reconstructed sequence of {len(ordered_tokens)} tokens.")
    generate_html_heatmap(ordered_tokens)