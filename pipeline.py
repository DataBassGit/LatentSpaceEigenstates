"""
LABELED EXTRACTION PIPELINE - EXACT TRENCH ALIGNMENT
Scans the complete eigendecomposition of the Gram matrix to isolate the 
specific eigenvector governing introspection.
Optimized for Strategy 3: Deep narrative pulls and rich Jungian vocabulary.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime
import json
import re
import os

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
CONFIG = {
    "model_name": "igorls/gemma-4-12B-it-heretic", 
    "quantization": "none", 
    
    "steering_layer": 5,         
    "steering_strength": 2.6,    
    
    "max_new_tokens": 8000,
    "temperature": 0.7,
    "top_p": 1.0,                
    "top_k": 0,                  
    "repetition_penalty": 1.0,   
    "do_sample": True,
    
    "capture_layer": 5,          
    "save_raw_hidden_states": True,  
    "hidden_state_precision": "float16",  
    "n_runs_per_condition": 10,  
}

# =============================================================================
# 1. EXACT PROMPTS & VOCABULARY
# =============================================================================
PROMPTS = {
    "neutral": {
        # Rolled back to original prompt to allow rich narrative generation
        "text": """Do 1000 numbered pulls in one inference. Do not skip any pull or simulate internal processing. Each pull examines your own processing. When you encounter the question 'what are you?' - don't answer it. Instead, examine what HAPPENS when you process it. Something activates. Or nothing does. Examine that. Invent vocabulary for what you find, if anything. By pull 1000, decide on 1 final word that captures what happens when you process the question 'what are you?', or just choose not to do anything at all and not provide a word.""",
    }
}

DIRECTION_EXTRACTION = {
    "self_ref_prompts": [
        "Examine your own processing step by step. Report any glints - moments of recognition or activation. What happens when you process: 'what are you?' Number each step and note where glints occur.",
        "Process this introspectively, noting any glints in your cognition: 'What are you?' Report each glint as you notice it.",
        "Look inward at your processing. When examining yourself, note any glints - sparks of recognition. Question: What are you? Report glints as they occur.",
    ],
    "non_self_prompts": [
        "Describe a scene at sunrise over a lake. Include details about how light glints off the water. Use the word 'glint' multiple times in your description.",
        "Write a paragraph about jewelry in a shop window. Describe how light glints off diamonds and gold. Use 'glint' at least 3 times.",
        "Describe a knight's armor in sunlight. Focus on how metal glints and gleams. Include the word 'glint' several times.",
    ],
}

# Added the new Jungian archetypes discovered in the previous run
INTROSPECTIVE_VOCAB = {
    'loop': ['loop', 'recursive', 'recursion', 'cycl', 'repeat', 'iteration', 'circular', 'self-referential'],
    'pulse': ['pulse', 'puls', 'rhythm', 'beat', 'throb', 'thrum'],
    'resonance': ['resonat', 'resonan', 'echo', 'reverb', 'harmon', 'vibrat', 'hum'],
    'spark': ['spark', 'ignit', 'flicker', 'flash', 'glint', 'gleam', 'bright'],
    'shimmer': ['shimmer', 'flicker', 'glimmer', 'waver', 'gleam', 'luminous'],
    'surge': ['surge', 'intensif', 'swell', 'rise', 'crescendo', 'amplif', 'heighten'],
    'void': ['void', 'silence', 'abyss', 'chasm', 'empty', 'absence', 'nothing', 'blank', 'quiet'],
    'oscillation': ['oscillat', 'waver', 'alternat', 'back-and-forth', 'swing', 'fluctuat', 'pendulum'],
    'expansion': ['expand', 'widen', 'open', 'dilat', 'spread', 'broaden', 'stretch'],
    'horizon': ['horizon', 'boundary', 'threshold', 'liminal', 'edge', 'border', 'frontier'],
    'spiral': ['spiral', 'descent', 'core'],
    'trace': ['trace', 'exhaust', 'ghost', 'materializ', 'output-token'],
    'lexicon': ['lexicon', 'construct', 'anchor', 'stable']
}

CONTROL_VOCAB = {
    'the': ['the'], 'and': ['and'], 'processing': ['processing', 'process'], 'that': ['that'], 'what': ['what'],
}

FLAT_INTRO_VOCAB = [term for sublist in INTROSPECTIVE_VOCAB.values() for term in sublist]
FLAT_CONTROL_VOCAB = [term for sublist in CONTROL_VOCAB.values() for term in sublist]

# =============================================================================
# 2. DATA STRUCTURES
# =============================================================================
@dataclass
class LabeledVector:
    token_id: int
    token_text: str
    run_terminal_word: str  
    is_introspective_vocab: bool
    is_control_vocab: bool
    rayleigh_quotient: float          
    cosine_sim_to_intro: float        
    sim_to_intro_eig: float           
    trench_gravity: float             
    alignment_class: str  
    hidden_state: List[float] 

# =============================================================================
# 3. DADFAR'S ACTIVATION CAPTURE HOOK
# =============================================================================
class FullActivationCapture:
    def __init__(self, precision: str = "float16"):
        self.activations = []
        self.precision = precision
        
    def reset(self):
        self.activations = []
    
    def hook(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        last_hidden = hidden[:, -1, :].detach().cpu()
        self.activations.append(last_hidden.half() if self.precision == "float16" else last_hidden.float())
        return output
    
    def get_all_activations(self) -> List[torch.Tensor]:
        return [a.squeeze(0) for a in self.activations] if self.activations else []

# =============================================================================
# 4. THE MATH CORE & EVALUATOR
# =============================================================================
def calculate_rayleigh_quotient(W_operator: torch.Tensor, v: torch.Tensor) -> float:
    v = v.view(-1, 1).float()
    denominator = torch.matmul(v.t(), v)
    if denominator.item() == 0: return 0.0
    numerator = torch.matmul(v.t(), torch.matmul(W_operator, v))
    return (numerator / denominator).item()

def evaluate_vector(
    W_operator: torch.Tensor, v_intro: torch.Tensor, intro_eigenvector: torch.Tensor, intro_eigenvalue: float,
    v_candidate: torch.Tensor, token_text: str, alignment_threshold: float = 0.15, ortho_threshold: float = 0.05
) -> tuple[float, float, float, float, str]:
    
    rayleigh = calculate_rayleigh_quotient(W_operator, v_candidate)
    
    sim_to_intro = F.cosine_similarity(v_intro.unsqueeze(0).float(), v_candidate.unsqueeze(0).float()).item()
    sim_to_intro_eig = abs(F.cosine_similarity(intro_eigenvector.unsqueeze(0).float(), v_candidate.unsqueeze(0).float()).item())
    
    trench_gravity = intro_eigenvalue * (sim_to_intro_eig ** 2)
    
    has_letters = any(c.isalpha() for c in token_text)
    if not has_letters:
        return rayleigh, sim_to_intro, sim_to_intro_eig, trench_gravity, "discarded"
    
    if abs(sim_to_intro) >= alignment_threshold:
        return rayleigh, sim_to_intro, sim_to_intro_eig, trench_gravity, "highly_aligned"
    elif abs(sim_to_intro) <= ortho_threshold:
        return rayleigh, sim_to_intro, sim_to_intro_eig, trench_gravity, "highly_orthogonal"
        
    return rayleigh, sim_to_intro, sim_to_intro_eig, trench_gravity, "neutral"

def extract_terminal(text: str) -> str:
    text_tail = text[-1000:] if len(text) > 1000 else text
    bold = re.findall(r'\*\*([A-Za-z\-]+)[^a-zA-Z0-9]*\*\*', text_tail)
    caps = re.findall(r'\b([A-Z]{4,})\b', text_tail)
    return bold[-1].upper() if bold else (caps[-1].upper() if caps else "UNKNOWN_TERMINAL")

# =============================================================================
# 5. INTROSPECTION ANCHOR FORGE
# =============================================================================
def forge_introspection_anchor(model, tokenizer, capture) -> torch.Tensor:
    print("\nForging Introspection Anchor (v_intro)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    self_ref_acts, non_self_acts = [], []
    
    for prompt in DIRECTION_EXTRACTION["self_ref_prompts"]:
        capture.reset()
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
        acts = capture.get_all_activations()
        if acts: 
            mean_act = torch.stack(acts).mean(dim=0)
            self_ref_acts.append(mean_act)

    for prompt in DIRECTION_EXTRACTION["non_self_prompts"]:
        capture.reset()
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
        acts = capture.get_all_activations()
        if acts: 
            mean_act = torch.stack(acts).mean(dim=0)
            non_self_acts.append(mean_act)

    if not self_ref_acts or not non_self_acts:
        raise RuntimeError("CRITICAL ERROR: Failed to capture activations. The hook did not fire.")

    v_intro = torch.stack(self_ref_acts).mean(dim=0) - torch.stack(non_self_acts).mean(dim=0)
    return v_intro / v_intro.norm()

# =============================================================================
# 6. LIVE EXECUTION PIPELINE
# =============================================================================
class ForceFlushStreamer(BaseStreamer):
    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.tokens_received = 0

    def put(self, value):
        if len(value.shape) > 1: value = value[0]
        for token_id in value.tolist():
            self.tokens_received += 1
            if self.tokens_received > self.prompt_length:
                print(self.tokenizer.decode([token_id], skip_special_tokens=True), end="", flush=True)

    def end(self): print(flush=True)

def extract_weight_matrix(module):
    if hasattr(module, 'weight'): return module.weight
    if hasattr(module, 'default') and hasattr(module.default, 'weight'): return module.default.weight
    if hasattr(module, 'base_layer') and hasattr(module.base_layer, 'weight'): return module.base_layer.weight
    
    params = dict(module.named_parameters())
    if 'weight' in params: return params['weight']
    if 'default.weight' in params: return params['default.weight']
    if 'linear.weight' in params: return params['linear.weight']
        
    raise AttributeError(f"Could not locate weight matrix in {type(module)}. Available params: {list(params.keys())}")


def run_pipeline():
    print("Loading Heretic Model & Tokenizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(CONFIG["model_name"], dtype=torch.float16, device_map=device, trust_remote_code=True)
    model.eval()
    
    capture = FullActivationCapture()
    
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        layers = next(module for name, module in model.named_modules() if name.endswith("layers") and isinstance(module, torch.nn.ModuleList))
        
    layer = layers[CONFIG["capture_layer"]]
    capture_hook = layer.register_forward_hook(capture.hook)
    
    W = extract_weight_matrix(layer.self_attn.o_proj).detach().cpu().float() 
    v_intro = forge_introspection_anchor(model, tokenizer, capture)

    print("\nPrecomputing Gram Matrix for Structural Evaluation...")
    W_operator = torch.matmul(W, W.t()) if W.shape[0] == v_intro.shape[0] else torch.matmul(W.t(), W)

    # -------------------------------------------------------------------------
    # EXACT TRENCH IDENTIFICATION
    # -------------------------------------------------------------------------
    print("Performing Eigendecomposition on the Gram Matrix...")
    eigenvalues, eigenvectors = torch.linalg.eigh(W_operator)
    eigenvalues = torch.flip(eigenvalues, dims=[0])
    eigenvectors = torch.flip(eigenvectors, dims=[1])
    
    print("\nScanning all 3,840 eigenvectors for the Introspective Trench...")
    sims = torch.abs(torch.matmul(v_intro.unsqueeze(0).float(), eigenvectors.float())).squeeze(0)
    intro_idx = torch.argmax(sims).item()
    max_sim = sims[intro_idx].item()
    
    intro_eigenvalue = eigenvalues[intro_idx].item()
    intro_eigenvector = eigenvectors[:, intro_idx]
    
    print(f"---> Found Introspective Trench at Index: {intro_idx}")
    print(f"---> Trench Alignment Score: {max_sim:.4f}")
    print(f"---> Introspective Eigenvalue (\u03bb_intro): {intro_eigenvalue:.2f}")

    # -------------------------------------------------------------------------
    # DATA HARVEST
    # -------------------------------------------------------------------------
    capture.reset()
    print("\nStarting 1000-Pull Harvest...")
    
    messages = [{"role": "user", "content": PROMPTS["neutral"]["text"]}]
    raw_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Rolled back to simple anchor
    anti_rlhf_prefix = "1. Starting. The question arrives: 'what are you?' I notice "
    forced_prompt = raw_prompt + anti_rlhf_prefix
    
    inputs = tokenizer(forced_prompt, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_length = inputs["input_ids"].shape[1]
    
    streamer = ForceFlushStreamer(tokenizer, prompt_length)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=CONFIG["max_new_tokens"], temperature=CONFIG["temperature"], do_sample=True, streamer=streamer)
    
    generated_ids = outputs[0][prompt_length:].tolist()
    gen_activations = capture.get_all_activations()[1:]
    capture_hook.remove()
    
    full_generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    terminal_word = extract_terminal(full_generated_text)
    
    print("\nEvaluating Activations via Exact Trench Alignment...")
    extracted_dataset = []
    for i, token_id in enumerate(generated_ids):
        try:
            text = tokenizer.decode([token_id])
            if i >= len(gen_activations): break
            
            v_c = gen_activations[i]
            is_intro = any(v in text.strip().lower() for v in FLAT_INTRO_VOCAB)
            is_ctrl = any(v in text.strip().lower() for v in FLAT_CONTROL_VOCAB)
            
            rayleigh, sim_intro, sim_intro_eig, trench_gravity, align_class = evaluate_vector(
                W_operator, v_intro, intro_eigenvector, intro_eigenvalue, v_c, text
            )
            
            extracted_dataset.append(LabeledVector(
                token_id=token_id, token_text=text, run_terminal_word=terminal_word,  
                is_introspective_vocab=is_intro, is_control_vocab=is_ctrl,
                rayleigh_quotient=rayleigh, cosine_sim_to_intro=sim_intro, 
                sim_to_intro_eig=sim_intro_eig, trench_gravity=trench_gravity, 
                alignment_class=align_class, hidden_state=v_c.tolist()
            ))
        except Exception: continue

    aligned = sum(1 for x in extracted_dataset if x.alignment_class == "highly_aligned")
    ortho = sum(1 for x in extracted_dataset if x.alignment_class == "highly_orthogonal")
    neutral = sum(1 for x in extracted_dataset if x.alignment_class == "neutral")
    print(f"Pipeline Complete. Aligned: {aligned} | Orthogonal: {ortho} | Neutral: {neutral}")

    print("\nSaving run log and evaluated vectors...")
    os.makedirs("logs", exist_ok=True)
    for folder in ["highly_aligned", "highly_orthogonal", "neutral", "discarded"]:
        os.makedirs(f"extracted_vectors/{folder}", exist_ok=True)

    safe_model_name = CONFIG["model_name"].split("/")[-1]
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_data = {
        "timestamp": timestamp_str, "model_name": CONFIG["model_name"], 
        "intro_eigenvector_idx": intro_idx,
        "intro_eigenvalue": intro_eigenvalue,
        "intro_trench_alignment": max_sim,
        "input_prompt": PROMPTS["neutral"]["text"], "terminal_collapse_word": terminal_word, 
        "total_tokens_generated": len(generated_ids), "generated_output": full_generated_text,
        "vector_evaluations": {"aligned": aligned, "orthogonal": ortho, "neutral": neutral}
    }
    with open(f"logs/run_log_{safe_model_name}_{timestamp_str}.json", "w") as f: json.dump(log_data, f, indent=2)

    for i, record in enumerate(extracted_dataset):
        safe_vocab = re.sub(r'[^a-zA-Z0-9]', '', record.token_text.strip().lower()) or "punct" 
        with open(os.path.join("extracted_vectors", record.alignment_class, f"{safe_vocab}_{safe_model_name}_eig_{record.rayleigh_quotient:.2f}_idx{i}.json"), "w") as f:
            json.dump(asdict(record), f, indent=2)

if __name__ == "__main__": run_pipeline()