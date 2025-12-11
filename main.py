import time
import csv
import os
from typing import Dict, List, Tuple

import requests
import matplotlib.pyplot as plt


# Configuration — adjust these to run different benchmarks
OLLAMA_GPU_URL = os.environ.get("OLLAMA_GPU_URL", "http://localhost:22545")
OLLAMA_CPU_URL = os.environ.get("OLLAMA_CPU_URL", "http://localhost:22546")

# List of models to test — change as needed
MODELS_TO_TEST: List[str] = [
    "llama3.2:3b",
    "llama3.1:8b",
    "phi3:3.8b",
]

# Prompt and generation options
PROMPT = (
    "Explique de façon concise les principes clés de la théorie de l'information. "
    "Réponse en 2-3 phrases."
)

# Number of timed runs per configuration
RUNS_PER_MODEL = 5

# Target prompt sizes in words to test (the prompt length varies, not the output)
WORD_COUNTS: List[int] = [i for i in range(10, 100, 2)]

# Ollama generation options; adjust as needed
GEN_OPTIONS: Dict = {
    "temperature": 0.2,
}

# Fixed output cap so that output is not the varying factor
OUTPUT_TOKENS = 128

# Output
RESULTS_CSV = "results.csv"
RESULTS_PNG = "results.png"


def get_next_run_dir() -> str:
    base_path = "Results"
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    existing_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    max_test_num = 0

    for d in existing_dirs:
        if d.startswith("Test"):
            try:
                num = int(d[4:])
                if num > max_test_num:
                    max_test_num = num
            except ValueError:
                pass

    new_dir_name = f"Test{max_test_num + 1}"
    new_dir_path = os.path.join(base_path, new_dir_name)
    os.makedirs(new_dir_path)
    return new_dir_path


def _post_json(url: str, payload: Dict) -> Dict:
    response = requests.post(url, json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def pull_model(base_url: str, model: str) -> None:
    # Ensure the model is available on the target Ollama instance
    _post_json(f"{base_url}/api/pull", {"name": model, "stream": False})


def warmup_model(base_url: str, model: str) -> None:
    """Précharge le modèle en mémoire avec une petite requête de warm-up."""
    payload = {
        "model": model,
        "prompt": "Hello",
        "stream": False,
        "options": {"num_predict": 5},  # Génération très courte
    }
    try:
        _post_json(f"{base_url}/api/generate", payload)
        print(f"  [Warm-up] Modèle {model} préchargé en mémoire")
    except Exception as e:
        print(f"  [Warm-up] Avertissement: {e}")


def generate_once(base_url: str, model: str, prompt: str, options: Dict) -> Tuple[float, str]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options or {},
    }
    start = time.perf_counter()
    result = _post_json(f"{base_url}/api/generate", payload)
    elapsed = time.perf_counter() - start
    return elapsed, result.get("response", "")


def build_prompt_with_word_count(target_words: int) -> str:
    # Create a prompt whose length in words is approximately equal to target_words
    # Start with a short instruction, then pad with filler tokens to reach the count
    instruction = (
        "Réponds à la consigne suivante de manière concise et claire."
    )
    # Split to count words and then add filler words to reach target
    words = instruction.split()
    if target_words <= len(words):
        return " ".join(words[:max(1, target_words)])
    filler_needed = target_words - len(words)
    filler = [f"mot{i}" for i in range(1, filler_needed + 1)]
    return instruction + " " + " ".join(filler)


def build_all_prompts(word_counts: List[int], runs: int) -> Dict[int, List[str]]:
    # Pre-generate prompts so all models are tested on identical inputs per run
    # For each word size, generate 'runs' different prompts
    prompts: Dict[int, List[str]] = {}
    for w in word_counts:
        prompts[w] = [build_prompt_with_word_count(w) for _ in range(runs)]
    return prompts


def benchmark_model(model: str, gpu_url: str, cpu_url: str, runs: int, word_counts: List[int], prompts_by_words: Dict[int, List[str]]) -> List[Dict]:
    # Pull the model on each backend
    pull_model(gpu_url, model)
    pull_model(cpu_url, model)

    # Warm-up: précharger le modèle en mémoire pour GPU et CPU
    print(f"  Préchargement du modèle {model} en mémoire...")
    warmup_model(gpu_url, model)
    warmup_model(cpu_url, model)

    rows: List[Dict] = []

    # For each target prompt size (in words), run sequentially GPU then CPU
    for words in word_counts:
        prompt_list = prompts_by_words[words]
        options = {**GEN_OPTIONS, "num_predict": OUTPUT_TOKENS}

        for i in range(1, runs + 1):
            prompt = prompt_list[i - 1]
            gpu_time, _ = generate_once(gpu_url, model, prompt, options)
            rows.append({
                "model": model,
                "backend": "GPU",
                "run": i,
                "words": words,
                "num_predict": OUTPUT_TOKENS,
                "seconds": gpu_time,
            })
            print(f"[{model}]:[GPU]:[{words}]")

            cpu_time, _ = generate_once(cpu_url, model, prompt, options)
            rows.append({
                "model": model,
                "backend": "CPU",
                "run": i,
                "words": words,
                "num_predict": OUTPUT_TOKENS,
                "seconds": cpu_time,
            })
            print(f"[{model}]:[CPU]:[{words}]")

    return rows


def save_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: List[Dict], output_dir: str) -> None:
    # Aggregate average seconds per (model, backend, words)
    agg: Dict[Tuple[str, str, int], List[float]] = {}
    for r in rows:
        key = (r["model"], r["backend"], int(r.get("words", 0)))
        agg.setdefault(key, []).append(float(r["seconds"]))

    models = sorted({m for (m, _, _) in agg.keys()})
    backends = ["CPU", "GPU"]

    for model in models:
        # Ensure consistent x ordering
        words_sorted = sorted({w for (m, _, w) in agg.keys() if m == model})
        series = {backend: [] for backend in backends}
        for backend in backends:
            for w in words_sorted:
                times = agg.get((model, backend, w), [])
                avg = sum(times) / len(times) if times else float("nan")
                series[backend].append(avg)

        plt.figure(figsize=(8, 5))
        plt.plot(words_sorted, series["CPU"], marker="o", label="CPU")
        plt.plot(words_sorted, series["GPU"], marker="o", label="GPU")
        plt.xlabel("Taille du prompt (nombre de mots)")
        plt.ylabel("Temps moyen (s)")
        plt.title(f"Temps de génération vs taille du prompt — {model}")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.4)
        plt.tight_layout()
        # Save per-model figure
        safe_model = model.replace(":", "_").replace("/", "_")
        plt.savefig(os.path.join(output_dir, f"results_{safe_model}.png"), dpi=150)
        plt.close()


def main() -> None:
    all_rows: List[Dict] = []
    
    # Determine output directory
    output_dir = get_next_run_dir()
    print(f"Results will be saved to: {output_dir}")

    # Build prompts once to guarantee identical inputs across models and per run
    prompts_by_words = build_all_prompts(WORD_COUNTS, RUNS_PER_MODEL)

    for model in MODELS_TO_TEST:
        print(f"\n=== Benchmark du modèle: {model} ===")
        rows = benchmark_model(model, OLLAMA_GPU_URL, OLLAMA_CPU_URL, RUNS_PER_MODEL, WORD_COUNTS, prompts_by_words)
        for r in rows:
            print(f"{r['model']} | {r['backend']} | {r['words']} mots | run {r['run']} -> {r['seconds']:.3f}s")
        # Plot immediately for this model only
        plot_results(rows, output_dir)
        print(f"Graphique enregistré pour {model}.")
        all_rows.extend(rows)

    save_csv(all_rows, os.path.join(output_dir, RESULTS_CSV))
    print(f"\nRésultats écrits dans: {os.path.join(output_dir, RESULTS_CSV)}")

    plot_results(all_rows, output_dir)
    print("Graphiques par modèle enregistrés: results_<model>.png")

if __name__ == "__main__":
    main()
