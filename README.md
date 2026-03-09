# LLM Performance Analysis: CPU vs GPU Benchmark

## 📊 Project Overview

This project provides a comprehensive, data-driven analysis comparing CPU and GPU performance for Large Language Model (LLM) inference using Ollama. The goal is to answer critical questions about hardware selection for LLM deployment and quantify the real-world performance benefits of GPU acceleration.

### 🎯 Research Objectives

1. **Performance Comparison**: Determine which platform (CPU or GPU) is faster for LLM inference
2. **Quantitative Analysis**: Measure the exact speedup factor provided by GPU acceleration
3. **Hardware Impact**: Demonstrate how GPU power correlates with performance gains
4. **Workload Dependency**: Analyze how model size and prompt length affect performance
5. **Statistical Rigor**: Provide evidence-based recommendations with statistical significance testing

### 🔬 Key Hypothesis

**More powerful GPUs yield disproportionately higher speedup factors compared to their CPU counterparts, demonstrating that GPU compute power is a critical multiplier for LLM inference performance.**

## 📁 Project Structure

```
Ollama_GPU_CPU_Test/
├── main.py                          # Data collection script
├── compose.yaml                     # Docker configuration for Ollama
├── Dockerfile                       # Container setup
├── GPU_vs_CPU Analysis/
│   └── Statistics_CPU_GPU.ipynb    # Main analysis notebook
├── Results/
│   ├── Laptop_Data/
│   │   └── results.csv             # Performance data from laptop (less powerful GPU)
│   ├── Workstation_Data/
│   │   └── results.csv             # Performance data from workstation (more powerful GPU)
│   └── Powerfull_workstation_data/
│       └── results.csv             # Performance data from powerful workstation (most powerful GPU)
└── Report/                          # Generated analysis outputs
    ├── Step_5_Correlation_Analysis/
    ├── Step_6_Performance_Visualizations/
    ├── Step_7_Speedup_Analysis/
    ├── Step_9_Predictive_Modeling/
    ├── Step_10_Multi_Machine_Comparison/
    └── *.csv                        # Exported statistics
```

## 📊 Understanding the Data

### Data Collection (`main.py`)

The benchmark script tests LLM inference across multiple configurations:

**Test Parameters:**
- **Models**: Multiple LLM variants (different sizes: 3b, 8b, 13b, 70b parameters)
- **Backends**: CPU vs GPU execution
- **Prompt Lengths**: Variable input sizes (measured in words)
- **Runs**: Multiple repetitions for statistical reliability

**Output Format (`results.csv`):**
```csv
model,backend,words,run,seconds
llama3.2:3b,cpu,10,1,2.543
llama3.2:3b,gpu,10,1,0.892
...
```

### Data Fields Explained

| Field | Description | Example Values |
|-------|-------------|----------------|
| `model` | LLM model identifier | `llama3.2:3b`, `mistral:7b`, `qwen2.5:14b` |
| `backend` | Execution platform | `cpu`, `gpu` |
| `words` | Prompt length in words | `10`, `50`, `100`, `200` |
| `run` | Repetition number | `1`, `2`, `3`, `4`, `5` |
| `seconds` | Execution time in seconds | `2.543`, `0.892` |

### Machine Configurations

#### Laptop (Machine A)
- **Purpose**: Baseline with consumer-grade GPU
- **Data Location**: `Results/Laptop_Data/results.csv`
- **Use Case**: Represents typical developer or light production environment

#### Workstation (Machine B)
- **Purpose**: High-performance configuration with powerful GPU
- **Data Location**: `Results/Workstation_Data/results.csv`
- **Use Case**: Represents production or research environment

#### Powerful Workstation (Machine C)
- **Purpose**: Top-tier configuration with the most powerful GPU
- **Data Location**: `Results/Powerfull_workstation_data/results.csv`
- **Use Case**: Represents high-end research or enterprise environment

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8+
python --version

# Required packages
pip install pandas matplotlib seaborn numpy scipy scikit-learn jupyter
```

### Running the Analysis

1. **Ensure data files exist:**
   ```bash
   ls Results/Laptop_Data/results.csv
   ls Results/Workstation_Data/results.csv
   ls Results/Powerfull_workstation_data/results.csv
   ```

2. **Open the analysis notebook:**
   ```bash
   jupyter notebook "GPU_vs_CPU Analysis/Statistics_CPU_GPU.ipynb"
   ```

3. **Run all cells** to generate the complete analysis

4. **Review results** in the `Report/` directory

### Starting Docker Containers

Two compose files are provided depending on your GPU vendor:

**NVIDIA GPU:**
```bash
docker compose -f compose.nvidia.yaml up -d
```

**AMD GPU (ROCm):**
```bash
docker compose -f compose.amd.yaml up -d
```

> ⚠️ For AMD, make sure the `rocm` drivers are installed on your host and that
> your GPU is [supported by ROCm](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).

### Collecting New Data

To run your own benchmarks:

```bash
# Configure your models and parameters in main.py
python main.py
```

The script will generate a `results.csv` file with performance measurements.

## 📈 Analysis Workflow

The notebook performs a comprehensive 12-step analysis:

### Step 1-4: Data Foundation
- **Data Loading & Validation**: Import and verify data integrity
- **Preprocessing**: Extract model sizes, normalize formats
- **Descriptive Statistics**: Calculate mean, median, std dev by configuration
- **Aggregation**: Group by backend, model, and prompt length

### Step 5-6: Exploratory Analysis
- **Correlation Analysis**: Identify relationships between variables
- **Performance Visualizations**: 
  - Execution time vs prompt length
  - Distribution plots (violin & box plots)
  - Performance heatmaps (CPU vs GPU)

### Step 7-8: Performance Metrics
- **Speedup Analysis**: Calculate GPU acceleration factor (CPU_time / GPU_time)
- **Statistical Testing**: T-tests to prove significance of differences

### Step 9: Predictive Modeling
- **Linear Regression**: Predict GPU performance from CPU measurements
- **R² Analysis**: Evaluate model fit quality

### Step 10: Multi-Machine Comparison ⭐
- **Cross-Hardware Analysis**: Compare laptop vs workstation
- **Hypothesis Testing**: Validate GPU power correlation
- **Comparative Visualizations**: Side-by-side speedup comparison

### Step 11-12: Conclusions
- **Final Verdict**: Data-driven recommendations
- **Results Export**: CSV files with all statistics

## 📊 Key Metrics Explained

### Speedup Factor
```
Speedup = CPU_Time / GPU_Time
```
- **Speedup = 1.0**: No performance difference
- **Speedup > 1.0**: GPU is faster (e.g., 5.0x = GPU is 5 times faster)
- **Speedup < 1.0**: CPU is faster (rare for LLM inference)

### Statistical Significance
- **p-value < 0.001**: Highly significant difference (99.9% confidence)
- **p-value < 0.05**: Significant difference (95% confidence)
- **p-value ≥ 0.05**: No statistically significant difference

### Time Saved Percentage
```
Time_Saved = ((CPU_Time - GPU_Time) / CPU_Time) × 100%
```
Represents the percentage reduction in inference time when using GPU.

## 🎯 Expected Findings

Based on the analysis methodology, you should observe:

1. **✅ GPU Superiority**: GPUs significantly outperform CPUs for LLM inference
2. **📈 Scaling Effects**: Larger models benefit more from GPU acceleration
3. **📊 Prompt Length Impact**: Performance difference may vary with input size
4. **💪 Hardware Correlation**: More powerful GPUs show disproportionately higher speedup
5. **🔬 Statistical Validity**: All findings backed by rigorous statistical tests

## 📂 Output Files

### Visualizations (`Report/Step_*/`)
- **Correlation heatmaps**: Variable relationships
- **Performance plots**: Time vs prompt length by model
- **Distribution plots**: Variance and outliers
- **Speedup analysis**: GPU acceleration visualization
- **Regression plots**: Predictive modeling
- **Comparison charts**: Multi-machine analysis

### Data Exports (`Report/*.csv`)
- `stats_summary_Laptop.csv`: Aggregated laptop statistics
- `stats_summary_Workstation.csv`: Aggregated workstation statistics
- `speedup_data_Laptop.csv`: Detailed speedup calculations (laptop)
- `speedup_data_Workstation.csv`: Detailed speedup calculations (workstation)
- `machine_comparison.csv`: Cross-hardware comparison results

## 🛠️ Customization

### Testing Different Models

Edit `main.py` to test your preferred models:
```python
models = ["llama3.2:3b", "mistral:7b", "your-model:size"]
```

### Adjusting Test Parameters

Modify test configurations:
```python
prompt_lengths = [10, 50, 100, 200, 500]  # Words per prompt
num_runs = 5  # Repetitions for reliability
```

### Analysis Modifications

The notebook is fully commented and structured for easy customization:
- Adjust visualization styles in plotting cells
- Modify statistical tests in Step 8
- Customize report structure in configuration cell

## 🔍 Interpreting Results

### Reading the Speedup Chart
```
Model Size     Laptop GPU    Workstation GPU
3b             2.5x           4.2x
8b             3.1x           6.8x
13b            3.8x           9.1x
```
- **Higher = Better**: Larger numbers indicate greater GPU advantage
- **Trend Analysis**: Increasing speedup with model size suggests better GPU utilization
- **Hardware Gap**: Difference between machines validates GPU power hypothesis

### Understanding Heatmaps
- **Color Intensity**: Darker = slower (higher execution time)
- **Patterns**: Look for diagonal gradients (performance scaling)
- **Comparisons**: CPU vs GPU side-by-side shows relative efficiency

## 🤝 Contributing

To contribute your own benchmark data:

1. Run `main.py` on your hardware
2. Save results to `Results/YourMachine_Data/results.csv`
3. Update `Statistics_CPU_GPU.ipynb` configuration cell
4. Submit a pull request with your findings

## 📄 License

This project is open-source and available for educational and research purposes.

## 🙏 Acknowledgments

- **Ollama**: For providing the LLM inference platform
- **Python Data Science Stack**: pandas, matplotlib, seaborn, scikit-learn
- **Community**: For contributing benchmark data and insights

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub.

---

**💡 Pro Tip**: Always run benchmarks multiple times and use the analysis tools provided to ensure statistical validity of your conclusions. Raw numbers can be misleading without proper statistical analysis!