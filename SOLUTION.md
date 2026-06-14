# NVIDIA Nemotron Model Reasoning Challenge - Solution Strategy

Our approach focuses on enhancing the reasoning capabilities of the **Nemotron-3-Nano-30B-A3B** model through high-quality synthetic data and LoRA fine-tuning.

## 1. Data Categorization & Rule Discovery
We analyzed the competition's training data and identified 6 core puzzle types:
- **Bit Manipulation:** Logic gates, shifts, and rotations on 8-bit strings.
- **Gravitational Constant:** Solving for distance using $d = 0.5 \cdot g \cdot t^2$ where $g$ is Wonderland-specific.
- **Unit Conversion:** Linear scaling between mysterious units.
- **Numeral Systems:** Mapping integers to Wonderland numerals (e.g., Roman).
- **Text Encryption:** Simple substitution ciphers.
- **Symbol Equations:** Abstract symbol transformations.

## 2. Synthetic Data Generation
To augment the provided 9,500 samples, we developed a generation pipeline that produced **7,500 additional samples** covering 5/6 categories. This helps the model generalize the underlying mathematical and logical operators.

## 3. Fine-tuning Strategy
We utilized **LoRA (Rank 32, Alpha 64)** for efficient adaptation.
- **Base Model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16`
- **Training Template:** We introduced a reasoning prompt using `<|im_start|>assistant\n<think>\n` to encourage the model to perform "Chain of Thought" analysis before outputting the final answer in a `\boxed{}` command.
- **Hyperparameters:**
  - Learning Rate: 1e-4
  - Batch Size: 1 (effective 8 with gradient accumulation)
  - Epochs: 2
  - Optimizer: AdamW with Cosine Schedule

## 4. Evaluation
Local validation on algorithmic categories (Gravity, Unit Conversion, Numerals) yielded 100% accuracy, while text encryption showed marked improvement with the reasoning-aware adapter.

## 5. Submission
The final submission is a LoRA adapter compatible with the vLLM inference engine, packaged with the required `adapter_config.json`.
