# LLM Inference Optimization

This repository contains code for optimizing Large Language Model (LLM) inference in cloud environments by determining the most cost-effective GPU instance selection and KV cache offloading strategy for batch processing tasks.

## Overview

The code implements a performance prediction model for LLM inference that considers:

- Model architecture characteristics (layers, hidden dimensions, etc.)
- Hardware specifications of different GPU instances
- KV cache offloading strategies
- Service Level Objectives (SLO) requirements
- Cost-efficiency metrics

## Key Features

- **Model Configuration**: Define various LLM architectures with different parameters
- **Hardware Configuration**: Specify GPU characteristics including memory, compute capabilities, and bandwidth
- **Memory Analysis**: Calculate memory requirements for model weights, activations, and KV cache
- **Offloading Strategy**: Determine optimal KV cache offloading ratio based on available GPU memory
- **Performance Prediction**: Estimate inference time for both prefill and decode phases
- **Cost Optimization**: Select the most cost-effective GPU instance that meets performance requirements

## Usage

```bash
python llm_inference_optimizer.py --model OPT-2.7B --batch-size 32 --input-len 128 --output-len 512 --safety-factor 0.87 --slo-tps 100
```

### Command-line Arguments

- `--model`: Model name (OPT-1.3B, OPT-2.7B, OPT-6.7B, OPT-13B, OPT-30B, OPT-66B)
- `--batch-size`: Batch size for inference
- `--input-len`: Length of input prompt in tokens
- `--output-len`: Length of generated output in tokens
- `--safety-factor`: Memory safety factor (0-1)
- `--adjust`: Apply experimental offloading adjustments
- `--precision`: Numerical precision in bytes (FP16=2, FP32=4)
- `--slo-tps`: Minimum required Tokens Per Second (SLO)

## How It Works

1. **Memory Calculation**: The code first calculates the total memory requirements for running the model, including model weights, activations, and KV cache.
2. **Offloading Determination**: It determines how much of the KV cache needs to be offloaded from GPU to CPU memory.
3. **Performance Prediction**: The code estimates processing time for both prefill and decode phases based on computation time and data transfer overhead.
4. **Instance Evaluation**: Each available GPU instance is evaluated against the requirements.
5. **Optimal Selection**: The most cost-effective instance that meets the performance criteria is selected.

## Example Output

```
Model: OPT-2.7B
Layers: 32
Hidden size: 2560
Batch size: 32
Input length: 128
Output length: 512
Safety factor: 0.87
SLO requirement: Minimum 100 TPS

Total KV Cache size: 15.72 GB

Optimal instance:
GPU: L4
KV Cache offloading ratio: 40.0%
TPS: 150.25 tokens/second
Hourly cost: $1.17
Cost efficiency: 128.66 TPS/$

Performance details:
Prefill time: 0.0123 seconds
Decode time: 3.2845 seconds
Total processing time: 3.2968 seconds
```

## Hardware Configurations

The repository includes configurations for several GPU instances:
- T4
- A10G
- L4
- L40S

You can extend this with additional hardware by modifying the `hardware_configs` list.

## Model Configurations

Predefined model configurations include various sizes of the OPT model family. You can add other LLM architectures by extending the `model_configs` dictionary.

## Dependencies

- NumPy
- Argparse

## Citation

If you use this code in your research, please cite our work:

```
@article{llm_inference_optimization,
  title={Cost-Effective LLM Inference in Cloud Environments with KV Cache Offloading},
  author={Your Name},
  journal={},
  year={2025}
}
```

## License

[MIT License](LICENSE)
