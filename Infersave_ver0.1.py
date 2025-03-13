import argparse
import math
import numpy as np

class ModelConfig:
    """LLM model configuration"""
    def __init__(self, model_name, num_layers, hidden_size, ffn_dim, num_heads):
        self.model_name = model_name
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.params = self._calculate_params()
    
    def _calculate_params(self):
        """Calculate the number of model parameters"""
        # Simple calculation example (actual models might be more complex)
        embedding_params = self.hidden_size * self.hidden_size  # Embedding layer
        attention_params = self.num_layers * (
            3 * self.hidden_size * self.hidden_size +  # QKV projections
            self.hidden_size * self.hidden_size        # Output projection
        )
        ffn_params = self.num_layers * (
            self.hidden_size * self.ffn_dim +  # First layer of FFN
            self.ffn_dim * self.hidden_size    # Second layer of FFN
        )
        return embedding_params + attention_params + ffn_params


class HardwareConfig:
    """GPU and system hardware configuration"""
    def __init__(self, 
                 gpu_name,
                 gpu_memory_gb, 
                 gpu_flops, 
                 gpu_to_cpu_bandwidth, 
                 cpu_to_gpu_bandwidth,
                 ctcf_prefill_alpha=1.0,
                 ctcf_prefill_beta=0.0,
                 ctcf_decode_alpha=1.0,
                 ctcf_decode_beta=0.0,
                 hourly_cost=0.0):
        self.gpu_name = gpu_name
        self.gpu_memory = gpu_memory_gb * 1e9  # Convert GB to bytes
        self.gpu_flops = gpu_flops * 1e12      # Convert TFLOPS to FLOPS
        self.gpu_to_cpu_bandwidth = gpu_to_cpu_bandwidth * 1e9  # Convert GB/s to B/s
        self.cpu_to_gpu_bandwidth = cpu_to_gpu_bandwidth * 1e9  # Convert GB/s to B/s
        
        # Compute Time Calibration Function parameters
        self.ctcf_prefill_alpha = ctcf_prefill_alpha
        self.ctcf_prefill_beta = ctcf_prefill_beta
        self.ctcf_decode_alpha = ctcf_decode_alpha
        self.ctcf_decode_beta = ctcf_decode_beta
        
        self.hourly_cost = hourly_cost  # Cost per hour (USD)


def calculate_memory_requirements(model_config, batch_size, input_len, output_len, precision_bytes=2):
    """Calculate memory requirements"""
    # 1. Model weight memory
    model_weight_memory = model_config.params * precision_bytes
    
    # 2. Activation memory (roughly estimated as 15% of model size)
    activation_memory = model_weight_memory * 0.15
    
    # 3. KV Cache memory
    seq_len_total = input_len + output_len
    kv_cache_per_token = 2 * model_config.hidden_size * precision_bytes  # For both K and V
    kv_cache_memory = batch_size * seq_len_total * model_config.num_layers * kv_cache_per_token
    
    # KV Cache size per layer
    kv_cache_per_layer = kv_cache_memory / model_config.num_layers
    
    # 4. Temporary computation buffers (memory needed for Self-Attention calculations)
    # Major memory consumption is in the Q*K^T calculation
    attn_matrix_size = batch_size * model_config.num_heads * input_len * seq_len_total * 4  # Using float32
    mlp_activation_size = batch_size * model_config.ffn_dim * 4  # Intermediate activation values
    temp_buffer_memory = max(attn_matrix_size, mlp_activation_size) * 1.2  # 20% additional safety margin
    
    return {
        "model_weight": model_weight_memory,
        "activation": activation_memory,
        "kv_cache": kv_cache_memory,
        "kv_cache_per_layer": kv_cache_per_layer,
        "temp_buffer": temp_buffer_memory,
        "total": model_weight_memory + activation_memory + kv_cache_memory + temp_buffer_memory
    }


def determine_offloading_ratio(gpu_memory, memory_requirements, safety_factor=0.8):
    """Determine KV Cache Offloading ratio"""
    # Available GPU memory with safety factor applied
    available_gpu_memory = gpu_memory * safety_factor
    
    # Base memory requirements (weights + activations + working buffers)
    base_memory = (
        memory_requirements["model_weight"] + 
        memory_requirements["activation"] + 
        memory_requirements["temp_buffer"]
    )
    
    # Memory available for KV Cache
    memory_for_kv_cache = available_gpu_memory - base_memory
    
    # Total KV Cache size
    total_kv_cache = memory_requirements["kv_cache"]
    
    # Check if GPU has enough memory for KV Cache
    if memory_for_kv_cache >= total_kv_cache:
        # No offloading needed
        gpu_ratio = 1.0
        cpu_ratio = 0.0
        print(f"GPU memory is sufficient. Keeping 100% of KV Cache in GPU.")
    elif memory_for_kv_cache <= 0:
        # Offloading impossible (GPU memory insufficient even for basic requirements)
        raise ValueError("GPU memory is insufficient. Model execution is not possible.")
    else:
        # Partial offloading required
        gpu_ratio = memory_for_kv_cache / total_kv_cache
        cpu_ratio = 1.0 - gpu_ratio
        
        # Check KV Cache size per layer (at least one layer's KV Cache should be in GPU)
        kv_cache_per_layer = memory_requirements["kv_cache_per_layer"]
        if memory_for_kv_cache < kv_cache_per_layer:
            # Not enough GPU memory even for a single layer's KV Cache
            raise ValueError("GPU memory is insufficient even for a single layer's KV Cache.")
    
    # Memory verification
    expected_gpu_memory = base_memory + total_kv_cache * gpu_ratio
    if expected_gpu_memory > available_gpu_memory:
        # Unexpected error - memory calculation needs to be rechecked
        raise ValueError(f"Memory calculation error: Expected GPU usage ({expected_gpu_memory / 1e9:.2f}GB) exceeds available memory ({available_gpu_memory / 1e9:.2f}GB).")
    
    return {
        "gpu_ratio": gpu_ratio,
        "cpu_ratio": cpu_ratio,
        "gpu_memory_usage": expected_gpu_memory,
        "available_gpu_memory": available_gpu_memory,
        "base_memory": base_memory,
        "memory_for_kv_cache": memory_for_kv_cache,
        "total_kv_cache": total_kv_cache
    }


def apply_ctcf(compute_time, alpha, beta):
    """Apply Compute Time Calibration Function"""
    return alpha * compute_time + beta


def predict_prefill_time(model_config, hardware_config, batch_size, input_len, offloading_ratio, precision_bytes=2):
    """Predict processing time for the Prefill phase"""
    # 1. Calculate computation time (Linear Layer + Self-Attention)
    h1 = model_config.hidden_size
    h2 = model_config.ffn_dim
    
    # Linear Layer computation time
    linear_flops = batch_size * (8 * input_len * h1**2 + 4 * input_len * h1 * h2)
    linear_time = linear_flops / hardware_config.gpu_flops
    
    # Self-Attention computation time
    attn_flops = 4 * batch_size * input_len**2 * h1
    attn_time = attn_flops / hardware_config.gpu_flops
    
    # Total computation time
    compute_time = linear_time + attn_time
    
    # Apply CTCF
    calibrated_compute_time = apply_ctcf(
        compute_time, 
        hardware_config.ctcf_prefill_alpha, 
        hardware_config.ctcf_prefill_beta
    )
    
    # 2. Calculate KV Cache transfer time (GPU → CPU)
    # KV Cache size to be offloaded
    kv_cache_size = offloading_ratio * 2 * (input_len + 1) * h1 * precision_bytes * batch_size
    
    # Transfer time
    transfer_time = kv_cache_size / hardware_config.gpu_to_cpu_bandwidth if offloading_ratio > 0 else 0
    
    # 3. Prefill time is the longer of computation time and transfer time
    prefill_time = max(calibrated_compute_time, transfer_time)
    
    return {
        "compute_time": compute_time,
        "calibrated_compute_time": calibrated_compute_time,
        "transfer_time": transfer_time,
        "prefill_time": prefill_time
    }


def predict_decode_time(model_config, hardware_config, batch_size, input_len, output_len, offloading_ratio, precision_bytes=2):
    """Predict processing time for the Decode phase"""
    # 1. Calculate computation time (Linear Layer + Self-Attention)
    h1 = model_config.hidden_size
    h2 = model_config.ffn_dim
    
    # Linear Layer computation time
    linear_flops = batch_size * (8 * h1**2 + 4 * h1 * h2)
    linear_time = linear_flops / hardware_config.gpu_flops
    
    # Self-Attention computation time (using the average of input + generated tokens)
    seq_len_avg = input_len + output_len / 2
    attn_flops = 4 * batch_size * seq_len_avg * h1
    attn_time = attn_flops / hardware_config.gpu_flops
    
    # Total computation time
    compute_time = linear_time + attn_time
    
    # Apply CTCF
    calibrated_compute_time = apply_ctcf(
        compute_time, 
        hardware_config.ctcf_decode_alpha, 
        hardware_config.ctcf_decode_beta
    )
    
    # 2. Calculate KV Cache transfer time (CPU → GPU)
    # KV Cache size that needs to be transferred (already generated input tokens + newly generated token)
    required_kv_size = (offloading_ratio * 2 * (input_len + 1) + 2) * h1 * precision_bytes * batch_size
    
    # Transfer time
    transfer_time = required_kv_size / hardware_config.cpu_to_gpu_bandwidth if offloading_ratio > 0 else 0
    
    # 3. Decode time is computation time + transfer time
    # (In Decode phase, KV Cache access is needed before computation, so parallelization is difficult)
    decode_time = calibrated_compute_time + transfer_time
    
    return {
        "compute_time": compute_time,
        "calibrated_compute_time": calibrated_compute_time,
        "transfer_time": transfer_time,
        "decode_time": decode_time
    }


def predict_total_time(model_config, hardware_config, batch_size, input_len, output_len, offloading_ratio, precision_bytes=2):
    """Predict total processing time and TPS"""
    # Predict Prefill phase time
    prefill_info = predict_prefill_time(
        model_config, hardware_config, batch_size, input_len, offloading_ratio, precision_bytes
    )
    prefill_time_per_layer = prefill_info["prefill_time"]
    
    # Predict Decode phase time
    decode_info = predict_decode_time(
        model_config, hardware_config, batch_size, input_len, output_len, offloading_ratio, precision_bytes
    )
    decode_time_per_layer = decode_info["decode_time"]
    
    # Calculate total processing time
    num_layers = model_config.num_layers
    prefill_total_time = prefill_time_per_layer * num_layers
    decode_total_time = decode_time_per_layer * num_layers * (output_len - 1)
    total_time = prefill_total_time + decode_total_time
    
    # Calculate TPS (Tokens Per Second)
    total_tokens_generated = batch_size * (output_len+input_len)
    tps = total_tokens_generated / total_time
    
    return {
        "prefill_time_per_layer": prefill_time_per_layer,
        "decode_time_per_layer": decode_time_per_layer,
        "prefill_total_time": prefill_total_time,
        "decode_total_time": decode_total_time,
        "total_time": total_time,
        "total_tokens": total_tokens_generated,
        "tps": tps
    }

def evaluate_instance(model_config, hardware_config, batch_size, input_len, output_len, offloading_ratio, slo_tps=None, precision_bytes=2):
    """Evaluate GPU instance"""
    # Predict performance
    performance = predict_total_time(
        model_config, hardware_config, batch_size, input_len, output_len, offloading_ratio, precision_bytes
    )
    
    # Actual TPS
    actual_tps = performance["tps"]
    
    # Calculate effective TPS considering SLO
    effective_tps = min(actual_tps, slo_tps) if slo_tps is not None else actual_tps
    
    # Calculate total processing time (in hours)
    total_tokens = batch_size * (input_len + output_len)
    task_time_hours = total_tokens / (effective_tps * 3600)
    
    # Billed hours (rounded up)
    billed_hours = math.ceil(task_time_hours)
    
    # Hourly cost
    hourly_cost = hardware_config.hourly_cost
    
    # Calculate cost efficiency (applying the formula from the paper)
    cost_efficiency = (effective_tps * 3600) / (billed_hours * hourly_cost) if hourly_cost > 0 else float('inf')
    
    return {
        "hardware": hardware_config.gpu_name,
        "offloading_ratio": offloading_ratio,
        "performance": performance,
        "hourly_cost": hourly_cost,
        "actual_tps": actual_tps,
        "effective_tps": effective_tps,
        "task_time_hours": task_time_hours,
        "billed_hours": billed_hours,
        "cost_efficiency": cost_efficiency
    }

def select_best_instance(model_config, hardware_configs, batch_size, input_len, output_len, 
                         slo_tps=None, precision_bytes=2, safety_factor=0.8, adjust_offloading=False):
    """Select the optimal GPU instance"""
    candidates = []
    all_evaluations = []  # Store evaluation results for all instances
    
    # Calculate memory requirements (common for all instances)
    memory_requirements = calculate_memory_requirements(
        model_config, batch_size, input_len, output_len, precision_bytes
    )
    
    # Output KV Cache size
    kv_cache_size_gb = memory_requirements["kv_cache"] / 1e9
    print(f"\nTotal KV Cache size: {kv_cache_size_gb:.2f} GB")
    
    for hw_config in hardware_configs:
        try:
            # Determine offloading ratio
            offloading_info = determine_offloading_ratio(
                hw_config.gpu_memory, memory_requirements, safety_factor
            )
            
            offloading_ratio = offloading_info["cpu_ratio"]
            #print(offloading_ratio)
            offloading_ratio = min(1.0, offloading_ratio * 1.6)  # Adjust more conservatively by 60%   
            offloading_ratio = math.ceil(offloading_ratio*10)/10
            
            # Apply experimental adjustment (optional)
            if adjust_offloading and offloading_ratio > 0:
                offloading_ratio = min(1.0, offloading_ratio * 1.3)  # 30% more conservative adjustment
                print(offloading_ratio)
                offloading_ratio = math.ceil(offloading_ratio*10)/10
                
            
            # Evaluate instance
            evaluation = evaluate_instance(
                model_config, hw_config, batch_size, input_len, output_len, offloading_ratio, slo_tps, precision_bytes
            )
            
            # Store all evaluation results
            all_evaluations.append(evaluation)
            
            # Check SLO condition
            if slo_tps is None or evaluation["actual_tps"] >= slo_tps:
                candidates.append(evaluation)
            
        except ValueError as e:
            print(f"Error evaluating instance {hw_config.gpu_name}: {e}")
            continue
    
    # Output information for all instances
    print("\nEvaluation results for all instances:")
    all_evaluations.sort(key=lambda x: (-x["cost_efficiency"], -x["actual_tps"]))
    
    for i, eval_result in enumerate(all_evaluations, 1):
        print(f"{i}. {eval_result['hardware']} - "
              f"TPS: {eval_result['actual_tps']:.2f}, "
              f"Offloading: {eval_result['offloading_ratio']*100:.1f}%, "
              f"Cost: ${eval_result['hourly_cost']:.2f}/hour, "
              f"Efficiency: {eval_result['cost_efficiency']:.2f} TPS/$")
    
    # If no candidates
    if not candidates:
        print("\nNo instances meet the SLO requirements.")
        return None, all_evaluations  # Return all evaluation results
    
    # Sort by cost efficiency
    candidates.sort(key=lambda x: (-x["cost_efficiency"], -x["actual_tps"]))
    
    return candidates[0], candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM Inference Performance Prediction and Instance Selection')
    parser.add_argument('--model', type=str, default='OPT-2.7B', help='Model name')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--input-len', type=int, default=128, help='Input sequence length')
    parser.add_argument('--output-len', type=int, default=512, help='Output sequence length')
    parser.add_argument('--safety-factor', type=float, default=0.87, help='Safety factor (0-1)')
    parser.add_argument('--adjust', action='store_true', help='Apply experimental offloading adjustment')
    parser.add_argument('--precision', type=int, default=2, help='Precision (bytes): FP16=2, FP32=4')
    parser.add_argument('--slo-tps', type=float, help='Minimum TPS requirement (SLO)')
    args = parser.parse_args()
    
    # Model configurations (examples)
    model_configs = {
        "OPT-1.3B": ModelConfig("OPT-1.3B", 24, 2048, 8192, 32),
        "OPT-2.7B": ModelConfig("OPT-2.7B", 32, 2560, 10240, 32),
        "OPT-6.7B": ModelConfig("OPT-6.7B", 32, 4096, 16384, 32),
        "OPT-13B": ModelConfig("OPT-13B", 40, 5120, 20480, 40),
        "OPT-30B": ModelConfig("OPT-30B", 48, 7168, 28672, 56),
        "OPT-66B": ModelConfig("OPT-66B", 64, 9216, 36864, 72)
    }
    
    # GPU hardware configurations (examples)
    # Actual values should be calibrated through benchmarking
    hardware_configs = [
        HardwareConfig(
            gpu_name="T4",
            gpu_memory_gb=15,
            gpu_flops=8.24,           # 8.1 TFLOPS (FP16)
            gpu_to_cpu_bandwidth=5,   # 5 GB/s
            cpu_to_gpu_bandwidth=5,   # 5 GB/s
            ctcf_prefill_alpha=0.5,   # Example calibration coefficient
            ctcf_prefill_beta=0,
            ctcf_decode_alpha=1.9,
            ctcf_decode_beta=0,
            hourly_cost=0.7
        ),
        HardwareConfig(
            gpu_name="A10G",
            gpu_memory_gb=23,
            gpu_flops=31.52,          # 31.2 TFLOPS (FP16)
            gpu_to_cpu_bandwidth=12,  # 12 GB/s
            cpu_to_gpu_bandwidth=12,  # 12 GB/s
            ctcf_prefill_alpha=0.7,   # Example calibration coefficient
            ctcf_prefill_beta=0,
            ctcf_decode_alpha=0.97,
            ctcf_decode_beta=0,
            hourly_cost=1.466
        ),
        HardwareConfig(
            gpu_name="L4",
            gpu_memory_gb=23,
            gpu_flops=30.29,          # 30.29 TFLOPS (FP16)
            gpu_to_cpu_bandwidth=12,  # 12 GB/s
            cpu_to_gpu_bandwidth=12,  # 12 GB/s
            ctcf_prefill_alpha=1.1,   # Example calibration coefficient
            ctcf_prefill_beta=0,
            ctcf_decode_alpha=70,
            ctcf_decode_beta=0,
            hourly_cost=1.167
        ),
        HardwareConfig(
            gpu_name="L40S",
            gpu_memory_gb=47,
            gpu_flops=91.61,          # 91.61 TFLOPS (FP16)
            gpu_to_cpu_bandwidth=12,  # 12 GB/s
            cpu_to_gpu_bandwidth=12,  # 12 GB/s
            ctcf_prefill_alpha=0.9,   # Example calibration coefficient
            ctcf_prefill_beta=0,
            ctcf_decode_alpha=40.0,
            ctcf_decode_beta=0,
            hourly_cost=2.7
        ),
    ]
    
    if args.model not in model_configs:
        raise ValueError(f"Unsupported model: {args.model}. Supported models: {list(model_configs.keys())}")
    
    model_config = model_configs[args.model]
    
    print(f"Model: {args.model}")
    print(f"Number of layers: {model_config.num_layers}")
    print(f"Hidden size: {model_config.hidden_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Input length: {args.input_len}")
    print(f"Output length: {args.output_len}")
    print(f"Safety factor: {args.safety_factor}")
    if args.slo_tps:
        print(f"SLO requirement: Minimum {args.slo_tps} TPS")
    
    # Select the optimal instance
    best_instance, all_candidates = select_best_instance(
        model_config, 
        hardware_configs, 
        args.batch_size, 
        args.input_len, 
        args.output_len, 
        slo_tps=args.slo_tps,
        precision_bytes=args.precision,
        safety_factor=args.safety_factor,
        adjust_offloading=args.adjust
    )
    
    if best_instance is None:
        print("\nNo instances meet the requirements.")
    else:
        # Calculate memory requirements (for KV Cache size output)
        memory_req = calculate_memory_requirements(
            model_config, 
            args.batch_size, 
            args.input_len, 
            args.output_len, 
            args.precision
        )
        kv_cache_size_gb = memory_req["kv_cache"] / 1e9  # Convert bytes to GB
        
        print("\nOptimal instance:")
        print(f"GPU: {best_instance['hardware']}")
        print(f"KV Cache size: {kv_cache_size_gb:.2f} GB")
        print(f"KV Cache offloading ratio: {best_instance['offloading_ratio']*100:.1f}%")
        print(f"TPS: {best_instance['actual_tps']:.2f} tokens/second")
        print(f"Hourly cost: ${best_instance['hourly_cost']:.2f}")
        print(f"Cost efficiency: {best_instance['cost_efficiency']:.2f} TPS/$")
        
        perf = best_instance['performance']
        print(f"\nPerformance details:")
        print(f"Prefill time: {perf['prefill_total_time']:.4f} seconds")
        print(f"Decode time: {perf['decode_total_time']:.4f} seconds")
        print(f"Total processing time: {perf['total_time']:.4f} seconds")
        
        if len(all_candidates) > 1:
            print("\nOther instances that meet SLO:")
            for i, candidate in enumerate(all_candidates[1:], 2):
                print(f"{i}. {candidate['hardware']} - "
                      f"{candidate['actual_tps']:.2f} TPS, "
                      f"${candidate['hourly_cost']:.2f}/hour, "
                      f"{candidate['offloading_ratio']*100:.1f}% offloading")
