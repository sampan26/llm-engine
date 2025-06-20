# paged-attention-engine

## LLM Inference Optimizations
Progressive implementation of increasingly complex LLM inference optimizations, from basic sequential processing to advanced batch operations with KV caching.
Optimization Stages
inference_0.py (15 tokens/second)
Basic sequential inference implementation
inference_1.py (15 tokens/second)
Extended to handle multiple prompts sequentially
inference_2.py (116 tokens/second)
Batch matrix multiplications for attention computation (all heads processed simultaneously)
inference_3.py (342 tokens/second)
Multiple prompts processed simultaneously with improved matrix operations and parallel token generation
inference_4.py (7,160 tokens/second)
KV caching implementation with proper batch processing, dynamically removing completed prompts from batches
baseline.py (10,514 tokens/second)
VLLM baseline for performance comparison
Performance Details

Hardware: Single RTX 4090
Model: Llama3.2-1B-Instruct
Batch Size: 50 prompts processed concurrently
Tokens Generated: 500 tokens per prompt

Notes
This codebase prioritizes educational clarity over production readiness. The implementations use single-file, function-free structures to maximize understanding of the underlying optimization techniques.
