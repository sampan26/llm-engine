import time
import torch
import torch.nn.functional as F
from prompts import prompts
from config import Config
from pathlib import Path
from typing import Tuple


class LlamaInferenceEngine:
    """Optimized Llama inference engine with improved code organization and performance."""
    
    def __init__(self, model_path: str = None):
        """Initialize the inference engine with model configuration."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)
        self._setup_cuda_optimizations()
        
        model_path = model_path or f"{Path.home()}/.llama/checkpoints/Llama3.2-1B-Instruct"
        self.config = Config(path=model_path, device=self.device)
        self.tokenizer = self.config.tokenizer
        self.model = self.config.model
        
        # Initialize embedding layer
        self.embed = torch.nn.Embedding(
            self.config.vocab_size, 
            self.config.dim, 
            _weight=self.model["tok_embeddings.weight"]
        ).to(self.device)
        
        # Cache frequently used token IDs
        self.pad_id = self.tokenizer.encode("<|end_of_text|>", allowed_special="all")[0]
        self.eot_id = self.tokenizer.encode("<|eot_id|>", allowed_special="all")[0]
    
    def _setup_cuda_optimizations(self):
        """Enable CUDA optimizations for better performance."""
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    @staticmethod
    @torch.jit.script
    def _compute_rope_frequencies(max_tokens: int, rope_theta: float, device: torch.device) -> torch.Tensor:
        """Compute RoPE (Rotary Position Embedding) frequencies."""
        idx = torch.arange(32, device=device) / 32.0
        freqs_for_each_token = torch.outer(
            torch.arange(max_tokens, device=device), 
            1.0 / (rope_theta ** idx)
        )
        return torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)
    
    @staticmethod
    @torch.jit.script
    def _rms_normalize(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """Apply RMS normalization to input tensor."""
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * weight
    
    def _apply_rotary_embedding(self, tensor: torch.Tensor, freqs: torch.Tensor, 
                               batch_size: int, rot_dim: int) -> torch.Tensor:
        """Apply rotary position embedding to query/key tensors."""
        t_rot, t_pass = tensor[..., :rot_dim], tensor[..., rot_dim:]
        t_rot = t_rot.float().view(batch_size, -1, rot_dim // 2, 2)
        t_rot = torch.view_as_real(torch.view_as_complex(t_rot) * freqs)
        return torch.cat(
            (t_rot.view(batch_size, -1, rot_dim), t_pass), 
            dim=-1
        ).to(torch.bfloat16)
    
    def _forward_attention_layer(self, x: torch.Tensor, layer_idx: int, pos: int,
                                k_cache: torch.Tensor, v_cache: torch.Tensor,
                                freqs_cis: torch.Tensor) -> torch.Tensor:
        """Forward pass through a single attention layer."""
        # Pre-attention normalization
        attn_norm_weight = self.model[f"layers.{layer_idx}.attention_norm.weight"]
        x_norm = self._rms_normalize(x, attn_norm_weight, self.config.norm_eps)
        
        # Compute Q, K, V projections
        wq = self.model[f"layers.{layer_idx}.attention.wq.weight"]
        wk = self.model[f"layers.{layer_idx}.attention.wk.weight"]
        wv = self.model[f"layers.{layer_idx}.attention.wv.weight"]
        
        q = (x_norm @ wq.T).view(self.config.batch_size, self.config.n_heads, self.config.head_dim)
        k = (x_norm @ wk.T).view(self.config.batch_size, self.config.n_kv_heads, self.config.head_dim)
        v = (x_norm @ wv.T).view(self.config.batch_size, self.config.n_kv_heads, self.config.head_dim)
        
        # Apply rotary position embedding
        freqs_layer = freqs_cis[pos : pos + 1, : self.config.rot_dim // 2]
        q = self._apply_rotary_embedding(q, freqs_layer, self.config.batch_size, self.config.rot_dim)
        k = self._apply_rotary_embedding(k, freqs_layer, self.config.batch_size, self.config.rot_dim)
        
        # Handle grouped query attention
        if self.config.kv_mult > 1:
            k = k.repeat_interleave(self.config.kv_mult, dim=1)
            v = v.repeat_interleave(self.config.kv_mult, dim=1)
        
        # Update KV cache
        k_cache[layer_idx, pos] = k
        v_cache[layer_idx, pos] = v
        
        # Retrieve cached keys and values
        k_all = k_cache[layer_idx, : pos + 1].permute(1, 2, 0, 3)
        v_all = v_cache[layer_idx, : pos + 1].permute(1, 2, 0, 3)
        
        # Compute attention
        scale = 1.0 / (self.config.head_dim ** 0.5)
        attn_out = F.scaled_dot_product_attention(
            q.unsqueeze(2), k_all, v_all, 
            scale=scale, is_causal=False
        ).squeeze(2)
        
        # Output projection
        attn_out = attn_out.reshape(self.config.batch_size, self.config.dim)
        wo = self.model[f"layers.{layer_idx}.attention.wo.weight"]
        
        return x + attn_out @ wo.T
    
    def _forward_feedforward_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Forward pass through feedforward network."""
        # Pre-FFN normalization
        ffn_norm_weight = self.model[f"layers.{layer_idx}.ffn_norm.weight"]
        x_ffn_norm = self._rms_normalize(x, ffn_norm_weight, self.config.norm_eps)
        
        # Get FFN weights
        w1 = self.model[f"layers.{layer_idx}.feed_forward.w1.weight"]
        w2 = self.model[f"layers.{layer_idx}.feed_forward.w2.weight"]
        w3 = self.model[f"layers.{layer_idx}.feed_forward.w3.weight"]
        
        # SwiGLU activation
        ffn_out = (F.silu(x_ffn_norm @ w1.T) * (x_ffn_norm @ w3.T)) @ w2.T
        
        return x + ffn_out
    
    def _forward_transformer_block(self, x: torch.Tensor, k_cache: torch.Tensor, 
                                  v_cache: torch.Tensor, pos: int, 
                                  freqs_cis: torch.Tensor) -> torch.Tensor:
        """Forward pass through all transformer layers."""
        for layer_idx in range(self.config.n_layers):
            # Attention layer
            x = self._forward_attention_layer(x, layer_idx, pos, k_cache, v_cache, freqs_cis)
            # Feedforward layer
            x = self._forward_feedforward_layer(x, layer_idx)
        
        return x
    
    def _prepare_prompts(self, prompts: list) -> Tuple[torch.Tensor, int]:
        """Prepare and tokenize input prompts."""
        formatted_prompts = [
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n" + 
            prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            for prompt in prompts
        ]
        
        # Tokenize all prompts
        encoded_prompts = [
            torch.tensor(self.tokenizer.encode(prompt, allowed_special="all"), dtype=torch.long)
            for prompt in formatted_prompts
        ]
        
        # Pad to same length
        max_len = max(len(tokens) for tokens in encoded_prompts)
        prompt_tokens = torch.full(
            (len(encoded_prompts), max_len), 
            self.pad_id, 
            dtype=torch.long, 
            device=self.device
        )
        
        for i, tokens in enumerate(encoded_prompts):
            prompt_tokens[i, :tokens.numel()] = tokens.to(self.device)
        
        return prompt_tokens, max_len
    
    def _initialize_kv_cache(self, max_total_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize key-value cache tensors."""
        cache_shape = (
            self.config.n_layers, 
            max_total_tokens, 
            self.config.batch_size, 
            self.config.n_heads, 
            self.config.head_dim
        )
        
        k_cache = torch.empty(*cache_shape, dtype=torch.bfloat16, device=self.device)
        v_cache = torch.empty_like(k_cache)
        
        return k_cache, v_cache
    
    def _prefill_phase(self, prompt_tokens: torch.Tensor, max_len: int,
                      k_cache: torch.Tensor, v_cache: torch.Tensor,
                      freqs_cis: torch.Tensor) -> torch.Tensor:
        """Process all prompt tokens in parallel (prefill phase)."""
        print(f"[PREFILL] Processing {max_len} prompt tokens...")
        prefill_start = time.time()
        
        for pos in range(max_len):
            token_step = prompt_tokens[:, pos]
            x = self.embed(token_step).to(torch.bfloat16)
            x = self._forward_transformer_block(x, k_cache, v_cache, pos, freqs_cis)
            
            # Only compute logits for the last position
            if pos == max_len - 1:
                x_norm = self._rms_normalize(x, self.model["norm.weight"], self.config.norm_eps)
                last_logits = (x_norm @ self.model["output.weight"].T).float()
        
        prefill_time = time.time() - prefill_start
        print(f"[PREFILL] Completed in {prefill_time:.2f}s")
        
        return last_logits
    
    def _generation_phase(self, initial_logits: torch.Tensor, max_len: int,
                         k_cache: torch.Tensor, v_cache: torch.Tensor,
                         freqs_cis: torch.Tensor, max_new_tokens: int) -> list:
        """Generate new tokens autoregressively."""
        next_token = torch.argmax(initial_logits, dim=-1)
        generated_sequences = [[] for _ in range(self.config.batch_size)]
        
        print(f"[GENERATION] Generating up to {max_new_tokens} tokens...")
        gen_start = time.time()
        
        for step in range(max_new_tokens):
            step_start = time.time()
            
            # Store generated tokens
            for i, token in enumerate(next_token.cpu().tolist()):
                generated_sequences[i].append(token)
            
            # Forward pass for next token
            x = self.embed(next_token).to(torch.bfloat16)
            pos = max_len + step
            x = self._forward_transformer_block(x, k_cache, v_cache, pos, freqs_cis)
            
            # Compute next token logits
            x_norm = self._rms_normalize(x, self.model["norm.weight"], self.config.norm_eps)
            logits = x_norm @ self.model["output.weight"].T
            next_token = torch.argmax(logits, dim=-1)
            
            step_time = time.time() - step_start
            tokens_per_sec = self.config.batch_size / step_time
            print(f"[STEP {step + 1}] Generated {self.config.batch_size} tokens "
                  f"in {step_time:.3f}s ({tokens_per_sec:.1f} tok/s)")
        
        gen_time = time.time() - gen_start
        total_tokens = sum(len(seq) for seq in generated_sequences)
        avg_speed = total_tokens / gen_time
        print(f"[GENERATION] Generated {total_tokens} tokens in {gen_time:.2f}s "
              f"({avg_speed:.1f} tok/s average)")
        
        return generated_sequences
    
    def generate(self, input_prompts: list, max_new_tokens: int = 500) -> list:
        """Generate text completions for input prompts."""
        with torch.no_grad():
            # Update batch size
            self.config.batch_size = len(input_prompts)
            
            # Prepare inputs
            prompt_tokens, max_len = self._prepare_prompts(input_prompts)
            max_total_tokens = max_len + max_new_tokens
            
            # Initialize components
            freqs_cis = self._compute_rope_frequencies(
                max_total_tokens, self.config.rope_theta, self.device
            )
            self.config.rot_dim = self.config.head_dim // 2 * 2
            k_cache, v_cache = self._initialize_kv_cache(max_total_tokens)
            
            # Prefill phase
            initial_logits = self._prefill_phase(
                prompt_tokens, max_len, k_cache, v_cache, freqs_cis
            )
            
            # Generation phase
            generated_sequences = self._generation_phase(
                initial_logits, max_len, k_cache, v_cache, freqs_cis, max_new_tokens
            )
            
            return self._format_outputs(prompt_tokens, generated_sequences)
    
    def _format_outputs(self, prompt_tokens: torch.Tensor, 
                       generated_sequences: list) -> list:
        """Format and decode the generated outputs."""
        results = []
        
        for i in range(self.config.batch_size):
            # Combine prompt and generated tokens
            prompt_list = prompt_tokens[i].cpu().tolist()
            # Remove padding tokens
            prompt_list = [t for t in prompt_list if t != self.pad_id]
            
            full_sequence = prompt_list + generated_sequences[i]
            decoded_text = self.tokenizer.decode(full_sequence)
            
            results.append({
                'prompt_tokens': len(prompt_list),
                'generated_tokens': len(generated_sequences[i]),
                'full_text': decoded_text
            })
            
            print(f"\n{'='*60}")
            print(f"OUTPUT {i + 1}")
            print(f"{'='*60}")
            print(decoded_text)
        
        return results


def main():
    """Main execution function."""
    # Initialize inference engine
    engine = LlamaInferenceEngine()
    
    # Generate completions
    results = engine.generate(prompts, max_new_tokens=500)
    
    # Print summary statistics
    total_prompt_tokens = sum(r['prompt_tokens'] for r in results)
    total_generated_tokens = sum(r['generated_tokens'] for r in results)
    
    print(f"\n{'='*60}")
    print("GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total prompt tokens: {total_prompt_tokens}")
    print(f"Total generated tokens: {total_generated_tokens}")
    print(f"Total sequences: {len(results)}")


if __name__ == "__main__":
    main()