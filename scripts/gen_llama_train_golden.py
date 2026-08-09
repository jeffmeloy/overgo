# Golden generator for the dense causal-LM training capability (rung 3).
# torch is sanctioned here solely as the golden oracle. Emits:
#   fixtures/llama_train_golden.json            full tiny-model step golden
#   fixtures/qwen2_train_golden.json            tiny qwen2 (qkv biases) step golden
#   fixtures/llama_attn_gqa_grad_golden.json    GQA causal attention fwd+bwd
#   fixtures/llama_rope_interleaved_grad_golden.json  interleaved (GGML) rope
#   fixtures/llama_gatedmlp_grad_golden.json    SiLU-gated MLP fwd+bwd
#   fixtures/llama_softmaxce_grad_golden.json   mean softmax-CE loss + grad
# Tiny dims only; every tensor seeded; f64 lists via json.dump.
# argv selects generators by function name; no argv = all.
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")


def dump(name, payload):
    path = os.path.join(FIXTURES, name)
    with open(path, "w") as f:
        json.dump(payload, f)
    print("wrote", path)


def flat(t):
    return t.detach().double().reshape(-1).tolist()


def emit_train_golden(name, schema, model, config, head_dim, attention_bias):
    # Shared tiny-model step emitter: seeded batch, causal-LM mean CE,
    # last-position logits, full f64 state_dict + every parameter grad.
    model.train()
    tokens = torch.randint(0, config.vocab_size, (1, 12), generator=torch.Generator().manual_seed(7))
    logits = model(input_ids=tokens).logits
    # Causal-LM objective: positions 0..n-2 predict tokens 1..n-1, mean CE.
    loss = F.cross_entropy(logits[0, :-1, :], tokens[0, 1:])
    loss.backward()
    params = {}
    grads = {}
    for pname, p in model.named_parameters():
        params[pname] = {"shape": list(p.shape), "values": flat(p)}
        grads[pname] = flat(p.grad)
    dump(name, {
        "schema": schema,
        "config": {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": head_dim,
            "rope_theta": config.rope_theta,
            "rms_norm_eps": config.rms_norm_eps,
            "attention_bias": attention_bias,
        },
        "tokens": tokens[0].tolist(),
        "loss": loss.item(),
        "last_logits": flat(logits[0, -1, :]),
        "params": params,
        "grads": grads,
    })


def gen_full_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(config).float()
    emit_train_golden("llama_train_golden.json", "llama_train_golden/v1",
                      model, config, config.head_dim, False)


def gen_qwen2_full_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        attention_dropout=0.0,
        use_sliding_window=False,
        attn_implementation="eager",
    )
    model = Qwen2ForCausalLM(config).float()
    names = [n for n, _ in model.named_parameters()]
    assert "model.layers.0.self_attn.q_proj.bias" in names, "qwen2 qkv biases missing"
    assert not any(n.endswith("o_proj.bias") for n in names), "unexpected o_proj bias"
    head_dim = config.hidden_size // config.num_attention_heads
    emit_train_golden("qwen2_train_golden.json", "qwen2_train_golden/v1",
                      model, config, head_dim, True)


def gen_attn_gqa():
    torch.manual_seed(1)
    seq, heads, kv, hd = 5, 4, 2, 8
    q = torch.randn(seq, heads, hd, dtype=torch.float64, requires_grad=True)
    k = torch.randn(seq, kv, hd, dtype=torch.float64, requires_grad=True)
    v = torch.randn(seq, kv, hd, dtype=torch.float64, requires_grad=True)
    dout = torch.randn(seq, heads, hd, dtype=torch.float64)
    group = heads // kv
    kx = k.repeat_interleave(group, dim=1)  # [seq, heads, hd]
    vx = v.repeat_interleave(group, dim=1)
    scale = 1.0 / math.sqrt(hd)
    scores = torch.einsum("qhd,khd->hqk", q, kx) * scale
    mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hqk,khd->qhd", probs, vx)
    (out * dout).sum().backward()
    dump("llama_attn_gqa_grad_golden.json", {
        "seq": seq, "heads": heads, "kv_heads": kv, "head_dim": hd,
        "q": flat(q), "k": flat(k), "v": flat(v),
        "out": flat(out), "dout": flat(dout),
        "grad_q": flat(q.grad), "grad_k": flat(k.grad), "grad_v": flat(v.grad),
    })


def gen_rope_interleaved():
    torch.manual_seed(2)
    hd, pos, theta = 8, 3, 10000.0
    x = torch.randn(hd, dtype=torch.float64, requires_grad=True)
    dout = torch.randn(hd, dtype=torch.float64)
    inv = torch.tensor([theta ** (-2.0 * i / hd) for i in range(hd // 2)], dtype=torch.float64)
    ang = pos * inv
    c, s = torch.cos(ang), torch.sin(ang)
    even, odd = x[0::2], x[1::2]
    out = torch.empty(hd, dtype=torch.float64)
    out[0::2] = even * c - odd * s
    out[1::2] = even * s + odd * c
    (out * dout).sum().backward()
    dump("llama_rope_interleaved_grad_golden.json", {
        "head_dim": hd, "pos": pos, "theta": theta,
        "x": flat(x), "out": flat(out), "dout": flat(dout), "grad_x": flat(x.grad),
    })


def gen_gated_mlp():
    torch.manual_seed(3)
    rows, d, inter = 3, 16, 32
    x = torch.randn(rows, d, dtype=torch.float64, requires_grad=True)
    wg = torch.randn(inter, d, dtype=torch.float64, requires_grad=True)
    wu = torch.randn(inter, d, dtype=torch.float64, requires_grad=True)
    wd = torch.randn(d, inter, dtype=torch.float64, requires_grad=True)
    dout = torch.randn(rows, d, dtype=torch.float64)
    out = (F.silu(x @ wg.T) * (x @ wu.T)) @ wd.T
    (out * dout).sum().backward()
    dump("llama_gatedmlp_grad_golden.json", {
        "rows": rows, "dim": d, "intermediate": inter,
        "x": flat(x), "w_gate": flat(wg), "w_up": flat(wu), "w_down": flat(wd),
        "out": flat(out), "dout": flat(dout),
        "grad_x": flat(x.grad), "grad_w_gate": flat(wg.grad),
        "grad_w_up": flat(wu.grad), "grad_w_down": flat(wd.grad),
    })


def gen_softmax_ce():
    torch.manual_seed(4)
    rows, classes = 4, 11
    logits = torch.randn(rows, classes, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, classes, (rows,), generator=torch.Generator().manual_seed(5))
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    dump("llama_softmaxce_grad_golden.json", {
        "rows": rows, "classes": classes,
        "logits": flat(logits), "targets": targets.tolist(),
        "loss": loss.item(), "grad_logits": flat(logits.grad),
    })


if __name__ == "__main__":
    import transformers
    print("torch", torch.__version__, "transformers", transformers.__version__)
    generators = [gen_full_model, gen_qwen2_full_model, gen_attn_gqa,
                  gen_rope_interleaved, gen_gated_mlp, gen_softmax_ce]
    selected = sys.argv[1:]
    for gen in generators:
        if selected and gen.__name__ not in selected:
            continue
        gen()
