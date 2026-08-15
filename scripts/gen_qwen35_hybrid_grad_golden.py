# Golden generator for the qwen3.5 hybrid decoder-layer training capability
# (rung 12). torch is sanctioned here solely as the autograd oracle: it emits
# fixtures/qwen35_hybrid_grad_golden.json holding, for BOTH layer variants
# (linear_attention with the gated-delta mix, and full_attention), the seeded
# weights + input + state + output cotangent, and every parameter gradient of
# L = sum(dOut * layer(x)) under torch autograd. The Go host VJP
# (HybridDecoderLayerBackward) is checked against these grads. The torch module
# below MIRRORS internal/hostmath host math op-for-op, in float64.
#
# Run: py -3.12 scripts/gen_qwen35_hybrid_grad_golden.py
import json
import math
import os

import torch

FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")
torch.set_default_dtype(torch.float64)


def flat(t):
    return t.detach().reshape(-1).tolist()


def seeded(shape, gen, scale=0.4):
    return (torch.randn(shape, generator=gen) * scale).requires_grad_(True)


def rms_norm(x, weight, eps):
    # x [..., d]; per-row x/sqrt(mean(x^2)+eps)*weight.
    inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * inv * weight


def l2_norm(x, eps):
    # per-row x/max(sqrt(sum(x^2)), eps); matches host L2NormForward.
    n = torch.sqrt(x.pow(2).sum(-1, keepdim=True)).clamp_min(eps)
    return x / n


def short_conv(x, w, bias, k):
    # x [ch, T]; causal depthwise conv (left pad k-1) then SiLU. w [ch,k].
    ch, T = x.shape
    xp = torch.cat([torch.zeros(ch, k - 1), x], dim=1)
    out = torch.zeros(ch, T)
    for j in range(k):
        out = out + xp[:, j:j + T] * w[:, j:j + 1]
    if bias is not None:
        out = out + bias[:, None]
    return out * torch.sigmoid(out)  # SiLU


def rotary_half(x, invfreq, pos):
    # x [hd]; rotate half-split pairs by pos*invfreq (returns new tensor).
    h = x.shape[0] // 2
    a = pos * invfreq
    c, s = torch.cos(a), torch.sin(a)
    x1, x2 = x[:h], x[h:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s])


def gated_delta_net(q, k, v, gate, beta, state, hk, hv, hd, T):
    # q,k [T,hk,hd]; v [T,hv,hd]; gate,beta [T,hv]; state [hv,hd,hd].
    scale = 1.0 / math.sqrt(hd)
    S = [state[h] for h in range(hv)]
    outs = []
    for t in range(T):
        row = []
        for h in range(hv):
            kh = k[t, h % hk]
            qh = q[t, h % hk]
            g = torch.exp(gate[t, h])
            Sh = S[h] * g
            dot = Sh @ kh
            delta = (v[t, h] - dot) * beta[t, h]
            Sh = Sh + torch.outer(delta, kh)
            S[h] = Sh
            row.append((Sh @ qh) * scale)
        outs.append(torch.stack(row))          # [hv, hd]
    return torch.stack(outs)                    # [T, hv, hd]


def gated_delta_mix(x, w, cfg):
    T, H, hk, hv, hd, K, eps = cfg["T"], cfg["H"], cfg["hk"], cfg["hv"], cfg["hd"], cfg["K"], cfg["eps"]
    keyDim, valDim = hk * hd, hv * hd
    qp = torch.nn.functional.linear(x, w["Wq"])   # [T, keyDim]
    kp = torch.nn.functional.linear(x, w["Wk"])
    vp = torch.nn.functional.linear(x, w["Wv"])   # [T, valDim]
    qc = short_conv(qp.t(), w["ConvQ"], w["ConvBiasQ"], K).t()
    kc = short_conv(kp.t(), w["ConvK"], w["ConvBiasK"], K).t()
    vc = short_conv(vp.t(), w["ConvV"], w["ConvBiasV"], K).t()
    ql = l2_norm(qc.reshape(T * hk, hd), eps).reshape(T, hk, hd)
    kl = l2_norm(kc.reshape(T * hk, hd), eps).reshape(T, hk, hd)
    vv = vc.reshape(T, hv, hd)
    beta = torch.sigmoid(torch.nn.functional.linear(x, w["Wbeta"]))     # [T,hv]
    alpha = torch.nn.functional.linear(x, w["Walpha"])                  # [T,hv]
    gate = torch.nn.functional.softplus(alpha + w["TimeStep"]) * w["A"]  # [T,hv]
    z = torch.nn.functional.linear(x, w["Wz"])                          # [T, valDim]
    gdn = gated_delta_net(ql, kl, vv, gate, beta, w["state"], hk, hv, hd, T)  # [T,hv,hd]
    normed = rms_norm(gdn, w["Norm"], eps).reshape(T, valDim)
    gated = normed * (z * torch.sigmoid(z))                            # * SiLU(z)
    return torch.nn.functional.linear(gated, w["Wout"])                # [T, H]


def attention_mix(x, w, cfg):
    T, H, heads, kv, hd, theta, eps = cfg["T"], cfg["H"], cfg["heads"], cfg["kv"], cfg["hd"], cfg["theta"], cfg["eps"]
    qDim, kvDim = heads * hd, kv * hd
    q = torch.nn.functional.linear(x, w["Wq"]).reshape(T, heads, hd)
    k = torch.nn.functional.linear(x, w["Wk"]).reshape(T, kv, hd)
    v = torch.nn.functional.linear(x, w["Wv"]).reshape(T, kv, hd)
    q = rms_norm(q, w["QNorm"], eps)
    k = rms_norm(k, w["KNorm"], eps)
    invfreq = torch.tensor([1.0 / theta ** (2 * i / hd) for i in range(hd // 2)])
    scale = 1.0 / math.sqrt(hd)
    qr = torch.stack([torch.stack([rotary_half(q[t, h], invfreq, t) * scale for h in range(heads)]) for t in range(T)])
    kr = torch.stack([torch.stack([rotary_half(k[t, h], invfreq, t) for h in range(kv)]) for t in range(T)])
    group = heads // kv
    out = torch.zeros(T, heads, hd)
    for h in range(heads):
        kvh = h // group
        for qi in range(T):
            scores = torch.stack([qr[qi, h] @ kr[ki, kvh] for ki in range(qi + 1)])
            probs = torch.softmax(scores, dim=0)
            out[qi, h] = sum(probs[ki] * v[ki, kvh] for ki in range(qi + 1))
    return torch.nn.functional.linear(out.reshape(T, qDim), w["Wo"])   # [T, H]


def mlp(x, w, cfg):
    gate = torch.nn.functional.linear(x, w["Gate"])
    up = torch.nn.functional.linear(x, w["Up"])
    return torch.nn.functional.linear((gate * torch.sigmoid(gate)) * up, w["Down"])


def hybrid_layer(x, w, cfg, mix_fn):
    xn = rms_norm(x, w["InputNorm"], cfg["eps"])
    h = x + mix_fn(xn, w["mix"], cfg)
    hn = rms_norm(h, w["PostNorm"], cfg["eps"])
    return h + mlp(hn, w["mlp"], cfg)


def build_common(gen, H, inter, eps):
    return {
        "InputNorm": seeded(H, gen), "PostNorm": seeded(H, gen),
        "mlp": {"Gate": seeded((inter, H), gen), "Up": seeded((inter, H), gen), "Down": seeded((H, inter), gen)},
    }


def emit():
    gen = torch.Generator().manual_seed(20260812)
    T, H, inter, hd, eps = 4, 8, 16, 4, 1e-6
    x = seeded((T, H), gen)
    dOut = (torch.randn((T, H), generator=gen) * 0.4)

    # linear_attention variant
    hk, hv = 2, 2
    keyDim, valDim, K = hk * hd, hv * hd, 3
    lin = build_common(gen, H, inter, eps)
    lin["mix"] = {
        "Wq": seeded((keyDim, H), gen), "Wk": seeded((keyDim, H), gen), "Wv": seeded((valDim, H), gen),
        "ConvQ": seeded((keyDim, K), gen), "ConvK": seeded((keyDim, K), gen), "ConvV": seeded((valDim, K), gen),
        "ConvBiasQ": seeded(keyDim, gen), "ConvBiasK": seeded(keyDim, gen), "ConvBiasV": seeded(valDim, gen),
        "Wbeta": seeded((hv, H), gen), "Walpha": seeded((hv, H), gen),
        "TimeStep": seeded(hv, gen), "A": seeded(hv, gen),
        "Wz": seeded((valDim, H), gen), "Norm": seeded(hd, gen), "Wout": seeded((H, valDim), gen),
        "state": seeded((hv, hd, hd), gen),
    }
    cfg_lin = {"T": T, "H": H, "hk": hk, "hv": hv, "hd": hd, "K": K, "inter": inter, "eps": eps}
    out_lin = hybrid_layer(x, lin, cfg_lin, gated_delta_mix)
    (dOut * out_lin).sum().backward()

    # full_attention variant (fresh input/dOut leaves not needed; reuse x by re-running)
    x2 = x.detach().clone().requires_grad_(True)
    heads, kv = 2, 1
    qDim, kvDim = heads * hd, kv * hd
    att = build_common(gen, H, inter, eps)
    att["mix"] = {
        "Wq": seeded((qDim, H), gen), "Wk": seeded((kvDim, H), gen), "Wv": seeded((kvDim, H), gen),
        "Wo": seeded((H, qDim), gen), "QNorm": seeded(hd, gen), "KNorm": seeded(hd, gen),
    }
    cfg_att = {"T": T, "H": H, "heads": heads, "kv": kv, "hd": hd, "theta": 10000.0, "inter": inter, "eps": eps}
    out_a = hybrid_layer(x2, att, cfg_att, attention_mix)
    (dOut * out_a).sum().backward()

    def dump_weights(w):
        flatw = {}

        def rec(prefix, d):
            for kk, vv in d.items():
                if isinstance(vv, dict):
                    rec(prefix + kk + ".", vv)
                else:
                    flatw[prefix + kk] = {"shape": list(vv.shape), "values": flat(vv), "grad": flat(vv.grad)}
        rec("", w)
        return flatw

    payload = {
        "schema": "qwen35_hybrid_grad_golden/v1",
        "dims": {"T": T, "H": H, "inter": inter, "hd": hd, "eps": eps,
                 "linear": {"hk": hk, "hv": hv, "K": K},
                 "attention": {"heads": heads, "kv": kv, "theta": 10000.0}},
        "x": {"shape": [T, H], "values": flat(x), "grad": flat(x.grad)},
        "x_attn": {"shape": [T, H], "values": flat(x2), "grad": flat(x2.grad)},
        "dOut": flat(dOut),
        "out_linear": flat(out_lin), "out_attention": flat(out_a),
        "linear": dump_weights(lin),
        "attention": dump_weights(att),
    }
    os.makedirs(FIXTURES, exist_ok=True)
    path = os.path.join(FIXTURES, "qwen35_hybrid_grad_golden.json")
    with open(path, "w") as f:
        json.dump(payload, f)
    print("wrote", path)


if __name__ == "__main__":
    emit()
