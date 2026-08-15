package routedlm

// Device decode graph: the 32-layer branch-routed MoT decode step composed as
// an overgo tensor.Builder graph for the CUDA generic executor. Decode is
// text-branch only (branch 0, the standard GQA + per-head QK-norm decode path);
// the graph mirrors the host reference DecodeRow op-for-op (input RMSNorm ->
// Q/K/V -> rope-then-QK-norm -> capacity KV append -> GQA attention -> gated
// output projection residual -> post-attn RMSNorm -> SwiGLU MLP residual) and
// closes with the terminal (final RMSNorm -> lm_head vocab projection).
//
// KV is device-resident, fixed-capacity: each layer appends the new token's
// K/V in place (WriteCache append at CacheTokenOffset) so a single compiled
// graph replays across steps with only the rope positions, attention window,
// and cache-append offset updated as runtime attributes.

import (
	"fmt"
	"math"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// DeviceLayerInputs: per-layer text-branch weight input nodes (device feeds).
type DeviceLayerInputs struct {
	InputNorm *tensor.Tensor // [H] F32
	Q         *tensor.Tensor // [H, qOut] BF16
	K         *tensor.Tensor // [H, kvOut] BF16
	V         *tensor.Tensor // [H, kvOut] BF16
	O         *tensor.Tensor // [qOut, H] BF16
	QNorm     *tensor.Tensor // [hd] F32
	KNorm     *tensor.Tensor // [hd] F32
	PostNorm  *tensor.Tensor // [H] F32
	Gate      *tensor.Tensor // [H, f] BF16
	Up        *tensor.Tensor // [H, f] BF16
	Down      *tensor.Tensor // [f, H] BF16
	PastKey   *tensor.Tensor // [hd, kvHeads, capacity] F32
	PastValue *tensor.Tensor // [hd, kvHeads, capacity] F32
}

// DeviceLayerNodes: per-layer graph nodes needed for runtime update / retain.
type DeviceLayerNodes struct {
	QRope       *tensor.Tensor // OpRoPENeoX (query) — runtime positions
	KRope       *tensor.Tensor // OpRoPENeoX (key)   — runtime positions
	KeyAppend   *tensor.Tensor // OpCacheAppend (key)   — runtime offset + retained target
	ValueAppend *tensor.Tensor // OpCacheAppend (value) — runtime offset + retained target
	Attention   *tensor.Tensor // OpAttention — runtime QueryStart/KeyValueTokens
}

// DeviceDecodeGraph: the full compiled-ready decode graph and its handles.
type DeviceDecodeGraph struct {
	Builder   *tensor.Builder
	Embedding *tensor.Tensor // [H, 1] F32 host feed (the token row)
	FinalNorm *tensor.Tensor // [H] F32 device feed
	Head      *tensor.Tensor // [H, vocab] BF16 device feed
	Logits    *tensor.Tensor // [vocab, 1] F32 output
	Inputs    []DeviceLayerInputs
	Nodes     []DeviceLayerNodes
	Capacity  uint32
	RopeBase  float32
}

// BuildDeviceDecodeGraph: text-branch decode stack + terminal at fixed KV
// capacity. Rope positions / attention window / append offset start at token 0
// and are overridden per step through the executor's runtime attributes.
func BuildDeviceDecodeGraph(cfg Config, capacity uint32) (*DeviceDecodeGraph, error) {
	if capacity < 2 {
		return nil, fmt.Errorf("routed lm device decode: capacity %d < 2", capacity)
	}
	base, err := RopeInvFreqBase(cfg)
	if err != nil {
		return nil, err
	}
	H := uint64(cfg.HiddenSize)
	hd := uint64(cfg.HeadDim)
	qHeads := uint64(cfg.NumAttentionHeads)
	kvHeads := uint64(cfg.NumKeyValueHeads)
	qOut := qHeads * hd
	kvOut := kvHeads * hd
	f := uint64(cfg.IntermediateSize)
	vocab := uint64(cfg.VocabSize)
	eps := float32(cfg.RMSNormEps)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	rot := uint32(hd)

	b := tensor.NewBuilder()
	// Fixed-capacity append plan: initial active tokens 0 (runtime overrides
	// the per-step offset / positions / attention window).
	b.SetCacheAppendPlan(tensor.CacheAppendPlan{
		ActiveTokens:         0,
		SourceCapacityTokens: capacity,
		CapacityTokens:       capacity,
	})

	g := &DeviceDecodeGraph{Builder: b, Capacity: capacity, RopeBase: float32(base)}
	g.Embedding = b.Input("embedding", dtype.F32, tensor.MustShape(H, 1))
	g.Inputs = make([]DeviceLayerInputs, cfg.NumHiddenLayers)
	g.Nodes = make([]DeviceLayerNodes, cfg.NumHiddenLayers)

	row := g.Embedding
	for layer := 0; layer < cfg.NumHiddenLayers; layer++ {
		li := DeviceLayerInputs{
			InputNorm: b.Input(fmt.Sprintf("l%d.input_norm", layer), dtype.F32, tensor.MustShape(H)),
			Q:         b.Input(fmt.Sprintf("l%d.q", layer), dtype.BF16, tensor.MustShape(H, qOut)),
			K:         b.Input(fmt.Sprintf("l%d.k", layer), dtype.BF16, tensor.MustShape(H, kvOut)),
			V:         b.Input(fmt.Sprintf("l%d.v", layer), dtype.BF16, tensor.MustShape(H, kvOut)),
			O:         b.Input(fmt.Sprintf("l%d.o", layer), dtype.BF16, tensor.MustShape(qOut, H)),
			QNorm:     b.Input(fmt.Sprintf("l%d.qnorm", layer), dtype.F32, tensor.MustShape(hd)),
			KNorm:     b.Input(fmt.Sprintf("l%d.knorm", layer), dtype.F32, tensor.MustShape(hd)),
			PostNorm:  b.Input(fmt.Sprintf("l%d.post_norm", layer), dtype.F32, tensor.MustShape(H)),
			Gate:      b.Input(fmt.Sprintf("l%d.gate", layer), dtype.BF16, tensor.MustShape(H, f)),
			Up:        b.Input(fmt.Sprintf("l%d.up", layer), dtype.BF16, tensor.MustShape(H, f)),
			Down:      b.Input(fmt.Sprintf("l%d.down", layer), dtype.BF16, tensor.MustShape(f, H)),
			PastKey:   b.Input(fmt.Sprintf("l%d.pastk", layer), dtype.F32, tensor.MustShape(hd, kvHeads, uint64(capacity))),
			PastValue: b.Input(fmt.Sprintf("l%d.pastv", layer), dtype.F32, tensor.MustShape(hd, kvHeads, uint64(capacity))),
		}
		g.Inputs[layer] = li

		normed := b.WeightedRMSNorm(row, li.InputNorm, eps)
		q := b.Reshape(b.MulMat(li.Q, normed), hd, qHeads, 1)
		k := b.Reshape(b.MulMat(li.K, normed), hd, kvHeads, 1)
		v := b.Reshape(b.MulMat(li.V, normed), hd, kvHeads, 1)

		// Reference order: rope THEN per-head QK-norm.
		qRope := b.RoPEWithOptions(q, tensor.RoPEOptions{Layout: tensor.RoPELayoutNeoX, Positions: []uint32{0}, RotaryDimensions: rot, FrequencyBase: g.RopeBase, FrequencyScale: 1})
		q = b.WeightedRMSNorm(qRope, li.QNorm, eps)
		kRope := b.RoPEWithOptions(k, tensor.RoPEOptions{Layout: tensor.RoPELayoutNeoX, Positions: []uint32{0}, RotaryDimensions: rot, FrequencyBase: g.RopeBase, FrequencyScale: 1})
		k = b.WeightedRMSNorm(kRope, li.KNorm, eps)

		queryStart := b.CacheTokenOffset(0)
		cacheKey := b.WriteCache(li.PastKey, k, 2, tensor.CacheWriteAppend)
		cacheValue := b.WriteCache(li.PastValue, v, 2, tensor.CacheWriteAppend)
		attn := b.AttentionWithOffset(q, cacheKey, cacheValue, scale, true, queryStart)
		attnFlat := b.Reshape(attn, qOut, 1)

		projected := b.MulMat(li.O, attnFlat)
		residual := b.Add(row, projected)
		postNorm := b.WeightedRMSNorm(residual, li.PostNorm, eps)
		gate := b.MulMat(li.Gate, postNorm)
		up := b.MulMat(li.Up, postNorm)
		ff := b.MulMat(li.Down, b.SwiGLU(gate, up))
		row = b.Add(residual, ff)

		g.Nodes[layer] = DeviceLayerNodes{
			QRope: qRope, KRope: kRope,
			KeyAppend: cacheKey, ValueAppend: cacheValue, Attention: attn,
		}
	}

	g.FinalNorm = b.Input("final_norm", dtype.F32, tensor.MustShape(H))
	g.Head = b.Input("head", dtype.BF16, tensor.MustShape(H, vocab))
	final := b.WeightedRMSNorm(row, g.FinalNorm, eps)
	g.Logits = b.MulMat(g.Head, final)
	if err := b.Err(); err != nil {
		return nil, fmt.Errorf("routed lm device decode graph: %w", err)
	}
	return g, nil
}
