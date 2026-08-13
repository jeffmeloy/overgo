package routedlm

import (
	"fmt"
	"math"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// DevicePrefixLayerGraph is one branch's multi-axis block-causal prefill.
type DevicePrefixLayerGraph struct {
	Builder                *tensor.Builder
	Row                    *tensor.Tensor
	Weights                DevicePrefillBranch
	Output, KeyKV, ValueKV *tensor.Tensor
	Tokens                 int
}

func BuildDevicePrefixLayer(
	cfg Config,
	rope RopePlan,
	positions []RowPosition,
) (*DevicePrefixLayerGraph, error) {
	if len(positions) == 0 {
		return nil, fmt.Errorf("routed lm prefix graph: empty positions")
	}
	H := uint64(cfg.HiddenSize)
	hd := uint64(cfg.HeadDim)
	qHeads := uint64(cfg.NumAttentionHeads)
	kvHeads := uint64(cfg.NumKeyValueHeads)
	qOut, kvOut := qHeads*hd, kvHeads*hd
	f, n := uint64(cfg.IntermediateSize), uint64(len(positions))
	eps := float32(cfg.RMSNormEps)
	scale := float32(1 / math.Sqrt(float64(hd)))

	b := tensor.NewBuilder()
	g := &DevicePrefixLayerGraph{Builder: b, Tokens: len(positions)}
	g.Row = b.Input("row", dtype.F32, tensor.MustShape(H, n))
	g.Weights = newPrefillBranch(b, "prefix", H, qOut, kvOut, hd, f)
	w := g.Weights
	normed := b.BF16Round(b.WeightedRMSNorm(g.Row, w.InputNorm, eps))
	query := b.Reshape(b.MulMat(w.Q, normed), hd, qHeads, n)
	key := b.Reshape(b.MulMat(w.K, normed), hd, kvHeads, n)
	value := b.BF16Round(b.Reshape(b.MulMat(w.V, normed), hd, kvHeads, n))
	var err error
	query, err = rope.DeviceNormalizeApply(b, query, w.QNorm, positions, eps)
	if err != nil {
		return nil, err
	}
	key, err = rope.DeviceNormalizeApply(b, key, w.KNorm, positions, eps)
	if err != nil {
		return nil, err
	}
	g.KeyKV, g.ValueKV = key, value

	strictCausal := true
	for index, position := range positions {
		if position.Time != index {
			strictCausal = false
			break
		}
	}
	var context *tensor.Tensor
	if strictCausal {
		context = b.AttentionWithOptions(
			b.BF16Round(query), b.BF16Round(key), b.BF16Round(value),
			tensor.AttentionOptions{Scale: scale, Causal: true},
		)
		context = b.Reshape(context, qOut, n)
	} else {
		for start := 0; start < len(positions); {
			end := start + 1
			for end < len(positions) && positions[end].Time == positions[start].Time {
				end++
			}
			count := uint64(end - start)
			queryBlock := b.FlatSlice(query, uint64(start)*hd*qHeads, hd, qHeads, count)
			keyBlock := b.FlatSlice(key, 0, hd, kvHeads, uint64(end))
			valueBlock := b.FlatSlice(value, 0, hd, kvHeads, uint64(end))
			attention := b.AttentionWithOptions(
				b.BF16Round(queryBlock), b.BF16Round(keyBlock), b.BF16Round(valueBlock),
				tensor.AttentionOptions{Scale: scale},
			)
			attention = b.Reshape(attention, qOut, count)
			if context == nil {
				context = attention
			} else {
				context = b.Concat(context, attention, 1)
			}
			start = end
		}
	}
	context = b.BF16Round(context)
	residual := b.Add(g.Row, b.MulMat(w.O, context))
	postNorm := b.WeightedRMSNorm(residual, w.PostNorm, eps)
	gate := b.MulMat(w.Gate, postNorm)
	up := b.MulMat(w.Up, postNorm)
	g.Output = b.Add(residual, b.MulMat(w.Down, b.SwiGLU(gate, up)))
	if err := b.Err(); err != nil {
		return nil, fmt.Errorf("routed lm prefix graph: %w", err)
	}
	return g, nil
}
