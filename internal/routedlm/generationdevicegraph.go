package routedlm

import (
	"fmt"
	"math"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// DeviceGenerationLayerGraph is one routed generation layer with prefix KV.
type DeviceGenerationLayerGraph struct {
	Builder                *tensor.Builder
	Row                    *tensor.Tensor
	PrefixKey, PrefixValue *tensor.Tensor
	Vision                 DevicePrefillBranch
	Output                 *tensor.Tensor
	Tokens, PrefixTokens   int
}

// BuildDeviceGenerationLayer emits neutral MoT generation math. Family policy
// enters only through derived config, rope, prefix extent, and image geometry.
func BuildDeviceGenerationLayer(
	cfg Config,
	rope RopePlan,
	prefixTokens, tokenHeight, tokenWidth, imageTime int,
) (*DeviceGenerationLayerGraph, error) {
	return buildDeviceGenerationLayer(cfg, rope, prefixTokens, tokenHeight, tokenWidth, imageTime, dtype.BF16)
}

func buildDeviceGenerationLayer(
	cfg Config,
	rope RopePlan,
	prefixTokens, tokenHeight, tokenWidth, imageTime int,
	weightType dtype.Type,
) (*DeviceGenerationLayerGraph, error) {
	if prefixTokens <= 0 || tokenHeight <= 0 || tokenWidth <= 0 || imageTime < 0 {
		return nil, fmt.Errorf("routed lm generation graph: invalid prefix or image geometry")
	}
	tokens := tokenHeight * tokenWidth
	H := uint64(cfg.HiddenSize)
	hd := uint64(cfg.HeadDim)
	qHeads := uint64(cfg.NumAttentionHeads)
	kvHeads := uint64(cfg.NumKeyValueHeads)
	qOut, kvOut := qHeads*hd, kvHeads*hd
	f := uint64(cfg.IntermediateSize)
	n, prefix := uint64(tokens), uint64(prefixTokens)
	eps := float32(cfg.RMSNormEps)
	scale := float32(1 / math.Sqrt(float64(hd)))

	positions := make([]RowPosition, tokens)
	for row := range positions {
		positions[row] = RowPosition{
			Branch: 1, Time: imageTime, H: row / tokenWidth, W: row % tokenWidth,
		}
	}

	b := tensor.NewBuilder()
	if weightType == dtype.BF16 {
		b.SetMulMatCompute(tensor.MulMatComputeBF16TensorCore)
	}
	g := &DeviceGenerationLayerGraph{
		Builder: b, Tokens: tokens, PrefixTokens: prefixTokens,
	}
	g.Row = b.Input("row", dtype.F32, tensor.MustShape(H, n))
	g.PrefixKey = b.Input("prefix_key", dtype.F32, tensor.MustShape(hd, kvHeads, prefix))
	g.PrefixValue = b.Input("prefix_value", dtype.F32, tensor.MustShape(hd, kvHeads, prefix))
	g.Vision = newTypedPrefillBranch(b, "generation", H, qOut, kvOut, hd, f, weightType)

	normed := b.BF16Round(b.WeightedRMSNorm(g.Row, g.Vision.InputNorm, eps))
	query := b.Reshape(b.MulMat(g.Vision.Q, normed), hd, qHeads, n)
	key := b.Reshape(b.MulMat(g.Vision.K, normed), hd, kvHeads, n)
	value := b.BF16Round(b.Reshape(b.MulMat(g.Vision.V, normed), hd, kvHeads, n))
	var err error
	query, err = rope.DeviceNormalizeApply(b, query, g.Vision.QNorm, positions, eps)
	if err != nil {
		return nil, err
	}
	key, err = rope.DeviceNormalizeApply(b, key, g.Vision.KNorm, positions, eps)
	if err != nil {
		return nil, err
	}
	key = b.Concat(g.PrefixKey, key, 2)
	value = b.Concat(g.PrefixValue, value, 2)
	context := b.AttentionWithOptions(
		b.BF16Round(query), b.BF16Round(key), b.BF16Round(value),
		tensor.AttentionOptions{Scale: scale},
	)
	context = b.Reshape(b.BF16Round(context), qOut, n)
	projected := b.BF16Round(b.MulMat(g.Vision.O, context))
	residual := b.BF16Round(b.Add(g.Row, projected))
	postNorm := b.BF16Round(b.WeightedRMSNorm(residual, g.Vision.PostNorm, eps))
	gate := b.BF16Round(b.MulMat(g.Vision.Gate, postNorm))
	up := b.BF16Round(b.MulMat(g.Vision.Up, postNorm))
	activated := b.BF16Round(b.Multiply(b.BF16Round(b.SiLU(gate)), up))
	down := b.BF16Round(b.MulMat(g.Vision.Down, activated))
	g.Output = b.BF16Round(b.Add(residual, down))
	if err := b.Err(); err != nil {
		return nil, fmt.Errorf("routed lm generation graph: %w", err)
	}
	return g, nil
}
