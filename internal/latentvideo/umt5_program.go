package latentvideo

import (
	"errors"
	"fmt"

	"overgo/internal/checked"
	"overgo/internal/pytorchzip"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// umt5Program is the text encoder's graph for one sequence of tokens: the tokens' embedding rows [Dim, seq] in,
// through every block, the final normalisation out, as encoderBlockForward computes them on the host. Its weights are
// the checkpoint's tensors by name, the blocks' matrices in BF16 as stored, read on the card's tensor cores as torch
// reads them, and the norms and each block's relative-position table in F32; a relative-position table is the
// attention's learned bias, read bidirectionally out to the policy's distance, and the scores are unscaled, as T5's
// are. Every value the host rounds to BF16, as torch's BF16 inference does, the graph rounds where the host does.
type umt5Program struct {
	Embed, Output *tensor.Tensor
	weights       tensor.WeightBindings
}

// compileUMT5Program builds the encoder's graph for seq tokens under config.
func compileUMT5Program(config encoderConfig, seq int) (*umt5Program, error) {
	if !checked.PositiveInts(seq, config.Layers, config.Dim, config.Heads, config.FFNDim, config.RelativeBuckets) {
		return nil, fmt.Errorf("umt5 program: tokens=%d and the encoder's geometry must be positive", seq)
	}
	headWidth, divides := checked.DivExactInt(config.Dim, config.Heads)
	if !divides {
		return nil, errors.New("umt5 program: the width does not divide across the heads")
	}
	b := tensor.NewBuilder()
	b.SetMulMatCompute(tensor.MulMatComputeBF16TensorCore)
	program := &umt5Program{}
	bind := tensor.WeightInputs{Builder: b, Bindings: &program.weights, MatrixType: dtype.BF16}
	width, heads, hidden, tokens := uint64(config.Dim), uint64(config.Heads), uint64(config.FFNDim), uint64(seq)
	eps := float32(config.NormEps)
	program.Embed = b.Input("umt5_embed", dtype.F32, tensor.MustShape(width, tokens))
	rows := program.Embed
	for layer := range config.Layers {
		prefix := fmt.Sprintf("blocks.%d.", layer)
		normed := b.BF16Round(b.WeightedRMSNorm(rows, bind.Input(prefix+"norm1.weight", width), eps))
		headed := func(name string) *tensor.Tensor {
			return b.Reshape(b.BF16Round(b.MulMat(bind.Input(prefix+name, width, width), normed)), uint64(headWidth), heads, tokens)
		}
		bias := b.Input(prefix+"pos_embedding.embedding.weight", dtype.F32, tensor.MustShape(heads, uint64(config.RelativeBuckets)))
		*bind.Bindings = append(*bind.Bindings, bias)
		attention := b.AttentionWithOptions(headed("attn.q.weight"), headed("attn.k.weight"), headed("attn.v.weight"),
			tensor.AttentionOptions{Bias: bias, Scale: tensor.UnitScale, RelativeBidirectional: true})
		projected := b.BF16Round(b.MulMat(bind.Input(prefix+"attn.o.weight", width, width), b.Reshape(attention, width, tokens)))
		rows = b.BF16Round(b.Add(rows, projected))
		normed = b.BF16Round(b.WeightedRMSNorm(rows, bind.Input(prefix+"norm2.weight", width), eps))
		gate := b.BF16Round(b.GELUTanhExact(b.BF16Round(b.MulMat(bind.Input(prefix+"ffn.gate.0.weight", width, hidden), normed))))
		up := b.BF16Round(b.MulMat(bind.Input(prefix+"ffn.fc1.weight", width, hidden), normed))
		ffn := b.BF16Round(b.MulMat(bind.Input(prefix+"ffn.fc2.weight", hidden, width), b.BF16Round(b.Multiply(gate, up))))
		rows = b.BF16Round(b.Add(rows, ffn))
	}
	program.Output = b.WeightedRMSNorm(rows, bind.Input("norm.weight", width), eps)
	if err := b.Err(); err != nil {
		return nil, fmt.Errorf("umt5 program graph: %w", err)
	}
	return program, nil
}

// encoderBindingsByName are the plan's checkpoint bindings by the tensor names the encoder reads, in encoderTensorNames'
// order, the order the plan binds them in.
func encoderBindingsByName(plan encoderPlan) (map[string]pytorchzip.TensorBinding, error) {
	names := encoderTensorNames(plan.Config.Layers)
	if len(names) != len(plan.bindings) {
		return nil, fmt.Errorf("umt5: the plan binds %d tensors of the encoder's %d", len(plan.bindings), len(names))
	}
	byName := make(map[string]pytorchzip.TensorBinding, len(names))
	for index, name := range names {
		byName[name] = plan.bindings[index]
	}
	return byName, nil
}
