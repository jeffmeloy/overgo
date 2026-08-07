package tensor

import (
	"errors"
	"math"
	"slices"

	"overgo/internal/tensor/dtype"
)

// SSMConv: applies channel-wise sliding convolution used by recurrent SSM
// blocks; Input is [kernel-1+tokens, channels, sequences] and weights are
// [kernel, channels]; output is [channels, tokens, sequences]
func (b *Builder) SSMConv(input, weights *Tensor) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || weights == nil {
		b.setError(errors.New("SSMConv input is nil"))
		return nil
	}
	if (input.Shape.Rank != 2 && input.Shape.Rank != 3) || weights.Shape.Rank != 2 {
		b.setError(errors.New("SSMConv requires rank-2/3 input and rank-2 weights"))
		return nil
	}
	if input.Type != dtype.F32 || weights.Type != dtype.F32 {
		b.setError(errors.New("SSMConv currently requires F32 inputs"))
		return nil
	}
	kernelSize := weights.Shape.Dims[0]
	if kernelSize == 0 || input.Shape.Dims[0] < kernelSize {
		b.setError(errors.New("SSMConv kernel exceeds input window"))
		return nil
	}
	if input.Shape.Dims[1] != weights.Shape.Dims[1] {
		b.setError(errors.New("SSMConv channel counts differ"))
		return nil
	}
	tokens := input.Shape.Dims[0] - kernelSize + 1
	var shape Shape
	var err error
	if input.Shape.Rank == 2 {
		shape, err = NewShape(input.Shape.Dims[1], tokens)
	} else {
		shape, err = NewShape(input.Shape.Dims[1], tokens, input.Shape.Dims[2])
	}
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", dtype.F32, shape, OpSSMConv, []*Tensor{input, weights}, nil)
}

// SSMScan: selective state update; packed output then final state.
func (b *Builder) SSMScan(state, x, dt, a, beta, c *Tensor) *Tensor {
	if b.err != nil {
		return nil
	}
	inputs := []*Tensor{state, x, dt, a, beta, c}
	for _, input := range inputs {
		if input == nil || input.Type != dtype.F32 {
			b.setError(errors.New("SSMScan requires six F32 inputs"))
			return nil
		}
	}
	if state.Shape.Rank != 4 || x.Shape.Rank != 4 || dt.Shape.Rank != 3 ||
		a.Shape.Rank != 2 || beta.Shape.Rank != 4 || c.Shape.Rank != 4 {
		b.setError(errors.New("SSMScan input ranks are invalid"))
		return nil
	}
	stateWidth := state.Shape.Dims[0]
	dimension := state.Shape.Dims[1]
	heads := state.Shape.Dims[2]
	sequences := state.Shape.Dims[3]
	tokens := x.Shape.Dims[2]
	groups := beta.Shape.Dims[1]
	if stateWidth == 0 || dimension == 0 || heads == 0 || sequences == 0 || tokens == 0 || groups == 0 ||
		x.Shape.Dims[0] != dimension || x.Shape.Dims[1] != heads || x.Shape.Dims[3] != sequences ||
		dt.Shape.Dims[0] != heads || dt.Shape.Dims[1] != tokens || dt.Shape.Dims[2] != sequences ||
		(a.Shape.Dims[0] != 1 && a.Shape.Dims[0] != stateWidth) || a.Shape.Dims[1] != heads || heads%groups != 0 ||
		beta.Shape.Dims[0] != stateWidth || beta.Shape.Dims[2] != tokens || beta.Shape.Dims[3] != sequences ||
		!beta.Shape.Equal(c.Shape) {
		b.setError(errors.New("SSMScan input shapes are incompatible"))
		return nil
	}
	attentionElements := dimension * heads * tokens * sequences
	stateElements := stateWidth * dimension * heads * sequences
	shape, err := NewShape(attentionElements + stateElements)
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", dtype.F32, shape, OpSSMScan, inputs, nil)
}

// GatedDeltaNet: applies llama.cpp's fused K=1 recurrent delta-net update
// Q/K/V are [state, heads, tokens, sequences], scalar or vector gate has
// [1|state, valueHeads, tokens, sequences], beta is [1,valueHeads,tokens,
// sequences], and state is [state,state,valueHeads,sequences]; output
// packs attention values followed by newest state snapshot
func (b *Builder) GatedDeltaNet(q, k, v, gate, beta, state *Tensor) *Tensor {
	return b.gatedDeltaNet(q, k, v, gate, beta, state, false)
}

func (b *Builder) GatedDeltaNetRepeatInterleave(q, k, v, gate, beta, state *Tensor) *Tensor {
	return b.gatedDeltaNet(q, k, v, gate, beta, state, true)
}

func (b *Builder) gatedDeltaNet(
	q, k, v, gate, beta, state *Tensor,
	repeatInterleave bool,
) *Tensor {
	if b.err != nil {
		return nil
	}
	inputs := []*Tensor{q, k, v, gate, beta, state}
	for _, input := range inputs {
		if input == nil || input.Type != dtype.F32 || input.Shape.Rank != 4 {
			b.setError(errors.New("GatedDeltaNet requires six rank-4 F32 inputs"))
			return nil
		}
	}
	size := v.Shape.Dims[0]
	heads := v.Shape.Dims[1]
	tokens := v.Shape.Dims[2]
	sequences := v.Shape.Dims[3]
	if q.Shape.Dims[0] != size || k.Shape.Dims[0] != size ||
		q.Shape.Dims[2] != tokens || k.Shape.Dims[2] != tokens ||
		q.Shape.Dims[3] != sequences || k.Shape.Dims[3] != sequences ||
		heads%q.Shape.Dims[1] != 0 || heads%k.Shape.Dims[1] != 0 {
		b.setError(errors.New("GatedDeltaNet Q/K/V shapes are incompatible"))
		return nil
	}
	if (gate.Shape.Dims[0] != 1 && gate.Shape.Dims[0] != size) ||
		gate.Shape.Dims[1] != heads || gate.Shape.Dims[2] != tokens ||
		gate.Shape.Dims[3] != sequences ||
		beta.Shape.Dims[0] != 1 || beta.Shape.Dims[1] != heads ||
		beta.Shape.Dims[2] != tokens || beta.Shape.Dims[3] != sequences ||
		state.Shape.Dims[0] != size || state.Shape.Dims[1] != size ||
		state.Shape.Dims[2] != heads || state.Shape.Dims[3] != sequences {
		b.setError(errors.New("GatedDeltaNet gate/beta/state shapes are incompatible"))
		return nil
	}
	shape, err := NewShape(size*heads, tokens*sequences+size*sequences)
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add(
		"", dtype.F32, shape, OpGatedDeltaNet, inputs,
		GatedDeltaNetAttributes{RepeatInterleave: repeatInterleave},
	)
}

// GatedLinearAttention: QRWKV linear attention; packed output and state.
func (b *Builder) GatedLinearAttention(key, value, receptance, decay, state *Tensor, scale float32) *Tensor {
	if b.err != nil {
		return nil
	}
	inputs := []*Tensor{key, value, receptance, decay, state}
	for _, input := range inputs {
		if input == nil || input.Type != dtype.F32 {
			b.setError(errors.New("GatedLinearAttention requires five F32 inputs"))
			return nil
		}
	}
	if key.Shape.Rank != 4 || value.Shape.Rank != 4 || receptance.Shape.Rank != 4 ||
		decay.Shape.Rank != 4 || state.Shape.Rank != 4 || !key.Shape.Equal(value.Shape) {
		b.setError(errors.New("GatedLinearAttention input shapes are incompatible"))
		return nil
	}
	width, heads := receptance.Shape.Dims[0], receptance.Shape.Dims[1]
	tokens, sequences := receptance.Shape.Dims[2], receptance.Shape.Dims[3]
	if width == 0 || heads == 0 || tokens == 0 || sequences == 0 ||
		key.Shape.Dims[0] != width || key.Shape.Dims[1] == 0 || heads%key.Shape.Dims[1] != 0 ||
		key.Shape.Dims[2] != tokens || key.Shape.Dims[3] != sequences ||
		!receptance.Shape.Equal(decay.Shape) ||
		state.Shape.Dims[0] != width || state.Shape.Dims[1] != width ||
		state.Shape.Dims[2] != heads || state.Shape.Dims[3] != sequences ||
		math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		b.setError(errors.New("GatedLinearAttention state or scale is invalid"))
		return nil
	}
	shape, err := NewShape(width*heads, tokens*sequences+width*sequences)
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", dtype.F32, shape, OpGatedLinearAttention, inputs, GatedLinearAttentionAttributes{Scale: scale})
}

// RWKV6: classic WKV6 recurrence; packed output and state.
func (b *Builder) RWKV6(key, value, receptance, first, decay, state *Tensor) *Tensor {
	if b.err != nil {
		return nil
	}
	inputs := []*Tensor{key, value, receptance, first, decay, state}
	for _, input := range inputs {
		if input == nil || input.Type != dtype.F32 {
			b.setError(errors.New("RWKV6 requires six F32 inputs"))
			return nil
		}
	}
	if key.Shape.Rank != 4 || !key.Shape.Equal(value.Shape) || !key.Shape.Equal(receptance.Shape) ||
		!key.Shape.Equal(decay.Shape) || first.Shape.Rank != 2 || state.Shape.Rank != 4 {
		b.setError(errors.New("RWKV6 input shapes are incompatible"))
		return nil
	}
	width, heads := key.Shape.Dims[0], key.Shape.Dims[1]
	tokens, sequences := key.Shape.Dims[2], key.Shape.Dims[3]
	if width == 0 || heads == 0 || tokens == 0 || sequences == 0 ||
		first.Shape.Dims[0] != width || first.Shape.Dims[1] != heads ||
		state.Shape.Dims[0] != width || state.Shape.Dims[1] != width ||
		state.Shape.Dims[2] != heads || state.Shape.Dims[3] != sequences {
		b.setError(errors.New("RWKV6 state or time-first shape is invalid"))
		return nil
	}
	shape, err := NewShape(width*heads, tokens*sequences+width*sequences)
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", dtype.F32, shape, OpRWKV6, inputs, nil)
}

// SumRows: first-dimension reduction.
func (b *Builder) SumRows(input *Tensor) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || input.Type != dtype.F32 || input.Shape.Rank == 0 || input.Shape.Dims[0] == 0 {
		b.setError(errors.New("SumRows requires non-empty F32 input"))
		return nil
	}
	dimensions := input.Shape.Slice()
	dimensions[0] = 1
	shape, err := NewShape(dimensions...)
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", dtype.F32, shape, OpSumRows, []*Tensor{input}, nil)
}

// FWHT: orthonormal Walsh-Hadamard transform over dimension zero.
func (b *Builder) FWHT(input *Tensor) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || input.Type != dtype.F32 || input.Shape.Rank == 0 {
		b.setError(errors.New("FWHT requires non-empty F32 input"))
		return nil
	}
	width := input.Shape.Dims[0]
	if width == 0 || width&(width-1) != 0 {
		b.setError(errors.New("FWHT dimension zero must be a power of two"))
		return nil
	}
	return b.add("", dtype.F32, input.Shape, OpFWHT, []*Tensor{input}, nil)
}

// TopK: descending dimension-zero indices; lower index wins ties.
func (b *Builder) TopK(input *Tensor, k uint32) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || input.Type != dtype.F32 || input.Shape.Rank == 0 {
		b.setError(errors.New("TopK requires non-empty F32 input"))
		return nil
	}
	if k == 0 || uint64(k) > input.Shape.Dims[0] {
		b.setError(errors.New("TopK count exceeds dimension zero"))
		return nil
	}
	if input.Shape.Dims[0] > maxExactFloat32Integer {
		b.setError(errors.New("TopK dimension zero exceeds exact F32 index range"))
		return nil
	}
	dimensions := input.Shape.Slice()
	dimensions[0] = uint64(k)
	shape, err := NewShape(dimensions...)
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", dtype.F32, shape, OpTopK, []*Tensor{input}, TopKAttributes{K: k})
}

// TopKPairs: interleaved descending token/logit pairs.
func (b *Builder) TopKPairs(input *Tensor, k uint32) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || input.Type != dtype.F32 || input.Shape.Rank == 0 {
		b.setError(errors.New("TopKPairs requires non-empty F32 input"))
		return nil
	}
	if k == 0 || k > MaxTopKPairs || uint64(k) > input.Shape.Dims[0] {
		b.setError(errors.New("TopKPairs count exceeds dimension zero"))
		return nil
	}
	if input.Shape.Dims[0] > maxExactFloat32Integer {
		b.setError(errors.New("TopKPairs dimension zero exceeds exact F32 index range"))
		return nil
	}
	const chunk = uint32(1024)
	chunks := (input.Shape.Dims[0] + uint64(chunk) - 1) / uint64(chunk)
	partialDimensions := make([]uint64, 0, int(input.Shape.Rank)+2)
	partialDimensions = append(partialDimensions, 2, uint64(k), chunks)
	partialDimensions = append(partialDimensions, input.Shape.Slice()[1:]...)
	partialShape, err := NewShape(partialDimensions...)
	if err != nil {
		b.setError(err)
		return nil
	}
	attributes := TopKAttributes{K: k, Chunk: chunk}
	partials := b.add("", dtype.F32, partialShape, OpTopKPartials, []*Tensor{input}, attributes)
	dimensions := make([]uint64, 0, int(input.Shape.Rank)+1)
	dimensions = append(dimensions, 2, uint64(k))
	dimensions = append(dimensions, input.Shape.Slice()[1:]...)
	shape, err := NewShape(dimensions...)
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", dtype.F32, shape, OpTopKPairs, []*Tensor{partials}, attributes)
}

// GatherLast: dynamic gather from final dimension; exact F32 indices.
func (b *Builder) GatherLast(input, indices *Tensor) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || indices == nil ||
		(input.Type != dtype.F32 && input.Type != dtype.Q8_0) || indices.Type != dtype.F32 ||
		input.Shape.Rank == 0 || indices.Shape.Rank == 0 {
		b.setError(errors.New("GatherLast requires F32 indices and F32/Q8_0 input"))
		return nil
	}
	if int(input.Shape.Rank)-1+int(indices.Shape.Rank) > MaxDimensions {
		b.setError(errors.New("GatherLast output rank exceeds limit"))
		return nil
	}
	dimensions := slices.Clone(input.Shape.Slice()[:input.Shape.Rank-1])
	dimensions = append(dimensions, indices.Shape.Slice()...)
	shape, err := NewShape(dimensions...)
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", dtype.F32, shape, OpGatherLast, []*Tensor{input, indices}, nil)
}

// RWKV7: vector-valued decay recurrence; packed output and state.
func (b *Builder) RWKV7(receptance, decay, key, value, a, bVector, state *Tensor) *Tensor {
	if b.err != nil {
		return nil
	}
	inputs := []*Tensor{receptance, decay, key, value, a, bVector, state}
	for _, input := range inputs {
		if input == nil || input.Type != dtype.F32 {
			b.setError(errors.New("RWKV7 requires seven F32 inputs"))
			return nil
		}
	}
	if receptance.Shape.Rank != 4 || !receptance.Shape.Equal(decay.Shape) ||
		!receptance.Shape.Equal(key.Shape) || !receptance.Shape.Equal(value.Shape) ||
		!receptance.Shape.Equal(a.Shape) || !receptance.Shape.Equal(bVector.Shape) || state.Shape.Rank != 4 {
		b.setError(errors.New("RWKV7 input shapes are incompatible"))
		return nil
	}
	width, heads := key.Shape.Dims[0], key.Shape.Dims[1]
	tokens, sequences := key.Shape.Dims[2], key.Shape.Dims[3]
	if width == 0 || heads == 0 || tokens == 0 || sequences == 0 ||
		state.Shape.Dims[0] != width || state.Shape.Dims[1] != width ||
		state.Shape.Dims[2] != heads || state.Shape.Dims[3] != sequences {
		b.setError(errors.New("RWKV7 state shape is invalid"))
		return nil
	}
	shape, err := NewShape(width*heads, tokens*sequences+width*sequences)
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", dtype.F32, shape, OpRWKV7, inputs, nil)
}
