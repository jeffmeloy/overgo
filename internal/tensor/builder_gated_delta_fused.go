package tensor

import (
	"errors"

	"overgo/internal/tensor/dtype"
)

const (
	// GatedDeltaFusedTokens is the most tokens a gated-delta layer prepares with GatedDeltaConv and GatedDeltaGates, a
	// decode step's or a verify span's; a prompt's tokens take the composed operations.
	GatedDeltaFusedTokens = 8
	// gatedDeltaFusedKernel is the largest convolution kernel the fused preparation holds, a thread's window.
	gatedDeltaFusedKernel = 8
	// gatedDeltaFusedStateWidth is the widest state the fused preparation holds, a block's head; a power of two below
	// it sums its norm as l2_norm_f32's tree does.
	gatedDeltaFusedStateWidth = 1024
)

// GatedDeltaConv's operands, in order.
const (
	// GatedDeltaConvState is the convolution's history.
	GatedDeltaConvState = iota
	// GatedDeltaConvQKV is the tokens' qkv rows.
	GatedDeltaConvQKV
	// GatedDeltaConvTaps is the convolution's taps.
	GatedDeltaConvTaps
	// GatedDeltaConvOperands is the operand count.
	GatedDeltaConvOperands
)

// GatedDeltaGates' operands, in order.
const (
	// GatedDeltaGatesInput is the layer's normalized input.
	GatedDeltaGatesInput = iota
	// GatedDeltaGatesAlpha is the decay gate's projection.
	GatedDeltaGatesAlpha
	// GatedDeltaGatesBeta is the beta's projection.
	GatedDeltaGatesBeta
	// GatedDeltaGatesTimeStep is the decay gate's time step.
	GatedDeltaGatesTimeStep
	// GatedDeltaGatesDecay is the decay gate's scale.
	GatedDeltaGatesDecay
	// GatedDeltaGatesOperands is the operand count.
	GatedDeltaGatesOperands
)

// GatedDeltaFused reports whether a gated-delta layer of this state width and convolution kernel, over these tokens and
// with its convolution, alpha and beta weights stored as conv, alpha and beta, prepares with GatedDeltaConv and
// GatedDeltaGates: F32 weights, a kernel of 2 to gatedDeltaFusedKernel taps, a power-of-two state width up to
// gatedDeltaFusedStateWidth and at most GatedDeltaFusedTokens tokens. The CUDA kernels take exactly these.
func GatedDeltaFused(stateWidth, kernel, tokens uint64, conv, alpha, beta dtype.Type) bool {
	return conv == dtype.F32 && alpha == dtype.F32 && beta == dtype.F32 && kernel >= PairedExtent && kernel <= gatedDeltaFusedKernel &&
		stateWidth > 0 && stateWidth&(stateWidth-1) == 0 && stateWidth <= gatedDeltaFusedStateWidth &&
		tokens > 0 && tokens <= GatedDeltaFusedTokens
}

// GatedDeltaConvLayout is where a GatedDeltaConv output holds each part, in elements: the query [width, key heads,
// tokens], the key [width, key heads, tokens], the value [width, value heads, tokens], then the convolution window after
// each token [kernel - 1, channels] in token order.
type GatedDeltaConvLayout struct {
	StateWidth, KeyHeads, ValueHeads, Kernel, Tokens uint64
}

// Channels is the layer's convolved channels: the query's and the key's heads and the value's.
func (l GatedDeltaConvLayout) Channels() uint64 {
	return (PairedExtent*l.KeyHeads + l.ValueHeads) * l.StateWidth
}

// Key is the key's first element; the query's is the first.
func (l GatedDeltaConvLayout) Key() uint64 { return l.KeyHeads * l.StateWidth * l.Tokens }

// Value is the value's first element.
func (l GatedDeltaConvLayout) Value() uint64 { return PairedExtent * l.Key() }

// Window is the first element of the convolution window after the token-th token (from zero).
func (l GatedDeltaConvLayout) Window(token uint64) uint64 {
	return l.Channels() * (l.Tokens + token*(l.Kernel-1))
}

// Elements is the output's element count.
func (l GatedDeltaConvLayout) Elements() uint64 { return l.Channels() * l.Tokens * l.Kernel }

// GatedDeltaConv prepares a gated-delta layer's query, key and value, the reads buildGatedDeltaMixCached composes from
// a transpose, a concatenation, slices, SSMConv, SiLU and L2Norm, in one operation. For state [kernel - 1, channels]
// (the history's last kernel - 1 rows a channel), qkv [channels, tokens] and weights [kernel, channels], each channel's
// value at a token is SiLU of the weights' taps over the kernel rows that end at it, the query's and the key's heads are
// each divided by max(their L2 norm, epsilon), and the output holds them in GatedDeltaConvLayout's order with the
// window of the last kernel - 1 rows after each token.
func (b *Builder) GatedDeltaConv(state, qkv, weights *Tensor, stateWidth, keyHeads, valueHeads uint32, epsilon float32) *Tensor {
	if b.err != nil {
		return nil
	}
	if state == nil || qkv == nil || weights == nil || state.Type != dtype.F32 || qkv.Type != dtype.F32 || weights.Type != dtype.F32 ||
		weights.Shape.Rank != PairedExtent || qkv.Shape.Rank != PairedExtent {
		b.setError(errors.New("gated-delta convolution operands are invalid"))
		return nil
	}
	layout := GatedDeltaConvLayout{
		StateWidth: uint64(stateWidth), KeyHeads: uint64(keyHeads), ValueHeads: uint64(valueHeads),
		Kernel: weights.Shape.Dims[0], Tokens: qkv.Shape.Dims[1],
	}
	channels := layout.Channels()
	if layout.Kernel < PairedExtent || keyHeads == 0 || valueHeads == 0 || weights.Shape.Dims[1] != channels ||
		qkv.Shape.Dims[0] != channels || layout.Tokens == 0 || state.Shape != MustShape(layout.Kernel-1, channels) {
		b.setError(errors.New("gated-delta convolution shapes are incompatible"))
		return nil
	}
	shape, err := NewShape(layout.Elements())
	if err != nil {
		b.setError(err)
		return nil
	}
	return b.add("", dtype.F32, shape, OpGatedDeltaConv, []*Tensor{state, qkv, weights}, GatedDeltaConvAttributes{
		StateWidth: stateWidth, KeyHeads: keyHeads, ValueHeads: valueHeads, Epsilon: epsilon,
	})
}

// GatedDeltaGates projects a gated-delta layer's decay gate and beta from input [embedding, tokens], the reads
// buildGatedDeltaMixCached composes from two products, an addition, Softplus, a product and Sigmoid, in one operation:
// with alpha and beta [embedding, heads] and timeStep and decay [heads], the output holds the gate softplus(alpha ·
// input + timeStep) * decay [heads, tokens], then the beta sigmoid(beta · input) [heads, tokens].
func (b *Builder) GatedDeltaGates(input, alpha, beta, timeStep, decay *Tensor) *Tensor {
	if b.err != nil {
		return nil
	}
	operands := []*Tensor{input, alpha, beta, timeStep, decay}
	for _, operand := range operands {
		if operand == nil || operand.Type != dtype.F32 {
			b.setError(errors.New("gated-delta gate operands are invalid"))
			return nil
		}
	}
	if input.Shape.Rank != PairedExtent || alpha.Shape.Rank != PairedExtent {
		b.setError(errors.New("gated-delta gate shapes are incompatible"))
		return nil
	}
	embedding, heads, tokens := input.Shape.Dims[0], alpha.Shape.Dims[1], input.Shape.Dims[1]
	if alpha.Shape.Dims[0] != embedding || beta.Shape != alpha.Shape || timeStep.Shape != MustShape(heads) ||
		decay.Shape != MustShape(heads) || heads == 0 || tokens == 0 {
		b.setError(errors.New("gated-delta gate shapes are incompatible"))
		return nil
	}
	return b.add("", dtype.F32, MustShape(PairedExtent*heads*tokens), OpGatedDeltaGates, operands, nil)
}
