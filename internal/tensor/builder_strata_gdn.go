package tensor

import (
	"errors"
	"math"

	"overgo/internal/tensor/dtype"
)

// Strata's verify kernels for a gated-delta layer's window (src/kernels/cuda/verify_kernels.cu, Strata at 296b59d2,
// MIT): the window's conv and recurrence read the layer's history and state and leave them; the commit advances both in
// place over the tokens a window keeps, a count read from device memory. The geometry is the one the kernels are
// compiled for: a head's state 128 wide, a convolution of 4 taps, at most 8 tokens a window (kVerifyMaxT).
const (
	// GatedDeltaWindowWidth is a head's state width, its keys and its values.
	GatedDeltaWindowWidth = 128
	// GatedDeltaWindowKernel is the convolution's taps.
	GatedDeltaWindowKernel = 4
	// GatedDeltaWindowTokens is the most tokens a window holds.
	GatedDeltaWindowTokens = 8
)

// GatedDeltaWindowConv's operands, in order.
const (
	// GatedDeltaWindowConvHistory is the convolution's history, [3, channels].
	GatedDeltaWindowConvHistory = iota
	// GatedDeltaWindowConvQKV is the window's qkv rows, [channels, tokens].
	GatedDeltaWindowConvQKV
	// GatedDeltaWindowConvWeights is the convolution's taps, [4, channels].
	GatedDeltaWindowConvWeights
	// GatedDeltaWindowConvOperands is the operand count.
	GatedDeltaWindowConvOperands
)

// GatedDeltaWindowStep's operands, in order.
const (
	// GatedDeltaWindowStepState is the layer's state, [width, width, value heads].
	GatedDeltaWindowStepState = iota
	// GatedDeltaWindowStepRows is the window's convolved rows, [channels, tokens].
	GatedDeltaWindowStepRows
	// GatedDeltaWindowStepGate is the decay gate, [value heads, tokens].
	GatedDeltaWindowStepGate
	// GatedDeltaWindowStepBeta is the update's strength, [value heads, tokens].
	GatedDeltaWindowStepBeta
	// GatedDeltaWindowStepZ is the output gate's projection, [width, value heads, tokens].
	GatedDeltaWindowStepZ
	// GatedDeltaWindowStepGamma is the output norm's weights, [width].
	GatedDeltaWindowStepGamma
	// GatedDeltaWindowStepOperands is the operand count.
	GatedDeltaWindowStepOperands
)

// GatedDeltaCommit's operands, in order; an earlier commit's word follows them when it is given.
const (
	// GatedDeltaCommitHistory is the convolution's history, advanced in place.
	GatedDeltaCommitHistory = iota
	// GatedDeltaCommitState is the layer's state, advanced in place.
	GatedDeltaCommitState
	// GatedDeltaCommitQKV is the window's qkv rows.
	GatedDeltaCommitQKV
	// GatedDeltaCommitRows is the window's convolved rows.
	GatedDeltaCommitRows
	// GatedDeltaCommitGate is the window's decay gate.
	GatedDeltaCommitGate
	// GatedDeltaCommitBeta is the window's update strength.
	GatedDeltaCommitBeta
	// GatedDeltaCommitKept is the kept count, an I32 [1].
	GatedDeltaCommitKept
	// GatedDeltaCommitOperands is the operand count without an earlier commit's word.
	GatedDeltaCommitOperands
)

// GatedDeltaWindowAttributes declares a gated-delta window's key and value heads and the epsilon of its norms.
type GatedDeltaWindowAttributes struct {
	KeyHeads, ValueHeads uint32
	Epsilon              float32
	// Commit has a one-token window's conv keep its token in the history (Strata's one-token self-commit).
	Commit bool
	// Layers is the layers a commit advances (Builder.GatedDeltaCommitLayers), zero where a node predates it.
	Layers uint32
}

// CommitLayers is the layers a commit advances, one for a node that names none.
func (a GatedDeltaWindowAttributes) CommitLayers() int {
	return max(int(a.Layers), SingletonExtent)
}

// channels is the layer's convolved channels: the query's and the key's heads and the value's.
func (a GatedDeltaWindowAttributes) channels() uint64 {
	return (PairedExtent*uint64(a.KeyHeads) + uint64(a.ValueHeads)) * GatedDeltaWindowWidth
}

// GatedDeltaWindowConv is Strata's gdn_conv_l2_multi over a window: for history [3, channels] (a channel's last three
// rows), qkv [channels, tokens] and weights [4, channels], each channel's value at a token is SiLU of the taps over the
// four rows that end at it, the query's and the key's heads each scaled by rsqrt(their squared norm + epsilon); the
// output is [channels, tokens]. The history is left as it was, except that a one-token window that commits keeps its
// token in it, in place.
func (b *Builder) GatedDeltaWindowConv(history, qkv, weights *Tensor, keyHeads, valueHeads uint32, epsilon float32, commit bool) *Tensor {
	if b.err != nil {
		return nil
	}
	attributes := GatedDeltaWindowAttributes{KeyHeads: keyHeads, ValueHeads: valueHeads, Epsilon: epsilon, Commit: commit}
	channels := attributes.channels()
	if history == nil || qkv == nil || weights == nil || history.Type != dtype.F32 || qkv.Type != dtype.F32 ||
		weights.Type != dtype.F32 || keyHeads == 0 || valueHeads == 0 || history.Shape != MustShape(GatedDeltaWindowKernel-1, channels) ||
		weights.Shape != MustShape(GatedDeltaWindowKernel, channels) || !gatedDeltaWindowRows(qkv, channels) ||
		commit && qkv.Shape.Dims[1] != SingletonExtent {
		b.setError(errors.New("gated-delta window convolution operands are invalid"))
		return nil
	}
	return b.add("", dtype.F32, qkv.Shape, OpGatedDeltaWindowConv, []*Tensor{history, qkv, weights}, attributes)
}

// GatedDeltaWindowStep is Strata's gdn_step_norm_multi with every output: for state [width, width, value heads] (a value
// head's values, each its keys), rows the window's convolved rows [channels, tokens], gate and beta [value heads,
// tokens], z [width, value heads, tokens] and gamma [width], it runs the recurrence from the state over the rows - the
// query and key head of value head h is h modulo the key heads - each output scaled by rsqrt(width), normed by its head's
// mean square (gamma, epsilon) and gated by sigmoid(z); the output is [width, value heads, tokens], and the state is
// left as it was. Given kept, an I32 [1] holding n, it is the kernel's kept form: the outputs of the first n tokens and
// the state after them, written in place (a one-token window's own commit).
func (b *Builder) GatedDeltaWindowStep(state, rows, gate, beta, z, gamma, kept *Tensor, keyHeads, valueHeads uint32, epsilon float32) *Tensor {
	if b.err != nil {
		return nil
	}
	attributes := GatedDeltaWindowAttributes{KeyHeads: keyHeads, ValueHeads: valueHeads, Epsilon: epsilon}
	channels := attributes.channels()
	if !gatedDeltaWindowState(state, valueHeads) || !gatedDeltaWindowRows(rows, channels) || gamma == nil ||
		gamma.Type != dtype.F32 || gamma.Shape != MustShape(GatedDeltaWindowWidth) ||
		kept != nil && (kept.Type != dtype.I32 || kept.Shape != MustShape(SingletonExtent)) {
		b.setError(errors.New("gated-delta window step operands are invalid"))
		return nil
	}
	tokens := rows.Shape.Dims[1]
	for _, operand := range []*Tensor{gate, beta} {
		if !gatedDeltaWindowElements(operand, uint64(valueHeads)*tokens) {
			b.setError(errors.New("gated-delta window step gates are invalid"))
			return nil
		}
	}
	if !gatedDeltaWindowElements(z, GatedDeltaWindowWidth*uint64(valueHeads)*tokens) {
		b.setError(errors.New("gated-delta window step gate projection is invalid"))
		return nil
	}
	inputs := []*Tensor{state, rows, gate, beta, z, gamma}
	if kept != nil {
		inputs = append(inputs, kept)
	}
	return b.add("", dtype.F32, MustShape(GatedDeltaWindowWidth, uint64(valueHeads), tokens), OpGatedDeltaWindowStep, inputs, attributes)
}

// GatedDeltaCommit is Strata's commit of a gated-delta layer (gdn_conv_commit and gdn_step_commit): with kept an I32
// [1] holding n, the history becomes the last three rows of [history | qkv's first n tokens] and the state the
// recurrence's over rows' first n tokens with their gate and beta, both in place; the output is an F32 [1] word holding
// the count's bits, written last, which a later commit may take as after (nil for none) so the commits run in order.
func (b *Builder) GatedDeltaCommit(after, history, state, qkv, rows, gate, beta, kept *Tensor, keyHeads, valueHeads uint32) *Tensor {
	return b.GatedDeltaCommitLayers(after, []GatedDeltaCommitLayer{{
		History: history, State: state, QKV: qkv, Rows: rows, Gate: gate, Beta: beta,
	}}, kept, keyHeads, valueHeads)
}

// GatedDeltaCommitLayer is a layer's operands in a commit of several (Builder.GatedDeltaCommitLayers), each as
// GatedDeltaCommit takes it.
type GatedDeltaCommitLayer struct {
	History, State, QKV, Rows, Gate, Beta *Tensor
}

// GatedDeltaCommitLayers is GatedDeltaCommit over several layers of the same heads and window in one node, Strata's
// gdn_commit_layers: each layer's history and state advanced in place over the kept count's tokens of its own rows. Its
// operands are each layer's in GatedDeltaCommit's order (GatedDeltaCommitKept of them a layer), then kept and after.
func (b *Builder) GatedDeltaCommitLayers(after *Tensor, layers []GatedDeltaCommitLayer, kept *Tensor, keyHeads, valueHeads uint32) *Tensor {
	if b.err != nil {
		return nil
	}
	attributes := GatedDeltaWindowAttributes{KeyHeads: keyHeads, ValueHeads: valueHeads, Layers: uint32(len(layers))}
	channels := attributes.channels()
	if len(layers) == 0 || uint64(len(layers)) > math.MaxUint32 || kept == nil || kept.Type != dtype.I32 ||
		kept.Shape != MustShape(SingletonExtent) || after != nil && (after.Type != dtype.F32 || after.Shape != kept.Shape) {
		b.setError(errors.New("gated-delta commit operands are invalid"))
		return nil
	}
	inputs := make([]*Tensor, 0, len(layers)*GatedDeltaCommitKept+PairedExtent)
	for _, layer := range layers {
		if layer.History == nil || layer.History.Type != dtype.F32 || layer.History.Shape != MustShape(GatedDeltaWindowKernel-1, channels) ||
			!gatedDeltaWindowState(layer.State, valueHeads) || !gatedDeltaWindowRows(layer.QKV, channels) ||
			!gatedDeltaWindowRows(layer.Rows, channels) || layer.QKV.Shape != layer.Rows.Shape || layer.Rows.Shape != layers[0].Rows.Shape {
			b.setError(errors.New("gated-delta commit operands are invalid"))
			return nil
		}
		tokens := layer.Rows.Shape.Dims[1]
		for _, operand := range []*Tensor{layer.Gate, layer.Beta} {
			if !gatedDeltaWindowElements(operand, uint64(valueHeads)*tokens) {
				b.setError(errors.New("gated-delta commit gates are invalid"))
				return nil
			}
		}
		inputs = append(inputs, layer.History, layer.State, layer.QKV, layer.Rows, layer.Gate, layer.Beta)
	}
	inputs = append(inputs, kept)
	if after != nil {
		inputs = append(inputs, after)
	}
	return b.add("", dtype.F32, kept.Shape, OpGatedDeltaCommit, inputs, attributes)
}

// gatedDeltaWindowState reports whether state is a gated-delta layer's F32 state [width, width, value heads].
func gatedDeltaWindowState(state *Tensor, valueHeads uint32) bool {
	return state != nil && state.Type == dtype.F32 &&
		(state.Shape == MustShape(GatedDeltaWindowWidth, GatedDeltaWindowWidth, uint64(valueHeads)) ||
			state.Shape == MustShape(GatedDeltaWindowWidth, GatedDeltaWindowWidth, uint64(valueHeads), SingletonExtent))
}

// gatedDeltaWindowElements reports whether operand is an F32 tensor of count elements.
func gatedDeltaWindowElements(operand *Tensor, count uint64) bool {
	if operand == nil || operand.Type != dtype.F32 {
		return false
	}
	elements, err := operand.Shape.Elements()
	return err == nil && elements == count
}

func (GatedDeltaWindowAttributes) validFor(op Op) bool {
	return sameOp(op, OpGatedDeltaWindowConv) || sameOp(op, OpGatedDeltaWindowStep) || sameOp(op, OpGatedDeltaCommit)
}

// gatedDeltaWindowRows reports whether rows is a window's F32 [channels, tokens], 1 to GatedDeltaWindowTokens tokens.
func gatedDeltaWindowRows(rows *Tensor, channels uint64) bool {
	return rows != nil && rows.Type == dtype.F32 && rows.Shape.Rank == PairedExtent && rows.Shape.Dims[0] == channels &&
		channels%GatedDeltaWindowWidth == 0 && rows.Shape.Dims[1] > 0 && rows.Shape.Dims[1] <= GatedDeltaWindowTokens
}

// CopyIndexed's operands, in order; an earlier commit's word follows them when it is given.
const (
	// CopyIndexedTarget is the values written in place.
	CopyIndexedTarget = iota
	// CopyIndexedSource is the rows copied from.
	CopyIndexedSource
	// CopyIndexedIndex is the first row copied, an I32 [1].
	CopyIndexedIndex
	// CopyIndexedOperands is the operand count without an earlier commit's word.
	CopyIndexedOperands
)

// CopyIndexedAttributes are a CopyIndexed's row stride, in values.
type CopyIndexedAttributes struct {
	Stride uint64
}

func (CopyIndexedAttributes) validFor(op Op) bool { return sameOp(op, OpCopyIndexed) }

// CopyIndexed is Strata's copy_indexed (verify_kernels.cu), a commit's copy of a window's history after its kept
// tokens: with index an I32 [1] holding i, target's values become source's from row i on, rows of stride values, in
// place; a negative index copies nothing. The output is an F32 [1] word holding the index's bits, written last, which
// a later commit may take as after (nil for none) so the commits run in order.
func (b *Builder) CopyIndexed(after, target, source, index *Tensor, stride uint64) *Tensor {
	if b == nil || b.err != nil {
		return nil
	}
	if target == nil || source == nil || index == nil {
		b.setError(errors.New("indexed copy operands are missing"))
		return nil
	}
	targetValues, targetErr := target.Shape.Elements()
	sourceValues, sourceErr := source.Shape.Elements()
	if target.Type != dtype.F32 || source.Type != dtype.F32 || targetErr != nil || sourceErr != nil || stride == 0 || targetValues > sourceValues ||
		index.Type != dtype.I32 || index.Shape != MustShape(SingletonExtent) ||
		after != nil && (after.Type != dtype.F32 || after.Shape != MustShape(SingletonExtent)) {
		b.setError(errors.New("indexed copy operands are invalid"))
		return nil
	}
	inputs := []*Tensor{target, source, index}
	if after != nil {
		inputs = append(inputs, after)
	}
	return b.add("", dtype.F32, MustShape(SingletonExtent), OpCopyIndexed, inputs, CopyIndexedAttributes{Stride: stride})
}
