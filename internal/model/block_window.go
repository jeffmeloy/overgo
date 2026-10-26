package model

import (
	"errors"

	"overgo/internal/gguf"
	"overgo/internal/tensor"
)

// WindowBinding is a verify window in Strata's verifier (src/core/verify.cpp record_window, Strata at 296b59d2, MIT):
// a layer reads its fixed state where it sits - PastKey, PastValue and PastStates are the sequence's own - and leaves
// it for the window's commit, except in a one-token window, whose layers advance it themselves (one_token_self_commit);
// each layer returns the rows its commit replays (DenseBlockResult.Window).
type WindowBinding struct {
	// One is an I32 [1] holding 1, a one-token window's kept count.
	One *tensor.Tensor
	// Snapshot is a QSA layer's tail snapshot (128, 3), where the window copies the tail it found.
	Snapshot *tensor.Tensor
}

// HoldsGatedDeltaWindow reports whether a recurrent layer steps a verify window by Strata's verify kernels
// (buildGatedDeltaWindow): a gated-delta layer of separate projections gated by sigmoid, at the state width and the
// convolution the kernels are compiled for, its convolution, alpha and beta stored F32 for the fused gates.
func (s Spec) HoldsGatedDeltaWindow(plan LayerPlan, weights LayerWeights) bool {
	f32 := func(info *gguf.TensorInfo) bool { return info != nil && info.Type == gguf.DTypeF32 }
	return plan.Recurrent && plan.AttentionGraph.deltaProjection == gatedDeltaSeparateSigmoid &&
		s.SSMStateSize == tensor.GatedDeltaWindowWidth && s.SSMConvKernel == tensor.GatedDeltaWindowKernel &&
		f32(weights.SSMConv1D) && f32(weights.SSMAlpha) && f32(weights.SSMBeta)
}

// LayerWindow is the rows a layer's verify window keeps for its commit (verify.cpp's qkv_L_, h_L_, gate_L_ and beta_L_,
// idx_raw_L_ and hist_snap_): a gated-delta layer's qkv rows, convolved rows and packed gates, a QSA layer's raw
// indexer keys, the lexical injection's convolution rows after its history. Kept is a one-token window's own commit of
// a state no other output reads, which the window's graph must run.
type LayerWindow struct {
	QKV, Rows, Gates *tensor.Tensor
	Raw              *tensor.Tensor
	Full             *tensor.Tensor
	Kept             *tensor.Tensor
}

// LayerCommitInputs are a layer's commit's inputs: the fixed state where it sits (Key, Value and States, as the layer's
// cache inputs bind it), the rows its window kept (Window), the kept count (Kept, an I32 [1]) and, for a QSA layer,
// each window row's cell (Cells, an I32 [tokens], -1 for a row not kept), its tail snapshot and its cells.
type LayerCommitInputs struct {
	Key, Value *tensor.Tensor
	States     CacheStates[*tensor.Tensor]
	Window     LayerWindow
	Kept       *tensor.Tensor
	Cells      *tensor.Tensor
	Snapshot   *tensor.Tensor
	QSA        *QSABinding
}

// Commit builds a layer's commit of a verify window, Strata's capture_commit for the layer: a gated-delta layer's
// history and state advanced over the kept tokens (tensor.Builder.GatedDeltaCommit), a QSA layer's indexer restored and
// its kept rows' keys appended again (tensor.Builder.QSACommit), the lexical injection's convolution history copied
// from after the kept tokens (tensor.Builder.CopyIndexed); each takes the commit before it, after, and answers its own
// word, or after itself where the layer keeps no fixed state of these kinds.
func (p CompiledLayerProgram) Commit(builder *tensor.Builder, after *tensor.Tensor, inputs LayerCommitInputs, weights LayerGraphWeights) (*tensor.Tensor, error) {
	spec := p.spec
	if inputs.QSA != nil {
		if inputs.Key == nil || inputs.Value == nil || inputs.Snapshot == nil || inputs.Window.Raw == nil || inputs.Cells == nil ||
			weights.IndexerKNorm == nil {
			return nil, errors.New("QSA commit inputs are missing")
		}
		raw := builder.Reshape(inputs.Window.Raw, tensor.QSAIndexerWidth, tensor.SingletonExtent, inputs.Window.Raw.Shape.Dims[1])
		after = builder.QSACommit(after, inputs.Key, inputs.Value, inputs.Snapshot, raw, inputs.Cells,
			weights.IndexerKNorm, inputs.QSA.State, spec.RMSNormEpsilon)
	}
	if inputs.Window.Full != nil {
		history := inputs.States[cacheStateNGramConvolution].Value
		if history == nil || inputs.Kept == nil {
			return nil, errors.New("lexical commit inputs are missing")
		}
		after = builder.CopyIndexed(after, history, inputs.Window.Full, inputs.Kept, inputs.Window.Full.Shape.Dims[0])
	}
	if inputs.Window.Rows != nil {
		if inputs.Key == nil || inputs.Value == nil || inputs.Window.QKV == nil || inputs.Window.Gates == nil || inputs.Kept == nil {
			return nil, errors.New("gated-delta commit inputs are missing")
		}
		gate, beta := windowGates(builder, inputs.Window.Gates, uint64(spec.SSMTimeStepRank), inputs.Window.Rows.Shape.Dims[1])
		after = builder.GatedDeltaCommit(after, inputs.Key, inputs.Value, inputs.Window.QKV, inputs.Window.Rows, gate, beta,
			inputs.Kept, spec.SSMGroupCount, spec.SSMTimeStepRank)
	}
	return after, builder.Err()
}

// buildGatedDeltaWindow is a gated-delta layer's verify window, verify.cpp record_window's gated-delta section: the qkv
// rows, Strata's window conv over the history where it sits, the fused gates, the gate projection and Strata's window
// recurrence over the state where it sits, normed and gated by sigmoid(z), then the output projection. A one-token
// window's conv and recurrence keep its token (the conv's commit, the recurrence's kept form at one); a longer window
// leaves both for its commit, which replays the rows it keeps.
func buildGatedDeltaWindow(
	builder *tensor.Builder, normalized *tensor.Tensor, spec Spec, weights LayerGraphWeights, tokens uint64,
	convState, ssmState *tensor.Tensor, deltaPolicy gatedDeltaPolicy, window *WindowBinding,
) (DenseBlockResult, error) {
	if err := (graphWeights{
		weights.AttentionQKV, weights.AttentionGate, weights.SSMConv1D, weights.SSMAlpha, weights.SSMBeta,
		weights.SSMTimeStep, weights.SSMA, weights.SSMNorm, weights.SSMOutput,
	}).validate("gated-delta window"); err != nil {
		return DenseBlockResult{}, err
	}
	stateWidth, keyHeads, valueHeads := uint64(spec.SSMStateSize), spec.SSMGroupCount, spec.SSMTimeStepRank
	if deltaPolicy != gatedDeltaSeparateSigmoid || stateWidth != tensor.GatedDeltaWindowWidth ||
		spec.SSMConvKernel != tensor.GatedDeltaWindowKernel || tokens == 0 || tokens > tensor.GatedDeltaWindowTokens ||
		!tensor.GatedDeltaFused(stateWidth, uint64(spec.SSMConvKernel), tokens, weights.SSMConv1D.Type, weights.SSMAlpha.Type, weights.SSMBeta.Type) ||
		window.One == nil {
		return DenseBlockResult{}, errors.New("gated-delta window takes the geometry and gates Strata's verify kernels hold")
	}
	commit := tokens == tensor.SingletonExtent
	qkv := builder.MulMat(weights.AttentionQKV, normalized)
	rows := builder.GatedDeltaWindowConv(convState, qkv, weights.SSMConv1D, keyHeads, valueHeads, spec.RMSNormEpsilon, commit)
	gates := builder.GatedDeltaGates(normalized, weights.SSMAlpha, weights.SSMBeta, weights.SSMTimeStep, weights.SSMA)
	gate, beta := windowGates(builder, gates, uint64(valueHeads), tokens)
	z := builder.MulMat(weights.AttentionGate, normalized)
	var kept *tensor.Tensor
	if commit {
		kept = window.One
	}
	outputs := builder.GatedDeltaWindowStep(ssmState, rows, gate, beta, z, weights.SSMNorm, kept, keyHeads, valueHeads, spec.RMSNormEpsilon)
	attention := builder.MulMat(weights.SSMOutput, builder.Reshape(outputs, uint64(spec.SSMInnerSize), tokens))
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: attention, Window: LayerWindow{QKV: qkv, Rows: rows, Gates: gates}}, nil
}

// windowGates are a window's gate and beta in its packed gates (tensor.Builder.GatedDeltaGates): [1, heads, tokens]
// each, the gate first.
func windowGates(builder *tensor.Builder, gates *tensor.Tensor, heads, tokens uint64) (gate, beta *tensor.Tensor) {
	gate = builder.FlatSlice(gates, tensor.FirstOffset, tensor.SingletonExtent, heads, tokens, tensor.SingletonExtent)
	beta = builder.FlatSlice(gates, heads*tokens, tensor.SingletonExtent, heads, tokens, tensor.SingletonExtent)
	return gate, beta
}
