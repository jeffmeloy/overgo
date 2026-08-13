package routedlm

// Device prefill graph: one branch-routed MoT prompt layer over all prompt rows
// composed as an overgo tensor.Builder graph for the CUDA generic executor.
// Branch routing (text vs vision weight sets, selected per row by the modality
// mask) is expressed as dual-compute-then-select: each branch-weighted op runs
// over ALL rows through its own weights, and a per-row 0/1 F32 mask selects the
// correct branch row-wise (multiply by exactly 1.0/0.0 and add is exact, so the
// selected rows are bit-identical to the host's per-branch scatter). Rope is the
// degenerate single full-width Time-axis section (RoPENeoX at sequential
// positions) — the exact arithmetic PromptLayerForward uses. Attention is the
// per-row segment/causal window pattern (SegmentWindows) expressed as a small
// list of block attentions concatenated on the token axis: bidirectional over a
// visual segment's rows, causal over the surrounding text rows.
//
// The graph mirrors PromptLayerForward op-for-op (input RMSNorm -> Q/K/V ->
// rope-then-QK-norm -> segment/causal attention -> gated output projection
// residual -> post-attn RMSNorm -> SwiGLU MLP residual), and also exposes the
// per-layer resident K (roped+QK-normed) and V that the device decode consumes,
// replacing the golden-seeded PromptResidentKV.

import (
	"fmt"
	"math"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// PrefillAttnBlock: one contiguous query-row run and its key window. Causal
// blocks attend [0, queryPosition] over the full key sequence; bidirectional
// blocks attend all rows of a sliced key window [KeyStart, KeyEnd).
type PrefillAttnBlock struct {
	QStart, QCount   int
	KeyStart, KeyEnd int
	Causal           bool
}

// PrefillAttnBlocks: the SegmentWindows pattern reduced to contiguous block
// attentions. Rows inside a visual segment share one bidirectional window;
// rows outside every segment are causal over the full prefix.
func PrefillAttnBlocks(segments [][2]int, tokens int) []PrefillAttnBlock {
	inSegment := func(row int) (int, int, bool) {
		for _, seg := range segments {
			if row >= seg[0] && row < seg[1] {
				return seg[0], seg[1], true
			}
		}
		return 0, 0, false
	}
	var blocks []PrefillAttnBlock
	row := 0
	for row < tokens {
		s, e, ok := inSegment(row)
		if ok {
			// One bidirectional block for the whole segment slice.
			blocks = append(blocks, PrefillAttnBlock{QStart: s, QCount: e - s, KeyStart: s, KeyEnd: e, Causal: false})
			row = e
			continue
		}
		// Causal run until the next segment (or end).
		start := row
		for row < tokens {
			if _, _, cov := inSegment(row); cov {
				break
			}
			row++
		}
		blocks = append(blocks, PrefillAttnBlock{QStart: start, QCount: row - start, KeyStart: 0, KeyEnd: tokens, Causal: true})
	}
	return blocks
}

// DevicePrefillBranch: one branch's weight feed nodes.
type DevicePrefillBranch struct {
	InputNorm *tensor.Tensor // [H] F32
	Q, K, V   *tensor.Tensor // [H,qOut]/[H,kvOut]/[H,kvOut] BF16
	O         *tensor.Tensor // [qOut,H] BF16
	QNorm     *tensor.Tensor // [hd] F32
	KNorm     *tensor.Tensor // [hd] F32
	PostNorm  *tensor.Tensor // [H] F32
	Gate      *tensor.Tensor // [H,f] BF16
	Up        *tensor.Tensor // [H,f] BF16
	Down      *tensor.Tensor // [f,H] BF16
}

// DevicePrefillLayerGraph: a single branch-routed prompt layer, replayed per
// layer with re-bound weight feeds.
type DevicePrefillLayerGraph struct {
	Builder  *tensor.Builder
	Row      *tensor.Tensor // [H, tokens] F32 host feed (layer input)
	MaskText *tensor.Tensor // [1, tokens] F32 (1 at text rows)
	MaskVis  *tensor.Tensor // [1, tokens] F32 (1 at vision rows)
	Text     DevicePrefillBranch
	Vision   DevicePrefillBranch
	Output   *tensor.Tensor // [H, tokens] F32 layer output
	KeyKV    *tensor.Tensor // [hd, kvHeads, tokens] resident K (roped+QK-normed)
	ValueKV  *tensor.Tensor // [hd, kvHeads, tokens] resident V
	Context  *tensor.Tensor // [qOut, tokens] bf16 attention context (probe)
	Tokens   int
}

func newPrefillBranch(bld *tensor.Builder, tag string, H, qOut, kvOut, hd, f uint64) DevicePrefillBranch {
	return newTypedPrefillBranch(bld, tag, H, qOut, kvOut, hd, f, dtype.BF16)
}

func newTypedPrefillBranch(
	bld *tensor.Builder,
	tag string,
	H, qOut, kvOut, hd, f uint64,
	weightType dtype.Type,
) DevicePrefillBranch {
	return DevicePrefillBranch{
		InputNorm: bld.Input(tag+".input_norm", dtype.F32, tensor.MustShape(H)),
		Q:         bld.Input(tag+".q", weightType, tensor.MustShape(H, qOut)),
		K:         bld.Input(tag+".k", weightType, tensor.MustShape(H, kvOut)),
		V:         bld.Input(tag+".v", weightType, tensor.MustShape(H, kvOut)),
		O:         bld.Input(tag+".o", weightType, tensor.MustShape(qOut, H)),
		QNorm:     bld.Input(tag+".qnorm", dtype.F32, tensor.MustShape(hd)),
		KNorm:     bld.Input(tag+".knorm", dtype.F32, tensor.MustShape(hd)),
		PostNorm:  bld.Input(tag+".post_norm", dtype.F32, tensor.MustShape(H)),
		Gate:      bld.Input(tag+".gate", weightType, tensor.MustShape(H, f)),
		Up:        bld.Input(tag+".up", weightType, tensor.MustShape(H, f)),
		Down:      bld.Input(tag+".down", weightType, tensor.MustShape(f, H)),
	}
}

// BuildDevicePrefillLayer: one branch-routed prompt layer over `tokens` rows
// with the given attention-block plan.
func BuildDevicePrefillLayer(cfg Config, tokens int, blocks []PrefillAttnBlock) (*DevicePrefillLayerGraph, error) {
	rope, err := ropePlanFromConfig(cfg)
	if err != nil {
		return nil, err
	}
	positions := make([]RowPosition, tokens)
	for row := range positions {
		positions[row] = RowPosition{Time: row}
	}
	return buildDevicePrefillLayer(cfg, rope, positions, blocks)
}

// BuildDeviceBlockPrefillLayer emits multi-axis block-causal prefix math.
func BuildDeviceBlockPrefillLayer(
	cfg Config,
	rope RopePlan,
	positions []RowPosition,
) (*DevicePrefillLayerGraph, error) {
	if len(positions) == 0 {
		return nil, fmt.Errorf("routed lm device prefill: empty positions")
	}
	blocks := make([]PrefillAttnBlock, 0, len(positions))
	for start := 0; start < len(positions); {
		end := start + 1
		for end < len(positions) && positions[end].Time == positions[start].Time {
			end++
		}
		blocks = append(blocks, PrefillAttnBlock{
			QStart: start, QCount: end - start, KeyStart: 0, KeyEnd: end,
		})
		start = end
	}
	return buildDevicePrefillLayer(cfg, rope, positions, blocks)
}

func buildDevicePrefillLayer(
	cfg Config,
	rope RopePlan,
	positions []RowPosition,
	blocks []PrefillAttnBlock,
) (*DevicePrefillLayerGraph, error) {
	tokens := len(positions)
	if tokens <= 0 {
		return nil, fmt.Errorf("routed lm device prefill: tokens=%d", tokens)
	}
	if len(blocks) == 0 {
		return nil, fmt.Errorf("routed lm device prefill: empty attention block plan")
	}
	H := uint64(cfg.HiddenSize)
	hd := uint64(cfg.HeadDim)
	qHeads := uint64(cfg.NumAttentionHeads)
	kvHeads := uint64(cfg.NumKeyValueHeads)
	qOut := qHeads * hd
	kvOut := kvHeads * hd
	f := uint64(cfg.IntermediateSize)
	eps := float32(cfg.RMSNormEps)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	n := uint64(tokens)

	b := tensor.NewBuilder()
	g := &DevicePrefillLayerGraph{Builder: b, Tokens: tokens}
	g.Row = b.Input("row", dtype.F32, tensor.MustShape(H, n))
	g.MaskText = b.Input("mask_text", dtype.F32, tensor.MustShape(1, n))
	g.MaskVis = b.Input("mask_vis", dtype.F32, tensor.MustShape(1, n))
	g.Text = newPrefillBranch(b, "text", H, qOut, kvOut, hd, f)
	g.Vision = newPrefillBranch(b, "vision", H, qOut, kvOut, hd, f)

	// select over [H, tokens] (mask [1,tokens] broadcasts on dim0).
	selectHidden := func(text, vis *tensor.Tensor) *tensor.Tensor {
		return b.Add(b.Multiply(text, g.MaskText), b.Multiply(vis, g.MaskVis))
	}
	// select over [hd, heads, tokens] (mask [1,1,tokens] broadcasts on dim0,1).
	maskTextQK := b.Reshape(g.MaskText, 1, 1, n)
	maskVisQK := b.Reshape(g.MaskVis, 1, 1, n)
	selectQK := func(text, vis *tensor.Tensor) *tensor.Tensor {
		return b.Add(b.Multiply(text, maskTextQK), b.Multiply(vis, maskVisQK))
	}

	// ---- input RMSNorm + Q/K/V projections, branch-selected -----------------
	normT := b.WeightedRMSNorm(g.Row, g.Text.InputNorm, eps)
	normV := b.WeightedRMSNorm(g.Row, g.Vision.InputNorm, eps)
	q := selectHidden(b.MulMat(g.Text.Q, normT), b.MulMat(g.Vision.Q, normV))
	k := selectHidden(b.MulMat(g.Text.K, normT), b.MulMat(g.Vision.K, normV))
	v := selectHidden(b.MulMat(g.Text.V, normT), b.MulMat(g.Vision.V, normV))

	qh := b.Reshape(q, hd, qHeads, n)
	kh := b.Reshape(k, hd, kvHeads, n)
	vh := b.Reshape(v, hd, kvHeads, n)

	// rope (shared) then per-branch QK-norm, selected.
	qRope, err := rope.DeviceApply(b, qh, positions)
	if err != nil {
		return nil, err
	}
	qh = selectQK(b.WeightedRMSNorm(qRope, g.Text.QNorm, eps), b.WeightedRMSNorm(qRope, g.Vision.QNorm, eps))
	kRope, err := rope.DeviceApply(b, kh, positions)
	if err != nil {
		return nil, err
	}
	kh = selectQK(b.WeightedRMSNorm(kRope, g.Text.KNorm, eps), b.WeightedRMSNorm(kRope, g.Vision.KNorm, eps))
	g.KeyKV = kh
	g.ValueKV = vh

	// ---- attention: contiguous block plan, concatenated on the token axis ---
	var context *tensor.Tensor
	for _, blk := range blocks {
		qBlk := b.FlatSlice(qh, uint64(blk.QStart)*hd*qHeads, hd, qHeads, uint64(blk.QCount))
		var kBlk, vBlk *tensor.Tensor
		opts := tensor.AttentionOptions{Scale: scale, NaiveF32: true}
		if blk.Causal {
			kBlk, vBlk = kh, vh
			opts.Causal = true
			opts.QueryStart = uint32(blk.QStart)
		} else {
			width := uint64(blk.KeyEnd - blk.KeyStart)
			kBlk = b.FlatSlice(kh, uint64(blk.KeyStart)*hd*kvHeads, hd, kvHeads, width)
			vBlk = b.FlatSlice(vh, uint64(blk.KeyStart)*hd*kvHeads, hd, kvHeads, width)
		}
		attn := b.Reshape(b.AttentionWithOptions(qBlk, kBlk, vBlk, opts), qOut, uint64(blk.QCount))
		if context == nil {
			context = attn
		} else {
			context = b.Concat(context, attn, 1)
		}
	}
	context = b.BF16Round(context) // host bf16-rounds each head's context
	g.Context = context

	// ---- output projection + residual + post-norm + SwiGLU MLP, selected ----
	proj := selectHidden(b.MulMat(g.Text.O, context), b.MulMat(g.Vision.O, context))
	residual := b.Add(g.Row, proj)
	postNorm := selectHidden(
		b.WeightedRMSNorm(residual, g.Text.PostNorm, eps),
		b.WeightedRMSNorm(residual, g.Vision.PostNorm, eps),
	)
	mlp := selectHidden(
		b.MulMat(g.Text.Down, b.SwiGLU(b.MulMat(g.Text.Gate, postNorm), b.MulMat(g.Text.Up, postNorm))),
		b.MulMat(g.Vision.Down, b.SwiGLU(b.MulMat(g.Vision.Gate, postNorm), b.MulMat(g.Vision.Up, postNorm))),
	)
	g.Output = b.Add(residual, mlp)

	if err := b.Err(); err != nil {
		return nil, fmt.Errorf("routed lm device prefill graph: %w", err)
	}
	return g, nil
}
