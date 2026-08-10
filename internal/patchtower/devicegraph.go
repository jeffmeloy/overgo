package patchtower

// Device vision-tower graph: the pre-norm attention blocks (pre_block0 ->
// block_last) composed as an overgo tensor.Builder graph for the CUDA generic
// executor. Front-end (conv patch embed + interpolated pos-embed) stays host
// (data-dependent gather; adaptive's device path is likewise host-front-end,
// device-blocks); this graph mirrors BlockForwardStages op-for-op (LayerNorm
// affine -> bf16 -> fused QKV + bias(bf16) -> split -> full bidirectional
// attention -> o-proj + bias(bf16) -> residual(bf16) -> LayerNorm affine ->
// bf16 -> fc1 + bias -> erf-GELU(bf16) -> fc2 + bias(bf16) -> residual(bf16)),
// which is 44/44 fixture-exact on the host. Weights are the checkpoint's F32
// tensors (torch [out,in] row-major == ggml [in,out]); activations stay F32
// carrying bf16-rounded values at the same points the host rounds.

import (
	"fmt"
	"math"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// DeviceBlockInputs: one block's F32 weight/bias feed nodes.
type DeviceBlockInputs struct {
	Norm1W, Norm1B *tensor.Tensor // [hidden]
	QKVW           *tensor.Tensor // [hidden, 3*hidden]
	QKVB           *tensor.Tensor // [3*hidden]
	ProjW          *tensor.Tensor // [hidden, hidden]
	ProjB          *tensor.Tensor // [hidden]
	Norm2W, Norm2B *tensor.Tensor // [hidden]
	FC1W           *tensor.Tensor // [hidden, inter]
	FC1B           *tensor.Tensor // [inter]
	FC2W           *tensor.Tensor // [inter, hidden]
	FC2B           *tensor.Tensor // [hidden]
}

// DeviceVisionGraph: the compiled-ready vision-blocks graph and its handles.
type DeviceVisionGraph struct {
	Builder   *tensor.Builder
	PreBlock0 *tensor.Tensor // [hidden, rows] F32 host feed
	// block0 sub-stage probes (bf16-rounded, [width, rows] column-major ==
	// host [rows, width] token-major flat).
	Block0Norm1 *tensor.Tensor // [hidden, rows]
	Block0QKV   *tensor.Tensor // [3*hidden, rows]
	Block0Attn  *tensor.Tensor // [hidden, rows] o-proj + bias
	Block0Out   *tensor.Tensor // [hidden, rows]
	BlockLast   *tensor.Tensor // [hidden, rows]
	Inputs      []DeviceBlockInputs
	Rows        int
}

// BuildDeviceVisionBlocks: the spec.Depth-block pre-norm tower over `rows`
// tokens. pre_block0 is the host-computed conv+pos embed feed [hidden, rows].
func BuildDeviceVisionBlocks(spec Spec, rows int) (*DeviceVisionGraph, error) {
	if rows <= 0 {
		return nil, fmt.Errorf("patch tower device blocks: rows=%d", rows)
	}
	if spec.Hidden <= 0 || spec.Heads <= 0 || spec.Hidden%spec.Heads != 0 || spec.Depth <= 0 || spec.BlockMLPInter <= 0 {
		return nil, fmt.Errorf("patch tower device blocks: bad spec %+v", spec)
	}
	h := uint64(spec.Hidden)
	heads := uint64(spec.Heads)
	hd := h / heads
	inter := uint64(spec.BlockMLPInter)
	n := uint64(rows)
	eps := float32(LayerNormEps)
	scale := float32(1.0 / math.Sqrt(float64(hd)))

	b := tensor.NewBuilder()
	g := &DeviceVisionGraph{Builder: b, Rows: rows}
	g.PreBlock0 = b.Input("pre_block0", dtype.F32, tensor.MustShape(h, n))
	g.Inputs = make([]DeviceBlockInputs, spec.Depth)

	row := g.PreBlock0
	for layer := 0; layer < spec.Depth; layer++ {
		in := DeviceBlockInputs{
			Norm1W: b.Input(fmt.Sprintf("b%d.norm1_w", layer), dtype.F32, tensor.MustShape(h)),
			Norm1B: b.Input(fmt.Sprintf("b%d.norm1_b", layer), dtype.F32, tensor.MustShape(h)),
			QKVW:   b.Input(fmt.Sprintf("b%d.qkv_w", layer), dtype.F32, tensor.MustShape(h, 3*h)),
			QKVB:   b.Input(fmt.Sprintf("b%d.qkv_b", layer), dtype.F32, tensor.MustShape(3*h)),
			ProjW:  b.Input(fmt.Sprintf("b%d.proj_w", layer), dtype.F32, tensor.MustShape(h, h)),
			ProjB:  b.Input(fmt.Sprintf("b%d.proj_b", layer), dtype.F32, tensor.MustShape(h)),
			Norm2W: b.Input(fmt.Sprintf("b%d.norm2_w", layer), dtype.F32, tensor.MustShape(h)),
			Norm2B: b.Input(fmt.Sprintf("b%d.norm2_b", layer), dtype.F32, tensor.MustShape(h)),
			FC1W:   b.Input(fmt.Sprintf("b%d.fc1_w", layer), dtype.F32, tensor.MustShape(h, inter)),
			FC1B:   b.Input(fmt.Sprintf("b%d.fc1_b", layer), dtype.F32, tensor.MustShape(inter)),
			FC2W:   b.Input(fmt.Sprintf("b%d.fc2_w", layer), dtype.F32, tensor.MustShape(inter, h)),
			FC2B:   b.Input(fmt.Sprintf("b%d.fc2_b", layer), dtype.F32, tensor.MustShape(h)),
		}
		g.Inputs[layer] = in

		// Attention half.
		n1 := b.BF16Round(b.AffineLayerNorm(row, in.Norm1W, in.Norm1B, eps))
		qkv := b.BF16Round(b.Add(b.MulMat(in.QKVW, n1), in.QKVB))
		q := b.Reshape(b.GroupSlice(qkv, 0, h, 1, 3*h), hd, heads, n)
		k := b.Reshape(b.GroupSlice(qkv, h, h, 1, 3*h), hd, heads, n)
		v := b.Reshape(b.GroupSlice(qkv, 2*h, h, 1, 3*h), hd, heads, n)
		// NaiveF32: reproduce the reference attn_fwd per-query reduction (the
		// blas-chunked scores path diverges enough over 27 blocks to breach the
		// block_last tolerance).
		attn := b.Reshape(b.AttentionWithOptions(q, k, v, tensor.AttentionOptions{Scale: scale, NaiveF32: true}), h, n)
		proj := b.BF16Round(b.Add(b.MulMat(in.ProjW, attn), in.ProjB))
		resid := b.BF16Round(b.Add(row, proj))

		// MLP half (erf-GELU).
		n2 := b.BF16Round(b.AffineLayerNorm(resid, in.Norm2W, in.Norm2B, eps))
		fc1 := b.BF16Round(b.GELUErf(b.Add(b.MulMat(in.FC1W, n2), in.FC1B)))
		fc2 := b.BF16Round(b.Add(b.MulMat(in.FC2W, fc1), in.FC2B))
		out := b.BF16Round(b.Add(resid, fc2))

		if layer == 0 {
			g.Block0Norm1 = n1
			g.Block0QKV = qkv
			g.Block0Attn = proj
			g.Block0Out = out
		}
		row = out
	}
	g.BlockLast = row
	if err := b.Err(); err != nil {
		return nil, fmt.Errorf("patch tower device blocks graph: %w", err)
	}
	return g, nil
}
