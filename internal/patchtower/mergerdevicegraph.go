package patchtower

// Device merger graph: the 4-matrix score-pooled merger (proj1 per 2x2 member,
// mean pool, score MLP with erf-GELU, per-channel softmax mix over the 4
// members, erf-GELU, proj2) composed as an overgo tensor.Builder graph for the
// CUDA generic executor. Mirrors MergerRowInto op-for-op over all image rows at
// once. block_last is the F32 host feed [vHidden, nPatch]; merger weights are
// the checkpoint F32 tensors (torch [out,in] row-major == ggml [in,out]);
// activations carry bf16-rounded values at the same points the host rounds.
//
// Layout trick: members are gathered POS-MAJOR (column = pos*rows + row) so a
// reshape to [out, rows, group] places the member index (pos) on the outermost
// axis, making each member's [out, rows] slab a contiguous FlatSlice and the
// per-channel softmax over the 4 members a Softmax over dim0 after stacking.

import (
	"fmt"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// DeviceMergerGraph: the compiled-ready merger graph and its handles.
type DeviceMergerGraph struct {
	Builder   *tensor.Builder
	BlockLast *tensor.Tensor // [vHidden, nPatch] F32 host feed
	Proj1W    *tensor.Tensor // [vHidden, out] F32
	Proj1B    *tensor.Tensor // [out]
	Proj2W    *tensor.Tensor // [out, out] F32
	Proj2B    *tensor.Tensor // [out]
	Pool0W    *tensor.Tensor // [2*out, out] F32
	Pool0B    *tensor.Tensor // [out]
	Pool2W    *tensor.Tensor // [out, out] F32
	Pool2B    *tensor.Tensor // [out]
	Merged    *tensor.Tensor // [out, imageRows] F32 output (bf16-carried)
	Rows      int
	// MemberCols: pos-major gather columns into block_last (len group*rows).
	MemberCols []uint32
}

// mergerMemberColumns: the block_last column for each (pos, out-row), matching
// MergerRowInto's src index, laid out pos-major (pos*rows + row).
func mergerMemberColumns(imageRows, gridH, gridW int, spec Spec) ([]uint32, error) {
	merge := spec.MergeSize
	outH, outW := gridH/merge, gridW/merge
	outSpatial := outH * outW
	group := merge * merge
	cols := make([]uint32, group*imageRows)
	for outRow := 0; outRow < imageRows; outRow++ {
		t := outRow / outSpatial
		spatial := outRow % outSpatial
		oh, ow := spatial/outW, spatial%outW
		for dy := 0; dy < merge; dy++ {
			for dx := 0; dx < merge; dx++ {
				pos := dy*merge + dx
				src := t*gridH*gridW + (oh*merge+dy)*gridW + (ow*merge + dx)
				cols[pos*imageRows+outRow] = uint32(src)
			}
		}
	}
	return cols, nil
}

// BuildDeviceMerger: the merger over imageRows output rows.
func BuildDeviceMerger(spec Spec, imageRows, gridT, gridH, gridW int) (*DeviceMergerGraph, error) {
	if spec.MergeSize != 2 {
		return nil, fmt.Errorf("patch tower device merger: merge=%d, only 2x pooler is implemented", spec.MergeSize)
	}
	if imageRows <= 0 || gridH%spec.MergeSize != 0 || gridW%spec.MergeSize != 0 {
		return nil, fmt.Errorf("patch tower device merger: bad rows=%d grid=[%d,%d,%d]", imageRows, gridT, gridH, gridW)
	}
	nPatch := gridT * gridH * gridW
	vHidden := uint64(spec.Hidden)
	out := uint64(spec.OutHidden)
	group := uint64(spec.MergeSize * spec.MergeSize)
	R := uint64(imageRows)

	cols, err := mergerMemberColumns(imageRows, gridH, gridW, spec)
	if err != nil {
		return nil, err
	}

	b := tensor.NewBuilder()
	g := &DeviceMergerGraph{Builder: b, Rows: imageRows, MemberCols: cols}
	g.BlockLast = b.Input("block_last", dtype.F32, tensor.MustShape(vHidden, uint64(nPatch)))
	g.Proj1W = b.Input("proj1_w", dtype.F32, tensor.MustShape(vHidden, out))
	g.Proj1B = b.Input("proj1_b", dtype.F32, tensor.MustShape(out))
	g.Proj2W = b.Input("proj2_w", dtype.F32, tensor.MustShape(out, out))
	g.Proj2B = b.Input("proj2_b", dtype.F32, tensor.MustShape(out))
	g.Pool0W = b.Input("pool0_w", dtype.F32, tensor.MustShape(2*out, out))
	g.Pool0B = b.Input("pool0_b", dtype.F32, tensor.MustShape(out))
	g.Pool2W = b.Input("pool2_w", dtype.F32, tensor.MustShape(out, out))
	g.Pool2B = b.Input("pool2_b", dtype.F32, tensor.MustShape(out))

	// Gather the group members pos-major -> [vHidden, group*rows].
	members := b.GetRows(g.BlockLast, cols)
	// proj1 per member -> [out, group*rows], bf16.
	proj := b.BF16Round(b.Add(b.MulMat(g.Proj1W, members), g.Proj1B))
	// Reshape [out, rows, group]: member index (pos) is the outermost axis.
	projRG := b.Reshape(proj, out, R, group)

	// Per-member [out, rows] slabs (contiguous FlatSlice on the outer axis).
	slab := make([]*tensor.Tensor, group)
	for pos := uint64(0); pos < group; pos++ {
		slab[pos] = b.FlatSlice(projRG, pos*out*R, out, R)
	}
	// Mean pool over the members -> [out, rows], bf16.
	sum := slab[0]
	for pos := uint64(1); pos < group; pos++ {
		sum = b.Add(sum, slab[pos])
	}
	pooled := b.BF16Round(b.Scale(sum, float32(1.0/float64(group))))

	// Score MLP per member: fused=[member;pooled] -> pool0 -> erf-GELU -> pool2.
	scoreStack := make([]*tensor.Tensor, group)
	projStack := make([]*tensor.Tensor, group)
	for pos := uint64(0); pos < group; pos++ {
		fused := b.Concat(slab[pos], pooled, 0) // [2*out, rows]
		hidden := b.BF16Round(b.Add(b.MulMat(g.Pool0W, fused), g.Pool0B))
		hidden = b.BF16Round(b.GELUErf(hidden))
		score := b.BF16Round(b.Add(b.MulMat(g.Pool2W, hidden), g.Pool2B)) // [out, rows]
		scoreStack[pos] = b.Reshape(score, 1, out, R)
		projStack[pos] = b.Reshape(slab[pos], 1, out, R)
	}
	// Stack members onto a leading axis -> [group, out, rows].
	scores := scoreStack[0]
	projs := projStack[0]
	for pos := uint64(1); pos < group; pos++ {
		scores = b.Concat(scores, scoreStack[pos], 0)
		projs = b.Concat(projs, projStack[pos], 0)
	}
	// Per-channel softmax over the members (dim0), weighted sum over members.
	probs := b.Softmax(scores)                      // [group, out, rows]
	weighted := b.SumRows(b.Multiply(probs, projs)) // [1, out, rows]
	weighted = b.Reshape(weighted, out, R)          // [out, rows]
	weighted = b.BF16Round(weighted)
	weighted = b.BF16Round(b.GELUErf(weighted))
	// proj2 -> [out, rows], bf16.
	g.Merged = b.BF16Round(b.Add(b.MulMat(g.Proj2W, weighted), g.Proj2B))

	if err := b.Err(); err != nil {
		return nil, fmt.Errorf("patch tower device merger graph: %w", err)
	}
	return g, nil
}
