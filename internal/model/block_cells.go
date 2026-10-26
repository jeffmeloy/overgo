package model

import (
	"slices"

	"overgo/internal/tensor"
)

// CellBinding is an attention layer's int8 cells for a step (tensor.CellAttention, Strata's K/V cells widened to an
// attention layer's geometry): the layer's state, its capacity and the step's first cell.
type CellBinding struct {
	State     tensor.CellState
	Capacity  uint32
	FirstCell uint32
}

// HoldsAttentionCells reports whether a runner may build the layer over Strata's int8 cells: an attention layer of
// gemma-4's shared-K/V adapter that holds its own keys and values, causal with no sinks, soft cap, ALiBi bias or block
// selection, its sliding window read as a range of cells, its keys and values one width at a geometry the cell
// attention's kernels are instantiated for.
func (p CompiledLayerProgram) HoldsAttentionCells() bool {
	plan, attention := p.plan, p.plan.AttentionGraph
	shapes := p.spec.TensorShapes(plan.Layer)
	return slices.Contains(plan.Program.Instructions, attentionLayerStage(layerOperatorAttentionSharedKVQKNorm)) &&
		plan.HasKV && !plan.SharedKV && !plan.Recurrent && attention.Causal && !attention.UseSinks &&
		!attention.ChunkedWindow && attention.Softcap == 0 && attention.MaxALiBiBias == 0 && attention.BlockSize == 0 &&
		shapes.Key == shapes.Value && tensor.CellAttentionHolds(shapes.Key, shapes.QueryHeads, shapes.KVHeads)
}

// AttentionCells is the layer's cells' geometry: its K/V heads, their width and the window a token reads.
func (p CompiledLayerProgram) AttentionCells() (kvHeads, width uint64, window uint32) {
	shapes := p.spec.TensorShapes(p.plan.Layer)
	return shapes.KVHeads, shapes.Key, p.plan.AttentionGraph.Window
}

// cellAttentionStep is the attention over a layer's cells for a step's query, key and value rows, which it appends.
func cellAttentionStep(builder *tensor.Builder, spec Spec, plan LayerPlan, cells *CellBinding, query, key, value *tensor.Tensor) *tensor.Tensor {
	tokens := query.Shape.Dims[tensor.CellTokenAxis]
	steps := make([]uint32, tokens)
	for index := range steps {
		steps[index] = cells.FirstCell + uint32(index)
	}
	return builder.CellAttention(query, key, value, cells.State, tensor.CellAttentionAttributes{
		Cells: steps, Capacity: cells.Capacity, Window: plan.AttentionGraph.Window, Scale: spec.AttentionScale,
	})
}
