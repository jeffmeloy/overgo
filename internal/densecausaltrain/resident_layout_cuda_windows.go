//go:build windows

package densecausaltrain

import (
	"fmt"
	"strings"

	"overgo/internal/densecausal"
	"overgo/internal/devicemath"
	"overgo/internal/optimizer"
)

// residentLayout is how a dense model's packed weights lie on the device: the block that holds every layer tensor,
// each layer's tensor names and offsets into that block, and each group's place in the packed weights.
type residentLayout struct {
	plan                 optimizer.Plan
	groupIndex           map[string]int
	layerStart, layerEnd int
	offsets              []devicemath.LayerTensorOffsets
	layerNames           []densecausal.LayerTensorNames
}

// group is a tensor's group in the plan, and offset its first element in the packed weights.
func (l residentLayout) group(name string) optimizer.Group {
	return l.plan.Groups()[l.groupIndex[name]]
}

func (l residentLayout) offset(name string) int { return l.group(name).Start }

// newResidentLayout lays out the plan of a model of d's dimensions: all "model.layers." tensors are contiguous in
// sorted order, so one block from the first to the last holds them, each layer's offsets rebased to it.
func newResidentLayout(d densecausal.Dims, plan optimizer.Plan) (residentLayout, error) {
	layout := residentLayout{plan: plan, groupIndex: make(map[string]int, plan.GroupCount()), layerStart: -1}
	for index, group := range plan.Groups() {
		layout.groupIndex[group.Name] = index
		if strings.HasPrefix(group.Name, "model.layers.") {
			if layout.layerStart < 0 {
				layout.layerStart = group.Start
			}
			layout.layerEnd = max(layout.layerEnd, group.End)
		}
	}
	if layout.layerStart < 0 {
		return residentLayout{}, fmt.Errorf("densecausal resident training: no layer tensors")
	}
	layout.offsets = make([]devicemath.LayerTensorOffsets, d.Layers)
	layout.layerNames = make([]densecausal.LayerTensorNames, d.Layers)
	for i := range layout.layerNames {
		names := densecausal.DenseLayerTensorNames(i)
		layout.layerNames[i] = names
		rebased := func(name string) int { return layout.offset(name) - layout.layerStart }
		layout.offsets[i] = devicemath.LayerTensorOffsets{
			InLN: rebased(names.InLN), PostLN: rebased(names.PostLN),
			Q: rebased(names.Q), K: rebased(names.K), V: rebased(names.V), O: rebased(names.O),
			Gate: rebased(names.Gate), Up: rebased(names.Up), Down: rebased(names.Down),
			AttnBias: d.AttnBias,
		}
		if d.AttnBias {
			layout.offsets[i].QBias, layout.offsets[i].KBias, layout.offsets[i].VBias = rebased(names.QB), rebased(names.KB), rebased(names.VB)
		}
	}
	return layout, nil
}

// vectorWeights are each layer's vector tensors read from the packed weights: its norms and, where the model has them,
// its attention biases.
func (l residentLayout) vectorWeights(d densecausal.Dims, weights []float32) []devicemath.LayerTrainVectors {
	vectors := make([]devicemath.LayerTrainVectors, d.Layers)
	for i, names := range l.layerNames {
		inNorm, postNorm := l.offset(names.InLN), l.offset(names.PostLN)
		vectors[i] = devicemath.LayerTrainVectors{InLN: weights[inNorm : inNorm+d.Hidden], PostLN: weights[postNorm : postNorm+d.Hidden]}
		if d.AttnBias {
			query, key, value := l.offset(names.QB), l.offset(names.KB), l.offset(names.VB)
			vectors[i].QBias = weights[query : query+d.Heads*d.HeadDim]
			vectors[i].KBias = weights[key : key+d.KVHeads*d.HeadDim]
			vectors[i].VBias = weights[value : value+d.KVHeads*d.HeadDim]
		}
	}
	return vectors
}
