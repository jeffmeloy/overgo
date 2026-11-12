//go:build windows

package routedlm

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/safetensors"
)

// DeviceBranchWeights is one branch's layer weights on the card, read and uploaded once and lent to each request's
// prefix stacks (branch 0) or generation session (branch 1) in place of their own upload: SenseNova's prepare read and
// uploaded both branches for every picture, about 16 GB each and 10.5 s of its 130 s at 2048 x 2048
// (guidance-ops/resident-branches). Each branch is read as its own path reads it, so the products are unchanged.
type DeviceBranchWeights struct {
	branch    int
	layers    []branchDeviceWeights
	resources device.AllocationSet
}

// branchLayerLoader reads a branch's layer as the path that uses it reads it: the prefix stacks through
// LoadBranchLayerWeights, the generation session through its compiled layer plans.
func branchLayerLoader(source *safetensors.Source, cfg Config, binding BranchBinding, branch int) (func(layer int) (BranchLayerWeights, error), error) {
	if branch == 0 {
		return func(layer int) (BranchLayerWeights, error) {
			return LoadBranchLayerWeights(source, cfg, binding, layer, 0)
		}, nil
	}
	plans, err := compileBranchLayerPlans(source, cfg, binding, branch)
	if err != nil {
		return nil, err
	}
	return func(layer int) (BranchLayerWeights, error) { return plans[layer].load() }, nil
}

// BranchWeightBytes is the card bytes a branch's layer weights take, its first layer's times the layers.
func BranchWeightBytes(source *safetensors.Source, cfg Config, binding BranchBinding, branch int) (uint64, error) {
	load, err := branchLayerLoader(source, cfg, binding, branch)
	if err != nil {
		return 0, err
	}
	first, err := load(0)
	if err != nil {
		return 0, err
	}
	var bytes uint64
	for _, part := range branchWeightParts(first) {
		bytes += uint64(len(part))
	}
	return bytes * uint64(cfg.NumHiddenLayers), nil
}

// NewDeviceBranchWeights reads and uploads a branch's every layer.
func NewDeviceBranchWeights(ctx context.Context, worker *device.Worker, source *safetensors.Source, cfg Config, binding BranchBinding, branch int) (*DeviceBranchWeights, error) {
	if worker == nil || source == nil || branch < 0 || branch > 1 {
		return nil, errors.New("routed lm branch weights: missing runtime or invalid branch")
	}
	load, err := branchLayerLoader(source, cfg, binding, branch)
	if err != nil {
		return nil, err
	}
	weights := &DeviceBranchWeights{branch: branch, layers: make([]branchDeviceWeights, cfg.NumHiddenLayers), resources: device.NewAllocationSet(worker)}
	for layer := range weights.layers {
		loaded, err := load(layer)
		if err == nil {
			weights.layers[layer], err = uploadBranchWeights(ctx, &weights.resources, loaded)
		}
		if err != nil {
			return nil, errors.Join(fmt.Errorf("routed lm branch %d layer %d: %w", branch, layer, err), weights.resources.Close(context.WithoutCancel(ctx)))
		}
	}
	return weights, nil
}

// layer is a branch's layer weights where w holds the branch and the layer, found false otherwise.
func (w *DeviceBranchWeights) layer(branch, layer int) (branchDeviceWeights, bool) {
	if w == nil || w.branch != branch || layer < 0 || layer >= len(w.layers) {
		return branchDeviceWeights{}, false
	}
	return w.layers[layer], true
}

// Close frees the branch's weights from the card.
func (w *DeviceBranchWeights) Close(ctx context.Context) error {
	if w == nil {
		return nil
	}
	w.layers = nil
	return w.resources.Close(context.WithoutCancel(ctx))
}
