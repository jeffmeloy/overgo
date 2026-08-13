//go:build windows

package routedlm

import (
	"context"
	"fmt"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/executor"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

// DevicePrefixInput declares one text conditioning branch.
type DevicePrefixInput struct {
	TokenIDs  []int
	ImageTime int
}

// DevicePrefixStackStats reports the shared prefix stream.
type DevicePrefixStackStats struct {
	Layers, Branches int
	Wall             time.Duration
}

type devicePrefixBranch struct {
	graph    *DevicePrefixLayerGraph
	compiled *executor.CompiledGraph
	state    *PrefixState
	hidden   []float32
}

// RunDevicePrefixStacks streams text weights once across conditioning branches.
func RunDevicePrefixStacks(
	ctx context.Context,
	worker *device.Worker,
	cuda *executor.Executor,
	source *safetensors.Source,
	cfg Config,
	binding BranchBinding,
	rope RopePlan,
	observe func(branch, layer int, hidden, key, value []float32),
	inputs ...DevicePrefixInput,
) ([]*PrefixState, DevicePrefixStackStats, error) {
	started := time.Now()
	stats := DevicePrefixStackStats{Layers: cfg.NumHiddenLayers, Branches: len(inputs)}
	if worker == nil || cuda == nil || source == nil || len(inputs) == 0 {
		return nil, stats, fmt.Errorf("routed lm prefix stacks: missing runtime or inputs")
	}
	branches := make([]devicePrefixBranch, len(inputs))
	for index, input := range inputs {
		if len(input.TokenIDs) == 0 || input.ImageTime <= 0 {
			return nil, stats, fmt.Errorf("routed lm prefix stacks: input %d invalid", index)
		}
		positions := make([]RowPosition, len(input.TokenIDs))
		for row := range positions {
			positions[row] = RowPosition{Time: row}
		}
		graph, err := BuildDevicePrefixLayer(cfg, rope, positions)
		if err != nil {
			return nil, stats, err
		}
		compiled, err := executor.Compile(graph.Output, graph.KeyKV, graph.ValueKV)
		if err != nil {
			return nil, stats, err
		}
		hidden, err := EmbeddingRows(source, cfg, binding, input.TokenIDs)
		if err != nil {
			return nil, stats, err
		}
		state, err := NewPrefixState(len(input.TokenIDs), cfg.NumKeyValueHeads, cfg.HeadDim, input.ImageTime, cfg.NumHiddenLayers)
		if err != nil {
			return nil, stats, err
		}
		branches[index] = devicePrefixBranch{graph: graph, compiled: compiled, state: state, hidden: hidden}
	}
	uploader := branchDeviceUploader{worker: worker, ctx: ctx}
	defer uploader.free()
	streamCtx, cancelStream := context.WithCancel(ctx)
	defer cancelStream()
	for load := range streamBranchLayers(streamCtx, source, cfg, binding, 0) {
		if load.err != nil {
			return nil, stats, load.err
		}
		layer := load.layer
		shared, err := uploader.uploadBranchWeights(load.weights)
		if err != nil {
			return nil, stats, err
		}
		for branchIndex := range branches {
			branch := &branches[branchIndex]
			result, err := cuda.ExecuteCompiledWithDeviceFeeds(ctx, branch.compiled, map[*tensor.Tensor]reference.Value{
				branch.graph.Row: {Shape: branch.graph.Row.Shape, Data: branch.hidden},
			}, bindBranchWeights(branch.graph.Weights, shared))
			if err != nil {
				return nil, stats, fmt.Errorf("routed lm prefix stacks: branch=%d layer=%d: %w", branchIndex, layer, err)
			}
			branch.hidden = result[branch.graph.Output].Data
			key, value := result[branch.graph.KeyKV].Data, result[branch.graph.ValueKV].Data
			if err := branch.state.SetLayer(layer, key, value); err != nil {
				return nil, stats, err
			}
			if observe != nil {
				observe(branchIndex, layer, branch.hidden, key, value)
			}
		}
		uploader.free()
	}
	states := make([]*PrefixState, len(branches))
	for index := range branches {
		states[index] = branches[index].state
	}
	stats.Wall = time.Since(started)
	return states, stats, nil
}
