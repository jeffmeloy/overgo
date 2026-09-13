//go:build windows

package routedlm

import (
	"context"
	"errors"
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
	*executor.IndexedGraph
	graph  *DevicePrefixLayerGraph
	inputs branchInputProgram
	host   map[*tensor.Tensor]reference.Value
	state  *PrefixState
	hidden []float32
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
) (_ []*PrefixState, stats DevicePrefixStackStats, result error) {
	started := time.Now()
	stats = DevicePrefixStackStats{Layers: cfg.NumHiddenLayers, Branches: len(inputs)}
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
		indexed, err := executor.CompileIndexed(graph.Output, graph.KeyKV, graph.ValueKV)
		if err != nil {
			return nil, stats, err
		}
		// The SenseNova prefix oracles were recorded against single-call
		// weight-staging SGEMM numerics; bounded tiling drifts the stack
		// past its committed floors.
		if err := indexed.Graph.ReserveExactWeightStaging(); err != nil {
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
		inputProgram, err := compileBranchInputProgram(indexed.Graph, graph.Weights)
		if err != nil {
			return nil, stats, err
		}
		branches[index] = devicePrefixBranch{
			IndexedGraph: indexed, graph: graph, inputs: inputProgram,
			host: make(map[*tensor.Tensor]reference.Value, 1), state: state, hidden: hidden,
		}
	}
	uploads := device.NewAllocationSet(worker)
	cleanup := context.WithoutCancel(ctx)
	defer func() {
		result = errors.Join(result, cuda.ReleaseWeightStaging(cleanup), uploads.Close(cleanup))
	}()
	for layer := 0; layer < cfg.NumHiddenLayers; layer++ {
		weights, err := LoadBranchLayerWeights(source, cfg, binding, layer, 0)
		if err != nil {
			return nil, stats, err
		}
		shared, err := uploadBranchWeights(ctx, &uploads, weights)
		if err != nil {
			return nil, stats, err
		}
		for branchIndex := range branches {
			branch := &branches[branchIndex]
			branch.inputs.bindWeights(branch.Inputs, shared)
			branch.host[branch.graph.Row] = reference.Value{Shape: branch.graph.Row.Shape, Data: branch.hidden}
			retained, err := cuda.ExecuteRetainedCompiled(
				ctx, branch.Graph, branch.host, branch.Inputs, nil, nil,
			)
			if err != nil {
				return nil, stats, fmt.Errorf("routed lm prefix stacks: branch=%d layer=%d: %w", branchIndex, layer, err)
			}
			hiddenValue, err := retained.CopyToHost(ctx, branch.graph.Output)
			if err != nil {
				_ = retained.Release(cleanup)
				return nil, stats, err
			}
			keyValue, err := retained.CopyToHost(ctx, branch.graph.KeyKV)
			if err != nil {
				_ = retained.Release(cleanup)
				return nil, stats, err
			}
			valueValue, err := retained.CopyToHost(ctx, branch.graph.ValueKV)
			if err != nil {
				_ = retained.Release(cleanup)
				return nil, stats, err
			}
			if err := retained.Release(cleanup); err != nil {
				return nil, stats, err
			}
			branch.hidden = hiddenValue.Data
			key, value := keyValue.Data, valueValue.Data
			if err := branch.state.SetLayer(layer, key, value); err != nil {
				return nil, stats, err
			}
			if observe != nil {
				observe(branchIndex, layer, branch.hidden, key, value)
			}
		}
		if err := uploads.Close(ctx); err != nil {
			return nil, stats, err
		}
	}
	states := make([]*PrefixState, len(branches))
	for index := range branches {
		states[index] = branches[index].state
	}
	stats.Wall = time.Since(started)
	return states, stats, nil
}
