//go:build windows

package routedlm

import (
	"context"
	"fmt"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

// DeviceGenerationStackStats reports the neutral body boundary.
type DeviceGenerationStackStats struct {
	Layers, Branches int
	Wall             time.Duration
}

type generationStackBranch struct {
	prefix   *PrefixState
	graph    *DeviceGenerationLayerGraph
	compiled *executor.CompiledGraph
	hidden   []float32
}

// RunDeviceGenerationStack streams one routed image branch per checkpoint
// layer, sharing each weight upload across all guidance branches.
func RunDeviceGenerationStack(
	ctx context.Context,
	worker *device.Worker,
	cuda *executor.Executor,
	source *safetensors.Source,
	cfg Config,
	binding BranchBinding,
	rope RopePlan,
	image FlowImagePlan,
	hidden []float32,
	prefixes ...*PrefixState,
) ([][]float32, DeviceGenerationStackStats, error) {
	return RunDeviceGenerationStackObserved(ctx, worker, cuda, source, cfg, binding, rope, image, hidden, nil, prefixes...)
}

// RunDeviceGenerationStackObserved exposes selected layer boundaries to gates.
func RunDeviceGenerationStackObserved(
	ctx context.Context,
	worker *device.Worker,
	cuda *executor.Executor,
	source *safetensors.Source,
	cfg Config,
	binding BranchBinding,
	rope RopePlan,
	image FlowImagePlan,
	hidden []float32,
	observe func(branch, layer int, hidden []float32),
	prefixes ...*PrefixState,
) ([][]float32, DeviceGenerationStackStats, error) {
	started := time.Now()
	stats := DeviceGenerationStackStats{Layers: cfg.NumHiddenLayers, Branches: len(prefixes)}
	if worker == nil || cuda == nil || source == nil || len(prefixes) == 0 {
		return nil, stats, fmt.Errorf("routed lm generation stack: missing runtime or prefixes")
	}
	if len(hidden) != image.Tokens*cfg.HiddenSize {
		return nil, stats, fmt.Errorf("routed lm generation stack: hidden elements=%d want=%d", len(hidden), image.Tokens*cfg.HiddenSize)
	}
	branches := make([]generationStackBranch, len(prefixes))
	for index, prefix := range prefixes {
		if prefix == nil || !prefix.Complete() || len(prefix.Layers) != cfg.NumHiddenLayers {
			return nil, stats, fmt.Errorf("routed lm generation stack: prefix %d incomplete", index)
		}
		graph, err := BuildDeviceGenerationLayer(cfg, rope, prefix.Rows, image.TokenHeight, image.TokenWidth, prefix.ImageTime)
		if err != nil {
			return nil, stats, err
		}
		compiled, err := executor.Compile(graph.Output)
		if err != nil {
			return nil, stats, err
		}
		branches[index] = generationStackBranch{
			prefix: prefix, graph: graph, compiled: compiled, hidden: append([]float32(nil), hidden...),
		}
	}
	uploader := branchDeviceUploader{worker: worker, ctx: ctx}
	defer uploader.free()
	catalog := source.Snapshot()
	for layer := 0; layer < cfg.NumHiddenLayers; layer++ {
		weights, err := LoadBranchLayerWeights(catalog, cfg, binding, layer, 1)
		if err != nil {
			return nil, stats, err
		}
		shared, err := uploader.uploadBranchWeights(weights)
		if err != nil {
			return nil, stats, err
		}
		for index := range branches {
			branch := &branches[index]
			feeds := bindBranchWeights(branch.graph.Vision, shared)
			kv := branch.prefix.Layers[layer]
			key, err := uploader.uploadF32(kv.Key)
			if err != nil {
				return nil, stats, err
			}
			value, err := uploader.uploadF32(kv.Value)
			if err != nil {
				return nil, stats, err
			}
			feeds[branch.graph.PrefixKey], feeds[branch.graph.PrefixValue] = key, value
			retained, err := cuda.ExecuteRetainedCompiledWithDeviceFeeds(ctx, branch.compiled, map[*tensor.Tensor]reference.Value{
				branch.graph.Row: {Shape: branch.graph.Row.Shape, Data: branch.hidden},
			}, feeds)
			if err != nil {
				return nil, stats, fmt.Errorf("routed lm generation stack: branch=%d layer=%d: %w", index, layer, err)
			}
			outputValue, err := retained.CopyToHost(ctx, branch.graph.Output)
			if err != nil {
				_ = retained.Release(ctx)
				return nil, stats, fmt.Errorf("routed lm generation stack: branch=%d layer=%d copy: %w", index, layer, err)
			}
			if err := retained.Release(ctx); err != nil {
				return nil, stats, err
			}
			branch.hidden = outputValue.Data
			if observe != nil {
				observe(index, layer, branch.hidden)
			}
			uploader.freeAfter(shared.count)
		}
		uploader.free()
	}
	out := make([][]float32, len(branches))
	for index := range branches {
		out[index] = branches[index].hidden
	}
	stats.Wall = time.Since(started)
	return out, stats, nil
}

type branchDeviceWeights struct {
	pointers [11]driver.DevicePtr
	count    int
}

type branchDeviceUploader struct {
	worker *device.Worker
	ctx    context.Context
	ptrs   []driver.DevicePtr
}

func (u *branchDeviceUploader) upload(raw []byte) (driver.DevicePtr, error) {
	var pointer driver.DevicePtr
	err := u.worker.Do(u.ctx, func(state *device.State) error {
		var err error
		pointer, err = state.Driver.MemAlloc(uint64(len(raw)))
		if err != nil {
			return err
		}
		if err = state.Driver.MemcpyHtoD(pointer, raw); err != nil {
			_ = state.Driver.MemFree(pointer)
			return err
		}
		return nil
	})
	if err == nil {
		u.ptrs = append(u.ptrs, pointer)
	}
	return pointer, err
}

func (u *branchDeviceUploader) uploadF32(values []float32) (driver.DevicePtr, error) {
	return u.upload(driver.Bytes(values))
}

func (u *branchDeviceUploader) uploadBranchWeights(w BranchLayerWeights) (branchDeviceWeights, error) {
	var out branchDeviceWeights
	raw := [][]byte{
		driver.Bytes(w.InputNorm), bf16MatrixBytes(w.Q), bf16MatrixBytes(w.K), bf16MatrixBytes(w.V),
		bf16MatrixBytes(w.O), driver.Bytes(w.QNorm), driver.Bytes(w.KNorm), driver.Bytes(w.PostNorm),
		bf16MatrixBytes(w.Gate), bf16MatrixBytes(w.Up), bf16MatrixBytes(w.Down),
	}
	for index := range raw {
		pointer, err := u.upload(raw[index])
		if err != nil {
			return out, err
		}
		out.pointers[index] = pointer
		out.count++
	}
	return out, nil
}

func bf16MatrixBytes(matrix BF16Matrix) []byte {
	if len(matrix.Raw) != 0 {
		return matrix.Raw
	}
	return driver.Bytes(matrix.Data)
}

func bindBranchWeights(nodes DevicePrefillBranch, weights branchDeviceWeights) map[*tensor.Tensor]driver.DevicePtr {
	inputs := []*tensor.Tensor{nodes.InputNorm, nodes.Q, nodes.K, nodes.V, nodes.O, nodes.QNorm, nodes.KNorm, nodes.PostNorm, nodes.Gate, nodes.Up, nodes.Down}
	feeds := make(map[*tensor.Tensor]driver.DevicePtr, len(inputs)+2)
	for index, input := range inputs {
		feeds[input] = weights.pointers[index]
	}
	return feeds
}

func (u *branchDeviceUploader) freeAfter(keep int) {
	if keep < 0 || keep > len(u.ptrs) {
		return
	}
	_ = u.worker.Do(u.ctx, func(state *device.State) error {
		for _, pointer := range u.ptrs[keep:] {
			_ = state.Driver.MemFree(pointer)
		}
		return nil
	})
	u.ptrs = u.ptrs[:keep]
}

func (u *branchDeviceUploader) free() {
	u.freeAfter(0)
}
