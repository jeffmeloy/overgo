//go:build windows

package routedlm

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
)

// DeviceGenerationStackStats reports the neutral body boundary.
type DeviceGenerationStackStats struct {
	Layers, Branches int
	Wall             time.Duration
	HostToDevice     uint64
	DeviceToHost     uint64
}

type generationStackBranch struct {
	graph        *DeviceGenerationLayerGraph
	compiled     *executor.CompiledGraph
	prefixKeys   []driver.DevicePtr
	prefixValues []driver.DevicePtr
	prefixKey    driver.DevicePtr
	prefixValue  driver.DevicePtr
	prefixBytes  uint64
	row          driver.DevicePtr
	output       driver.DevicePtr
	target       *executor.RetainedTargets
}

// DeviceGenerationSession retains invariant generation state.
type DeviceGenerationSession struct {
	mu        sync.Mutex
	worker    *device.Worker
	cuda      *executor.Executor
	cfg       Config
	image     FlowImagePlan
	layers    []branchLayerPlan
	branches  []generationStackBranch
	resources branchDeviceUploader
	setupWall time.Duration
	prefixB   uint64
	closed    bool
}

// DeviceGenerationSessionStats reports retained setup cost.
type DeviceGenerationSessionStats struct {
	Graphs      int
	SetupWall   time.Duration
	PrefixBytes uint64
}

// NewDeviceGenerationSession compiles graphs and uploads reusable prefix KV.
func NewDeviceGenerationSession(
	ctx context.Context,
	worker *device.Worker,
	cuda *executor.Executor,
	source *safetensors.Source,
	cfg Config,
	binding BranchBinding,
	rope RopePlan,
	image FlowImagePlan,
	prefixes ...*PrefixState,
) (*DeviceGenerationSession, error) {
	started := time.Now()
	if worker == nil || cuda == nil || source == nil || len(prefixes) == 0 {
		return nil, fmt.Errorf("routed lm generation session: missing runtime or prefixes")
	}
	layers, err := compileBranchLayerPlans(source, cfg, binding, 1)
	if err != nil {
		return nil, err
	}
	session := &DeviceGenerationSession{
		worker: worker, cuda: cuda, cfg: cfg, image: image, layers: layers,
		branches:  make([]generationStackBranch, len(prefixes)),
		resources: branchDeviceUploader{worker: worker, ctx: context.WithoutCancel(ctx)},
	}
	fail := func(err error) (*DeviceGenerationSession, error) {
		session.resources.free()
		return nil, err
	}
	inputBytes := uint64(image.Tokens) * uint64(cfg.HiddenSize) * 4
	for index, prefix := range prefixes {
		if prefix == nil || !prefix.Complete() || len(prefix.Layers) != cfg.NumHiddenLayers {
			return fail(fmt.Errorf("routed lm generation session: prefix %d incomplete", index))
		}
		graph, err := BuildDeviceGenerationLayer(cfg, rope, prefix.Rows, image.TokenHeight, image.TokenWidth, prefix.ImageTime)
		if err != nil {
			return fail(err)
		}
		compiled, err := executor.Compile(graph.Output)
		if err != nil {
			return fail(err)
		}
		branch := generationStackBranch{
			graph: graph, compiled: compiled,
			prefixKeys: make([]driver.DevicePtr, cfg.NumHiddenLayers), prefixValues: make([]driver.DevicePtr, cfg.NumHiddenLayers),
		}
		branch.row, err = session.resources.allocate(inputBytes)
		if err != nil {
			return fail(err)
		}
		branch.output, err = session.resources.allocate(inputBytes)
		if err != nil {
			return fail(err)
		}
		branch.prefixBytes = uint64(len(prefix.Layers[0].Key)) * 4
		branch.prefixKey, err = session.resources.allocate(branch.prefixBytes)
		if err != nil {
			return fail(err)
		}
		branch.prefixValue, err = session.resources.allocate(branch.prefixBytes)
		if err != nil {
			return fail(err)
		}
		branch.target = compiled.NewRetainedTargets()
		if err := branch.target.Set(graph.Output, executor.DeviceValue{
			Pointer: branch.output, Shape: graph.Output.Shape, CapacityBytes: inputBytes,
		}); err != nil {
			return fail(err)
		}
		for layer, kv := range prefix.Layers {
			if uint64(len(kv.Key))*4 != branch.prefixBytes || len(kv.Value) != len(kv.Key) {
				return fail(fmt.Errorf("routed lm generation session: prefix %d layer %d changed geometry", index, layer))
			}
			branch.prefixKeys[layer], err = session.resources.uploadF32(kv.Key)
			if err != nil {
				return fail(err)
			}
			branch.prefixValues[layer], err = session.resources.uploadF32(kv.Value)
			if err != nil {
				return fail(err)
			}
			session.prefixB += uint64(len(kv.Key)+len(kv.Value)) * 4
		}
		session.branches[index] = branch
	}
	session.setupWall = time.Since(started)
	return session, nil
}

// Stats returns immutable setup evidence.
func (s *DeviceGenerationSession) Stats() DeviceGenerationSessionStats {
	if s == nil {
		return DeviceGenerationSessionStats{}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return DeviceGenerationSessionStats{Graphs: len(s.branches), SetupWall: s.setupWall, PrefixBytes: s.prefixB}
}

// Run streams layer weights while hidden state stays device-resident.
func (s *DeviceGenerationSession) Run(
	ctx context.Context,
	hidden []float32,
	observedLayers []int,
	observe func(branch, layer int, hidden []float32),
) ([][]float32, DeviceGenerationStackStats, error) {
	started := time.Now()
	if s == nil {
		return nil, DeviceGenerationStackStats{}, fmt.Errorf("routed lm generation session: unavailable")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	stats := DeviceGenerationStackStats{Layers: s.cfg.NumHiddenLayers, Branches: len(s.branches)}
	if s.closed {
		return nil, stats, fmt.Errorf("routed lm generation session: closed")
	}
	if len(hidden) != s.image.Tokens*s.cfg.HiddenSize {
		return nil, stats, fmt.Errorf("routed lm generation session: hidden elements=%d want=%d", len(hidden), s.image.Tokens*s.cfg.HiddenSize)
	}
	selected := make([]bool, s.cfg.NumHiddenLayers)
	for _, layer := range observedLayers {
		if observe == nil || layer < 0 || layer >= len(selected) || selected[layer] {
			return nil, stats, fmt.Errorf("routed lm generation session: invalid observed layer %d", layer)
		}
		selected[layer] = true
	}
	if err := s.worker.Do(ctx, func(state *device.State) error {
		raw := driver.Bytes(hidden)
		for index := range s.branches {
			if err := state.Driver.MemcpyHtoD(s.branches[index].row, raw); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		return nil, stats, err
	}
	stats.HostToDevice += uint64(len(hidden)*len(s.branches)) * 4
	uploader := branchDeviceUploader{worker: s.worker, ctx: ctx}
	defer uploader.free()
	out := make([][]float32, len(s.branches))
	type layerLoad struct {
		weights BranchLayerWeights
		err     error
	}
	load := func(layer int) <-chan layerLoad {
		result := make(chan layerLoad, 1)
		go func() {
			weights, err := s.layers[layer].load()
			result <- layerLoad{weights: weights, err: err}
		}()
		return result
	}
	pending := load(0)
	for layer := 0; layer < s.cfg.NumHiddenLayers; layer++ {
		loaded := <-pending
		if loaded.err != nil {
			return nil, stats, loaded.err
		}
		if layer+1 < s.cfg.NumHiddenLayers {
			pending = load(layer + 1)
		}
		if err := s.worker.Do(ctx, func(state *device.State) error {
			for index := range s.branches {
				branch := &s.branches[index]
				if err := state.Driver.MemcpyDtoD(branch.prefixKey, branch.prefixKeys[layer], branch.prefixBytes); err != nil {
					return err
				}
				if err := state.Driver.MemcpyDtoD(branch.prefixValue, branch.prefixValues[layer], branch.prefixBytes); err != nil {
					return err
				}
			}
			return nil
		}); err != nil {
			return nil, stats, err
		}
		shared, err := uploader.uploadBranchWeights(loaded.weights)
		if err != nil {
			return nil, stats, err
		}
		for index := range s.branches {
			branch := &s.branches[index]
			feeds := bindBranchWeights(branch.graph.Vision, shared)
			feeds[branch.graph.PrefixKey], feeds[branch.graph.PrefixValue] = branch.prefixKey, branch.prefixValue
			feeds[branch.graph.Row] = branch.row
			retained, err := s.cuda.ExecuteRetainedCompiledWithTargets(ctx, branch.compiled, nil, feeds, branch.target)
			if err != nil {
				return nil, stats, fmt.Errorf("routed lm generation stack: branch=%d layer=%d: %w", index, layer, err)
			}
			if selected[layer] || layer == s.cfg.NumHiddenLayers-1 {
				value, err := retained.CopyToHost(ctx, branch.graph.Output)
				if err != nil {
					return nil, stats, fmt.Errorf("routed lm generation stack: branch=%d layer=%d copy: %w", index, layer, err)
				}
				stats.DeviceToHost += uint64(len(value.Data)) * 4
				if selected[layer] {
					observe(index, layer, value.Data)
				}
				if layer == s.cfg.NumHiddenLayers-1 {
					out[index] = value.Data
				}
			}
			if err := retained.Release(ctx); err != nil {
				return nil, stats, err
			}
		}
		if layer < s.cfg.NumHiddenLayers-1 {
			if err := s.worker.Do(ctx, func(state *device.State) error {
				for index := range s.branches {
					branch := &s.branches[index]
					if err := state.Driver.MemcpyDtoD(branch.row, branch.output, uint64(len(hidden))*4); err != nil {
						return err
					}
				}
				return nil
			}); err != nil {
				return nil, stats, err
			}
		}
		uploader.free()
	}
	stats.Wall = time.Since(started)
	return out, stats, nil
}

// Close releases retained prefix KV and input storage.
func (s *DeviceGenerationSession) Close(ctx context.Context) error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	s.resources.ctx = context.WithoutCancel(ctx)
	s.branches = nil
	return s.resources.releaseAfter(0)
}

type branchDeviceWeights struct {
	pointers [11]driver.DevicePtr
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

func (u *branchDeviceUploader) allocate(bytes uint64) (driver.DevicePtr, error) {
	var pointer driver.DevicePtr
	err := u.worker.Do(u.ctx, func(state *device.State) error {
		var err error
		pointer, err = state.Driver.MemAlloc(bytes)
		return err
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

func (u *branchDeviceUploader) releaseAfter(keep int) error {
	if keep < 0 || keep > len(u.ptrs) {
		return fmt.Errorf("routed lm device release: keep=%d allocations=%d", keep, len(u.ptrs))
	}
	err := u.worker.Do(u.ctx, func(state *device.State) error {
		var releaseErr error
		for _, pointer := range u.ptrs[keep:] {
			releaseErr = errors.Join(releaseErr, state.Driver.MemFree(pointer))
		}
		return releaseErr
	})
	u.ptrs = u.ptrs[:keep]
	return err
}

func (u *branchDeviceUploader) free() {
	_ = u.releaseAfter(0)
}
