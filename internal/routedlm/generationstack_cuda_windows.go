//go:build windows

package routedlm

import (
	"context"
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
	*executor.IndexedGraph
	graph        *DeviceGenerationLayerGraph
	inputs       branchInputProgram
	prefixKey    executor.InputSlot
	prefixValue  executor.InputSlot
	prefixKeys   []driver.DevicePtr
	prefixValues []driver.DevicePtr
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
	weights   []branchDeviceWeights
	branches  []generationStackBranch
	resources device.AllocationSet
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
		worker: worker, cuda: cuda, cfg: cfg, image: image,
		weights:   make([]branchDeviceWeights, len(layers)),
		branches:  make([]generationStackBranch, len(prefixes)),
		resources: device.NewAllocationSet(worker),
	}
	fail := func(err error) (*DeviceGenerationSession, error) {
		_ = session.resources.Close(context.WithoutCancel(ctx))
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
		indexed, err := executor.CompileIndexed(graph.Output)
		if err != nil {
			return fail(err)
		}
		if err := cuda.PrepareCompiled(ctx, indexed.Graph); err != nil {
			return fail(fmt.Errorf("prepare branch %d: %w", index, err))
		}
		branch := generationStackBranch{
			IndexedGraph: indexed, graph: graph,
			prefixKeys: make([]driver.DevicePtr, cfg.NumHiddenLayers), prefixValues: make([]driver.DevicePtr, cfg.NumHiddenLayers),
		}
		branch.inputs, err = compileBranchInputProgram(indexed.Graph, graph.Vision)
		if err != nil {
			return fail(err)
		}
		branch.prefixKey, err = compiledInputSlot(indexed.Graph, graph.PrefixKey)
		if err != nil {
			return fail(err)
		}
		branch.prefixValue, err = compiledInputSlot(indexed.Graph, graph.PrefixValue)
		if err != nil {
			return fail(err)
		}
		rowSlot, err := compiledInputSlot(indexed.Graph, graph.Row)
		if err != nil {
			return fail(err)
		}
		branch.row, err = session.resources.Allocate(ctx, inputBytes)
		if err != nil {
			return fail(err)
		}
		branch.Inputs.Pointers[rowSlot] = branch.row
		branch.output, err = session.resources.Allocate(ctx, inputBytes)
		if err != nil {
			return fail(err)
		}
		prefixBytes := uint64(len(prefix.Layers[0].Key)) * 4
		branch.target = indexed.Graph.NewRetainedTargets()
		if err := branch.target.Set(graph.Output, executor.DeviceValue{
			Pointer: branch.output, Shape: graph.Output.Shape, CapacityBytes: inputBytes,
		}); err != nil {
			return fail(err)
		}
		for layer, kv := range prefix.Layers {
			if uint64(len(kv.Key))*4 != prefixBytes || len(kv.Value) != len(kv.Key) {
				return fail(fmt.Errorf("routed lm generation session: prefix %d layer %d changed geometry", index, layer))
			}
			branch.prefixKeys[layer], err = session.resources.Upload(ctx, driver.Bytes(kv.Key))
			if err != nil {
				return fail(err)
			}
			branch.prefixValues[layer], err = session.resources.Upload(ctx, driver.Bytes(kv.Value))
			if err != nil {
				return fail(err)
			}
			session.prefixB += uint64(len(kv.Key)+len(kv.Value)) * 4
		}
		session.branches[index] = branch
	}
	for layer := range layers {
		loaded, err := layers[layer].load()
		if err != nil {
			return fail(fmt.Errorf("load layer %d: %w", layer, err))
		}
		session.weights[layer], err = uploadBranchWeights(ctx, &session.resources, loaded)
		if err != nil {
			return fail(fmt.Errorf("retain layer %d: %w", layer, err))
		}
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

// Run reuses retained layer weights and device-resident hidden state.
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
	out := make([][]float32, len(s.branches))
	for layer := 0; layer < s.cfg.NumHiddenLayers; layer++ {
		for index := range s.branches {
			branch := &s.branches[index]
			branch.inputs.bindWeights(branch.Inputs, s.weights[layer])
			branch.bindPrefix(layer)
			retained, err := s.cuda.ExecuteRetainedCompiled(
				ctx, branch.Graph, nil, branch.Inputs, branch.target, nil,
			)
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
			for index := range s.branches {
				branch := &s.branches[index]
				branch.row, branch.output = branch.output, branch.row
				rowSlot, err := compiledInputSlot(branch.Graph, branch.graph.Row)
				if err != nil {
					return nil, stats, err
				}
				branch.Inputs.Pointers[rowSlot] = branch.row
				if err := branch.target.Set(branch.graph.Output, executor.DeviceValue{
					Pointer: branch.output, Shape: branch.graph.Output.Shape, CapacityBytes: uint64(len(hidden)) * 4,
				}); err != nil {
					return nil, stats, err
				}
			}
		}
	}
	stats.Wall = time.Since(started)
	return out, stats, nil
}

func (b *generationStackBranch) bindPrefix(layer int) {
	b.Inputs.Pointers[b.prefixKey] = b.prefixKeys[layer]
	b.Inputs.Pointers[b.prefixValue] = b.prefixValues[layer]
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
	s.weights = nil
	s.branches = nil
	return s.resources.Close(context.WithoutCancel(ctx))
}

type branchDeviceWeights struct {
	pointers [11]driver.DevicePtr
}

func uploadBranchWeights(ctx context.Context, allocations *device.AllocationSet, w BranchLayerWeights) (branchDeviceWeights, error) {
	var out branchDeviceWeights
	raw := [][]byte{
		driver.Bytes(w.InputNorm), bf16MatrixBytes(w.Q), bf16MatrixBytes(w.K), bf16MatrixBytes(w.V),
		bf16MatrixBytes(w.O), driver.Bytes(w.QNorm), driver.Bytes(w.KNorm), driver.Bytes(w.PostNorm),
		bf16MatrixBytes(w.Gate), bf16MatrixBytes(w.Up), bf16MatrixBytes(w.Down),
	}
	for index := range raw {
		pointer, err := allocations.Upload(ctx, raw[index])
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

type branchInputProgram struct {
	weights [11]executor.InputSlot
}

func compileBranchInputProgram(compiled *executor.CompiledGraph, nodes DevicePrefillBranch) (branchInputProgram, error) {
	program := branchInputProgram{}
	weightNodes := [...]*tensor.Tensor{
		nodes.InputNorm, nodes.Q, nodes.K, nodes.V, nodes.O, nodes.QNorm,
		nodes.KNorm, nodes.PostNorm, nodes.Gate, nodes.Up, nodes.Down,
	}
	for index, node := range weightNodes {
		slot, err := compiledInputSlot(compiled, node)
		if err != nil {
			return branchInputProgram{}, err
		}
		program.weights[index] = slot
	}
	return program, nil
}

func compiledInputSlot(compiled *executor.CompiledGraph, node *tensor.Tensor) (executor.InputSlot, error) {
	slot, ok := compiled.InputSlot(node)
	if !ok {
		return 0, fmt.Errorf("routed lm: input %q is not compiled", node.Name)
	}
	return slot, nil
}

func (p *branchInputProgram) bindWeights(inputs *executor.DeviceInputs, weights branchDeviceWeights) {
	for index, slot := range p.weights {
		inputs.Pointers[slot] = weights.pointers[index]
	}
}
