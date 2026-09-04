package executor

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"
	"sync"
	"unsafe"

	"overgo/internal/checked"
	"overgo/internal/cuda/cublas"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/planner"
	"overgo/internal/tensor/reference"
)

// Executor: evaluates initial F32 tensor graph on one CUDA worker
type Executor struct {
	mu sync.RWMutex

	worker     *device.Worker
	ownsWorker bool
	closed     bool
	resources  executorResources
}

type executorResources struct {
	module     driver.Module
	functions  functionSet
	blas       *blasState
	q8Input    q8InputState
	arena      driver.DevicePtr
	arenaSize  uint64
	scratch    executionScratch
	buffers    deviceBufferPool
	graphExecs graphExecCache
	arenaUse   arenaMetrics
}

// ExecutionMetrics is a point-in-time, executor-owned evidence view. Counters
// are monotonic for the executor lifetime; byte fields describe the most
// recent arena requirement and the retained allocation that serves it.
type ExecutionMetrics struct {
	GraphCacheHits         uint64 `json:"graphCacheHits"`
	GraphCacheMisses       uint64 `json:"graphCacheMisses"`
	GraphCaptures          uint64 `json:"graphCaptures"`
	GraphUpdates           uint64 `json:"graphUpdates"`
	GraphInstantiations    uint64 `json:"graphInstantiations"`
	GraphEvictions         uint64 `json:"graphEvictions"`
	GraphUpdateFallbacks   uint64 `json:"graphUpdateFallbacks"`
	GraphDrops             uint64 `json:"graphDrops"`
	GraphCacheEntries      uint64 `json:"graphCacheEntries"`
	GraphCacheCapacity     uint64 `json:"graphCacheCapacity"`
	ArenaRequiredBytes     uint64 `json:"arenaRequiredBytes"`
	ArenaPeakRequiredBytes uint64 `json:"arenaPeakRequiredBytes"`
	ArenaCommittedBytes    uint64 `json:"arenaCommittedBytes"`
	ArenaUnusedBytes       uint64 `json:"arenaUnusedBytes"`
	ArenaGrowths           uint64 `json:"arenaGrowths"`
}

type arenaMetrics struct {
	requiredBytes     uint64
	peakRequiredBytes uint64
	growths           uint64
}

type executionScratch struct {
	pointers          []driver.DevicePtr
	attributePointers []driver.DevicePtr
	attributeWords    []uint32
	attributes        []tensor.Attributes
	retainedStorage   []bool
	retainedOffsets   []uint64
	replayFrame       []driver.DevicePtr
}

const (
	gemmProductScale       = float32(1)
	gemmAccumulatorScale   = float32(0)
	graphExecCacheCapacity = 4
)
const nativeWeightStagingLimitBytes = uint64(32 << 20)

const deviceAllocationAlignment uint64 = 256

type DeviceValue struct {
	Pointer       driver.DevicePtr
	Shape         tensor.Shape
	CapacityBytes uint64
}

// DeviceBuffer: pooled capacity allocation for stable device values.
type DeviceBuffer struct {
	mu       sync.Mutex
	executor *Executor
	lease    deviceBufferLease
	released bool
}

func (e *Executor) AllocateDeviceBuffer(ctx context.Context, bytes uint64) (*DeviceBuffer, error) {
	if e == nil || bytes == 0 {
		return nil, errors.New("CUDA device buffer request is invalid")
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.worker == nil {
		return nil, errors.New("CUDA executor is closed")
	}
	var lease deviceBufferLease
	err := e.worker.Do(ctx, func(state *device.State) error {
		var allocateErr error
		lease, allocateErr = e.resources.buffers.acquireExact(state, bytes)
		return allocateErr
	})
	if err != nil {
		return nil, err
	}
	return &DeviceBuffer{executor: e, lease: lease}, nil
}

func (b *DeviceBuffer) Value(shape tensor.Shape) (DeviceValue, error) {
	if b == nil {
		return DeviceValue{}, errors.New("CUDA device buffer is unavailable")
	}
	bytes, err := shape.Bytes(dtype.F32)
	if err != nil {
		return DeviceValue{}, err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.released || b.lease.pointer == 0 || bytes > b.lease.size {
		return DeviceValue{}, errors.New("CUDA device buffer capacity is insufficient")
	}
	return DeviceValue{Pointer: b.lease.pointer, Shape: shape, CapacityBytes: b.lease.size}, nil
}

func (b *DeviceBuffer) Release(ctx context.Context) error {
	return ReleaseDeviceBuffers(ctx, b)
}

// ReleaseDeviceBuffers: returns one ownership group through one worker transaction.
func ReleaseDeviceBuffers(ctx context.Context, buffers ...*DeviceBuffer) error {
	active := make([]*DeviceBuffer, 0, len(buffers))
	seen := make(map[*DeviceBuffer]struct{}, len(buffers))
	for _, buffer := range buffers {
		if buffer == nil {
			continue
		}
		if _, duplicate := seen[buffer]; duplicate {
			return errors.New("CUDA device buffer release contains a duplicate")
		}
		seen[buffer] = struct{}{}
		active = append(active, buffer)
	}
	sort.Slice(active, func(left, right int) bool {
		return uintptr(unsafe.Pointer(active[left])) < uintptr(unsafe.Pointer(active[right]))
	})
	for _, buffer := range active {
		buffer.mu.Lock()
	}
	defer func() {
		for index := len(active) - 1; index >= 0; index-- {
			active[index].mu.Unlock()
		}
	}()
	var executor *Executor
	leases := make([]deviceBufferLease, 0, len(active))
	owners := make([]*DeviceBuffer, 0, len(active))
	for _, buffer := range active {
		if buffer.released {
			continue
		}
		if buffer.executor == nil {
			return errors.New("CUDA executor is closed")
		}
		if executor == nil {
			executor = buffer.executor
		} else if buffer.executor != executor {
			return errors.New("CUDA device buffer release spans executors")
		}
		leases = append(leases, buffer.lease)
		owners = append(owners, buffer)
	}
	if len(leases) == 0 {
		return nil
	}
	executor.mu.RLock()
	defer executor.mu.RUnlock()
	if executor.closed || executor.worker == nil {
		return errors.New("CUDA executor is closed")
	}
	if err := executor.worker.Do(ctx, func(*device.State) error {
		for _, lease := range leases {
			executor.resources.buffers.release(lease)
		}
		return nil
	}); err != nil {
		return err
	}
	for _, buffer := range owners {
		buffer.released = true
		buffer.lease = deviceBufferLease{}
	}
	return nil
}

type DeviceCopySegment struct {
	Source driver.DevicePtr
	Bytes  uint64
}

type DeviceCopy struct {
	Shape    tensor.Shape
	Storage  dtype.Type
	Segments []DeviceCopySegment
}

// RetainedOutputs: owns selected graph outputs in standalone device
// allocations; Call Release when values are no longer used
type RetainedOutputs struct {
	mu       sync.Mutex
	executor *Executor
	compiled *CompiledGraph
	values   []DeviceValue
	leases   []deviceBufferLease
	released bool
}

func (r *RetainedOutputs) Value(output *tensor.Tensor) (DeviceValue, bool) {
	if r == nil {
		return DeviceValue{}, false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.released {
		return DeviceValue{}, false
	}
	if r.compiled == nil {
		return DeviceValue{}, false
	}
	index, ok := r.compiled.outputIndexes[output]
	if !ok || index >= len(r.values) || r.values[index].Pointer == 0 {
		return DeviceValue{}, false
	}
	return r.values[index], true
}

func (r *RetainedOutputs) CopyToHost(
	ctx context.Context,
	output *tensor.Tensor,
) (reference.Value, error) {
	if r == nil {
		return reference.Value{}, errors.New("CUDA retained output is unavailable")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.released {
		return reference.Value{}, errors.New("CUDA retained output is unavailable")
	}
	if r.compiled == nil {
		return reference.Value{}, errors.New("CUDA retained output is unavailable")
	}
	index, ok := r.compiled.outputIndexes[output]
	if !ok || index >= len(r.values) || r.values[index].Pointer == 0 {
		return reference.Value{}, errors.New("CUDA retained output is unavailable")
	}
	value := r.values[index]
	elements, err := value.Shape.Elements()
	if err != nil {
		return reference.Value{}, err
	}
	if elements > uint64(math.MaxInt) {
		return reference.Value{}, errors.New("CUDA retained output is too large")
	}
	data := make([]float32, int(elements))
	r.executor.mu.RLock()
	defer r.executor.mu.RUnlock()
	if r.executor.closed || r.executor.worker == nil {
		return reference.Value{}, errors.New("CUDA executor is closed")
	}
	err = r.executor.worker.Do(ctx, func(state *device.State) error {
		return state.Driver.MemcpyDtoH(driver.Bytes(data), value.Pointer)
	})
	return reference.Value{Shape: value.Shape, Data: data}, err
}

func (r *RetainedOutputs) Release(ctx context.Context) error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.released {
		return nil
	}
	var releaseErr error
	if r.executor == nil {
		releaseErr = errors.New("CUDA executor is closed")
	} else {
		r.executor.mu.RLock()
		if r.executor.closed || r.executor.worker == nil {
			releaseErr = errors.New("CUDA executor is closed")
		} else {
			releaseErr = r.executor.worker.Do(ctx, func(state *device.State) error {
				for _, lease := range r.leases {
					r.executor.resources.buffers.release(lease)
				}
				return nil
			})
		}
		r.executor.mu.RUnlock()
	}
	if releaseErr != nil && (errors.Is(releaseErr, context.Canceled) || errors.Is(releaseErr, context.DeadlineExceeded)) {
		return releaseErr
	}
	r.released = true
	r.values = nil
	r.leases = nil
	return releaseErr
}

// CopyDeviceValues creates independently owned device values by concatenating
// caller-selected source segments; used for persistent state edits that
// cannot be represented by pointer view
func (e *Executor) CopyDeviceValues(
	ctx context.Context,
	copies []DeviceCopy,
) (*RetainedOutputs, []DeviceValue, error) {
	if e == nil {
		return nil, nil, errors.New("CUDA executor is closed")
	}
	if len(copies) == 0 {
		return nil, nil, errors.New("CUDA device copy list is empty")
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.worker == nil {
		return nil, nil, errors.New("CUDA executor is closed")
	}
	values := make([]DeviceValue, len(copies))
	leases := make([]deviceBufferLease, 0, len(copies))
	err := e.worker.Do(ctx, func(state *device.State) error {
		fail := func(cause error) error {
			for _, lease := range leases {
				e.resources.buffers.release(lease)
			}
			leases = nil
			return cause
		}
		for index, copySpec := range copies {
			expected, shapeErr := copySpec.Shape.Bytes(copySpec.Storage)
			if shapeErr != nil {
				return fail(shapeErr)
			}
			var bytes uint64
			for _, segment := range copySpec.Segments {
				if segment.Source == 0 || segment.Bytes == 0 {
					return fail(errors.New(
						"CUDA device copy segment is empty",
					))
				}
				if bytes > math.MaxUint64-segment.Bytes {
					return fail(errors.New(
						"CUDA device copy segment size overflows",
					))
				}
				bytes += segment.Bytes
			}
			if bytes != expected {
				return fail(fmt.Errorf(
					"CUDA device copy has %d bytes for shape %v; want %d",
					bytes,
					copySpec.Shape.Slice(),
					expected,
				))
			}
			lease, allocErr := e.resources.buffers.acquire(state, bytes)
			if allocErr != nil {
				return fail(allocErr)
			}
			leases = append(leases, lease)
			pointer := lease.pointer
			var offset uint64
			for _, segment := range copySpec.Segments {
				if uint64(pointer) > math.MaxUint64-offset {
					return fail(errors.New(
						"CUDA device copy destination overflows",
					))
				}
				if copyErr := state.Driver.MemcpyDtoD(
					pointer+driver.DevicePtr(offset),
					segment.Source,
					segment.Bytes,
				); copyErr != nil {
					return fail(copyErr)
				}
				offset += segment.Bytes
			}
			values[index] = DeviceValue{
				Pointer: pointer,
				Shape:   copySpec.Shape,
			}
		}
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	return &RetainedOutputs{
		executor: e,
		leases:   leases,
	}, values, nil
}

type executionResult struct {
	host   map[*tensor.Tensor]reference.Value
	values []DeviceValue
	leases []deviceBufferLease
}

func hostResult(result *executionResult, err error) (map[*tensor.Tensor]reference.Value, error) {
	if err != nil {
		return nil, err
	}
	return result.host, nil
}

// RetainedTargets: graph-indexed stable output destinations.
type RetainedTargets struct {
	compiled *CompiledGraph
	values   []DeviceValue
}

// InputSlot: compiled input ordinal.
type InputSlot uint32

// DeviceInputs: graph-indexed resident input pointers.
type DeviceInputs struct {
	compiled *CompiledGraph
	Pointers []driver.DevicePtr
}

func (c *CompiledGraph) InputSlot(input *tensor.Tensor) (InputSlot, bool) {
	if c == nil || input == nil || input.Op != tensor.OpInput {
		return 0, false
	}
	index, ok := c.orderIndexes[input]
	if !ok || c.nodes[index].operandOffset >= 0 {
		return 0, false
	}
	return InputSlot(-c.nodes[index].operandOffset - 1), true
}

// InputBytes reports the byte extent the compiled graph reads from one
// input slot: the slot's compiled shape in its dtype. A caller binding a
// resident buffer into the slot compares it with the buffer's capacity; a
// smaller buffer would be read past its end by the kernels that consume
// the slot.
func (c *CompiledGraph) InputBytes(slot InputSlot) (uint64, bool) {
	if c == nil {
		return 0, false
	}
	for index, node := range c.order {
		if node == nil || node.Op != tensor.OpInput || c.nodes[index].operandOffset >= 0 ||
			InputSlot(-c.nodes[index].operandOffset-1) != slot {
			continue
		}
		bytes, err := node.Shape.Bytes(node.Type)
		if err != nil {
			return 0, false
		}
		return bytes, true
	}
	return 0, false
}

func (c *CompiledGraph) NewDeviceInputs() *DeviceInputs {
	if c == nil {
		return nil
	}
	return &DeviceInputs{compiled: c, Pointers: make([]driver.DevicePtr, c.inputCount)}
}

// Set binds one compiled input node's device pointer into its indexed slot.
// The indexed slots are the ONLY operand contract: the former tensor-keyed
// map layer (BindDeviceInputs) is deleted, so callers bind directly as they
// produce pointers instead of accumulating a map to convert.
func (i *DeviceInputs) Set(node *tensor.Tensor, pointer driver.DevicePtr) error {
	if i == nil || i.compiled == nil {
		return errors.New("CUDA device inputs are unavailable")
	}
	slot, ok := i.compiled.InputSlot(node)
	if !ok {
		return fmt.Errorf("CUDA device input %q is not compiled", node.Name)
	}
	i.Pointers[slot] = pointer
	return nil
}

// Bind resolves ordered graph bindings into compiled input slots.
func (i *DeviceInputs) Bind(bindings tensor.InputBindings[driver.DevicePtr]) error {
	for _, binding := range bindings {
		if err := i.Set(binding.Node, binding.Value); err != nil {
			return err
		}
	}
	return nil
}

// RuntimeAttributes: graph-indexed per-execution attribute overrides.
type RuntimeAttributes struct {
	compiled *CompiledGraph
	values   []tensor.Attributes
}

// NewRuntimeAttributes: indexed override slots.
func (c *CompiledGraph) NewRuntimeAttributes() *RuntimeAttributes {
	if c == nil {
		return nil
	}
	return &RuntimeAttributes{compiled: c, values: make([]tensor.Attributes, len(c.order))}
}

// Set: validated runtime attribute override.
func (a *RuntimeAttributes) Set(node *tensor.Tensor, attributes tensor.Attributes) error {
	if a == nil || a.compiled == nil {
		return errors.New("CUDA runtime attributes are unavailable")
	}
	index, ok := a.compiled.orderIndexes[node]
	if !ok || attributes == nil {
		return errors.New("CUDA runtime attribute target is invalid")
	}
	resolved, err := resolveRuntimeAttributes(attributes)
	if err != nil {
		return err
	}
	if err := tensor.ValidateOperationAttributes(node.Op, resolved); err != nil {
		return err
	}
	if err := validateRuntimeAttributes(node, resolved); err != nil {
		return err
	}
	a.values[index] = attributes
	return nil
}

func resolveRuntimeAttributes(attributes tensor.Attributes) (tensor.Attributes, error) {
	switch value := attributes.(type) {
	case *tensor.GetRowsAttributes:
		if value != nil {
			return *value, nil
		}
	case *tensor.RoPEAttributes:
		if value != nil {
			return *value, nil
		}
	case *tensor.RoPEMultiAttributes:
		if value != nil {
			return *value, nil
		}
	case *tensor.AttentionAttributes:
		if value != nil {
			return *value, nil
		}
	case *tensor.CacheAppendAttributes:
		if value != nil {
			return *value, nil
		}
	default:
		return attributes, nil
	}
	return nil, errors.New("CUDA runtime attribute binding is nil")
}

func validateRuntimeAttributes(node *tensor.Tensor, attributes tensor.Attributes) error {
	sameLength := func(left, right []uint32) bool { return len(left) == len(right) }
	switch node.Op {
	case tensor.OpGetRows:
		initial, next := node.Attrs.(tensor.GetRowsAttributes), attributes.(tensor.GetRowsAttributes)
		if !sameLength(initial.Rows, next.Rows) {
			return errors.New("CUDA runtime row count changes compiled shape")
		}
		rowCount := node.Inputs[0].Shape.Dims[node.Inputs[0].Shape.Rank-1]
		for _, row := range next.Rows {
			if uint64(row) >= rowCount {
				return errors.New("CUDA runtime row exceeds input table")
			}
		}
	case tensor.OpRoPENeoX, tensor.OpRoPENormal:
		initial, next := node.Attrs.(tensor.RoPEAttributes), attributes.(tensor.RoPEAttributes)
		if !sameLength(node.Attrs.(tensor.RoPEAttributes).Positions, attributes.(tensor.RoPEAttributes).Positions) ||
			initial.RotaryDimensions != next.RotaryDimensions ||
			initial.FrequencyBase != next.FrequencyBase ||
			initial.FrequencyScale != next.FrequencyScale ||
			initial.OriginalContext != next.OriginalContext ||
			initial.ExtFactor != next.ExtFactor ||
			initial.AttentionFactor != next.AttentionFactor ||
			initial.BetaFast != next.BetaFast || initial.BetaSlow != next.BetaSlow {
			return errors.New("CUDA runtime RoPE override changes compiled policy")
		}
	case tensor.OpRoPEMulti:
		initial, next := node.Attrs.(tensor.RoPEMultiAttributes), attributes.(tensor.RoPEMultiAttributes)
		for axis := range initial.Positions {
			if !sameLength(initial.Positions[axis], next.Positions[axis]) {
				return errors.New("CUDA runtime multi-RoPE position count changes compiled shape")
			}
		}
		if initial.Sections != next.Sections || initial.RotaryDimensions != next.RotaryDimensions ||
			initial.FrequencyBase != next.FrequencyBase || initial.FrequencyScale != next.FrequencyScale {
			return errors.New("CUDA runtime multi-RoPE override changes compiled policy")
		}
	case tensor.OpAttention:
		initial, next := node.Attrs.(tensor.AttentionAttributes), attributes.(tensor.AttentionAttributes)
		queryStart, logicalTokens := uint64(next.QueryStart), uint64(next.KeyValueTokens)
		initial.QueryStart, next.QueryStart = 0, 0
		initial.KeyValueTokens, next.KeyValueTokens = 0, 0
		capacity := node.Inputs[1].Shape.Dims[2]
		logicalTokens = cmp.Or(logicalTokens, capacity)
		queryTokens := node.Inputs[0].Shape.Dims[2]
		if initial != next || logicalTokens > capacity ||
			(next.Causal && queryStart+queryTokens > logicalTokens) {
			return errors.New("CUDA runtime attention override changes compiled policy or exceeds capacity")
		}
	case tensor.OpCacheAppend:
		initial, next := node.Attrs.(tensor.CacheAppendAttributes), attributes.(tensor.CacheAppendAttributes)
		if initial.Axis != next.Axis || next.Axis >= uint32(node.Shape.Rank) ||
			uint64(next.Offset)+node.Inputs[1].Shape.Dims[next.Axis] > node.Shape.Dims[next.Axis] {
			return errors.New("CUDA runtime cache append exceeds compiled capacity")
		}
	default:
		return fmt.Errorf("CUDA operation %s has no runtime parameters", node.Op)
	}
	return nil
}

// OutputSlot: compiled output ordinal.
type OutputSlot uint32

// OutputSlot resolves one graph output to its stable ordinal.
func (c *CompiledGraph) OutputSlot(output *tensor.Tensor) (OutputSlot, bool) {
	if c == nil {
		return 0, false
	}
	index, ok := c.outputIndexes[output]
	return OutputSlot(index), ok
}

// NewRetainedTargets allocates target slots for this compiled graph.
func (c *CompiledGraph) NewRetainedTargets() *RetainedTargets {
	if c == nil {
		return nil
	}
	return &RetainedTargets{compiled: c, values: make([]DeviceValue, len(c.outputs))}
}

// Set assigns one compiled output slot.
func (t *RetainedTargets) Set(output *tensor.Tensor, value DeviceValue) error {
	if t == nil || t.compiled == nil {
		return errors.New("CUDA retained targets are unavailable")
	}
	index, ok := t.compiled.outputIndexes[output]
	if !ok {
		return errors.New("CUDA retained target is not a compiled output")
	}
	return t.SetSlot(OutputSlot(index), value)
}

// SetSlot assigns one precompiled output slot.
func (t *RetainedTargets) SetSlot(slot OutputSlot, value DeviceValue) error {
	if t == nil || t.compiled == nil || uint64(slot) >= uint64(len(t.values)) {
		return errors.New("CUDA retained target slot is invalid")
	}
	t.values[slot] = value
	return nil
}

// CompiledGraph: validated order and memory plan for repeated execution.
type CompiledGraph struct {
	// serial is this compilation's unique number; see compiledSerials.
	serial         uint64
	outputs        []*tensor.Tensor
	outputIndexes  map[*tensor.Tensor]int
	outputViews    []retainedStorageView
	outputAliases  []bool
	order          []*tensor.Tensor
	orderIndexes   map[*tensor.Tensor]int
	inputCount     uint32
	operandSlots   []int
	nodes          []compiledNode
	launches       []int
	attributeSlots []dynamicAttributeSlot
	attributeWords int
	memory         planner.Plan
	// fusions: rewrite-only descriptors; released after launch compilation.
	fusions         map[*tensor.Tensor]*compiledFusion
	elided          map[*tensor.Tensor]struct{}
	q8Emit          map[*tensor.Tensor]struct{}
	targetContracts []tensor.OutputTargetContract
	externalOutputs bool
	skipped         map[*tensor.Tensor]struct{}
	needBlas        bool
	q8InputBytes    uint64
	// matmulStagingBytes: shared exact-weight or tensor-core-input scratch.
	matmulStagingBytes uint64
	// attentionScoreBytes: cuBLAS attention score staging ([heads][chunk][keys] F32)
	attentionScoreBytes uint64
}

// IndexedGraph owns compiled topology and indexed device inputs.
type IndexedGraph struct {
	Graph  *CompiledGraph
	Inputs *DeviceInputs
}

type compiledNode struct {
	operandOffset int
	fusion        *compiledFusion
	view          tensor.StorageView
	aliases       bool
	skipped       bool
	launcher      nodeLauncher
}

type compiledFusionKind uint8

const (
	compiledFusionWeightedRMS compiledFusionKind = iota + 1
	compiledFusionActivatedGate
	compiledFusionGELUTanh
	compiledFusionLayerNormModulate
	compiledFusionBroadcastGateAdd
	compiledFusionWeightedRMSGate
	compiledFusionBF16Append
	compiledFusionBF16ArgmaxPartials
	compiledFusionBF16ArgmaxReduction
	compiledFusionRopeAppend
	compiledFusionBF16Gate
	compiledFusionBF16ProjAdd
	compiledFusionBF16Attention
	compiledFusionQ8ArgmaxPartials
	compiledFusionQ8ArgmaxReduction
)

type compiledFusion struct {
	kind          compiledFusionKind
	operandOffset int
	operandCount  int
	operands      []*tensor.Tensor
	emitQ8        bool
	peer          *tensor.Tensor
	weightedRMS   weightedRMSFusion
	activatedGate activatedGateFusion
	geluTanh      geluTanhFusion
	layerNorm     layerNormModulateFusion
	broadcastGate broadcastGateAddFusion
	weightedGate  weightedRMSGateFusion
	bf16Append    bf16AppendFusion
	ropeAppend    ropeAppendFusion
	bf16Gate      bf16GateFusion
	bf16ProjAdd   bf16ProjAddFusion
	bf16Attention bf16AttentionFusion
}

type dynamicAttributeSlot struct {
	node   *tensor.Tensor
	index  int
	offset int
	words  int
}

// graphPointerTable resolves setup-only tensor addresses.
type graphPointerTable struct {
	indexes map[*tensor.Tensor]int
	values  []driver.DevicePtr
}

func newGraphPointerTable(compiled *CompiledGraph, scratch *[]driver.DevicePtr) graphPointerTable {
	values := prepareAttributePointers(scratch, len(compiled.order))
	return graphPointerTable{indexes: compiled.orderIndexes, values: values}
}

func (p graphPointerTable) get(node *tensor.Tensor) driver.DevicePtr {
	return p.values[p.indexes[node]]
}

func (p graphPointerTable) set(node *tensor.Tensor, pointer driver.DevicePtr) {
	p.values[p.indexes[node]] = pointer
}

// prepareAttributePointers resets graph-indexed invocation addresses.
func prepareAttributePointers(scratch *[]driver.DevicePtr, count int) []driver.DevicePtr {
	if cap(*scratch) < count {
		*scratch = make([]driver.DevicePtr, count)
	}
	values := (*scratch)[:count]
	clear(values)
	return values
}

// launchPointerFrame: precompiled output/input slots.
type launchPointerFrame struct {
	values    []driver.DevicePtr
	slots     []int
	attribute driver.DevicePtr
}

func (p launchPointerFrame) output() driver.DevicePtr {
	return p.values[p.slots[0]]
}

func (p launchPointerFrame) input(index int) driver.DevicePtr {
	return p.values[p.slots[index+1]]
}

func compileFusion(compiled *CompiledGraph, node *tensor.Tensor) (*compiledFusion, error) {
	fusion := compiled.fusions[node]
	if fusion == nil {
		return nil, nil
	}
	fusion.operandOffset = len(compiled.operandSlots)
	fusion.operandCount = len(fusion.operands) + 1
	fusion.emitQ8 = hasTensor(compiled.q8Emit, node)
	appendSlot := func(operand *tensor.Tensor) error {
		slot, ok := compiled.orderIndexes[operand]
		if !ok {
			return errors.New("CUDA fusion operand is outside the compiled graph")
		}
		compiled.operandSlots = append(compiled.operandSlots, slot)
		return nil
	}
	if err := appendSlot(node); err != nil {
		return nil, err
	}
	for _, operand := range fusion.operands {
		if err := appendSlot(operand); err != nil {
			return nil, err
		}
	}
	fusion.operands = nil
	return fusion, nil
}

func hasTensor[T any](values map[*tensor.Tensor]T, node *tensor.Tensor) bool {
	_, ok := values[node]
	return ok
}

const conv2DStagingBytes = uint64(128 << 20)

type retainedStorageView struct {
	source      *tensor.Tensor
	sourceIndex int
	byteOffset  uint64
}

func resolveRetainedStorageView(output *tensor.Tensor) (retainedStorageView, bool, error) {
	current := output
	var byteOffset uint64
	for {
		view, aliases, err := tensor.ResolveStorageView(current)
		if err != nil {
			return retainedStorageView{}, false, err
		}
		if !aliases {
			break
		}
		offset, err := view.ByteOffset(current.Type)
		if err != nil || byteOffset > math.MaxUint64-offset {
			return retainedStorageView{}, false, errors.New("CUDA retained view offset overflows")
		}
		byteOffset += offset
		current = current.Inputs[view.Input]
	}
	if current == output || current.Op == tensor.OpInput {
		return retainedStorageView{}, false, nil
	}
	return retainedStorageView{source: current, byteOffset: byteOffset}, true, nil
}

type deviceAddressRange struct {
	first uint64
	last  uint64
}

func newDeviceAddressRange(pointer driver.DevicePtr, bytes uint64) (deviceAddressRange, error) {
	first := uint64(pointer)
	if pointer == 0 || bytes == 0 || first > math.MaxUint64-bytes {
		return deviceAddressRange{}, errors.New("CUDA device range is invalid")
	}
	return deviceAddressRange{first: first, last: first + bytes}, nil
}

func deviceRangesOverlap(left, right deviceAddressRange) bool {
	return left.first < right.last && right.first < left.last
}

func validateRetainedTargetAlias(
	node *tensor.Tensor,
	target DeviceValue,
	contract tensor.OutputTargetContract,
	pointers graphPointerTable,
) error {
	outputRange, err := newDeviceAddressRange(target.Pointer, contract.Bytes)
	if err != nil {
		return err
	}
	for index, input := range node.Inputs {
		inputBytes, sizeErr := input.Shape.Bytes(input.Type)
		if sizeErr != nil {
			return sizeErr
		}
		inputRange, rangeErr := newDeviceAddressRange(pointers.get(input), inputBytes)
		if rangeErr != nil {
			return rangeErr
		}
		if !deviceRangesOverlap(outputRange, inputRange) {
			continue
		}
		alias := contract.Alias
		if alias == nil || alias.Input != index || target.Pointer != pointers.get(input) ||
			alias.InitializedBytes != inputBytes ||
			alias.WriteOffsetBytes > contract.Bytes ||
			alias.WriteBytes > contract.Bytes-alias.WriteOffsetBytes {
			return fmt.Errorf(
				"CUDA retained target for tensor %d overlaps input %d without an exact alias contract",
				node.ID, index,
			)
		}
	}
	return nil
}

func retainedOutputLayout(
	compiled *CompiledGraph,
	storage []bool,
	offsets []uint64,
) (uint64, error) {
	var total uint64
	for index, node := range compiled.order {
		if node.Op == tensor.OpInput || !storage[index] {
			continue
		}
		offset, ok := checked.Align(total, deviceAllocationAlignment)
		if !ok {
			return 0, errors.New("CUDA retained output offset overflows")
		}
		bytes, err := node.Shape.Bytes(node.Type)
		if err != nil || offset > math.MaxUint64-bytes {
			return 0, errors.New("CUDA retained output size overflows")
		}
		offsets[index] = offset
		total = offset + bytes
	}
	return total, nil
}

// Compile: validates and plans an immutable tensor graph.
func Compile(outputs ...*tensor.Tensor) (*CompiledGraph, error) {
	program, err := tensor.CompileProgram(outputs...)
	if err != nil {
		return nil, err
	}
	return CompileProgram(program)
}

// CompileExternal plans outputs in caller-owned device buffers.
func CompileExternal(outputs ...*tensor.Tensor) (*CompiledGraph, error) {
	program, err := tensor.CompileProgram(outputs...)
	if err != nil {
		return nil, err
	}
	return CompileExternalProgram(program)
}

// CompileIndexed compiles managed outputs and allocates indexed inputs.
func CompileIndexed(outputs ...*tensor.Tensor) (*IndexedGraph, error) {
	return indexedGraph(Compile(outputs...))
}

// CompileExternalIndexed compiles caller-owned outputs and indexed inputs.
func CompileExternalIndexed(outputs ...*tensor.Tensor) (*IndexedGraph, error) {
	return indexedGraph(CompileExternal(outputs...))
}

func indexedGraph(graph *CompiledGraph, err error) (*IndexedGraph, error) {
	if err != nil {
		return nil, err
	}
	return &IndexedGraph{Graph: graph, Inputs: graph.NewDeviceInputs()}, nil
}

// CompileProgram plans one validated neutral tensor program for CUDA.
func CompileProgram(program tensor.Program) (*CompiledGraph, error) {
	return compileGraph(false, program)
}

// CompileExternalProgram plans a neutral program with caller-owned outputs.
func CompileExternalProgram(program tensor.Program) (*CompiledGraph, error) {
	return compileGraph(true, program)
}

func compileGraph(externalOutputs bool, program tensor.Program) (*CompiledGraph, error) {
	if err := program.RequireBackend(tensor.BackendCUDA); err != nil {
		return nil, err
	}
	outputs := program.Outputs()
	order := program.Order()
	compiled := &CompiledGraph{
		serial:          nextCompiledSerial(),
		outputs:         slices.Clone(outputs),
		outputIndexes:   make(map[*tensor.Tensor]int, len(outputs)),
		outputViews:     make([]retainedStorageView, len(outputs)),
		outputAliases:   make([]bool, len(outputs)),
		targetContracts: make([]tensor.OutputTargetContract, len(outputs)),
		externalOutputs: externalOutputs,
		order:           order,
		orderIndexes:    make(map[*tensor.Tensor]int, len(order)),
	}
	for index, node := range order {
		compiled.orderIndexes[node] = index
		words, slotErr := tensor.RuntimeAttributeWords(node, node.Attrs, nil)
		if slotErr != nil {
			return nil, slotErr
		}
		if words > 0 {
			compiled.attributeSlots = append(compiled.attributeSlots, dynamicAttributeSlot{
				node: node, index: index, offset: compiled.attributeWords, words: words,
			})
			compiled.attributeWords += words
		}
	}
	uses := make(map[*tensor.Tensor]int, len(order))
	consumers := make(map[*tensor.Tensor][]*tensor.Tensor, len(order))
	outputSet := make(map[*tensor.Tensor]struct{}, len(outputs))
	for index, output := range outputs {
		if _, duplicate := compiled.outputIndexes[output]; duplicate {
			return nil, errors.New("CUDA compiled graph contains a duplicate output")
		}
		compiled.outputIndexes[output] = index
		outputSet[output] = struct{}{}
		view, aliases, viewErr := resolveRetainedStorageView(output)
		if viewErr != nil {
			return nil, viewErr
		}
		compiled.outputViews[index] = view
		compiled.outputAliases[index] = aliases
		if aliases {
			compiled.outputViews[index].sourceIndex = compiled.orderIndexes[view.source]
		}
		contract, contractErr := tensor.CompileOutputTargetContract(output)
		if contractErr != nil {
			return nil, contractErr
		}
		compiled.targetContracts[index] = contract
	}
	for _, node := range order {
		for _, input := range node.Inputs {
			uses[input]++
			consumers[input] = append(consumers[input], node)
		}
	}
	for _, node := range order {
		if (node.Op == tensor.OpMulMat || node.Op == tensor.OpGroupedMulMat) &&
			(node.Inputs[0].Type == dtype.F32 || node.Inputs[0].Type == dtype.BF16) {
			compiled.needBlas = true
		}
		if node.Op == tensor.OpMulMat &&
			(node.Inputs[0].Type == dtype.F16 || node.Inputs[0].Type == dtype.BF16 ||
				node.Inputs[0].Type == dtype.F8E4M3) {
			// Multi-token exact paths stage F32 weights. Explicit BF16 tensor-core
			// paths stage BF16 activations. Decode reads native weights directly.
			inner := node.Inputs[0].Shape.Dims[0]
			leftRows := node.Inputs[0].Shape.Dims[1]
			rightRows := node.Inputs[1].Shape.Dims[1]
			if rightRows != 1 {
				compiled.needBlas = true
				stagingRows, stagingWidth := leftRows, f32ScalarBytes
				if tensorCoreMulMat(node) {
					// Tensor-core mul_mat stages the activation in the
					// weight's 2-byte dtype, never the weight.
					stagingRows, stagingWidth = rightRows, bf16ScalarBytes
				}
				if inner > math.MaxUint64/stagingRows {
					return nil, errors.New("half-precision mul_mat staging size overflows")
				}
				stagingElements := inner * stagingRows
				if stagingElements > math.MaxUint64/stagingWidth {
					return nil, errors.New("half-precision mul_mat staging size overflows")
				}
				stagingBytes := stagingElements * stagingWidth
				if stagingWidth == f32ScalarBytes {
					stagingBytes = nativeWeightStagingBytes(inner, stagingRows)
				}
				compiled.matmulStagingBytes = max(compiled.matmulStagingBytes, stagingBytes)
			}
		}
		if quantStagedMulMat(node) {
			// Quantized weights past the column floor prefill through f16
			// staging and the tensor-core GEMM; the reservation covers one
			// weight chunk and the packed activation.
			inner := node.Inputs[0].Shape.Dims[0]
			leftRows := node.Inputs[0].Shape.Dims[1]
			rightRows := node.Inputs[1].Shape.Dims[1]
			if rightRows >= uint64(quantStagedColumnFloor) {
				if inner > math.MaxUint64/max(leftRows, rightRows)/f32ScalarBytes {
					return nil, errors.New("quantized mul_mat staging size overflows")
				}
				compiled.needBlas = true
				compiled.matmulStagingBytes = max(
					compiled.matmulStagingBytes, quantStagedStagingBytes(inner, leftRows, rightRows),
				)
			}
		}
		if node.Op == tensor.OpConv2D {
			attributes, ok := node.Attrs.(tensor.Conv2DAttributes)
			if ok && !attributes.Depthwise && node.Inputs[0].Type == dtype.F32 &&
				node.Inputs[1].Type == dtype.F32 {
				compiled.needBlas = true
				compiled.matmulStagingBytes = max(compiled.matmulStagingBytes, conv2DStagingBytes)
			}
		}
		// Every column count stages: prefill chunks its columns through
		// the span kernel, so the workspace holds the whole right operand.
		if fast := node.Op == tensor.OpMulMat && len(node.Inputs) == 2 &&
			q8InputFastPathType(node.Inputs[0].Type); fast &&
			node.Inputs[1].Shape.Rank == 2 {
			elements, elementErr := node.Inputs[1].Shape.Elements()
			if elementErr != nil || elements%q8InputTraits.BlockSize != 0 ||
				elements/q8InputTraits.BlockSize > math.MaxUint64/q8InputTraits.TypeSize {
				return nil, fmt.Errorf("%s mul_mat input storage overflows", node.Inputs[0].Type)
			}
			compiled.q8InputBytes = max(
				compiled.q8InputBytes,
				elements/q8InputTraits.BlockSize*q8InputTraits.TypeSize,
			)
		}
	}
	dependencies := compileGraphRewrites(compiled, order, uses, consumers, outputSet)
	for _, node := range order {
		if node.Op != tensor.OpAttention {
			continue
		}
		descriptor := compiled.fusions[node]
		if descriptor != nil && descriptor.kind == compiledFusionBF16Attention {
			bytes, ok := blasBF16AttentionStagingBytes(descriptor.bf16Attention)
			if !ok {
				return nil, errors.New("fused attention BF16 staging size overflows")
			}
			compiled.needBlas = true
			compiled.attentionScoreBytes = max(compiled.attentionScoreBytes, bytes)
			continue
		}
		if bytes, ok := blasAttentionScoreBytes(node); ok {
			compiled.needBlas = true
			compiled.attentionScoreBytes = max(compiled.attentionScoreBytes, bytes)
		}
	}
	plannerExcluded := compiled.skipped
	if externalOutputs {
		plannerExcluded = make(map[*tensor.Tensor]struct{}, len(compiled.skipped)+len(outputs))
		for node := range compiled.skipped {
			plannerExcluded[node] = struct{}{}
		}
		for _, output := range outputs {
			plannerExcluded[output] = struct{}{}
		}
	}
	memory, err := planner.BuildWithRewrites(outputs, deviceAllocationAlignment, dependencies, plannerExcluded)
	if err != nil {
		return nil, err
	}
	compiled.memory = memory
	compiled.nodes = make([]compiledNode, len(order))
	for index, node := range order {
		view, aliases, viewErr := tensor.ResolveStorageView(node)
		if viewErr != nil {
			return nil, viewErr
		}
		_, skipped := compiled.skipped[node]
		offset := -int(compiled.inputCount) - 1
		if node.Op != tensor.OpInput {
			offset = len(compiled.operandSlots)
			compiled.operandSlots = append(compiled.operandSlots, index)
			for _, input := range node.Inputs {
				compiled.operandSlots = append(compiled.operandSlots, compiled.orderIndexes[input])
			}
		} else {
			compiled.inputCount++
		}
		fusion, fusionErr := compileFusion(compiled, node)
		if fusionErr != nil {
			return nil, fusionErr
		}
		frame := compiledNode{
			operandOffset: offset, fusion: fusion,
			view: view, aliases: aliases, skipped: skipped,
		}
		if node.Op != tensor.OpInput {
			descriptor, ok := tensor.DescribeOperation(node.Op)
			if !ok || descriptor.CUDA == tensor.CUDAProgramNone {
				return nil, fmt.Errorf("CUDA operation %s has no compiled launch program", node.Op)
			}
			if int(descriptor.CUDA) >= len(nodeLaunchers) || nodeLaunchers[descriptor.CUDA] == nil {
				return nil, fmt.Errorf("CUDA operation %s has no compiled launcher", node.Op)
			}
			frame.launcher = nodeLaunchers[descriptor.CUDA]
		}
		compiled.nodes[index] = frame
		_, elided := compiled.elided[node]
		if node.Op != tensor.OpInput && !skipped && !elided {
			compiled.launches = append(compiled.launches, index)
		}
	}
	dumpOpCounts(compiled)
	compiled.fusions = nil
	compiled.skipped = nil
	compiled.elided = nil
	compiled.q8Emit = nil
	return compiled, nil
}

func nativeWeightStagingBytes(inner, rows uint64) uint64 {
	rowBytes := inner * 4
	return min(inner*rows*4, max(rowBytes, nativeWeightStagingLimitBytes))
}

func New(deviceOrdinal int) (*Executor, error) {
	worker, err := device.New(deviceOrdinal)
	if err != nil {
		return nil, err
	}
	return &Executor{worker: worker, ownsWorker: true}, nil
}

// NewWithWorker: binds executor to caller-owned CUDA worker/context
func NewWithWorker(worker *device.Worker) (*Executor, error) {
	if worker == nil {
		return nil, errors.New("CUDA executor requires a worker")
	}
	return &Executor{worker: worker}, nil
}

func (e *Executor) Close() error {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	if e.worker == nil {
		return nil
	}
	var errs []error
	if err := e.worker.Do(context.Background(), func(state *device.State) error {
		return e.closeResources(state)
	}); err != nil {
		errs = append(errs, err)
	}
	if e.ownsWorker {
		if err := e.worker.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	e.worker = nil
	return errors.Join(errs...)
}

// DropGraphExecs destroys every instantiated graph exec. Callers that free
// device memory a captured graph may reference (a resident session releasing
// a program's weights) invoke it on the executor's worker turn, with that
// turn's state, before the free; the next execution captures afresh.
func (e *Executor) DropGraphExecs(state *device.State) error {
	if e == nil || state == nil {
		return errors.New("CUDA executor drop requires the worker state")
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed {
		return nil
	}
	return e.resources.graphExecs.drop(state)
}

// Metrics returns executor-local graph replay and arena evidence. Reading the
// snapshot is serialized on the CUDA worker with resource mutations.
func (e *Executor) Metrics(ctx context.Context) (ExecutionMetrics, error) {
	if e == nil {
		return ExecutionMetrics{}, errors.New("CUDA executor is closed")
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.worker == nil {
		return ExecutionMetrics{}, errors.New("CUDA executor is closed")
	}
	var metrics ExecutionMetrics
	err := e.worker.Do(ctx, func(_ *device.State) error {
		cache := &e.resources.graphExecs
		arena := e.resources.arenaUse
		metrics = ExecutionMetrics{
			GraphCacheHits:         cache.hits,
			GraphCacheMisses:       cache.misses,
			GraphCaptures:          cache.captures,
			GraphUpdates:           cache.updates,
			GraphInstantiations:    cache.instantiations,
			GraphEvictions:         cache.evictions,
			GraphUpdateFallbacks:   cache.updateFallbacks,
			GraphDrops:             cache.drops,
			GraphCacheEntries:      uint64(len(cache.entries)),
			GraphCacheCapacity:     uint64(graphExecCacheCapacity),
			ArenaRequiredBytes:     arena.requiredBytes,
			ArenaPeakRequiredBytes: arena.peakRequiredBytes,
			ArenaCommittedBytes:    e.resources.arenaSize,
			ArenaGrowths:           arena.growths,
		}
		if metrics.ArenaCommittedBytes >= metrics.ArenaRequiredBytes {
			metrics.ArenaUnusedBytes = metrics.ArenaCommittedBytes - metrics.ArenaRequiredBytes
		}
		return nil
	})
	return metrics, err
}

// Execute: evaluates outputs and returns host copies of those tensors
func (e *Executor) Execute(
	ctx context.Context,
	outputs []*tensor.Tensor,
	feeds map[*tensor.Tensor]reference.Value,
) (map[*tensor.Tensor]reference.Value, error) {
	program, err := tensor.CompileProgram(outputs...)
	if err != nil {
		return nil, err
	}
	return e.ExecuteProgram(ctx, program, feeds)
}

// ExecuteProgram evaluates one validated neutral tensor program on CUDA.
func (e *Executor) ExecuteProgram(
	ctx context.Context,
	program tensor.Program,
	feeds map[*tensor.Tensor]reference.Value,
) (map[*tensor.Tensor]reference.Value, error) {
	compiled, err := CompileProgram(program)
	if err != nil {
		return nil, err
	}
	return e.ExecuteCompiled(ctx, compiled, feeds, nil)
}

// ExecuteCompiled: reuses validated topology and memory planning.
func (e *Executor) ExecuteCompiled(
	ctx context.Context,
	compiled *CompiledGraph,
	feeds map[*tensor.Tensor]reference.Value,
	inputs *DeviceInputs,
) (map[*tensor.Tensor]reference.Value, error) {
	return hostResult(e.runCompiled(ctx, compiled, feeds, inputs, nil, nil, false, false))
}

// ExecuteRetainedCompiled: indexed retained execution.
func (e *Executor) ExecuteRetainedCompiled(
	ctx context.Context,
	compiled *CompiledGraph,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceInputs *DeviceInputs,
	targets *RetainedTargets,
	attributes *RuntimeAttributes,
) (*RetainedOutputs, error) {
	execution, err := e.runCompiled(ctx, compiled, hostFeeds, deviceInputs, targets, attributes, true, true)
	if err != nil {
		return nil, err
	}
	return &RetainedOutputs{
		executor: e, compiled: compiled, values: execution.values, leases: execution.leases,
	}, nil
}

// ExecuteRetainedCompiledOnce: retained outputs without replay-cache ownership.
func (e *Executor) ExecuteRetainedCompiledOnce(
	ctx context.Context,
	compiled *CompiledGraph,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceInputs *DeviceInputs,
	targets *RetainedTargets,
	attributes *RuntimeAttributes,
) (*RetainedOutputs, error) {
	execution, err := e.runCompiled(ctx, compiled, hostFeeds, deviceInputs, targets, attributes, true, false)
	if err != nil {
		return nil, err
	}
	return &RetainedOutputs{
		executor: e, compiled: compiled, values: execution.values, leases: execution.leases,
	}, nil
}

// PrepareCompiled loads modules and sizes persistent execution storage.
func (e *Executor) PrepareCompiled(ctx context.Context, compiled *CompiledGraph) error {
	if e == nil || compiled == nil {
		return errors.New("CUDA executor or compiled graph absent")
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.worker == nil {
		return errors.New("CUDA executor is closed")
	}
	return e.worker.Do(ctx, func(state *device.State) error {
		resources, err := e.ensureResources(
			state, compiled.needBlas, compiled.q8InputBytes,
			compiled.attentionScoreBytes, compiled.matmulStagingBytes,
		)
		if err != nil {
			return err
		}
		_, err = resources.ensureArena(state, compiled.memory.ArenaSize)
		return err
	})
}

func (e *Executor) runCompiled(
	ctx context.Context,
	compiled *CompiledGraph,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceInputs *DeviceInputs,
	targets *RetainedTargets,
	attributes *RuntimeAttributes,
	retain, cacheGraph bool,
) (*executionResult, error) {
	if e == nil {
		return nil, errors.New("CUDA executor is closed")
	}
	if compiled == nil {
		return nil, errors.New("CUDA compiled graph is nil")
	}
	if deviceInputs != nil && (deviceInputs.compiled != compiled || len(deviceInputs.Pointers) != int(compiled.inputCount)) {
		return nil, errors.New("CUDA device inputs belong to another compiled graph")
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.worker == nil {
		return nil, errors.New("CUDA executor is closed")
	}
	var result *executionResult
	err := e.worker.Do(ctx, func(state *device.State) error {
		resources, resourceErr := e.ensureResources(
			state, compiled.needBlas, compiled.q8InputBytes,
			compiled.attentionScoreBytes, compiled.matmulStagingBytes,
		)
		if resourceErr != nil {
			return resourceErr
		}
		arena, arenaErr := resources.ensureArena(state, compiled.memory.ArenaSize)
		if arenaErr != nil {
			return arenaErr
		}
		var executeErr error
		var graphExecs *graphExecCache
		if cacheGraph {
			graphExecs = &resources.graphExecs
		}
		result, executeErr = execute(
			state,
			compiled,
			hostFeeds,
			deviceInputs,
			targets,
			attributes,
			resources.functions,
			resources.blas,
			&resources.q8Input,
			arena,
			&resources.scratch,
			&resources.buffers,
			graphExecs,
			retain,
		)
		if executeErr == nil && !cacheGraph && compiled.matmulStagingBytes > 0 &&
			resources.blas != nil && resources.blas.staging != 0 {
			// A one-shot staged run (no replay-cache ownership -- prefill,
			// not steady-state decode) releases the multi-token weight
			// staging reservation at completion instead of riding the
			// whole session: nothing decode-resident coexists with prefill
			// scratch, and the next staged run re-reserves. Replay-cached
			// graphs keep their reservation because their captured frames
			// key on the staging pointer. The free follows the grow path's
			// established free-after-submission contract on this stream.
			if freeErr := state.Driver.MemFree(resources.blas.staging); freeErr != nil {
				return freeErr
			}
			resources.blas.staging = 0
			resources.blas.stagingBytes = 0
			resources.blas.stagedNode = nil
		}
		dumpResourceMemory(resources)
		return executeErr
	})
	return result, err
}

func launchCompiledFusion(
	state *device.State,
	functions functionSet,
	blas *blasState,
	q8Input *q8InputState,
	node *tensor.Tensor,
	fusion *compiledFusion,
	pointers launchPointerFrame,
	auxiliaryAttributePointer driver.DevicePtr,
) (string, error) {
	if fusion == nil {
		return "", nil
	}
	switch fusion.kind {
	case compiledFusionWeightedRMS:
		return "weighted_rms_norm", launchWeightedRMSNorm(state, functions, q8Input, node, fusion.weightedRMS, fusion.emitQ8, pointers)
	case compiledFusionActivatedGate:
		return "activated_gate", launchActivatedGate(state, functions, q8Input, node, fusion.activatedGate, fusion.emitQ8, pointers)
	case compiledFusionGELUTanh:
		return "gelu_tanh", launchGELUTanh(state, functions, node, fusion.geluTanh, pointers)
	case compiledFusionLayerNormModulate:
		return "layer_norm_modulate", launchLayerNormModulate(state, functions, node, fusion.layerNorm, pointers)
	case compiledFusionBroadcastGateAdd:
		return "broadcast_gate_add", launchBroadcastGateAdd(state, functions, node, fusion.broadcastGate, pointers)
	case compiledFusionWeightedRMSGate:
		return "weighted_rms_gate", launchWeightedRMSGate(state, functions, q8Input, node, fusion.weightedGate, fusion.emitQ8, pointers)
	case compiledFusionBF16Append:
		return "bf16_append", launchBF16Append(
			state, functions, node, fusion.bf16Append, pointers, pointers.attribute,
		)
	case compiledFusionBF16ArgmaxPartials:
		return "bf16_argmax", launchBF16ArgmaxPartials(state, functions, node, pointers)
	case compiledFusionBF16ArgmaxReduction:
		return "bf16_argmax", launchQ8ArgmaxReduction(state, functions, fusion.peer, node, pointers)
	case compiledFusionRopeAppend:
		return "rope_append", launchRopeAppend(
			state, functions, node, fusion.ropeAppend, pointers,
			auxiliaryAttributePointer,
		)
	case compiledFusionBF16Gate:
		return "bf16_gate", launchBF16Gate(state, functions, node, fusion.bf16Gate, pointers)
	case compiledFusionBF16ProjAdd:
		return "bf16_projection_add", launchBF16ProjAdd(state, functions, node, fusion.bf16ProjAdd, pointers)
	case compiledFusionBF16Attention:
		return "bf16_attention", launchBF16Attention(state, functions, blas, node, fusion.bf16Attention, pointers)
	case compiledFusionQ8ArgmaxPartials:
		return "q8_argmax", launchQ8ArgmaxPartials(state, functions, q8Input, node, pointers)
	case compiledFusionQ8ArgmaxReduction:
		return "q8_argmax", launchQ8ArgmaxReduction(state, functions, fusion.peer, node, pointers)
	default:
		return "", errors.New("compiled CUDA fusion is invalid")
	}
}

func execute(
	state *device.State,
	compiled *CompiledGraph,
	feeds map[*tensor.Tensor]reference.Value,
	deviceInputs *DeviceInputs,
	retainedTargets *RetainedTargets,
	runtimeAttributes *RuntimeAttributes,
	functions functionSet,
	blas *blasState,
	q8Input *q8InputState,
	arena driver.DevicePtr,
	scratch *executionScratch,
	buffers *deviceBufferPool,
	execCache *graphExecCache,
	retainOutputs bool,
) (_ *executionResult, err error) {
	outputs := compiled.outputs
	order := compiled.order
	plan := compiled.memory
	if runtimeAttributes != nil &&
		(runtimeAttributes.compiled != compiled || len(runtimeAttributes.values) != len(order)) {
		return nil, errors.New("CUDA runtime attributes belong to another compiled graph")
	}
	var resolvedAttributes []tensor.Attributes
	if runtimeAttributes != nil {
		if cap(scratch.attributes) < len(order) {
			scratch.attributes = make([]tensor.Attributes, len(order))
		}
		resolvedAttributes = scratch.attributes[:len(order)]
		clear(resolvedAttributes)
		for index, attributes := range runtimeAttributes.values {
			if attributes == nil {
				continue
			}
			resolved, resolveErr := resolveRuntimeAttributes(attributes)
			if resolveErr != nil {
				return nil, resolveErr
			}
			resolvedAttributes[index] = resolved
		}
	}

	if plan.ArenaSize > 0 && arena == 0 {
		return nil, errors.New("CUDA executor arena is unavailable")
	}
	submitted := false
	defer func() {
		if submitted {
			err = errors.Join(err, state.Driver.StreamSynchronize(state.Stream))
		}
	}()

	pointers := newGraphPointerTable(compiled, &scratch.pointers)
	var targetValues []DeviceValue
	if retainedTargets != nil {
		if retainedTargets.compiled != compiled || len(retainedTargets.values) != len(outputs) {
			return nil, errors.New("CUDA retained targets belong to another compiled graph")
		}
		targetValues = retainedTargets.values
	}
	if compiled.externalOutputs {
		if len(targetValues) != len(outputs) {
			return nil, errors.New("CUDA external-output graph requires retained targets")
		}
		for _, target := range targetValues {
			if target.Pointer == 0 {
				return nil, errors.New("CUDA external-output graph has an unbound target")
			}
		}
	}
	ownsOutput := func(index int) bool {
		return len(targetValues) == 0 || targetValues[index].Pointer == 0
	}
	retainedValues := make([]DeviceValue, len(outputs))
	retainedLeases := make([]deviceBufferLease, 0, 1)
	for index, target := range targetValues {
		if target.Pointer == 0 {
			continue
		}
		node := outputs[index]
		if !target.Shape.Equal(node.Shape) {
			return nil, errors.New("CUDA retained output target is invalid")
		}
		contract := compiled.targetContracts[index]
		if contract.Alignment == 0 || uint64(target.Pointer)%contract.Alignment != 0 ||
			target.CapacityBytes < contract.Bytes {
			return nil, errors.New("CUDA retained output target capacity is insufficient")
		}
	}
	if cap(scratch.retainedStorage) < len(order) {
		scratch.retainedStorage = make([]bool, len(order))
	}
	if cap(scratch.retainedOffsets) < len(order) {
		scratch.retainedOffsets = make([]uint64, len(order))
	}
	retainedStorage := scratch.retainedStorage[:len(order)]
	retainedOffsets := scratch.retainedOffsets[:len(order)]
	clear(retainedStorage)
	clear(retainedOffsets)
	for index, output := range outputs {
		if !ownsOutput(index) {
			continue
		}
		if compiled.outputAliases[index] {
			retainedStorage[compiled.outputViews[index].sourceIndex] = true
			continue
		}
		retainedStorage[compiled.orderIndexes[output]] = true
	}
	retainedBytes, err := retainedOutputLayout(compiled, retainedStorage, retainedOffsets)
	if err != nil {
		return nil, err
	}
	var retainedBase driver.DevicePtr
	if retainOutputs && retainedBytes > 0 {
		lease, allocateErr := buffers.acquire(state, retainedBytes)
		err = allocateErr
		if err != nil {
			return nil, err
		}
		retainedBase = lease.pointer
		retainedLeases = append(retainedLeases, lease)
	}
	retained := false
	defer func() {
		if retained {
			return
		}
		for _, lease := range retainedLeases {
			buffers.release(lease)
		}
	}()
	inputLeases := make([]deviceBufferLease, 0)
	defer func() {
		for _, lease := range inputLeases {
			buffers.release(lease)
		}
	}()
	for nodeIndex, frame := range compiled.nodes {
		node := compiled.order[nodeIndex]
		if node.Op != tensor.OpInput && node.Type != dtype.F32 {
			return nil, fmt.Errorf("CUDA executor does not support %s for tensor %d", node.Type, node.ID)
		}
		if node.Op != tensor.OpInput {
			if frame.skipped {
				continue
			}
			outputIndex, retainedOutput := compiled.outputIndexes[node]
			retainStorage := retainedStorage[nodeIndex]
			if frame.aliases {
				offset, offsetErr := frame.view.ByteOffset(node.Type)
				input := pointers.get(node.Inputs[frame.view.Input])
				if offsetErr != nil || uint64(input) > math.MaxUint64-offset {
					return nil, errors.New("CUDA storage view offset is invalid")
				}
				retainedAlias := retainedOutput && ownsOutput(outputIndex) && compiled.outputAliases[outputIndex]
				retainedView := compiled.outputViews[outputIndex]
				if retainedAlias ||
					!(retainOutputs && retainedOutput) {
					pointer := input + driver.DevicePtr(offset)
					pointers.set(node, pointer)
					if retainedAlias {
						if retainedView.source == nil {
							return nil, errors.New("CUDA retained storage view is invalid")
						}
						bytes, sizeErr := node.Shape.Bytes(node.Type)
						if sizeErr != nil {
							return nil, sizeErr
						}
						retainedValues[outputIndex] = DeviceValue{
							Pointer: pointer, Shape: node.Shape, CapacityBytes: bytes,
						}
					}
					continue
				}
			}
			if retainOutputs && (retainedOutput || retainStorage) {
				if retainedOutput && len(targetValues) != 0 && targetValues[outputIndex].Pointer != 0 {
					target := targetValues[outputIndex]
					if err := validateRetainedTargetAlias(
						node, target, compiled.targetContracts[outputIndex], pointers,
					); err != nil {
						return nil, err
					}
					retainedValues[outputIndex] = target
					pointers.set(node, target.Pointer)
					continue
				}
				offset := retainedOffsets[nodeIndex]
				if !retainStorage || uint64(retainedBase) > math.MaxUint64-offset {
					return nil, errors.New("CUDA retained output layout is invalid")
				}
				pointer := retainedBase + driver.DevicePtr(offset)
				if retainedOutput {
					bytes, sizeErr := node.Shape.Bytes(node.Type)
					if sizeErr != nil {
						return nil, sizeErr
					}
					retainedValues[outputIndex] = DeviceValue{
						Pointer: pointer, Shape: node.Shape, CapacityBytes: bytes,
					}
				}
				pointers.set(node, pointer)
				continue
			}
			allocation, ok := plan.Allocations[node]
			if !ok {
				return nil, fmt.Errorf("tensor %d has no arena allocation", node.ID)
			}
			if uint64(arena) > math.MaxUint64-allocation.Offset {
				return nil, errors.New("device pointer offset overflows uint64")
			}
			pointers.set(node, arena+driver.DevicePtr(allocation.Offset))
			continue
		}
		pointer, deviceFed := driver.DevicePtr(0), false
		if inputSlot := -frame.operandOffset - 1; deviceInputs != nil && deviceInputs.Pointers[inputSlot] != 0 {
			pointer, deviceFed = deviceInputs.Pointers[inputSlot], true
		}
		if deviceFed {
			if node.Type != dtype.F32 && node.Type != dtype.BF16 && node.Type != dtype.F16 &&
				node.Type != dtype.F8E4M3 && !nativeQuantizedType(node.Type) {
				return nil, fmt.Errorf("CUDA device feed %q has unsupported type %s", node.Name, node.Type)
			}
			if pointer == 0 {
				return nil, fmt.Errorf("device feed for input %q is null", node.Name)
			}
			if _, duplicate := feeds[node]; duplicate {
				return nil, fmt.Errorf("input %q has both host and device feeds", node.Name)
			}
			pointers.set(node, pointer)
			continue
		}
		if node.Type != dtype.F32 {
			return nil, fmt.Errorf("host feed %q has unsupported type %s", node.Name, node.Type)
		}
		value, ok := feeds[node]
		if !ok {
			if embedded, embeddedOK := node.Attrs.(tensor.EmbeddedInputAttributes); embeddedOK {
				value = reference.Value{Shape: node.Shape, Data: embedded.Data}
				ok = true
			}
		}
		if !ok {
			return nil, fmt.Errorf("missing feed for input %q", node.Name)
		}
		if !value.Shape.Equal(node.Shape) {
			return nil, fmt.Errorf("feed shape for %q does not match graph", node.Name)
		}
		elements, storageErr := value.Shape.Elements()
		if storageErr != nil {
			return nil, storageErr
		}
		if value.Storage == reference.ValueMaterialized && elements != uint64(len(value.Data)) {
			return nil, fmt.Errorf("feed storage for %q has %d elements, need %d", node.Name, len(value.Data), elements)
		}
		if value.Storage == reference.ValueImplicitZero && len(value.Data) != 0 {
			return nil, fmt.Errorf("implicit-zero feed %q has materialized data", node.Name)
		}
		bytes, err := node.Shape.Bytes(node.Type)
		if err != nil {
			return nil, err
		}
		lease, err := buffers.acquire(state, bytes)
		if err != nil {
			return nil, err
		}
		inputLeases = append(inputLeases, lease)
		pointers.set(node, lease.pointer)
		if value.Storage == reference.ValueImplicitZero {
			if err := state.Driver.MemsetD32Async(
				lease.pointer,
				0,
				elements,
				state.Stream,
			); err != nil {
				return nil, err
			}
			submitted = true
		} else if value.Storage == reference.ValueMaterialized {
			if err := state.Driver.MemcpyHtoD(lease.pointer, driver.Bytes(value.Data)); err != nil {
				return nil, err
			}
		} else {
			return nil, fmt.Errorf("feed storage for %q is invalid", node.Name)
		}
	}

	attributePointers := prepareAttributePointers(&scratch.attributePointers, len(compiled.order))
	auxiliaryLeases := make([]deviceBufferLease, 0)
	defer func() {
		for _, lease := range auxiliaryLeases {
			buffers.release(lease)
		}
	}()
	if compiled.attributeWords > 0 {
		if cap(scratch.attributeWords) < compiled.attributeWords {
			scratch.attributeWords = make([]uint32, compiled.attributeWords)
		}
		words := scratch.attributeWords[:compiled.attributeWords]
		clear(words)
		bytes := uint64(compiled.attributeWords) * uint64(unsafe.Sizeof(uint32(0)))
		lease, allocateErr := buffers.acquire(state, bytes)
		if allocateErr != nil {
			return nil, allocateErr
		}
		auxiliaryLeases = append(auxiliaryLeases, lease)
		for _, slot := range compiled.attributeSlots {
			attributes := slot.node.Attrs
			if resolvedAttributes != nil && resolvedAttributes[slot.index] != nil {
				attributes = resolvedAttributes[slot.index]
			}
			destination := words[slot.offset : slot.offset+slot.words]
			written, writeErr := tensor.RuntimeAttributeWords(slot.node, attributes, destination)
			if writeErr != nil {
				return nil, writeErr
			}
			if written != slot.words {
				return nil, errors.New("CUDA runtime attribute changes compiled slot size")
			}
			offset := uint64(slot.offset) * uint64(unsafe.Sizeof(uint32(0)))
			attributePointers[slot.index] = lease.pointer + driver.DevicePtr(offset)
		}
		if copyErr := state.Driver.MemcpyHtoD(lease.pointer, driver.Bytes(words)); copyErr != nil {
			return nil, copyErr
		}
	}

	capturing := retainOutputs && execCache != nil
	resetStaged := func() {
		if blas != nil {
			blas.stagedNode = nil
		}
		if q8Input != nil {
			q8Input.stagedNode = nil
		}
	}
	resetStaged()
	runLaunches := func() error {
		for _, nodeIndex := range compiled.launches {
			frame := compiled.nodes[nodeIndex]
			node := compiled.order[nodeIndex]
			operands := launchPointerFrame{
				values:    pointers.values,
				slots:     compiled.operandSlots[frame.operandOffset : frame.operandOffset+len(node.Inputs)+1],
				attribute: attributePointers[nodeIndex],
			}
			if frame.aliases {
				outputIndex, retainedOutput := compiled.outputIndexes[node]
				retainedAlias := retainedOutput && ownsOutput(outputIndex) && compiled.outputAliases[outputIndex]
				if retainedAlias || !(retainOutputs && retainedOutput) {
					continue
				}
			}
			var fusionOperands launchPointerFrame
			if frame.fusion != nil {
				// The fusion frame carries the HOST node's own attribute
				// pointer (a cache append's offset word); dropping it left
				// every fused append reading a zero offset address.
				fusionOperands = launchPointerFrame{
					values:    pointers.values,
					slots:     compiled.operandSlots[frame.fusion.operandOffset : frame.fusion.operandOffset+frame.fusion.operandCount],
					attribute: attributePointers[nodeIndex],
				}
			}
			var auxiliaryAttributePointer driver.DevicePtr
			if frame.fusion != nil && frame.fusion.kind == compiledFusionRopeAppend {
				auxiliaryAttributePointer = attributePointers[frame.fusion.ropeAppend.attributeIndex]
			}
			if label, err := launchCompiledFusion(
				state, functions, blas, q8Input, node, frame.fusion, fusionOperands,
				auxiliaryAttributePointer,
			); label != "" {
				if err != nil {
					return fmt.Errorf("launch tensor %d (%s): %w", node.ID, label, err)
				}
				submitted = true
				continue
			}
			attributes := node.Attrs
			if resolvedAttributes != nil && resolvedAttributes[nodeIndex] != nil {
				attributes = resolvedAttributes[nodeIndex]
			}
			if err := frame.launcher(state, functions, blas, q8Input, node, attributes, operands); err != nil {
				return fmt.Errorf("launch tensor %d (%s): %w", node.ID, node.Op, err)
			}
			submitted = true
		}
		return nil
	}
	var blasStaging, blasScores, q8Staging driver.DevicePtr
	if blas != nil {
		blasStaging, blasScores = blas.staging, blas.scores
	}
	if q8Input != nil {
		q8Staging = q8Input.staging
	}
	scratch.replayFrame = append(
		scratch.replayFrame[:0], arena, blasStaging, blasScores, q8Staging,
	)
	scratch.replayFrame = append(scratch.replayFrame, pointers.values...)
	scratch.replayFrame = append(scratch.replayFrame, attributePointers...)
	frame := scratch.replayFrame
	replayed := false
	if capturing {
		if exec, ok := execCache.match(compiled, frame); ok {
			if err := state.Driver.GraphLaunch(exec, state.Stream); err != nil {
				return nil, err
			}
			submitted = true
			replayed = true
		}
	}
	if !replayed {
		if capturing {
			if captureErr := state.Driver.StreamBeginCapture(state.Stream); captureErr != nil {
				return nil, captureErr
			}
		}
		launchErr := runLaunches()
		if launchErr != nil {
			if capturing {
				graph, _ := state.Driver.StreamEndCapture(state.Stream)
				_ = state.Driver.GraphDestroy(graph)
			}
			return nil, launchErr
		}
		if capturing {
			graph, captureErr := state.Driver.StreamEndCapture(state.Stream)
			if captureErr != nil {
				return nil, captureErr
			}
			defer state.Driver.GraphDestroy(graph)
			exec, storeErr := execCache.store(state, graph, compiled, frame)
			if storeErr != nil {
				return nil, storeErr
			}
			if err := state.Driver.GraphLaunch(exec, state.Stream); err != nil {
				return nil, err
			}
		}
	}
	if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
		return nil, err
	}
	submitted = false
	if retainOutputs {
		retained = true
		return &executionResult{
			values: retainedValues,
			leases: retainedLeases,
		}, nil
	}

	results := make(map[*tensor.Tensor]reference.Value, len(outputs))
	for _, output := range outputs {
		elements, err := output.Shape.Elements()
		if err != nil {
			return nil, err
		}
		if elements > uint64(math.MaxInt) {
			return nil, errors.New("output is too large for host memory")
		}
		data := make([]float32, int(elements))
		if err := state.Driver.MemcpyDtoH(driver.Bytes(data), pointers.get(output)); err != nil {
			return nil, err
		}
		results[output] = reference.Value{Shape: output.Shape, Data: data}
	}
	return &executionResult{host: results}, nil
}

type quantKernelDescriptor struct {
	label   string
	getRows kernelFunctionID
	mulMat  kernelFunctionID
}

var quantKernels = [dtype.Count]quantKernelDescriptor{
	dtype.Q8_0:   {"Q8_0", kernelGetRowsQ80F32, kernelMulMatQ80F32},
	dtype.Q8_1:   {"Q8_1", kernelGetRowsQ81F32, kernelMulMatQ81F32},
	dtype.Q8K:    {"Q8_K", kernelGetRowsQ8KF32, kernelMulMatQ8KF32},
	dtype.Q4_0:   {"Q4_0", kernelGetRowsQ40F32, kernelMulMatQ40F32},
	dtype.Q4_1:   {"Q4_1", kernelGetRowsQ41F32, kernelMulMatQ41F32},
	dtype.Q5_0:   {"Q5_0", kernelGetRowsQ50F32, kernelMulMatQ50F32},
	dtype.Q5_1:   {"Q5_1", kernelGetRowsQ51F32, kernelMulMatQ51F32},
	dtype.Q1_0:   {"Q1_0", kernelGetRowsQ10F32, kernelMulMatQ10F32},
	dtype.Q2_0:   {"Q2_0", kernelGetRowsQ20F32, kernelMulMatQ20F32},
	dtype.TQ2_0:  {"TQ2_0", kernelGetRowsTq20F32, kernelMulMatTq20F32},
	dtype.TQ1_0:  {"TQ1_0", kernelGetRowsTq10F32, kernelMulMatTq10F32},
	dtype.Q2K:    {"Q2_K", kernelGetRowsQ2KF32, kernelMulMatQ2KF32},
	dtype.Q3K:    {"Q3_K", kernelGetRowsQ3KF32, kernelMulMatQ3KF32},
	dtype.Q4K:    {"Q4_K", kernelGetRowsQ4KF32, kernelMulMatQ4KF32},
	dtype.Q5K:    {"Q5_K", kernelGetRowsQ5KF32, kernelMulMatQ5KF32},
	dtype.IQ4XS:  {"IQ4_XS", kernelGetRowsIq4XsF32, kernelMulMatIq4XsF32},
	dtype.IQ4NL:  {"IQ4_NL", kernelGetRowsIq4NlF32, kernelMulMatIq4NlF32},
	dtype.IQ2XXS: {"IQ2_XXS", kernelGetRowsIq2XxsF32, kernelMulMatIq2XxsF32},
	dtype.IQ2XS:  {"IQ2_XS", kernelGetRowsIq2XsF32, kernelMulMatIq2XsF32},
	dtype.IQ2S:   {"IQ2_S", kernelGetRowsIq2SF32, kernelMulMatIq2SF32},
	dtype.IQ3XXS: {"IQ3_XXS", kernelGetRowsIq3XxsF32, kernelMulMatIq3XxsF32},
	dtype.IQ3S:   {"IQ3_S", kernelGetRowsIq3SF32, kernelMulMatIq3SF32},
	dtype.IQ1S:   {"IQ1_S", kernelGetRowsIq1SF32, kernelMulMatIq1SF32},
	dtype.IQ1M:   {"IQ1_M", kernelGetRowsIq1MF32, kernelMulMatIq1MF32},
	dtype.MXFP4:  {"MXFP4", kernelGetRowsMxfp4F32, kernelMulMatMxfp4F32},
	dtype.NVFP4:  {"NVFP4", kernelGetRowsNvfp4F32, kernelMulMatNvfp4F32},
	dtype.Q6K:    {"Q6_K", kernelGetRowsQ6KF32, kernelMulMatQ6KF32},
}

func quantKernel(dataType dtype.Type) (quantKernelDescriptor, bool) {
	if dataType >= dtype.Count {
		return quantKernelDescriptor{}, false
	}
	descriptor := quantKernels[dataType]
	return descriptor, descriptor.label != ""
}

type blasState struct {
	library      *cublas.Library
	handle       cublas.Handle
	staging      driver.DevicePtr
	stagingBytes uint64
	stagedNode   *tensor.Tensor
	// stagedType is the dtype the staged activation was packed to; a
	// node feeding both F16 and BF16 weights repacks when it switches.
	stagedType dtype.Type
	// scores: attention score staging for the strided-batched SGEMM path
	scores     driver.DevicePtr
	scoreBytes uint64
}

var (
	q8InputTraits, _   = dtype.Q8_1.Traits()
	f32ScalarBytes, _  = dtype.F32.ScalarBytes()
	bf16ScalarBytes, _ = dtype.BF16.ScalarBytes()
)

type q8InputState struct {
	staging      driver.DevicePtr
	stagingBytes uint64
	stagedNode   *tensor.Tensor
}

func (e *Executor) ensureResources(
	state *device.State,
	needBlas bool,
	q8InputBytes uint64,
	attentionScoreBytes uint64,
	matmulStagingBytes uint64,
) (*executorResources, error) {
	if e.resources.module == 0 {
		if err := kernel.ValidateAssets(); err != nil {
			return nil, err
		}
		module, err := state.Driver.ModuleLoadData(kernel.OpsF32PTX)
		if err != nil {
			return nil, err
		}
		functions, err := loadFunctions(state.Driver, module)
		if err != nil {
			_ = state.Driver.ModuleUnload(module)
			return nil, err
		}
		if err := configureLargeSharedKernels(state.Driver, functions); err != nil {
			_ = state.Driver.ModuleUnload(module)
			return nil, err
		}
		e.resources.module = module
		e.resources.functions = functions
	}
	if needBlas && e.resources.blas == nil {
		library, err := cublas.Open()
		if err != nil {
			return nil, err
		}
		handle, err := library.Create()
		if err != nil {
			_ = library.Close()
			return nil, err
		}
		if err := library.SetStream(handle, state.Stream); err != nil {
			_ = library.Destroy(handle)
			_ = library.Close()
			return nil, err
		}
		e.resources.blas = &blasState{library: library, handle: handle}
	}
	if matmulStagingBytes > 0 && e.resources.blas != nil && e.resources.blas.stagingBytes < matmulStagingBytes {
		staging, err := state.Driver.MemAlloc(matmulStagingBytes)
		if err != nil {
			return nil, err
		}
		if e.resources.blas.staging != 0 {
			if err := e.resources.graphExecs.drop(state); err != nil {
				_ = state.Driver.MemFree(staging)
				return nil, err
			}
			if err := state.Driver.MemFree(e.resources.blas.staging); err != nil {
				_ = state.Driver.MemFree(staging)
				return nil, err
			}
		}
		e.resources.blas.staging = staging
		e.resources.blas.stagingBytes = matmulStagingBytes
	}
	if attentionScoreBytes > 0 && e.resources.blas != nil && e.resources.blas.scoreBytes < attentionScoreBytes {
		scores, err := state.Driver.MemAlloc(attentionScoreBytes)
		if err != nil {
			return nil, err
		}
		if e.resources.blas.scores != 0 {
			if err := e.resources.graphExecs.drop(state); err != nil {
				_ = state.Driver.MemFree(scores)
				return nil, err
			}
			if err := state.Driver.MemFree(e.resources.blas.scores); err != nil {
				_ = state.Driver.MemFree(scores)
				return nil, err
			}
		}
		e.resources.blas.scores = scores
		e.resources.blas.scoreBytes = attentionScoreBytes
	}
	if q8InputBytes > e.resources.q8Input.stagingBytes {
		staging, err := state.Driver.MemAlloc(q8InputBytes)
		if err != nil {
			return nil, err
		}
		if e.resources.q8Input.staging != 0 {
			if err := e.resources.graphExecs.drop(state); err != nil {
				_ = state.Driver.MemFree(staging)
				return nil, err
			}
			if err := state.Driver.MemFree(e.resources.q8Input.staging); err != nil {
				_ = state.Driver.MemFree(staging)
				return nil, err
			}
		}
		e.resources.q8Input.staging = staging
		e.resources.q8Input.stagingBytes = q8InputBytes
	}
	return &e.resources, nil
}

func (r *executorResources) ensureArena(state *device.State, size uint64) (driver.DevicePtr, error) {
	r.arenaUse.requiredBytes = size
	r.arenaUse.peakRequiredBytes = max(r.arenaUse.peakRequiredBytes, size)
	if size == 0 {
		return 0, nil
	}
	if r.arena != 0 && r.arenaSize >= size {
		return r.arena, nil
	}
	if r.arena != 0 {
		// Captured graphs address the arena directly; none may replay
		// over the range being freed.
		if err := r.graphExecs.drop(state); err != nil {
			return 0, err
		}
		if err := state.Driver.MemFree(r.arena); err != nil {
			return 0, err
		}
		r.arena, r.arenaSize = 0, 0
	}
	next, err := state.Driver.MemAlloc(size)
	if err != nil {
		return 0, err
	}
	r.arena, r.arenaSize = next, size
	r.arenaUse.growths++
	return r.arena, nil
}

func (e *Executor) closeResources(state *device.State) error {
	var errs []error
	if err := e.resources.buffers.close(state); err != nil {
		errs = append(errs, err)
	}
	if e.resources.arena != 0 {
		if err := state.Driver.MemFree(e.resources.arena); err != nil {
			errs = append(errs, err)
		}
		e.resources.arena = 0
		e.resources.arenaSize = 0
	}
	if err := e.resources.graphExecs.close(state); err != nil {
		errs = append(errs, err)
	}
	if e.resources.blas != nil {
		if e.resources.blas.staging != 0 {
			errs = append(errs, state.Driver.MemFree(e.resources.blas.staging))
			e.resources.blas.staging = 0
			e.resources.blas.stagingBytes = 0
			e.resources.blas.stagedNode = nil
		}
		if e.resources.blas.scores != 0 {
			errs = append(errs, state.Driver.MemFree(e.resources.blas.scores))
			e.resources.blas.scores = 0
			e.resources.blas.scoreBytes = 0
		}
		if err := e.resources.blas.library.Destroy(e.resources.blas.handle); err != nil {
			errs = append(errs, err)
		}
		if err := e.resources.blas.library.Close(); err != nil {
			errs = append(errs, err)
		}
		e.resources.blas = nil
	}
	if e.resources.q8Input.staging != 0 {
		errs = append(errs, state.Driver.MemFree(e.resources.q8Input.staging))
		e.resources.q8Input = q8InputState{}
	}
	if e.resources.module != 0 {
		if err := state.Driver.ModuleUnload(e.resources.module); err != nil {
			errs = append(errs, err)
		}
		e.resources.module = 0
		e.resources.functions = functionSet{}
	}
	return errors.Join(errs...)
}
