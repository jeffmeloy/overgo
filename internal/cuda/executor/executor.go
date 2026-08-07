package executor

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/bits"
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
	module    driver.Module
	functions functionSet
	blas      *blasState
	q8Input   q8InputState
	arena     driver.DevicePtr
	arenaSize uint64
	buffers   deviceBufferPool
	graphExec driver.GraphExec
}

const minimumDeviceBufferBytes uint64 = 256

type deviceBufferLease struct {
	pointer driver.DevicePtr
	size    uint64
}

type deviceBufferPool struct {
	free        map[uint64][]driver.DevicePtr
	allocations []driver.DevicePtr
}

func (p *deviceBufferPool) acquire(state *device.State, size uint64) (deviceBufferLease, error) {
	bucket, err := deviceBufferBucket(size)
	if err != nil {
		return deviceBufferLease{}, err
	}
	return p.acquireBucket(state, bucket)
}

func (p *deviceBufferPool) acquireExact(state *device.State, size uint64) (deviceBufferLease, error) {
	bucket, ok := checked.Align(size, minimumDeviceBufferBytes)
	if !ok || bucket == 0 {
		return deviceBufferLease{}, errors.New("CUDA buffer size is invalid")
	}
	return p.acquireBucket(state, bucket)
}

func (p *deviceBufferPool) acquireBucket(
	state *device.State,
	bucket uint64,
) (deviceBufferLease, error) {
	if available := p.free[bucket]; len(available) > 0 {
		pointer := available[len(available)-1]
		p.free[bucket] = available[:len(available)-1]
		return deviceBufferLease{pointer: pointer, size: bucket}, nil
	}
	pointer, err := state.Driver.MemAlloc(bucket)
	if err != nil {
		return deviceBufferLease{}, err
	}
	if p.free == nil {
		p.free = make(map[uint64][]driver.DevicePtr)
	}
	p.allocations = append(p.allocations, pointer)
	return deviceBufferLease{pointer: pointer, size: bucket}, nil
}

func (p *deviceBufferPool) release(lease deviceBufferLease) {
	if lease.pointer != 0 {
		p.free[lease.size] = append(p.free[lease.size], lease.pointer)
	}
}

func (p *deviceBufferPool) close(state *device.State) error {
	var errs []error
	for _, pointer := range p.allocations {
		if err := state.Driver.MemFree(pointer); err != nil {
			errs = append(errs, err)
		}
	}
	p.free = nil
	p.allocations = nil
	return errors.Join(errs...)
}

func deviceBufferBucket(size uint64) (uint64, error) {
	if size == 0 {
		return 0, errors.New("CUDA buffer size is zero")
	}
	if size <= minimumDeviceBufferBytes {
		return minimumDeviceBufferBytes, nil
	}
	if size > uint64(1)<<63 {
		return size, nil
	}
	return uint64(1) << bits.Len64(size-1), nil
}

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
			elements, shapeErr := copySpec.Shape.Elements()
			if shapeErr != nil {
				return fail(shapeErr)
			}
			if elements > math.MaxUint64/4 {
				return fail(errors.New("CUDA device copy size overflows"))
			}
			expected := elements * 4
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

// RetainedTargets: graph-indexed stable output destinations.
type RetainedTargets struct {
	compiled *CompiledGraph
	values   []DeviceValue
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
	if err := tensor.ValidateOperationAttributes(node.Op, attributes); err != nil {
		return err
	}
	if err := validateRuntimeAttributes(node, attributes); err != nil {
		return err
	}
	a.values[index] = attributes
	return nil
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
		if logicalTokens == 0 {
			logicalTokens = capacity
		}
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
	outputs         []*tensor.Tensor
	outputIndexes   map[*tensor.Tensor]int
	order           []*tensor.Tensor
	orderIndexes    map[*tensor.Tensor]int
	memory          planner.Plan
	weightedRMS     map[*tensor.Tensor]weightedRMSFusion
	activatedGate   map[*tensor.Tensor]activatedGateFusion
	weightedRMSGate map[*tensor.Tensor]weightedRMSGateFusion
	q8Emit          map[*tensor.Tensor]struct{}
	q8Argmax        map[*tensor.Tensor]*tensor.Tensor
	targetContracts []tensor.OutputTargetContract
	skipped         map[*tensor.Tensor]struct{}
	needBlas        bool
	bf16InputBytes  uint64
	q8InputBytes    uint64
}

const graphArenaAlignment = 256

type retainedStorageView struct {
	source     *tensor.Tensor
	byteOffset uint64
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
	pointers map[*tensor.Tensor]driver.DevicePtr,
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
		inputRange, rangeErr := newDeviceAddressRange(pointers[input], inputBytes)
		if rangeErr != nil {
			return rangeErr
		}
		if !deviceRangesOverlap(outputRange, inputRange) {
			continue
		}
		alias := contract.Alias
		if alias == nil || alias.Input != index || target.Pointer != pointers[input] ||
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
	order []*tensor.Tensor,
	outputs map[*tensor.Tensor]struct{},
) (map[*tensor.Tensor]uint64, uint64, error) {
	offsets := make(map[*tensor.Tensor]uint64, len(outputs))
	var total uint64
	for _, node := range order {
		if node.Op == tensor.OpInput {
			continue
		}
		if _, keep := outputs[node]; !keep {
			continue
		}
		offset, ok := checked.Align(total, graphArenaAlignment)
		if !ok {
			return nil, 0, errors.New("CUDA retained output offset overflows")
		}
		bytes, err := node.Shape.Bytes(node.Type)
		if err != nil || offset > math.MaxUint64-bytes {
			return nil, 0, errors.New("CUDA retained output size overflows")
		}
		offsets[node] = offset
		total = offset + bytes
	}
	return offsets, total, nil
}

// Compile: validates and plans an immutable tensor graph.
func Compile(outputs ...*tensor.Tensor) (*CompiledGraph, error) {
	order, err := tensor.Topological(outputs...)
	if err != nil {
		return nil, err
	}
	compiled := &CompiledGraph{
		outputs:         slices.Clone(outputs),
		outputIndexes:   make(map[*tensor.Tensor]int, len(outputs)),
		targetContracts: make([]tensor.OutputTargetContract, len(outputs)),
		order:           order,
		orderIndexes:    make(map[*tensor.Tensor]int, len(order)),
	}
	for index, node := range order {
		compiled.orderIndexes[node] = index
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
		if node.Op == tensor.OpMulMat && node.Inputs[0].Type == dtype.BF16 {
			elements, elementErr := node.Inputs[1].Shape.Elements()
			if elementErr != nil || elements > math.MaxUint64/2 {
				return nil, errors.New("BF16 mul_mat input size overflows")
			}
			compiled.bf16InputBytes = max(compiled.bf16InputBytes, elements*2)
		}
		if node.Op == tensor.OpMulMat && node.Inputs[0].Type == dtype.Q8_0 &&
			node.Inputs[1].Shape.Rank == 2 && node.Inputs[1].Shape.Dims[1] == 1 {
			elements, elementErr := node.Inputs[1].Shape.Elements()
			if elementErr != nil || elements%q8InputBlockWidth != 0 ||
				elements/q8InputBlockWidth > math.MaxUint64/q8InputBlockBytes {
				return nil, errors.New("Q8_0 mul_mat input storage overflows")
			}
			compiled.q8InputBytes = max(
				compiled.q8InputBytes,
				elements/q8InputBlockWidth*q8InputBlockBytes,
			)
		}
	}
	dependencies := compileGraphRewrites(compiled, order, uses, consumers, outputSet)
	memory, err := planner.BuildWithRewrites(
		outputs, graphArenaAlignment, dependencies, compiled.skipped,
	)
	if err != nil {
		return nil, err
	}
	compiled.memory = memory
	return compiled, nil
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

// Execute: evaluates outputs and returns host copies of those tensors
func (e *Executor) Execute(
	ctx context.Context,
	outputs []*tensor.Tensor,
	feeds map[*tensor.Tensor]reference.Value,
) (map[*tensor.Tensor]reference.Value, error) {
	compiled, err := Compile(outputs...)
	if err != nil {
		return nil, err
	}
	return e.ExecuteCompiled(ctx, compiled, feeds)
}

// ExecuteCompiled: reuses validated topology and memory planning.
func (e *Executor) ExecuteCompiled(
	ctx context.Context,
	compiled *CompiledGraph,
	feeds map[*tensor.Tensor]reference.Value,
) (map[*tensor.Tensor]reference.Value, error) {
	result, err := e.runCompiled(ctx, compiled, feeds, nil, nil, nil, false)
	if err != nil {
		return nil, err
	}
	return result.host, nil
}

// ExecuteWithDeviceFeeds: evaluates graph with selected F32 input nodes
// already resident in executor's CUDA context
func (e *Executor) ExecuteWithDeviceFeeds(
	ctx context.Context,
	outputs []*tensor.Tensor,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
) (map[*tensor.Tensor]reference.Value, error) {
	compiled, err := Compile(outputs...)
	if err != nil {
		return nil, err
	}
	return e.ExecuteCompiledWithDeviceFeeds(ctx, compiled, hostFeeds, deviceFeeds)
}

// ExecuteCompiledWithDeviceFeeds: compiled graph plus resident inputs.
func (e *Executor) ExecuteCompiledWithDeviceFeeds(
	ctx context.Context,
	compiled *CompiledGraph,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
) (map[*tensor.Tensor]reference.Value, error) {
	result, err := e.runCompiled(ctx, compiled, hostFeeds, deviceFeeds, nil, nil, false)
	if err != nil {
		return nil, err
	}
	return result.host, nil
}

// ExecuteRetainedWithDeviceFeeds: evaluates graph but leaves each requested
// output in individually owned device allocation
func (e *Executor) ExecuteRetainedWithDeviceFeeds(
	ctx context.Context,
	outputs []*tensor.Tensor,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
) (*RetainedOutputs, error) {
	compiled, err := Compile(outputs...)
	if err != nil {
		return nil, err
	}
	return e.ExecuteRetainedCompiledWithDeviceFeeds(ctx, compiled, hostFeeds, deviceFeeds)
}

// ExecuteRetainedCompiledWithDeviceFeeds: compiled retained execution.
func (e *Executor) ExecuteRetainedCompiledWithDeviceFeeds(
	ctx context.Context,
	compiled *CompiledGraph,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
) (*RetainedOutputs, error) {
	return e.ExecuteRetainedCompiledWithTargets(ctx, compiled, hostFeeds, deviceFeeds, nil)
}

// ExecuteRetainedCompiledWithTargets: retained execution into selected stable buffers.
func (e *Executor) ExecuteRetainedCompiledWithTargets(
	ctx context.Context,
	compiled *CompiledGraph,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
	targets *RetainedTargets,
) (*RetainedOutputs, error) {
	return e.ExecuteRetainedCompiledParameterized(
		ctx, compiled, hostFeeds, deviceFeeds, targets, nil,
	)
}

// ExecuteRetainedCompiledParameterized: indexed targets and attributes.
func (e *Executor) ExecuteRetainedCompiledParameterized(
	ctx context.Context,
	compiled *CompiledGraph,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
	targets *RetainedTargets,
	attributes *RuntimeAttributes,
) (*RetainedOutputs, error) {
	execution, err := e.runCompiled(
		ctx, compiled, hostFeeds, deviceFeeds, targets, attributes, true,
	)
	if err != nil {
		return nil, err
	}
	return &RetainedOutputs{
		executor: e, compiled: compiled, values: execution.values, leases: execution.leases,
	}, nil
}

func (e *Executor) runCompiled(
	ctx context.Context,
	compiled *CompiledGraph,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
	targets *RetainedTargets,
	attributes *RuntimeAttributes,
	retain bool,
) (*executionResult, error) {
	if e == nil {
		return nil, errors.New("CUDA executor is closed")
	}
	if compiled == nil {
		return nil, errors.New("CUDA compiled graph is nil")
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.worker == nil {
		return nil, errors.New("CUDA executor is closed")
	}
	var result *executionResult
	err := e.worker.Do(ctx, func(state *device.State) error {
		resources, resourceErr := e.ensureResources(
			state, compiled.needBlas, compiled.bf16InputBytes, compiled.q8InputBytes,
		)
		if resourceErr != nil {
			return resourceErr
		}
		arena, arenaErr := resources.ensureArena(state, compiled.memory.ArenaSize)
		if arenaErr != nil {
			return arenaErr
		}
		var executeErr error
		result, executeErr = execute(
			state,
			compiled,
			hostFeeds,
			deviceFeeds,
			targets,
			attributes,
			resources.functions,
			resources.blas,
			&resources.q8Input,
			arena,
			&resources.buffers,
			&resources.graphExec,
			retain,
		)
		return executeErr
	})
	return result, err
}

func execute(
	state *device.State,
	compiled *CompiledGraph,
	feeds map[*tensor.Tensor]reference.Value,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
	retainedTargets *RetainedTargets,
	runtimeAttributes *RuntimeAttributes,
	functions functionSet,
	blas *blasState,
	q8Input *q8InputState,
	arena driver.DevicePtr,
	buffers *deviceBufferPool,
	graphExec *driver.GraphExec,
	retainOutputs bool,
) (result *executionResult, err error) {
	outputs := compiled.outputs
	order := compiled.order
	plan := compiled.memory
	if runtimeAttributes != nil &&
		(runtimeAttributes.compiled != compiled || len(runtimeAttributes.values) != len(order)) {
		return nil, errors.New("CUDA runtime attributes belong to another compiled graph")
	}
	attributesFor := func(node *tensor.Tensor) tensor.Attributes {
		if runtimeAttributes != nil {
			if index, ok := compiled.orderIndexes[node]; ok && runtimeAttributes.values[index] != nil {
				return runtimeAttributes.values[index]
			}
		}
		return node.Attrs
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

	pointers := make(map[*tensor.Tensor]driver.DevicePtr, len(order))
	var targetValues []DeviceValue
	if retainedTargets != nil {
		if retainedTargets.compiled != compiled || len(retainedTargets.values) != len(outputs) {
			return nil, errors.New("CUDA retained targets belong to another compiled graph")
		}
		targetValues = retainedTargets.values
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
	ownedOutputs := make(map[*tensor.Tensor]struct{}, len(outputs))
	for index, output := range outputs {
		if len(targetValues) == 0 || targetValues[index].Pointer == 0 {
			ownedOutputs[output] = struct{}{}
		}
	}
	retainedStorage := make(map[*tensor.Tensor]struct{}, len(ownedOutputs))
	retainedViews := make(map[*tensor.Tensor]retainedStorageView)
	for output := range ownedOutputs {
		view, aliases, viewErr := resolveRetainedStorageView(output)
		if viewErr != nil {
			return nil, viewErr
		}
		if aliases {
			retainedViews[output] = view
			retainedStorage[view.source] = struct{}{}
			continue
		}
		retainedStorage[output] = struct{}{}
	}
	retainedOffsets, retainedBytes, err := retainedOutputLayout(order, retainedStorage)
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
	for _, node := range order {
		if node.Op != tensor.OpInput && node.Type != dtype.F32 {
			return nil, fmt.Errorf("CUDA executor does not support %s for tensor %d", node.Type, node.ID)
		}
		if node.Op != tensor.OpInput {
			if _, skipped := compiled.skipped[node]; skipped {
				continue
			}
			outputIndex, retainedOutput := compiled.outputIndexes[node]
			_, retainStorage := retainedStorage[node]
			view, aliases, viewErr := tensor.ResolveStorageView(node)
			if viewErr != nil {
				return nil, viewErr
			}
			if aliases {
				offset, offsetErr := view.ByteOffset(node.Type)
				input := pointers[node.Inputs[view.Input]]
				if offsetErr != nil || uint64(input) > math.MaxUint64-offset {
					return nil, errors.New("CUDA storage view offset is invalid")
				}
				if retainedView, retainedAlias := retainedViews[node]; retainedAlias ||
					!(retainOutputs && retainedOutput) {
					pointer := input + driver.DevicePtr(offset)
					pointers[node] = pointer
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
					pointers[node] = target.Pointer
					continue
				}
				offset, present := retainedOffsets[node]
				if !present || uint64(retainedBase) > math.MaxUint64-offset {
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
				pointers[node] = pointer
				continue
			}
			allocation, ok := plan.Allocations[node]
			if !ok {
				return nil, fmt.Errorf("tensor %d has no arena allocation", node.ID)
			}
			if uint64(arena) > math.MaxUint64-allocation.Offset {
				return nil, errors.New("device pointer offset overflows uint64")
			}
			pointers[node] = arena + driver.DevicePtr(allocation.Offset)
			continue
		}
		if pointer, ok := deviceFeeds[node]; ok {
			if node.Type != dtype.F32 && node.Type != dtype.BF16 && !nativeQuantizedType(node.Type) {
				return nil, fmt.Errorf("CUDA device feed %q has unsupported type %s", node.Name, node.Type)
			}
			if pointer == 0 {
				return nil, fmt.Errorf("device feed for input %q is null", node.Name)
			}
			if _, duplicate := feeds[node]; duplicate {
				return nil, fmt.Errorf("input %q has both host and device feeds", node.Name)
			}
			pointers[node] = pointer
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
		pointers[node] = lease.pointer
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

	attributePointers := make(map[*tensor.Tensor]driver.DevicePtr)
	auxiliaryLeases := make([]deviceBufferLease, 0)
	sharedAttributes := make(map[string]driver.DevicePtr)
	defer func() {
		for _, lease := range auxiliaryLeases {
			buffers.release(lease)
		}
	}()
	for _, node := range order {
		var values []uint32
		switch node.Op {
		case tensor.OpGetRows:
			attributes, ok := attributesFor(node).(tensor.GetRowsAttributes)
			if !ok {
				return nil, errors.New("invalid get_rows attributes")
			}
			values = attributes.Rows
		case tensor.OpRoPENeoX:
			attributes, ok := attributesFor(node).(tensor.RoPEAttributes)
			if !ok {
				return nil, errors.New("invalid rope_neox attributes")
			}
			values = attributes.Positions
		case tensor.OpRoPENormal:
			attributes, ok := attributesFor(node).(tensor.RoPEAttributes)
			if !ok {
				return nil, errors.New("invalid rope_normal attributes")
			}
			values = attributes.Positions
		case tensor.OpRoPEMulti:
			attributes, ok := attributesFor(node).(tensor.RoPEMultiAttributes)
			if !ok {
				return nil, errors.New("invalid rope_multi attributes")
			}
			for axis := range attributes.Positions {
				values = append(values, attributes.Positions[axis]...)
			}
		default:
			continue
		}
		encoded := driver.Bytes(values)
		key := string(encoded)
		if pointer, ok := sharedAttributes[key]; ok {
			attributePointers[node] = pointer
			continue
		}
		bytes := uint64(len(values)) * uint64(unsafe.Sizeof(uint32(0)))
		lease, allocateErr := buffers.acquire(state, bytes)
		if allocateErr != nil {
			return nil, allocateErr
		}
		auxiliaryLeases = append(auxiliaryLeases, lease)
		sharedAttributes[key] = lease.pointer
		attributePointers[node] = lease.pointer
		if copyErr := state.Driver.MemcpyHtoD(lease.pointer, encoded); copyErr != nil {
			return nil, copyErr
		}
	}

	capturing := retainOutputs && graphExec != nil
	if blas != nil {
		blas.stagedNode = nil
	}
	if q8Input != nil {
		q8Input.stagedNode = nil
	}
	if capturing {
		if captureErr := state.Driver.StreamBeginCapture(state.Stream); captureErr != nil {
			return nil, captureErr
		}
	}
	abortCapture := func() {
		if !capturing {
			return
		}
		graph, _ := state.Driver.StreamEndCapture(state.Stream)
		_ = state.Driver.GraphDestroy(graph)
		capturing = false
	}
	for _, node := range order {
		if node.Op == tensor.OpInput {
			continue
		}
		if _, skipped := compiled.skipped[node]; skipped {
			continue
		}
		_, aliases, viewErr := tensor.ResolveStorageView(node)
		if viewErr != nil {
			abortCapture()
			return nil, viewErr
		}
		if aliases {
			_, retainedOutput := compiled.outputIndexes[node]
			_, retainedAlias := retainedViews[node]
			if retainedAlias || !(retainOutputs && retainedOutput) {
				continue
			}
		}
		if fusion, ok := compiled.weightedRMS[node]; ok {
			_, emitQ8 := compiled.q8Emit[node]
			if err := launchWeightedRMSNorm(
				state, functions, q8Input, node, fusion, emitQ8, pointers,
			); err != nil {
				abortCapture()
				return nil, fmt.Errorf("launch tensor %d (weighted_rms_norm): %w", node.ID, err)
			}
			submitted = true
			continue
		}
		if fusion, ok := compiled.activatedGate[node]; ok {
			_, emitQ8 := compiled.q8Emit[node]
			if err := launchActivatedGate(
				state, functions, q8Input, node, fusion, emitQ8, pointers,
			); err != nil {
				abortCapture()
				return nil, fmt.Errorf("launch tensor %d (activated_gate): %w", node.ID, err)
			}
			submitted = true
			continue
		}
		if fusion, ok := compiled.weightedRMSGate[node]; ok {
			_, emitQ8 := compiled.q8Emit[node]
			if err := launchWeightedRMSGate(
				state, functions, q8Input, node, fusion, emitQ8, pointers,
			); err != nil {
				abortCapture()
				return nil, fmt.Errorf("launch tensor %d (weighted_rms_gate): %w", node.ID, err)
			}
			submitted = true
			continue
		}
		if paired, ok := compiled.q8Argmax[node]; ok {
			var launchErr error
			if node.Op == tensor.OpMulMat {
				launchErr = launchQ8ArgmaxPartials(
					state, functions, q8Input, node, pointers,
				)
			} else {
				launchErr = launchQ8ArgmaxReduction(
					state, functions, paired, node, pointers,
				)
			}
			if launchErr != nil {
				abortCapture()
				return nil, fmt.Errorf("launch tensor %d (q8_argmax): %w", node.ID, launchErr)
			}
			submitted = true
			continue
		}
		if err := launchNode(
			state, functions, blas, q8Input, node, attributesFor(node), pointers, attributePointers,
		); err != nil {
			abortCapture()
			return nil, fmt.Errorf("launch tensor %d (%s): %w", node.ID, node.Op, err)
		}
		submitted = true
	}
	if capturing {
		graph, captureErr := state.Driver.StreamEndCapture(state.Stream)
		capturing = false
		if captureErr != nil {
			return nil, captureErr
		}
		defer state.Driver.GraphDestroy(graph)
		if *graphExec != 0 {
			updated, updateErr := state.Driver.GraphExecUpdate(*graphExec, graph)
			if updateErr != nil || !updated {
				_ = state.Driver.GraphExecDestroy(*graphExec)
				*graphExec = 0
			}
		}
		if *graphExec == 0 {
			*graphExec, err = state.Driver.GraphInstantiate(graph)
			if err != nil {
				return nil, err
			}
		}
		if err := state.Driver.GraphLaunch(*graphExec, state.Stream); err != nil {
			return nil, err
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
		if err := state.Driver.MemcpyDtoH(driver.Bytes(data), pointers[output]); err != nil {
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

var quantKernels = map[dtype.Type]quantKernelDescriptor{
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

type blasState struct {
	library      *cublas.Library
	handle       cublas.Handle
	staging      driver.DevicePtr
	stagingBytes uint64
	stagedNode   *tensor.Tensor
}

const (
	q8InputBlockWidth = uint64(32)
	q8InputBlockBytes = uint64(36)
)

type q8InputState struct {
	staging      driver.DevicePtr
	stagingBytes uint64
	stagedNode   *tensor.Tensor
}

func (e *Executor) ensureResources(
	state *device.State,
	needBlas bool,
	bf16InputBytes uint64,
	q8InputBytes uint64,
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
	if bf16InputBytes > 0 && e.resources.blas != nil && e.resources.blas.stagingBytes < bf16InputBytes {
		staging, err := state.Driver.MemAlloc(bf16InputBytes)
		if err != nil {
			return nil, err
		}
		if e.resources.blas.staging != 0 {
			if err := state.Driver.MemFree(e.resources.blas.staging); err != nil {
				_ = state.Driver.MemFree(staging)
				return nil, err
			}
		}
		e.resources.blas.staging = staging
		e.resources.blas.stagingBytes = bf16InputBytes
	}
	if q8InputBytes > e.resources.q8Input.stagingBytes {
		staging, err := state.Driver.MemAlloc(q8InputBytes)
		if err != nil {
			return nil, err
		}
		if e.resources.q8Input.staging != 0 {
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
	if size == 0 {
		return 0, nil
	}
	if r.arena != 0 && r.arenaSize >= size {
		return r.arena, nil
	}
	next, err := state.Driver.MemAlloc(size)
	if err != nil {
		return 0, err
	}
	if r.arena != 0 {
		if err := state.Driver.MemFree(r.arena); err != nil {
			_ = state.Driver.MemFree(next)
			return 0, err
		}
	}
	r.arena, r.arenaSize = next, size
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
	if e.resources.graphExec != 0 {
		errs = append(errs, state.Driver.GraphExecDestroy(e.resources.graphExec))
		e.resources.graphExec = 0
	}
	if e.resources.blas != nil {
		if e.resources.blas.staging != 0 {
			errs = append(errs, state.Driver.MemFree(e.resources.blas.staging))
			e.resources.blas.staging = 0
			e.resources.blas.stagingBytes = 0
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
