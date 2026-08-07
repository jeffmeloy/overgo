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
	values   map[*tensor.Tensor]DeviceValue
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
	value, ok := r.values[output]
	return value, ok
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
	value, ok := r.values[output]
	if !ok {
		return reference.Value{}, errors.New("CUDA retained output is unavailable")
	}
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
	values map[*tensor.Tensor]DeviceValue
	leases []deviceBufferLease
}

// CompiledGraph: validated order and memory plan for repeated execution.
type CompiledGraph struct {
	outputs         []*tensor.Tensor
	order           []*tensor.Tensor
	memory          planner.Plan
	weightedRMS     map[*tensor.Tensor]weightedRMSFusion
	activatedGate   map[*tensor.Tensor]activatedGateFusion
	weightedRMSGate map[*tensor.Tensor]weightedRMSGateFusion
	q8Emit          map[*tensor.Tensor]struct{}
	q8Argmax        map[*tensor.Tensor]*tensor.Tensor
	targetContracts map[*tensor.Tensor]tensor.OutputTargetContract
	skipped         map[*tensor.Tensor]struct{}
	needBlas        bool
	bf16InputBytes  uint64
	q8InputBytes    uint64
}

type weightedRMSFusion struct {
	normalization *tensor.Tensor
	weight        *tensor.Tensor
	addLeft       *tensor.Tensor
	addRight      *tensor.Tensor
}

type activatedGateKind uint32

const (
	activatedGateSiLU activatedGateKind = iota + 1
	activatedGateSigmoid
)

type activatedGateFusion struct {
	gate       *tensor.Tensor
	up         *tensor.Tensor
	activation *tensor.Tensor
	kind       activatedGateKind
}

type weightedRMSGateFusion struct {
	weightedRMSFusion
	gate *tensor.Tensor
	kind activatedGateKind
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
			alias.WriteOffsetBytes != inputBytes ||
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
		outputs: slices.Clone(outputs), order: order,
	}
	uses := make(map[*tensor.Tensor]int, len(order))
	consumers := make(map[*tensor.Tensor][]*tensor.Tensor, len(order))
	outputSet := make(map[*tensor.Tensor]struct{}, len(outputs))
	for _, output := range outputs {
		outputSet[output] = struct{}{}
		contract, contractErr := tensor.CompileOutputTargetContract(output)
		if contractErr != nil {
			return nil, contractErr
		}
		if compiled.targetContracts == nil {
			compiled.targetContracts = make(map[*tensor.Tensor]tensor.OutputTargetContract, len(outputs))
		}
		compiled.targetContracts[output] = contract
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
		if node.Op != tensor.OpMultiply || len(node.Inputs) != 2 {
			continue
		}
		normalization, weight := node.Inputs[0], node.Inputs[1]
		if normalization.Op != tensor.OpRMSNorm {
			normalization, weight = weight, normalization
		}
		if normalization.Op != tensor.OpRMSNorm || uses[normalization] != 1 {
			continue
		}
		if _, retained := outputSet[normalization]; retained || !rmsWeightCompatible(normalization, weight) {
			continue
		}
		if compiled.weightedRMS == nil {
			compiled.weightedRMS = make(map[*tensor.Tensor]weightedRMSFusion)
			compiled.skipped = make(map[*tensor.Tensor]struct{})
		}
		fusion := weightedRMSFusion{normalization: normalization, weight: weight}
		if source := normalization.Inputs[0]; source.Op == tensor.OpAdd && uses[source] == 1 {
			if _, retained := outputSet[source]; !retained {
				fusion.addLeft, fusion.addRight = source.Inputs[0], source.Inputs[1]
				compiled.skipped[source] = struct{}{}
			}
		}
		compiled.weightedRMS[node] = fusion
		compiled.skipped[normalization] = struct{}{}
	}
	for _, node := range order {
		if node.Op != tensor.OpMultiply || len(node.Inputs) != 2 {
			continue
		}
		activation, up := node.Inputs[0], node.Inputs[1]
		kind, ok := activatedGateKindFor(activation)
		if !ok {
			activation, up = up, activation
			kind, ok = activatedGateKindFor(activation)
		}
		if !ok || uses[activation] != 1 ||
			!activation.Shape.Equal(up.Shape) || !node.Shape.Equal(up.Shape) {
			continue
		}
		if _, retained := outputSet[activation]; retained {
			continue
		}
		if weighted, fused := compiled.weightedRMS[up]; fused && uses[up] == 1 {
			if _, retained := outputSet[up]; !retained {
				if compiled.weightedRMSGate == nil {
					compiled.weightedRMSGate = make(map[*tensor.Tensor]weightedRMSGateFusion)
				}
				compiled.weightedRMSGate[node] = weightedRMSGateFusion{
					weightedRMSFusion: weighted,
					gate:              activation.Inputs[0],
					kind:              kind,
				}
				delete(compiled.weightedRMS, up)
				compiled.skipped[up] = struct{}{}
				compiled.skipped[activation] = struct{}{}
				continue
			}
		}
		if compiled.activatedGate == nil {
			compiled.activatedGate = make(map[*tensor.Tensor]activatedGateFusion)
		}
		if compiled.skipped == nil {
			compiled.skipped = make(map[*tensor.Tensor]struct{})
		}
		compiled.activatedGate[node] = activatedGateFusion{
			gate: activation.Inputs[0], up: up, activation: activation, kind: kind,
		}
		compiled.skipped[activation] = struct{}{}
	}
	for _, node := range order {
		if node.Op != tensor.OpMulMat || node.Inputs[0].Type != dtype.Q8_0 ||
			node.Inputs[1].Shape.Rank != 2 || node.Inputs[1].Shape.Dims[1] != 1 {
			continue
		}
		producer := node.Inputs[1]
		if _, weighted := compiled.weightedRMS[producer]; !weighted {
			if _, activated := compiled.activatedGate[producer]; !activated {
				if _, gatedNorm := compiled.weightedRMSGate[producer]; !gatedNorm {
					continue
				}
			}
		}
		if compiled.q8Emit == nil {
			compiled.q8Emit = make(map[*tensor.Tensor]struct{})
		}
		compiled.q8Emit[producer] = struct{}{}
	}
	for _, projection := range order {
		if projection.Op != tensor.OpMulMat || projection.Inputs[0].Type != dtype.Q8_0 ||
			projection.Inputs[1].Shape.Rank != 2 || projection.Inputs[1].Shape.Dims[1] != 1 ||
			uses[projection] != 1 {
			continue
		}
		if _, retained := outputSet[projection]; retained {
			continue
		}
		selection := consumers[projection][0]
		attributes, ok := selection.Attrs.(tensor.TopKAttributes)
		rows := projection.Shape.Dims[0]
		partials, partialsOK := q8ArgmaxPartialCount(rows)
		if selection.Op != tensor.OpTopK || !ok || attributes.K != 1 || rows < 2 ||
			!partialsOK || q8ArgmaxPartialValues*uint64(partials) > rows {
			continue
		}
		if compiled.q8Argmax == nil {
			compiled.q8Argmax = make(map[*tensor.Tensor]*tensor.Tensor)
		}
		compiled.q8Argmax[projection] = selection
		compiled.q8Argmax[selection] = projection
	}
	dependencies := make(map[*tensor.Tensor][]*tensor.Tensor)
	for node, fusion := range compiled.weightedRMS {
		if fusion.addLeft != nil {
			dependencies[node] = append(dependencies[node], fusion.addLeft, fusion.addRight)
		} else {
			dependencies[node] = append(dependencies[node], fusion.normalization.Inputs[0])
		}
	}
	for node, fusion := range compiled.activatedGate {
		dependencies[node] = append(dependencies[node], fusion.gate)
	}
	for node, fusion := range compiled.weightedRMSGate {
		dependencies[node] = append(dependencies[node], fusion.gate, fusion.weight)
		if fusion.addLeft != nil {
			dependencies[node] = append(dependencies[node], fusion.addLeft, fusion.addRight)
		} else {
			dependencies[node] = append(dependencies[node], fusion.normalization.Inputs[0])
		}
	}
	memory, err := planner.BuildWithRewrites(
		outputs, graphArenaAlignment, dependencies, compiled.skipped,
	)
	if err != nil {
		return nil, err
	}
	compiled.memory = memory
	return compiled, nil
}

func activatedGateKindFor(node *tensor.Tensor) (activatedGateKind, bool) {
	if node == nil || len(node.Inputs) != 1 {
		return 0, false
	}
	switch node.Op {
	case tensor.OpSiLU:
		return activatedGateSiLU, true
	case tensor.OpSigmoid:
		return activatedGateSigmoid, true
	default:
		return 0, false
	}
}

func rmsWeightCompatible(normalization, weight *tensor.Tensor) bool {
	if normalization == nil || weight == nil || len(normalization.Inputs) != 1 || weight.Type != dtype.F32 {
		return false
	}
	width := normalization.Shape.Dims[0]
	elements, err := weight.Shape.Elements()
	return err == nil && elements == width && weight.Shape.Dims[0] == width
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
	result, err := e.runCompiled(ctx, compiled, feeds, nil, nil, false)
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
	result, err := e.runCompiled(ctx, compiled, hostFeeds, deviceFeeds, nil, false)
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
	targets map[*tensor.Tensor]DeviceValue,
) (*RetainedOutputs, error) {
	execution, err := e.runCompiled(ctx, compiled, hostFeeds, deviceFeeds, targets, true)
	if err != nil {
		return nil, err
	}
	return &RetainedOutputs{
		executor: e, values: execution.values, leases: execution.leases,
	}, nil
}

func (e *Executor) runCompiled(
	ctx context.Context,
	compiled *CompiledGraph,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
	targets map[*tensor.Tensor]DeviceValue,
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
	retainedTargets map[*tensor.Tensor]DeviceValue,
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
	outputSet := make(map[*tensor.Tensor]struct{}, len(outputs))
	for _, output := range outputs {
		outputSet[output] = struct{}{}
	}
	retainedValues := make(map[*tensor.Tensor]DeviceValue, len(outputs))
	retainedLeases := make([]deviceBufferLease, 0, 1)
	for node, target := range retainedTargets {
		if _, ok := outputSet[node]; !ok || target.Pointer == 0 || !target.Shape.Equal(node.Shape) {
			return nil, errors.New("CUDA retained output target is invalid")
		}
		contract, ok := compiled.targetContracts[node]
		if !ok || contract.Alignment == 0 || uint64(target.Pointer)%contract.Alignment != 0 ||
			target.CapacityBytes < contract.Bytes {
			return nil, errors.New("CUDA retained output target capacity is insufficient")
		}
	}
	ownedOutputs := make(map[*tensor.Tensor]struct{}, len(outputSet))
	for output := range outputSet {
		if _, targeted := retainedTargets[output]; !targeted {
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
			_, retainedOutput := outputSet[node]
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
						retainedValues[node] = DeviceValue{
							Pointer: pointer, Shape: node.Shape, CapacityBytes: bytes,
						}
					}
					continue
				}
			}
			if retainOutputs && (retainedOutput || retainStorage) {
				if target, targeted := retainedTargets[node]; targeted {
					if err := validateRetainedTargetAlias(
						node, target, compiled.targetContracts[node], pointers,
					); err != nil {
						return nil, err
					}
					retainedValues[node] = target
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
					retainedValues[node] = DeviceValue{
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
			attributes, ok := node.Attrs.(tensor.GetRowsAttributes)
			if !ok {
				return nil, errors.New("invalid get_rows attributes")
			}
			values = attributes.Rows
		case tensor.OpRoPENeoX:
			attributes, ok := node.Attrs.(tensor.RoPEAttributes)
			if !ok {
				return nil, errors.New("invalid rope_neox attributes")
			}
			values = attributes.Positions
		case tensor.OpRoPENormal:
			attributes, ok := node.Attrs.(tensor.RoPEAttributes)
			if !ok {
				return nil, errors.New("invalid rope_normal attributes")
			}
			values = attributes.Positions
		case tensor.OpRoPEMulti:
			attributes, ok := node.Attrs.(tensor.RoPEMultiAttributes)
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
			_, retainedOutput := outputSet[node]
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
		if err := launchNode(state, functions, blas, q8Input, node, pointers, attributePointers); err != nil {
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

type functionSet struct {
	add                  driver.Function
	multiply             driver.Function
	divide               driver.Function
	broadcastAdd         driver.Function
	broadcastMultiply    driver.Function
	broadcastDivide      driver.Function
	scale                driver.Function
	clamp                driver.Function
	bf16Round            driver.Function
	f32ToBF16            driver.Function
	quantizeQ8Input      driver.Function
	copy                 driver.Function
	silu                 driver.Function
	gelu                 driver.Function
	geluErf              driver.Function
	xielu                driver.Function
	reluSquared          driver.Function
	relu                 driver.Function
	conv1DSame           driver.Function
	conv2D               driver.Function
	windowPartition2D    driver.Function
	windowUnpartition2D  driver.Function
	samAttention         driver.Function
	groupNorm            driver.Function
	sigmoid              driver.Function
	softplus             driver.Function
	tanh                 driver.Function
	exp                  driver.Function
	l2Norm               driver.Function
	ssmConv              driver.Function
	ssmScan              driver.Function
	gatedDeltaNet        driver.Function
	gatedLinearAttn      driver.Function
	rwkv6                driver.Function
	sumRows              driver.Function
	fwht                 driver.Function
	argmax               driver.Function
	topK                 driver.Function
	topKPairs            driver.Function
	topKPartials         driver.Function
	gatherLast           driver.Function
	gatherLastQ8         driver.Function
	sparseAttention      driver.Function
	indexerScore         driver.Function
	rwkv7                driver.Function
	moe                  driver.Function
	moeGrouped           driver.Function
	loraMerge            driver.Function
	repeatHeads          driver.Function
	transpose2D          driver.Function
	groupSlice           driver.Function
	flatSlice            driver.Function
	rmsNorm              driver.Function
	weightedRMSNorm      driver.Function
	weightedRMSNormQ8    driver.Function
	weightedRMSNormAdd   driver.Function
	weightedRMSNormAddQ8 driver.Function
	activatedGate        driver.Function
	activatedGateQ8      driver.Function
	weightedRMSGate      driver.Function
	weightedRMSGateQ8    driver.Function
	layerNorm            driver.Function
	softmax              driver.Function
	mulMat               driver.Function
	getRows              driver.Function
	ropeNeoX             driver.Function
	ropeNormal           driver.Function
	ropeMulti            driver.Function
	attention            driver.Function
	attentionDecode      driver.Function
	attentionOnline      driver.Function
	concat               driver.Function
	getRowsQ8            driver.Function
	mulMatQ8             driver.Function
	mulMatQ8Input        driver.Function
	mulMatQ8Argmax       driver.Function
	q8ArgmaxReduction    driver.Function
	getRowsQ81           driver.Function
	mulMatQ81            driver.Function
	getRowsQ8K           driver.Function
	mulMatQ8K            driver.Function
	getRowsQ40           driver.Function
	mulMatQ40            driver.Function
	getRowsQ41           driver.Function
	mulMatQ41            driver.Function
	getRowsQ50           driver.Function
	mulMatQ50            driver.Function
	getRowsQ51           driver.Function
	mulMatQ51            driver.Function
	getRowsQ10           driver.Function
	mulMatQ10            driver.Function
	getRowsQ20           driver.Function
	mulMatQ20            driver.Function
	getRowsTQ20          driver.Function
	mulMatTQ20           driver.Function
	getRowsTQ10          driver.Function
	mulMatTQ10           driver.Function
	getRowsQ2K           driver.Function
	mulMatQ2K            driver.Function
	getRowsQ3K           driver.Function
	mulMatQ3K            driver.Function
	getRowsQ4K           driver.Function
	mulMatQ4K            driver.Function
	getRowsQ5K           driver.Function
	mulMatQ5K            driver.Function
	getRowsIQ4XS         driver.Function
	mulMatIQ4XS          driver.Function
	getRowsIQ4NL         driver.Function
	mulMatIQ4NL          driver.Function
	getRowsIQ2XXS        driver.Function
	mulMatIQ2XXS         driver.Function
	getRowsIQ2XS         driver.Function
	mulMatIQ2XS          driver.Function
	getRowsIQ2S          driver.Function
	mulMatIQ2S           driver.Function
	getRowsIQ3XXS        driver.Function
	mulMatIQ3XXS         driver.Function
	getRowsIQ3S          driver.Function
	mulMatIQ3S           driver.Function
	getRowsIQ1S          driver.Function
	mulMatIQ1S           driver.Function
	getRowsIQ1M          driver.Function
	mulMatIQ1M           driver.Function
	getRowsMXFP4         driver.Function
	mulMatMXFP4          driver.Function
	getRowsNVFP4         driver.Function
	mulMatNVFP4          driver.Function
	getRowsQ6K           driver.Function
	mulMatQ6K            driver.Function
}

type quantKernelDescriptor struct {
	label   string
	getRows func(functionSet) driver.Function
	mulMat  func(functionSet) driver.Function
}

var quantKernels = map[dtype.Type]quantKernelDescriptor{
	dtype.Q8_0:   {"Q8_0", func(f functionSet) driver.Function { return f.getRowsQ8 }, func(f functionSet) driver.Function { return f.mulMatQ8 }},
	dtype.Q8_1:   {"Q8_1", func(f functionSet) driver.Function { return f.getRowsQ81 }, func(f functionSet) driver.Function { return f.mulMatQ81 }},
	dtype.Q8K:    {"Q8_K", func(f functionSet) driver.Function { return f.getRowsQ8K }, func(f functionSet) driver.Function { return f.mulMatQ8K }},
	dtype.Q4_0:   {"Q4_0", func(f functionSet) driver.Function { return f.getRowsQ40 }, func(f functionSet) driver.Function { return f.mulMatQ40 }},
	dtype.Q4_1:   {"Q4_1", func(f functionSet) driver.Function { return f.getRowsQ41 }, func(f functionSet) driver.Function { return f.mulMatQ41 }},
	dtype.Q5_0:   {"Q5_0", func(f functionSet) driver.Function { return f.getRowsQ50 }, func(f functionSet) driver.Function { return f.mulMatQ50 }},
	dtype.Q5_1:   {"Q5_1", func(f functionSet) driver.Function { return f.getRowsQ51 }, func(f functionSet) driver.Function { return f.mulMatQ51 }},
	dtype.Q1_0:   {"Q1_0", func(f functionSet) driver.Function { return f.getRowsQ10 }, func(f functionSet) driver.Function { return f.mulMatQ10 }},
	dtype.Q2_0:   {"Q2_0", func(f functionSet) driver.Function { return f.getRowsQ20 }, func(f functionSet) driver.Function { return f.mulMatQ20 }},
	dtype.TQ2_0:  {"TQ2_0", func(f functionSet) driver.Function { return f.getRowsTQ20 }, func(f functionSet) driver.Function { return f.mulMatTQ20 }},
	dtype.TQ1_0:  {"TQ1_0", func(f functionSet) driver.Function { return f.getRowsTQ10 }, func(f functionSet) driver.Function { return f.mulMatTQ10 }},
	dtype.Q2K:    {"Q2_K", func(f functionSet) driver.Function { return f.getRowsQ2K }, func(f functionSet) driver.Function { return f.mulMatQ2K }},
	dtype.Q3K:    {"Q3_K", func(f functionSet) driver.Function { return f.getRowsQ3K }, func(f functionSet) driver.Function { return f.mulMatQ3K }},
	dtype.Q4K:    {"Q4_K", func(f functionSet) driver.Function { return f.getRowsQ4K }, func(f functionSet) driver.Function { return f.mulMatQ4K }},
	dtype.Q5K:    {"Q5_K", func(f functionSet) driver.Function { return f.getRowsQ5K }, func(f functionSet) driver.Function { return f.mulMatQ5K }},
	dtype.IQ4XS:  {"IQ4_XS", func(f functionSet) driver.Function { return f.getRowsIQ4XS }, func(f functionSet) driver.Function { return f.mulMatIQ4XS }},
	dtype.IQ4NL:  {"IQ4_NL", func(f functionSet) driver.Function { return f.getRowsIQ4NL }, func(f functionSet) driver.Function { return f.mulMatIQ4NL }},
	dtype.IQ2XXS: {"IQ2_XXS", func(f functionSet) driver.Function { return f.getRowsIQ2XXS }, func(f functionSet) driver.Function { return f.mulMatIQ2XXS }},
	dtype.IQ2XS:  {"IQ2_XS", func(f functionSet) driver.Function { return f.getRowsIQ2XS }, func(f functionSet) driver.Function { return f.mulMatIQ2XS }},
	dtype.IQ2S:   {"IQ2_S", func(f functionSet) driver.Function { return f.getRowsIQ2S }, func(f functionSet) driver.Function { return f.mulMatIQ2S }},
	dtype.IQ3XXS: {"IQ3_XXS", func(f functionSet) driver.Function { return f.getRowsIQ3XXS }, func(f functionSet) driver.Function { return f.mulMatIQ3XXS }},
	dtype.IQ3S:   {"IQ3_S", func(f functionSet) driver.Function { return f.getRowsIQ3S }, func(f functionSet) driver.Function { return f.mulMatIQ3S }},
	dtype.IQ1S:   {"IQ1_S", func(f functionSet) driver.Function { return f.getRowsIQ1S }, func(f functionSet) driver.Function { return f.mulMatIQ1S }},
	dtype.IQ1M:   {"IQ1_M", func(f functionSet) driver.Function { return f.getRowsIQ1M }, func(f functionSet) driver.Function { return f.mulMatIQ1M }},
	dtype.MXFP4:  {"MXFP4", func(f functionSet) driver.Function { return f.getRowsMXFP4 }, func(f functionSet) driver.Function { return f.mulMatMXFP4 }},
	dtype.NVFP4:  {"NVFP4", func(f functionSet) driver.Function { return f.getRowsNVFP4 }, func(f functionSet) driver.Function { return f.mulMatNVFP4 }},
	dtype.Q6K:    {"Q6_K", func(f functionSet) driver.Function { return f.getRowsQ6K }, func(f functionSet) driver.Function { return f.mulMatQ6K }},
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

func loadFunctions(lib *driver.Library, module driver.Module) (functionSet, error) {
	var result functionSet
	items := []struct {
		name string
		dst  *driver.Function
	}{
		{"add_f32", &result.add},
		{"multiply_f32", &result.multiply},
		{"divide_f32", &result.divide},
		{"broadcast_add_f32", &result.broadcastAdd},
		{"broadcast_multiply_f32", &result.broadcastMultiply},
		{"broadcast_divide_f32", &result.broadcastDivide},
		{"scale_f32", &result.scale},
		{"clamp_f32", &result.clamp},
		{"bf16_round_f32", &result.bf16Round},
		{"f32_to_bf16", &result.f32ToBF16},
		{"quantize_q8_0_input_f32", &result.quantizeQ8Input},
		{"copy_f32", &result.copy},
		{"silu_f32", &result.silu},
		{"gelu_f32", &result.gelu},
		{"gelu_erf_f32", &result.geluErf},
		{"xielu_f32", &result.xielu},
		{"relu_squared_f32", &result.reluSquared},
		{"relu_f32", &result.relu},
		{"conv_1d_same_f32", &result.conv1DSame},
		{"conv_2d_f32", &result.conv2D},
		{"window_partition_2d_f32", &result.windowPartition2D},
		{"window_unpartition_2d_f32", &result.windowUnpartition2D},
		{"sam_attention_f32", &result.samAttention},
		{"group_norm_f32", &result.groupNorm},
		{"sigmoid_f32", &result.sigmoid},
		{"softplus_f32", &result.softplus},
		{"tanh_f32", &result.tanh},
		{"exp_f32", &result.exp},
		{"l2_norm_f32", &result.l2Norm},
		{"ssm_conv_f32", &result.ssmConv},
		{"ssm_scan_f32", &result.ssmScan},
		{"gated_delta_net_f32", &result.gatedDeltaNet},
		{"gated_linear_attention_f32", &result.gatedLinearAttn},
		{"rwkv6_f32", &result.rwkv6},
		{"sum_rows_f32", &result.sumRows},
		{"fwht_f32", &result.fwht},
		{"argmax_f32", &result.argmax},
		{"top_k_f32", &result.topK},
		{"top_k_pairs_f32", &result.topKPairs},
		{"top_k_partials_f32", &result.topKPartials},
		{"gather_last_f32", &result.gatherLast},
		{"gather_last_q8_0_f32", &result.gatherLastQ8},
		{"sparse_attention_f32", &result.sparseAttention},
		{"indexer_score_f32", &result.indexerScore},
		{"rwkv7_f32", &result.rwkv7},
		{"moe_f32", &result.moe},
		{"moe_grouped_f32", &result.moeGrouped},
		{"lora_merge_f32", &result.loraMerge},
		{"repeat_heads_f32", &result.repeatHeads},
		{"transpose_2d_f32", &result.transpose2D},
		{"group_slice_f32", &result.groupSlice},
		{"flat_slice_f32", &result.flatSlice},
		{"rms_norm_f32", &result.rmsNorm},
		{"weighted_rms_norm_f32", &result.weightedRMSNorm},
		{"weighted_rms_norm_q8_0_f32", &result.weightedRMSNormQ8},
		{"weighted_rms_norm_add_f32", &result.weightedRMSNormAdd},
		{"weighted_rms_norm_add_q8_0_f32", &result.weightedRMSNormAddQ8},
		{"activated_gate_f32", &result.activatedGate},
		{"activated_gate_q8_0_f32", &result.activatedGateQ8},
		{"weighted_rms_gate_f32", &result.weightedRMSGate},
		{"weighted_rms_gate_q8_0_f32", &result.weightedRMSGateQ8},
		{"layer_norm_f32", &result.layerNorm},
		{"softmax_f32", &result.softmax},
		{"mul_mat_f32", &result.mulMat},
		{"get_rows_f32", &result.getRows},
		{"rope_neox_f32", &result.ropeNeoX},
		{"rope_normal_f32", &result.ropeNormal},
		{"rope_multi_f32", &result.ropeMulti},
		{"attention_f32", &result.attention},
		{"attention_decode_f32", &result.attentionDecode},
		{"attention_online_f32", &result.attentionOnline},
		{"concat_f32", &result.concat},
		{"get_rows_q8_0_f32", &result.getRowsQ8},
		{"mul_mat_q8_0_f32", &result.mulMatQ8},
		{"mul_mat_q8_0_input_f32", &result.mulMatQ8Input},
		{"mul_mat_q8_0_input_argmax_partials_f32", &result.mulMatQ8Argmax},
		{"argmax_q8_0_input_partials_f32", &result.q8ArgmaxReduction},
		{"get_rows_q8_1_f32", &result.getRowsQ81},
		{"mul_mat_q8_1_f32", &result.mulMatQ81},
		{"get_rows_q8_K_f32", &result.getRowsQ8K},
		{"mul_mat_q8_K_f32", &result.mulMatQ8K},
		{"get_rows_q4_0_f32", &result.getRowsQ40},
		{"mul_mat_q4_0_f32", &result.mulMatQ40},
		{"get_rows_q4_1_f32", &result.getRowsQ41},
		{"mul_mat_q4_1_f32", &result.mulMatQ41},
		{"get_rows_q5_0_f32", &result.getRowsQ50},
		{"mul_mat_q5_0_f32", &result.mulMatQ50},
		{"get_rows_q5_1_f32", &result.getRowsQ51},
		{"mul_mat_q5_1_f32", &result.mulMatQ51},
		{"get_rows_q1_0_f32", &result.getRowsQ10},
		{"mul_mat_q1_0_f32", &result.mulMatQ10},
		{"get_rows_q2_0_f32", &result.getRowsQ20},
		{"mul_mat_q2_0_f32", &result.mulMatQ20},
		{"get_rows_tq2_0_f32", &result.getRowsTQ20},
		{"mul_mat_tq2_0_f32", &result.mulMatTQ20},
		{"get_rows_tq1_0_f32", &result.getRowsTQ10},
		{"mul_mat_tq1_0_f32", &result.mulMatTQ10},
		{"get_rows_q2_K_f32", &result.getRowsQ2K},
		{"mul_mat_q2_K_f32", &result.mulMatQ2K},
		{"get_rows_q3_K_f32", &result.getRowsQ3K},
		{"mul_mat_q3_K_f32", &result.mulMatQ3K},
		{"get_rows_q4_K_f32", &result.getRowsQ4K},
		{"mul_mat_q4_K_f32", &result.mulMatQ4K},
		{"get_rows_q5_K_f32", &result.getRowsQ5K},
		{"mul_mat_q5_K_f32", &result.mulMatQ5K},
		{"get_rows_iq4_xs_f32", &result.getRowsIQ4XS},
		{"mul_mat_iq4_xs_f32", &result.mulMatIQ4XS},
		{"get_rows_iq4_nl_f32", &result.getRowsIQ4NL},
		{"mul_mat_iq4_nl_f32", &result.mulMatIQ4NL},
		{"get_rows_iq2_xxs_f32", &result.getRowsIQ2XXS},
		{"mul_mat_iq2_xxs_f32", &result.mulMatIQ2XXS},
		{"get_rows_iq2_xs_f32", &result.getRowsIQ2XS},
		{"mul_mat_iq2_xs_f32", &result.mulMatIQ2XS},
		{"get_rows_iq2_s_f32", &result.getRowsIQ2S},
		{"mul_mat_iq2_s_f32", &result.mulMatIQ2S},
		{"get_rows_iq3_xxs_f32", &result.getRowsIQ3XXS},
		{"mul_mat_iq3_xxs_f32", &result.mulMatIQ3XXS},
		{"get_rows_iq3_s_f32", &result.getRowsIQ3S},
		{"mul_mat_iq3_s_f32", &result.mulMatIQ3S},
		{"get_rows_iq1_s_f32", &result.getRowsIQ1S},
		{"mul_mat_iq1_s_f32", &result.mulMatIQ1S},
		{"get_rows_iq1_m_f32", &result.getRowsIQ1M},
		{"mul_mat_iq1_m_f32", &result.mulMatIQ1M},
		{"get_rows_mxfp4_f32", &result.getRowsMXFP4},
		{"mul_mat_mxfp4_f32", &result.mulMatMXFP4},
		{"get_rows_nvfp4_f32", &result.getRowsNVFP4},
		{"mul_mat_nvfp4_f32", &result.mulMatNVFP4},
		{"get_rows_q6_K_f32", &result.getRowsQ6K},
		{"mul_mat_q6_K_f32", &result.mulMatQ6K},
	}
	for _, item := range items {
		function, err := lib.ModuleFunction(module, item.name)
		if err != nil {
			return functionSet{}, err
		}
		*item.dst = function
	}
	return result, nil
}
