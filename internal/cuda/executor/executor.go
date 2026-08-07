package executor

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/bits"
	"slices"
	"sync"
	"unsafe"

	"llamacpp2go/internal/checked"
	"llamacpp2go/internal/cuda/cublas"
	"llamacpp2go/internal/cuda/device"
	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/cuda/kernel"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/planner"
	"llamacpp2go/internal/tensor/reference"
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
	Pointer driver.DevicePtr
	Shape   tensor.Shape
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
	mu          sync.Mutex
	executor    *Executor
	values      map[*tensor.Tensor]DeviceValue
	allocations []driver.DevicePtr
	released    bool
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
				var errs []error
				for _, pointer := range r.allocations {
					if err := state.Driver.MemFree(pointer); err != nil {
						errs = append(errs, err)
					}
				}
				return errors.Join(errs...)
			})
		}
		r.executor.mu.RUnlock()
	}
	r.released = true
	r.values = nil
	r.allocations = nil
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
	allocations := make([]driver.DevicePtr, 0, len(copies))
	err := e.worker.Do(ctx, func(state *device.State) error {
		fail := func(cause error) error {
			var errs []error
			errs = append(errs, cause)
			for _, pointer := range allocations {
				if releaseErr := state.Driver.MemFree(pointer); releaseErr != nil {
					errs = append(errs, releaseErr)
				}
			}
			allocations = nil
			return errors.Join(errs...)
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
			pointer, allocErr := state.Driver.MemAlloc(bytes)
			if allocErr != nil {
				return fail(allocErr)
			}
			allocations = append(allocations, pointer)
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
		executor:    e,
		allocations: allocations,
	}, values, nil
}

type executionResult struct {
	host        map[*tensor.Tensor]reference.Value
	values      map[*tensor.Tensor]DeviceValue
	allocations []driver.DevicePtr
}

// CompiledGraph: validated order and memory plan for repeated execution.
type CompiledGraph struct {
	outputs        []*tensor.Tensor
	order          []*tensor.Tensor
	memory         planner.Plan
	weightedRMS    map[*tensor.Tensor]weightedRMSFusion
	skipped        map[*tensor.Tensor]struct{}
	needBlas       bool
	bf16InputBytes uint64
}

type weightedRMSFusion struct {
	normalization *tensor.Tensor
	weight        *tensor.Tensor
}

const graphArenaAlignment = 256

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
	memory, err := planner.Build(outputs, graphArenaAlignment)
	if err != nil {
		return nil, err
	}
	compiled := &CompiledGraph{
		outputs: slices.Clone(outputs), order: order, memory: memory,
	}
	uses := make(map[*tensor.Tensor]int, len(order))
	outputSet := make(map[*tensor.Tensor]struct{}, len(outputs))
	for _, output := range outputs {
		outputSet[output] = struct{}{}
	}
	for _, node := range order {
		for _, input := range node.Inputs {
			uses[input]++
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
		compiled.weightedRMS[node] = weightedRMSFusion{normalization: normalization, weight: weight}
		compiled.skipped[normalization] = struct{}{}
	}
	return compiled, nil
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
	result, err := e.runCompiled(ctx, compiled, feeds, nil, false)
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
	result, err := e.runCompiled(ctx, compiled, hostFeeds, deviceFeeds, false)
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
	execution, err := e.runCompiled(ctx, compiled, hostFeeds, deviceFeeds, true)
	if err != nil {
		return nil, err
	}
	return &RetainedOutputs{
		executor: e, values: execution.values, allocations: execution.allocations,
	}, nil
}

func (e *Executor) runCompiled(
	ctx context.Context,
	compiled *CompiledGraph,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
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
		resources, resourceErr := e.ensureResources(state, compiled.needBlas, compiled.bf16InputBytes)
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
			resources.functions,
			resources.blas,
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
	functions functionSet,
	blas *blasState,
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
	retainedAllocations := make([]driver.DevicePtr, 0, 1)
	retainedOffsets, retainedBytes, err := retainedOutputLayout(order, outputSet)
	if err != nil {
		return nil, err
	}
	var retainedBase driver.DevicePtr
	if retainOutputs && retainedBytes > 0 {
		retainedBase, err = state.Driver.MemAlloc(retainedBytes)
		if err != nil {
			return nil, err
		}
		retainedAllocations = append(retainedAllocations, retainedBase)
	}
	retained := false
	defer func() {
		if retained {
			return
		}
		for _, pointer := range retainedAllocations {
			_ = state.Driver.MemFree(pointer)
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
			_, retainedOutput := outputSet[node]
			if node.Op == tensor.OpReshape && !(retainOutputs && retainedOutput) {
				pointers[node] = pointers[node.Inputs[0]]
				continue
			}
			if retainOutputs && retainedOutput {
				offset, present := retainedOffsets[node]
				if !present || uint64(retainedBase) > math.MaxUint64-offset {
					return nil, errors.New("CUDA retained output layout is invalid")
				}
				pointer := retainedBase + driver.DevicePtr(offset)
				retainedValues[node] = DeviceValue{Pointer: pointer, Shape: node.Shape}
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
		if node.Op == tensor.OpReshape {
			if _, retainedOutput := outputSet[node]; !(retainOutputs && retainedOutput) {
				continue
			}
		}
		if fusion, ok := compiled.weightedRMS[node]; ok {
			if err := launchWeightedRMSNorm(state, functions, node, fusion, pointers); err != nil {
				abortCapture()
				return nil, fmt.Errorf("launch tensor %d (weighted_rms_norm): %w", node.ID, err)
			}
			submitted = true
			continue
		}
		if err := launchNode(state, functions, blas, node, pointers, attributePointers); err != nil {
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
			values:      retainedValues,
			allocations: retainedAllocations,
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
	add                 driver.Function
	multiply            driver.Function
	divide              driver.Function
	broadcastAdd        driver.Function
	broadcastMultiply   driver.Function
	broadcastDivide     driver.Function
	scale               driver.Function
	clamp               driver.Function
	bf16Round           driver.Function
	f32ToBF16           driver.Function
	copy                driver.Function
	silu                driver.Function
	gelu                driver.Function
	geluErf             driver.Function
	xielu               driver.Function
	reluSquared         driver.Function
	relu                driver.Function
	conv1DSame          driver.Function
	conv2D              driver.Function
	windowPartition2D   driver.Function
	windowUnpartition2D driver.Function
	samAttention        driver.Function
	groupNorm           driver.Function
	sigmoid             driver.Function
	softplus            driver.Function
	tanh                driver.Function
	exp                 driver.Function
	l2Norm              driver.Function
	ssmConv             driver.Function
	ssmScan             driver.Function
	gatedDeltaNet       driver.Function
	gatedLinearAttn     driver.Function
	rwkv6               driver.Function
	sumRows             driver.Function
	fwht                driver.Function
	argmax              driver.Function
	topK                driver.Function
	gatherLast          driver.Function
	gatherLastQ8        driver.Function
	sparseAttention     driver.Function
	indexerScore        driver.Function
	rwkv7               driver.Function
	moe                 driver.Function
	loraMerge           driver.Function
	repeatHeads         driver.Function
	transpose2D         driver.Function
	groupSlice          driver.Function
	flatSlice           driver.Function
	rmsNorm             driver.Function
	weightedRMSNorm     driver.Function
	layerNorm           driver.Function
	softmax             driver.Function
	mulMat              driver.Function
	getRows             driver.Function
	ropeNeoX            driver.Function
	ropeNormal          driver.Function
	ropeMulti           driver.Function
	attention           driver.Function
	attentionDecode     driver.Function
	concat              driver.Function
	getRowsQ8           driver.Function
	mulMatQ8            driver.Function
	getRowsQ81          driver.Function
	mulMatQ81           driver.Function
	getRowsQ8K          driver.Function
	mulMatQ8K           driver.Function
	getRowsQ40          driver.Function
	mulMatQ40           driver.Function
	getRowsQ41          driver.Function
	mulMatQ41           driver.Function
	getRowsQ50          driver.Function
	mulMatQ50           driver.Function
	getRowsQ51          driver.Function
	mulMatQ51           driver.Function
	getRowsQ10          driver.Function
	mulMatQ10           driver.Function
	getRowsQ20          driver.Function
	mulMatQ20           driver.Function
	getRowsTQ20         driver.Function
	mulMatTQ20          driver.Function
	getRowsTQ10         driver.Function
	mulMatTQ10          driver.Function
	getRowsQ2K          driver.Function
	mulMatQ2K           driver.Function
	getRowsQ3K          driver.Function
	mulMatQ3K           driver.Function
	getRowsQ4K          driver.Function
	mulMatQ4K           driver.Function
	getRowsQ5K          driver.Function
	mulMatQ5K           driver.Function
	getRowsIQ4XS        driver.Function
	mulMatIQ4XS         driver.Function
	getRowsIQ4NL        driver.Function
	mulMatIQ4NL         driver.Function
	getRowsIQ2XXS       driver.Function
	mulMatIQ2XXS        driver.Function
	getRowsIQ2XS        driver.Function
	mulMatIQ2XS         driver.Function
	getRowsIQ2S         driver.Function
	mulMatIQ2S          driver.Function
	getRowsIQ3XXS       driver.Function
	mulMatIQ3XXS        driver.Function
	getRowsIQ3S         driver.Function
	mulMatIQ3S          driver.Function
	getRowsIQ1S         driver.Function
	mulMatIQ1S          driver.Function
	getRowsIQ1M         driver.Function
	mulMatIQ1M          driver.Function
	getRowsMXFP4        driver.Function
	mulMatMXFP4         driver.Function
	getRowsNVFP4        driver.Function
	mulMatNVFP4         driver.Function
	getRowsQ6K          driver.Function
	mulMatQ6K           driver.Function
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

func (e *Executor) ensureResources(
	state *device.State,
	needBlas bool,
	bf16InputBytes uint64,
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
		{"gather_last_f32", &result.gatherLast},
		{"gather_last_q8_0_f32", &result.gatherLastQ8},
		{"sparse_attention_f32", &result.sparseAttention},
		{"indexer_score_f32", &result.indexerScore},
		{"rwkv7_f32", &result.rwkv7},
		{"moe_f32", &result.moe},
		{"lora_merge_f32", &result.loraMerge},
		{"repeat_heads_f32", &result.repeatHeads},
		{"transpose_2d_f32", &result.transpose2D},
		{"group_slice_f32", &result.groupSlice},
		{"flat_slice_f32", &result.flatSlice},
		{"rms_norm_f32", &result.rmsNorm},
		{"weighted_rms_norm_f32", &result.weightedRMSNorm},
		{"layer_norm_f32", &result.layerNorm},
		{"softmax_f32", &result.softmax},
		{"mul_mat_f32", &result.mulMat},
		{"get_rows_f32", &result.getRows},
		{"rope_neox_f32", &result.ropeNeoX},
		{"rope_normal_f32", &result.ropeNormal},
		{"rope_multi_f32", &result.ropeMulti},
		{"attention_f32", &result.attention},
		{"attention_decode_f32", &result.attentionDecode},
		{"concat_f32", &result.concat},
		{"get_rows_q8_0_f32", &result.getRowsQ8},
		{"mul_mat_q8_0_f32", &result.mulMatQ8},
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
