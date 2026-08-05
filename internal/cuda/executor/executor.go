package executor

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
	"unsafe"

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
	outputs  []*tensor.Tensor
	order    []*tensor.Tensor
	memory   planner.Plan
	needBlas bool
}

const graphArenaAlignment = 256

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
	for _, node := range order {
		if (node.Op == tensor.OpMulMat || node.Op == tensor.OpGroupedMulMat) &&
			node.Inputs[0].Type == dtype.F32 {
			compiled.needBlas = true
			break
		}
	}
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
		resources, resourceErr := e.ensureResources(state, compiled.needBlas)
		if resourceErr != nil {
			return resourceErr
		}
		var executeErr error
		result, executeErr = execute(
			state,
			compiled,
			hostFeeds,
			deviceFeeds,
			resources.functions,
			resources.blas,
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
	retainOutputs bool,
) (*executionResult, error) {
	outputs := compiled.outputs
	order := compiled.order
	plan := compiled.memory

	var arena driver.DevicePtr
	if plan.ArenaSize > 0 {
		var err error
		arena, err = state.Driver.MemAlloc(plan.ArenaSize)
		if err != nil {
			return nil, err
		}
		defer state.Driver.MemFree(arena)
	}

	pointers := make(map[*tensor.Tensor]driver.DevicePtr, len(order))
	outputSet := make(map[*tensor.Tensor]struct{}, len(outputs))
	for _, output := range outputs {
		outputSet[output] = struct{}{}
	}
	retainedValues := make(map[*tensor.Tensor]DeviceValue, len(outputs))
	retainedAllocations := make([]driver.DevicePtr, 0, len(outputs))
	retained := false
	defer func() {
		if retained {
			return
		}
		for _, pointer := range retainedAllocations {
			_ = state.Driver.MemFree(pointer)
		}
	}()
	inputPointers := make([]driver.DevicePtr, 0)
	defer func() {
		for _, pointer := range inputPointers {
			_ = state.Driver.MemFree(pointer)
		}
	}()
	for _, node := range order {
		if node.Op != tensor.OpInput && node.Type != dtype.F32 {
			return nil, fmt.Errorf("CUDA executor does not support %s for tensor %d", node.Type, node.ID)
		}
		if node.Op != tensor.OpInput {
			if _, keep := outputSet[node]; retainOutputs && keep {
				bytes, bytesErr := node.Shape.Bytes(node.Type)
				if bytesErr != nil {
					return nil, bytesErr
				}
				pointer, allocateErr := state.Driver.MemAlloc(bytes)
				if allocateErr != nil {
					return nil, allocateErr
				}
				retainedAllocations = append(retainedAllocations, pointer)
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
			if node.Type != dtype.F32 && !nativeQuantizedType(node.Type) {
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
		bytes, err := node.Shape.Bytes(node.Type)
		if err != nil {
			return nil, err
		}
		pointer, err := state.Driver.MemAlloc(bytes)
		if err != nil {
			return nil, err
		}
		inputPointers = append(inputPointers, pointer)
		pointers[node] = pointer
		if allZeroFloat32(value.Data) {
			if err := state.Driver.MemsetD32Async(
				pointer,
				0,
				uint64(len(value.Data)),
				state.Stream,
			); err != nil {
				return nil, err
			}
		} else {
			if err := state.Driver.MemcpyHtoD(pointer, driver.Bytes(value.Data)); err != nil {
				return nil, err
			}
		}
	}

	attributePointers := make(map[*tensor.Tensor]driver.DevicePtr)
	auxiliaryPointers := make([]driver.DevicePtr, 0)
	sharedAttributes := make(map[string]driver.DevicePtr)
	defer func() {
		for _, pointer := range auxiliaryPointers {
			_ = state.Driver.MemFree(pointer)
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
		pointer, allocateErr := state.Driver.MemAlloc(bytes)
		if allocateErr != nil {
			return nil, allocateErr
		}
		auxiliaryPointers = append(auxiliaryPointers, pointer)
		sharedAttributes[key] = pointer
		attributePointers[node] = pointer
		if copyErr := state.Driver.MemcpyHtoD(pointer, encoded); copyErr != nil {
			return nil, copyErr
		}
	}

	for _, node := range order {
		if node.Op == tensor.OpInput {
			continue
		}
		if err := launchNode(state, functions, blas, node, pointers, attributePointers); err != nil {
			return nil, fmt.Errorf("launch tensor %d (%s): %w", node.ID, node.Op, err)
		}
	}
	if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
		return nil, err
	}
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

func allZeroFloat32(values []float32) bool {
	if len(values) == 0 {
		return false
	}
	for _, value := range values {
		if value != 0 {
			return false
		}
	}
	return true
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
	topK                driver.Function
	gatherLast          driver.Function
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
	layerNorm           driver.Function
	softmax             driver.Function
	mulMat              driver.Function
	getRows             driver.Function
	ropeNeoX            driver.Function
	ropeNormal          driver.Function
	ropeMulti           driver.Function
	attention           driver.Function
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
	library *cublas.Library
	handle  cublas.Handle
}

func (e *Executor) ensureResources(
	state *device.State,
	needBlas bool,
) (executorResources, error) {
	if e.resources.module == 0 {
		if err := kernel.ValidateAssets(); err != nil {
			return executorResources{}, err
		}
		module, err := state.Driver.ModuleLoadData(kernel.OpsF32PTX)
		if err != nil {
			return executorResources{}, err
		}
		functions, err := loadFunctions(state.Driver, module)
		if err != nil {
			_ = state.Driver.ModuleUnload(module)
			return executorResources{}, err
		}
		e.resources.module = module
		e.resources.functions = functions
	}
	if needBlas && e.resources.blas == nil {
		library, err := cublas.Open()
		if err != nil {
			return executorResources{}, err
		}
		handle, err := library.Create()
		if err != nil {
			_ = library.Close()
			return executorResources{}, err
		}
		if err := library.SetStream(handle, state.Stream); err != nil {
			_ = library.Destroy(handle)
			_ = library.Close()
			return executorResources{}, err
		}
		e.resources.blas = &blasState{library: library, handle: handle}
	}
	return e.resources, nil
}

func (e *Executor) closeResources(state *device.State) error {
	var errs []error
	if e.resources.blas != nil {
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
		{"top_k_f32", &result.topK},
		{"gather_last_f32", &result.gatherLast},
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
		{"layer_norm_f32", &result.layerNorm},
		{"softmax_f32", &result.softmax},
		{"mul_mat_f32", &result.mulMat},
		{"get_rows_f32", &result.getRows},
		{"rope_neox_f32", &result.ropeNeoX},
		{"rope_normal_f32", &result.ropeNormal},
		{"rope_multi_f32", &result.ropeMulti},
		{"attention_f32", &result.attention},
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
