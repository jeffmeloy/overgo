package executor

import (
	"context"
	"errors"
	"fmt"
	"math"
	"runtime"
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

// Executor evaluates the initial F32 tensor graph on one CUDA worker.
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

// RetainedOutputs owns selected graph outputs in standalone device
// allocations. Call Release when the values are no longer used.
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
	if elements > uint64(maxInt()) {
		return reference.Value{}, errors.New("CUDA retained output is too large")
	}
	data := make([]float32, int(elements))
	r.executor.mu.RLock()
	defer r.executor.mu.RUnlock()
	if r.executor.closed || r.executor.worker == nil {
		return reference.Value{}, errors.New("CUDA executor is closed")
	}
	err = r.executor.worker.Do(ctx, func(state *device.State) error {
		return state.Driver.MemcpyDtoH(float32Bytes(data), value.Pointer)
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
// caller-selected source segments. It is used for persistent state edits that
// cannot be represented by a pointer view.
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

func New(deviceOrdinal int) (*Executor, error) {
	worker, err := device.New(deviceOrdinal)
	if err != nil {
		return nil, err
	}
	return &Executor{worker: worker, ownsWorker: true}, nil
}

// NewWithWorker binds an executor to a caller-owned CUDA worker/context.
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

// Execute evaluates outputs and returns host copies of those tensors.
func (e *Executor) Execute(
	ctx context.Context,
	outputs []*tensor.Tensor,
	feeds map[*tensor.Tensor]reference.Value,
) (map[*tensor.Tensor]reference.Value, error) {
	if e == nil {
		return nil, errors.New("CUDA executor is closed")
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.worker == nil {
		return nil, errors.New("CUDA executor is closed")
	}
	var results map[*tensor.Tensor]reference.Value
	err := e.worker.Do(ctx, func(state *device.State) error {
		needBlas, graphErr := graphRequiresBlas(outputs)
		if graphErr != nil {
			return graphErr
		}
		resources, resourceErr := e.ensureResources(state, needBlas)
		if resourceErr != nil {
			return resourceErr
		}
		var executeErr error
		execution, executeErr := execute(
			state,
			outputs,
			feeds,
			nil,
			resources.functions,
			resources.blas,
			false,
		)
		if executeErr == nil {
			results = execution.host
		}
		return executeErr
	})
	return results, err
}

// ExecuteWithDeviceFeeds evaluates a graph with selected F32 input nodes
// already resident in the executor's CUDA context.
func (e *Executor) ExecuteWithDeviceFeeds(
	ctx context.Context,
	outputs []*tensor.Tensor,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
) (map[*tensor.Tensor]reference.Value, error) {
	if e == nil {
		return nil, errors.New("CUDA executor is closed")
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.worker == nil {
		return nil, errors.New("CUDA executor is closed")
	}
	var results map[*tensor.Tensor]reference.Value
	err := e.worker.Do(ctx, func(state *device.State) error {
		needBlas, graphErr := graphRequiresBlas(outputs)
		if graphErr != nil {
			return graphErr
		}
		resources, resourceErr := e.ensureResources(state, needBlas)
		if resourceErr != nil {
			return resourceErr
		}
		var executeErr error
		execution, executeErr := execute(
			state,
			outputs,
			hostFeeds,
			deviceFeeds,
			resources.functions,
			resources.blas,
			false,
		)
		if executeErr == nil {
			results = execution.host
		}
		return executeErr
	})
	return results, err
}

// ExecuteRetainedWithDeviceFeeds evaluates a graph but leaves each requested
// output in an individually owned device allocation.
func (e *Executor) ExecuteRetainedWithDeviceFeeds(
	ctx context.Context,
	outputs []*tensor.Tensor,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
) (*RetainedOutputs, error) {
	if e == nil {
		return nil, errors.New("CUDA executor is closed")
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.worker == nil {
		return nil, errors.New("CUDA executor is closed")
	}
	var result *RetainedOutputs
	err := e.worker.Do(ctx, func(state *device.State) error {
		needBlas, graphErr := graphRequiresBlas(outputs)
		if graphErr != nil {
			return graphErr
		}
		resources, resourceErr := e.ensureResources(state, needBlas)
		if resourceErr != nil {
			return resourceErr
		}
		execution, executeErr := execute(
			state,
			outputs,
			hostFeeds,
			deviceFeeds,
			resources.functions,
			resources.blas,
			true,
		)
		if executeErr != nil {
			return executeErr
		}
		result = &RetainedOutputs{
			executor:    e,
			values:      execution.values,
			allocations: execution.allocations,
		}
		return nil
	})
	return result, err
}

func execute(
	state *device.State,
	outputs []*tensor.Tensor,
	feeds map[*tensor.Tensor]reference.Value,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
	functions functionSet,
	blas *blasState,
	retainOutputs bool,
) (*executionResult, error) {
	order, err := tensor.Topological(outputs...)
	if err != nil {
		return nil, err
	}
	plan, err := planner.Build(outputs, 256)
	if err != nil {
		return nil, err
	}

	var arena driver.DevicePtr
	if plan.ArenaSize > 0 {
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
			if err := state.Driver.MemcpyHtoD(pointer, float32Bytes(value.Data)); err != nil {
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
		encoded := uint32Bytes(values)
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
		if elements > uint64(maxInt()) {
			return nil, errors.New("output is too large for host memory")
		}
		data := make([]float32, int(elements))
		if err := state.Driver.MemcpyDtoH(float32Bytes(data), pointers[output]); err != nil {
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
	add               driver.Function
	multiply          driver.Function
	broadcastAdd      driver.Function
	broadcastMultiply driver.Function
	scale             driver.Function
	copy              driver.Function
	silu              driver.Function
	gelu              driver.Function
	xielu             driver.Function
	reluSquared       driver.Function
	sigmoid           driver.Function
	softplus          driver.Function
	l2Norm            driver.Function
	ssmConv           driver.Function
	gatedDeltaNet     driver.Function
	moe               driver.Function
	repeatHeads       driver.Function
	transpose2D       driver.Function
	groupSlice        driver.Function
	flatSlice         driver.Function
	rmsNorm           driver.Function
	layerNorm         driver.Function
	softmax           driver.Function
	mulMat            driver.Function
	getRows           driver.Function
	ropeNeoX          driver.Function
	ropeNormal        driver.Function
	ropeMulti         driver.Function
	attention         driver.Function
	concat            driver.Function
	getRowsQ8         driver.Function
	mulMatQ8          driver.Function
	getRowsQ81        driver.Function
	mulMatQ81         driver.Function
	getRowsQ8K        driver.Function
	mulMatQ8K         driver.Function
	getRowsQ40        driver.Function
	mulMatQ40         driver.Function
	getRowsQ41        driver.Function
	mulMatQ41         driver.Function
	getRowsQ50        driver.Function
	mulMatQ50         driver.Function
	getRowsQ51        driver.Function
	mulMatQ51         driver.Function
	getRowsQ10        driver.Function
	mulMatQ10         driver.Function
	getRowsQ20        driver.Function
	mulMatQ20         driver.Function
	getRowsTQ20       driver.Function
	mulMatTQ20        driver.Function
	getRowsTQ10       driver.Function
	mulMatTQ10        driver.Function
	getRowsQ2K        driver.Function
	mulMatQ2K         driver.Function
	getRowsQ3K        driver.Function
	mulMatQ3K         driver.Function
	getRowsQ4K        driver.Function
	mulMatQ4K         driver.Function
	getRowsQ5K        driver.Function
	mulMatQ5K         driver.Function
	getRowsIQ4XS      driver.Function
	mulMatIQ4XS       driver.Function
	getRowsIQ4NL      driver.Function
	mulMatIQ4NL       driver.Function
	getRowsIQ2XXS     driver.Function
	mulMatIQ2XXS      driver.Function
	getRowsIQ2XS      driver.Function
	mulMatIQ2XS       driver.Function
	getRowsIQ2S       driver.Function
	mulMatIQ2S        driver.Function
	getRowsIQ3XXS     driver.Function
	mulMatIQ3XXS      driver.Function
	getRowsIQ3S       driver.Function
	mulMatIQ3S        driver.Function
	getRowsIQ1S       driver.Function
	mulMatIQ1S        driver.Function
	getRowsIQ1M       driver.Function
	mulMatIQ1M        driver.Function
	getRowsMXFP4      driver.Function
	mulMatMXFP4       driver.Function
	getRowsNVFP4      driver.Function
	mulMatNVFP4       driver.Function
	getRowsQ6K        driver.Function
	mulMatQ6K         driver.Function
}

type blasState struct {
	library *cublas.Library
	handle  cublas.Handle
}

func graphRequiresBlas(outputs []*tensor.Tensor) (bool, error) {
	nodes, err := tensor.Topological(outputs...)
	if err != nil {
		return false, err
	}
	for _, node := range nodes {
		if node.Op == tensor.OpMulMat && node.Inputs[0].Type == dtype.F32 {
			return true, nil
		}
	}
	return false, nil
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
		{"broadcast_add_f32", &result.broadcastAdd},
		{"broadcast_multiply_f32", &result.broadcastMultiply},
		{"scale_f32", &result.scale},
		{"copy_f32", &result.copy},
		{"silu_f32", &result.silu},
		{"gelu_f32", &result.gelu},
		{"xielu_f32", &result.xielu},
		{"relu_squared_f32", &result.reluSquared},
		{"sigmoid_f32", &result.sigmoid},
		{"softplus_f32", &result.softplus},
		{"l2_norm_f32", &result.l2Norm},
		{"ssm_conv_f32", &result.ssmConv},
		{"gated_delta_net_f32", &result.gatedDeltaNet},
		{"moe_f32", &result.moe},
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

func launchNode(
	state *device.State,
	functions functionSet,
	blas *blasState,
	node *tensor.Tensor,
	pointers map[*tensor.Tensor]driver.DevicePtr,
	attributePointers map[*tensor.Tensor]driver.DevicePtr,
) error {
	output := pointers[node]
	switch node.Op {
	case tensor.OpAdd, tensor.OpMultiply:
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		left := pointers[node.Inputs[0]]
		right := pointers[node.Inputs[1]]
		if node.Inputs[0].Shape.Equal(node.Shape) && node.Inputs[1].Shape.Equal(node.Shape) {
			function := functions.add
			if node.Op == tensor.OpMultiply {
				function = functions.multiply
			}
			args := []unsafe.Pointer{
				unsafe.Pointer(&left),
				unsafe.Pointer(&right),
				unsafe.Pointer(&output),
				unsafe.Pointer(&count),
			}
			err = launch1D(state, function, count, args)
			runtime.KeepAlive(left)
			runtime.KeepAlive(right)
			runtime.KeepAlive(output)
			runtime.KeepAlive(count)
			return err
		}
		leftDimensions, err := shapeDimensions32(node.Inputs[0].Shape)
		if err != nil {
			return err
		}
		rightDimensions, err := shapeDimensions32(node.Inputs[1].Shape)
		if err != nil {
			return err
		}
		outputDimensions, err := shapeDimensions32(node.Shape)
		if err != nil {
			return err
		}
		function := functions.broadcastAdd
		if node.Op == tensor.OpMultiply {
			function = functions.broadcastMultiply
		}
		args := []unsafe.Pointer{
			unsafe.Pointer(&left),
			unsafe.Pointer(&right),
			unsafe.Pointer(&output),
			unsafe.Pointer(&count),
			unsafe.Pointer(&leftDimensions[0]),
			unsafe.Pointer(&leftDimensions[1]),
			unsafe.Pointer(&leftDimensions[2]),
			unsafe.Pointer(&leftDimensions[3]),
			unsafe.Pointer(&rightDimensions[0]),
			unsafe.Pointer(&rightDimensions[1]),
			unsafe.Pointer(&rightDimensions[2]),
			unsafe.Pointer(&rightDimensions[3]),
			unsafe.Pointer(&outputDimensions[0]),
			unsafe.Pointer(&outputDimensions[1]),
			unsafe.Pointer(&outputDimensions[2]),
		}
		err = launch1D(state, function, count, args)
		runtime.KeepAlive(left)
		runtime.KeepAlive(right)
		runtime.KeepAlive(output)
		runtime.KeepAlive(count)
		runtime.KeepAlive(leftDimensions)
		runtime.KeepAlive(rightDimensions)
		runtime.KeepAlive(outputDimensions)
		return err
	case tensor.OpScale:
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		attributes, ok := node.Attrs.(tensor.ScaleAttributes)
		if !ok {
			return errors.New("invalid scale attributes")
		}
		input := pointers[node.Inputs[0]]
		scale := attributes.Value
		args := []unsafe.Pointer{
			unsafe.Pointer(&input),
			unsafe.Pointer(&output),
			unsafe.Pointer(&scale),
			unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.scale, count, args)
		runtime.KeepAlive(input)
		runtime.KeepAlive(output)
		runtime.KeepAlive(scale)
		runtime.KeepAlive(count)
		return err
	case tensor.OpSiLU, tensor.OpGELU, tensor.OpReLUSquared, tensor.OpSigmoid, tensor.OpSoftplus:
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		args := []unsafe.Pointer{
			unsafe.Pointer(&input),
			unsafe.Pointer(&output),
			unsafe.Pointer(&count),
		}
		function := functions.silu
		if node.Op == tensor.OpGELU {
			function = functions.gelu
		} else if node.Op == tensor.OpReLUSquared {
			function = functions.reluSquared
		} else if node.Op == tensor.OpSigmoid {
			function = functions.sigmoid
		} else if node.Op == tensor.OpSoftplus {
			function = functions.softplus
		}
		err = launch1D(state, function, count, args)
		runtime.KeepAlive(input)
		runtime.KeepAlive(output)
		runtime.KeepAlive(count)
		return err
	case tensor.OpXIELU:
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		attributes, ok := node.Attrs.(tensor.XIELUAttributes)
		if !ok {
			return errors.New("invalid xIELU attributes")
		}
		input := pointers[node.Inputs[0]]
		alphaN := attributes.AlphaN
		alphaP := attributes.AlphaP
		beta := attributes.Beta
		epsilon := attributes.Epsilon
		args := []unsafe.Pointer{
			unsafe.Pointer(&input),
			unsafe.Pointer(&output),
			unsafe.Pointer(&alphaN),
			unsafe.Pointer(&alphaP),
			unsafe.Pointer(&beta),
			unsafe.Pointer(&epsilon),
			unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.xielu, count, args)
		runtime.KeepAlive(input)
		runtime.KeepAlive(output)
		runtime.KeepAlive(alphaN)
		runtime.KeepAlive(alphaP)
		runtime.KeepAlive(beta)
		runtime.KeepAlive(epsilon)
		runtime.KeepAlive(count)
		return err
	case tensor.OpL2Norm:
		attributes, ok := node.Attrs.(tensor.L2NormAttributes)
		if !ok {
			return errors.New("invalid L2Norm attributes")
		}
		width, rows, err := rowDimensions32(node.Shape)
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		epsilon := attributes.Epsilon
		args := []unsafe.Pointer{
			unsafe.Pointer(&input),
			unsafe.Pointer(&output),
			unsafe.Pointer(&width),
			unsafe.Pointer(&rows),
			unsafe.Pointer(&epsilon),
		}
		err = launch1D(state, functions.l2Norm, rows, args)
		runtime.KeepAlive(input)
		runtime.KeepAlive(output)
		runtime.KeepAlive(width)
		runtime.KeepAlive(rows)
		runtime.KeepAlive(epsilon)
		return err
	case tensor.OpSSMConv:
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		window, err := uint32Checked(node.Inputs[0].Shape.Dims[0], "SSMConv window")
		if err != nil {
			return err
		}
		channels, err := uint32Checked(node.Shape.Dims[0], "SSMConv channels")
		if err != nil {
			return err
		}
		tokens, err := uint32Checked(node.Shape.Dims[1], "SSMConv tokens")
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		weights := pointers[node.Inputs[1]]
		args := []unsafe.Pointer{
			unsafe.Pointer(&input),
			unsafe.Pointer(&weights),
			unsafe.Pointer(&output),
			unsafe.Pointer(&window),
			unsafe.Pointer(&channels),
			unsafe.Pointer(&tokens),
			unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.ssmConv, count, args)
		runtime.KeepAlive(input)
		runtime.KeepAlive(weights)
		runtime.KeepAlive(output)
		runtime.KeepAlive(window)
		runtime.KeepAlive(channels)
		runtime.KeepAlive(tokens)
		runtime.KeepAlive(count)
		return err
	case tensor.OpGatedDeltaNet:
		size, err := uint32Checked(node.Inputs[2].Shape.Dims[0], "GatedDeltaNet state width")
		if err != nil {
			return err
		}
		qHeads, err := uint32Checked(node.Inputs[0].Shape.Dims[1], "GatedDeltaNet Q heads")
		if err != nil {
			return err
		}
		kHeads, err := uint32Checked(node.Inputs[1].Shape.Dims[1], "GatedDeltaNet K heads")
		if err != nil {
			return err
		}
		heads, err := uint32Checked(node.Inputs[2].Shape.Dims[1], "GatedDeltaNet value heads")
		if err != nil {
			return err
		}
		tokens, err := uint32Checked(node.Inputs[2].Shape.Dims[2], "GatedDeltaNet tokens")
		if err != nil {
			return err
		}
		sequences, err := uint32Checked(node.Inputs[2].Shape.Dims[3], "GatedDeltaNet sequences")
		if err != nil {
			return err
		}
		gateWidth, err := uint32Checked(node.Inputs[3].Shape.Dims[0], "GatedDeltaNet gate width")
		if err != nil {
			return err
		}
		if uint64(heads)*uint64(sequences) > uint64(^uint32(0)) {
			return errors.New("GatedDeltaNet launch count exceeds uint32")
		}
		count := heads * sequences
		query := pointers[node.Inputs[0]]
		key := pointers[node.Inputs[1]]
		value := pointers[node.Inputs[2]]
		gate := pointers[node.Inputs[3]]
		beta := pointers[node.Inputs[4]]
		stateInput := pointers[node.Inputs[5]]
		args := []unsafe.Pointer{
			unsafe.Pointer(&query),
			unsafe.Pointer(&key),
			unsafe.Pointer(&value),
			unsafe.Pointer(&gate),
			unsafe.Pointer(&beta),
			unsafe.Pointer(&stateInput),
			unsafe.Pointer(&output),
			unsafe.Pointer(&size),
			unsafe.Pointer(&qHeads),
			unsafe.Pointer(&kHeads),
			unsafe.Pointer(&heads),
			unsafe.Pointer(&tokens),
			unsafe.Pointer(&sequences),
			unsafe.Pointer(&gateWidth),
		}
		err = launch1D(state, functions.gatedDeltaNet, count, args)
		runtime.KeepAlive(query)
		runtime.KeepAlive(key)
		runtime.KeepAlive(value)
		runtime.KeepAlive(gate)
		runtime.KeepAlive(beta)
		runtime.KeepAlive(stateInput)
		runtime.KeepAlive(output)
		runtime.KeepAlive(size)
		runtime.KeepAlive(qHeads)
		runtime.KeepAlive(kHeads)
		runtime.KeepAlive(heads)
		runtime.KeepAlive(tokens)
		runtime.KeepAlive(sequences)
		runtime.KeepAlive(gateWidth)
		return err
	case tensor.OpMoE:
		attributes, ok := node.Attrs.(tensor.MoEAttributes)
		if !ok || (len(node.Inputs) != 5 && len(node.Inputs) != 6) ||
			(attributes.Routing != tensor.MoERoutingSoftmax && attributes.Routing != tensor.MoERoutingSigmoid) {
			return errors.New("invalid MoE attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		hidden, err := uint32Checked(node.Shape.Dims[0], "MoE hidden width")
		if err != nil {
			return err
		}
		tokens, err := uint32Checked(node.Shape.Dims[1], "MoE token count")
		if err != nil {
			return err
		}
		intermediate, err := uint32Checked(node.Inputs[2].Shape.Dims[1], "MoE intermediate width")
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		router := pointers[node.Inputs[1]]
		gate := pointers[node.Inputs[2]]
		up := pointers[node.Inputs[3]]
		down := pointers[node.Inputs[4]]
		var selectionBias driver.DevicePtr
		if len(node.Inputs) == 6 {
			selectionBias = pointers[node.Inputs[5]]
		}
		experts := attributes.Experts
		topK := attributes.TopK
		var normalize uint32
		if attributes.NormalizeTopKProb {
			normalize = 1
		}
		scale := attributes.Scale
		routing := uint32(attributes.Routing)
		args := []unsafe.Pointer{
			unsafe.Pointer(&input), unsafe.Pointer(&router), unsafe.Pointer(&gate),
			unsafe.Pointer(&up), unsafe.Pointer(&down), unsafe.Pointer(&selectionBias), unsafe.Pointer(&output),
			unsafe.Pointer(&hidden), unsafe.Pointer(&tokens), unsafe.Pointer(&experts),
			unsafe.Pointer(&topK), unsafe.Pointer(&intermediate), unsafe.Pointer(&normalize),
			unsafe.Pointer(&routing), unsafe.Pointer(&scale), unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.moe, count, args)
		runtime.KeepAlive(args)
		return err
	case tensor.OpRepeatHeads:
		attributes, ok := node.Attrs.(tensor.RepeatHeadsAttributes)
		if !ok || attributes.Heads == 0 || len(node.Inputs) != 1 {
			return errors.New("invalid RepeatHeads attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		width, err := uint32Checked(node.Shape.Dims[0], "RepeatHeads width")
		if err != nil {
			return err
		}
		tokens, err := uint32Checked(node.Shape.Dims[2], "RepeatHeads token count")
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		heads := attributes.Heads
		args := []unsafe.Pointer{
			unsafe.Pointer(&input), unsafe.Pointer(&output), unsafe.Pointer(&width),
			unsafe.Pointer(&heads), unsafe.Pointer(&tokens), unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.repeatHeads, count, args)
		runtime.KeepAlive(args)
		return err
	case tensor.OpTranspose2D:
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		width, err := uint32Checked(node.Inputs[0].Shape.Dims[0], "Transpose2D width")
		if err != nil {
			return err
		}
		rows, err := uint32Checked(node.Inputs[0].Shape.Dims[1], "Transpose2D rows")
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		args := []unsafe.Pointer{
			unsafe.Pointer(&input),
			unsafe.Pointer(&output),
			unsafe.Pointer(&width),
			unsafe.Pointer(&rows),
			unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.transpose2D, count, args)
		runtime.KeepAlive(input)
		runtime.KeepAlive(output)
		runtime.KeepAlive(width)
		runtime.KeepAlive(rows)
		runtime.KeepAlive(count)
		return err
	case tensor.OpGroupSlice:
		attributes, ok := node.Attrs.(tensor.GroupSliceAttributes)
		if !ok {
			return errors.New("invalid GroupSlice attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		inputWidth, err := uint32Checked(node.Inputs[0].Shape.Dims[0], "GroupSlice input width")
		if err != nil {
			return err
		}
		offset, err := uint32Checked(attributes.Offset, "GroupSlice offset")
		if err != nil {
			return err
		}
		width, err := uint32Checked(attributes.Width, "GroupSlice width")
		if err != nil {
			return err
		}
		groups, err := uint32Checked(attributes.Groups, "GroupSlice groups")
		if err != nil {
			return err
		}
		stride, err := uint32Checked(attributes.Stride, "GroupSlice stride")
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		args := []unsafe.Pointer{
			unsafe.Pointer(&input),
			unsafe.Pointer(&output),
			unsafe.Pointer(&inputWidth),
			unsafe.Pointer(&offset),
			unsafe.Pointer(&width),
			unsafe.Pointer(&groups),
			unsafe.Pointer(&stride),
			unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.groupSlice, count, args)
		runtime.KeepAlive(input)
		runtime.KeepAlive(output)
		runtime.KeepAlive(inputWidth)
		runtime.KeepAlive(offset)
		runtime.KeepAlive(width)
		runtime.KeepAlive(groups)
		runtime.KeepAlive(stride)
		runtime.KeepAlive(count)
		return err
	case tensor.OpFlatSlice:
		attributes, ok := node.Attrs.(tensor.FlatSliceAttributes)
		if !ok {
			return errors.New("invalid FlatSlice attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		offset, err := uint32Checked(attributes.Offset, "FlatSlice offset")
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		args := []unsafe.Pointer{
			unsafe.Pointer(&input),
			unsafe.Pointer(&output),
			unsafe.Pointer(&offset),
			unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.flatSlice, count, args)
		runtime.KeepAlive(input)
		runtime.KeepAlive(output)
		runtime.KeepAlive(offset)
		runtime.KeepAlive(count)
		return err
	case tensor.OpRMSNorm:
		attributes, ok := node.Attrs.(tensor.RMSNormAttributes)
		if !ok {
			return errors.New("invalid RMSNorm attributes")
		}
		width, rows, err := rowDimensions32(node.Shape)
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		epsilon := attributes.Epsilon
		args := []unsafe.Pointer{
			unsafe.Pointer(&input),
			unsafe.Pointer(&output),
			unsafe.Pointer(&width),
			unsafe.Pointer(&rows),
			unsafe.Pointer(&epsilon),
		}
		err = launch1D(state, functions.rmsNorm, rows, args)
		runtime.KeepAlive(input)
		runtime.KeepAlive(output)
		runtime.KeepAlive(width)
		runtime.KeepAlive(rows)
		runtime.KeepAlive(epsilon)
		return err
	case tensor.OpLayerNorm:
		attributes, ok := node.Attrs.(tensor.LayerNormAttributes)
		if !ok {
			return errors.New("invalid LayerNorm attributes")
		}
		width, rows, err := rowDimensions32(node.Shape)
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		epsilon := attributes.Epsilon
		args := []unsafe.Pointer{
			unsafe.Pointer(&input),
			unsafe.Pointer(&output),
			unsafe.Pointer(&width),
			unsafe.Pointer(&rows),
			unsafe.Pointer(&epsilon),
		}
		err = launch1D(state, functions.layerNorm, rows, args)
		runtime.KeepAlive(input)
		runtime.KeepAlive(output)
		runtime.KeepAlive(width)
		runtime.KeepAlive(rows)
		runtime.KeepAlive(epsilon)
		return err
	case tensor.OpSoftmax:
		width, rows, err := rowDimensions32(node.Shape)
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		args := []unsafe.Pointer{
			unsafe.Pointer(&input),
			unsafe.Pointer(&output),
			unsafe.Pointer(&width),
			unsafe.Pointer(&rows),
		}
		err = launch1D(state, functions.softmax, rows, args)
		runtime.KeepAlive(input)
		runtime.KeepAlive(output)
		runtime.KeepAlive(width)
		runtime.KeepAlive(rows)
		return err
	case tensor.OpMulMat:
		leftNode := node.Inputs[0]
		rightNode := node.Inputs[1]
		inner, err := uint32Checked(leftNode.Shape.Dims[0], "mul_mat inner dimension")
		if err != nil {
			return err
		}
		leftRows, err := uint32Checked(leftNode.Shape.Dims[1], "mul_mat left rows")
		if err != nil {
			return err
		}
		rightRows, err := uint32Checked(rightNode.Shape.Dims[1], "mul_mat right rows")
		if err != nil {
			return err
		}
		left := pointers[leftNode]
		right := pointers[rightNode]
		if nativeQuantizedType(leftNode.Type) {
			if rightNode.Type != dtype.F32 {
				return fmt.Errorf("%s mul_mat right input has type %s", leftNode.Type, rightNode.Type)
			}
			traits, _ := leftNode.Type.Traits()
			if uint64(inner)%traits.BlockSize != 0 {
				return fmt.Errorf("%s mul_mat inner dimension is not block aligned", leftNode.Type)
			}
			if uint64(leftRows)*uint64(rightRows) > math.MaxUint32 {
				return fmt.Errorf("%s mul_mat output element count exceeds uint32", leftNode.Type)
			}
			count := leftRows * rightRows
			args := []unsafe.Pointer{
				unsafe.Pointer(&left),
				unsafe.Pointer(&right),
				unsafe.Pointer(&output),
				unsafe.Pointer(&inner),
				unsafe.Pointer(&leftRows),
				unsafe.Pointer(&rightRows),
			}
			function := functions.mulMatQ8
			switch leftNode.Type {
			case dtype.Q8_1:
				function = functions.mulMatQ81
			case dtype.Q8K:
				function = functions.mulMatQ8K
			case dtype.Q4_0:
				function = functions.mulMatQ40
			case dtype.Q4_1:
				function = functions.mulMatQ41
			case dtype.Q5_0:
				function = functions.mulMatQ50
			case dtype.Q5_1:
				function = functions.mulMatQ51
			case dtype.Q1_0:
				function = functions.mulMatQ10
			case dtype.Q2_0:
				function = functions.mulMatQ20
			case dtype.TQ2_0:
				function = functions.mulMatTQ20
			case dtype.TQ1_0:
				function = functions.mulMatTQ10
			case dtype.Q2K:
				function = functions.mulMatQ2K
			case dtype.Q3K:
				function = functions.mulMatQ3K
			case dtype.Q4K:
				function = functions.mulMatQ4K
			case dtype.Q5K:
				function = functions.mulMatQ5K
			case dtype.IQ4XS:
				function = functions.mulMatIQ4XS
			case dtype.IQ4NL:
				function = functions.mulMatIQ4NL
			case dtype.IQ2XXS:
				function = functions.mulMatIQ2XXS
			case dtype.IQ2XS:
				function = functions.mulMatIQ2XS
			case dtype.IQ2S:
				function = functions.mulMatIQ2S
			case dtype.IQ3XXS:
				function = functions.mulMatIQ3XXS
			case dtype.IQ3S:
				function = functions.mulMatIQ3S
			case dtype.IQ1S:
				function = functions.mulMatIQ1S
			case dtype.IQ1M:
				function = functions.mulMatIQ1M
			case dtype.MXFP4:
				function = functions.mulMatMXFP4
			case dtype.NVFP4:
				function = functions.mulMatNVFP4
			case dtype.Q6K:
				function = functions.mulMatQ6K
			}
			err = launch1D(state, function, count, args)
			runtime.KeepAlive(left)
			runtime.KeepAlive(right)
			runtime.KeepAlive(output)
			runtime.KeepAlive(inner)
			runtime.KeepAlive(leftRows)
			runtime.KeepAlive(rightRows)
			return err
		}
		if blas == nil {
			return errors.New("cuBLAS is unavailable for F32 mul_mat")
		}
		err = blas.library.SGEMM(
			blas.handle,
			cublas.OperationTranspose,
			cublas.OperationNone,
			int32(leftRows),
			int32(rightRows),
			int32(inner),
			1,
			left,
			int32(inner),
			right,
			int32(inner),
			0,
			output,
			int32(leftRows),
		)
		runtime.KeepAlive(left)
		runtime.KeepAlive(right)
		runtime.KeepAlive(output)
		return err
	case tensor.OpGetRows:
		attributes, ok := node.Attrs.(tensor.GetRowsAttributes)
		if !ok {
			return errors.New("invalid get_rows attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		width, err := uint32Checked(node.Shape.Dims[0], "get_rows width")
		if err != nil {
			return err
		}
		table := pointers[node.Inputs[0]]
		rows, ok := attributePointers[node]
		if !ok {
			return errors.New("get_rows row storage is unavailable")
		}
		if len(attributes.Rows) == 0 {
			return errors.New("get_rows row list is empty")
		}
		args := []unsafe.Pointer{
			unsafe.Pointer(&table),
			unsafe.Pointer(&rows),
			unsafe.Pointer(&output),
			unsafe.Pointer(&width),
			unsafe.Pointer(&count),
		}
		function := functions.getRows
		if node.Inputs[0].Type == dtype.Q8_0 {
			if width%32 != 0 {
				return errors.New("Q8_0 get_rows width is not block aligned")
			}
			function = functions.getRowsQ8
		} else if node.Inputs[0].Type == dtype.Q8_1 {
			if width%32 != 0 {
				return errors.New("Q8_1 get_rows width is not block aligned")
			}
			function = functions.getRowsQ81
		} else if node.Inputs[0].Type == dtype.Q8K {
			if width%256 != 0 {
				return errors.New("Q8_K get_rows width is not block aligned")
			}
			function = functions.getRowsQ8K
		} else if node.Inputs[0].Type == dtype.Q4_0 {
			if width%32 != 0 {
				return errors.New("Q4_0 get_rows width is not block aligned")
			}
			function = functions.getRowsQ40
		} else if node.Inputs[0].Type == dtype.Q4_1 {
			if width%32 != 0 {
				return errors.New("Q4_1 get_rows width is not block aligned")
			}
			function = functions.getRowsQ41
		} else if node.Inputs[0].Type == dtype.Q5_0 {
			if width%32 != 0 {
				return errors.New("Q5_0 get_rows width is not block aligned")
			}
			function = functions.getRowsQ50
		} else if node.Inputs[0].Type == dtype.Q5_1 {
			if width%32 != 0 {
				return errors.New("Q5_1 get_rows width is not block aligned")
			}
			function = functions.getRowsQ51
		} else if node.Inputs[0].Type == dtype.Q1_0 {
			if width%128 != 0 {
				return errors.New("Q1_0 get_rows width is not block aligned")
			}
			function = functions.getRowsQ10
		} else if node.Inputs[0].Type == dtype.Q2_0 {
			if width%64 != 0 {
				return errors.New("Q2_0 get_rows width is not block aligned")
			}
			function = functions.getRowsQ20
		} else if node.Inputs[0].Type == dtype.TQ2_0 {
			if width%256 != 0 {
				return errors.New("TQ2_0 get_rows width is not block aligned")
			}
			function = functions.getRowsTQ20
		} else if node.Inputs[0].Type == dtype.TQ1_0 {
			if width%256 != 0 {
				return errors.New("TQ1_0 get_rows width is not block aligned")
			}
			function = functions.getRowsTQ10
		} else if node.Inputs[0].Type == dtype.Q2K {
			if width%256 != 0 {
				return errors.New("Q2_K get_rows width is not block aligned")
			}
			function = functions.getRowsQ2K
		} else if node.Inputs[0].Type == dtype.Q3K {
			if width%256 != 0 {
				return errors.New("Q3_K get_rows width is not block aligned")
			}
			function = functions.getRowsQ3K
		} else if node.Inputs[0].Type == dtype.Q4K {
			if width%256 != 0 {
				return errors.New("Q4_K get_rows width is not block aligned")
			}
			function = functions.getRowsQ4K
		} else if node.Inputs[0].Type == dtype.Q5K {
			if width%256 != 0 {
				return errors.New("Q5_K get_rows width is not block aligned")
			}
			function = functions.getRowsQ5K
		} else if node.Inputs[0].Type == dtype.IQ4XS {
			if width%256 != 0 {
				return errors.New("IQ4_XS get_rows width is not block aligned")
			}
			function = functions.getRowsIQ4XS
		} else if node.Inputs[0].Type == dtype.IQ4NL {
			if width%32 != 0 {
				return errors.New("IQ4_NL get_rows width is not block aligned")
			}
			function = functions.getRowsIQ4NL
		} else if node.Inputs[0].Type == dtype.IQ2XXS {
			if width%256 != 0 {
				return errors.New("IQ2_XXS get_rows width is not block aligned")
			}
			function = functions.getRowsIQ2XXS
		} else if node.Inputs[0].Type == dtype.IQ2XS {
			if width%256 != 0 {
				return errors.New("IQ2_XS get_rows width is not block aligned")
			}
			function = functions.getRowsIQ2XS
		} else if node.Inputs[0].Type == dtype.IQ2S {
			if width%256 != 0 {
				return errors.New("IQ2_S get_rows width is not block aligned")
			}
			function = functions.getRowsIQ2S
		} else if node.Inputs[0].Type == dtype.IQ3XXS {
			if width%256 != 0 {
				return errors.New("IQ3_XXS get_rows width is not block aligned")
			}
			function = functions.getRowsIQ3XXS
		} else if node.Inputs[0].Type == dtype.IQ3S {
			if width%256 != 0 {
				return errors.New("IQ3_S get_rows width is not block aligned")
			}
			function = functions.getRowsIQ3S
		} else if node.Inputs[0].Type == dtype.IQ1S {
			if width%256 != 0 {
				return errors.New("IQ1_S get_rows width is not block aligned")
			}
			function = functions.getRowsIQ1S
		} else if node.Inputs[0].Type == dtype.IQ1M {
			if width%256 != 0 {
				return errors.New("IQ1_M get_rows width is not block aligned")
			}
			function = functions.getRowsIQ1M
		} else if node.Inputs[0].Type == dtype.MXFP4 {
			if width%32 != 0 {
				return errors.New("MXFP4 get_rows width is not block aligned")
			}
			function = functions.getRowsMXFP4
		} else if node.Inputs[0].Type == dtype.NVFP4 {
			if width%64 != 0 {
				return errors.New("NVFP4 get_rows width is not block aligned")
			}
			function = functions.getRowsNVFP4
		} else if node.Inputs[0].Type == dtype.Q6K {
			if width%256 != 0 {
				return errors.New("Q6_K get_rows width is not block aligned")
			}
			function = functions.getRowsQ6K
		}
		err = launch1D(state, function, count, args)
		runtime.KeepAlive(table)
		runtime.KeepAlive(rows)
		runtime.KeepAlive(output)
		runtime.KeepAlive(width)
		runtime.KeepAlive(count)
		return err
	case tensor.OpRoPENeoX, tensor.OpRoPENormal:
		attributes, ok := node.Attrs.(tensor.RoPEAttributes)
		if !ok {
			return errors.New("invalid RoPE attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		width, err := uint32Checked(node.Shape.Dims[0], "RoPE width")
		if err != nil {
			return err
		}
		heads, err := uint32Checked(node.Shape.Dims[1], "RoPE heads")
		if err != nil {
			return err
		}
		tokens, err := uint32Checked(node.Shape.Dims[2], "RoPE tokens")
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		var frequencyFactors driver.DevicePtr
		if len(node.Inputs) == 2 {
			frequencyFactors = pointers[node.Inputs[1]]
		}
		positions, ok := attributePointers[node]
		if !ok {
			return errors.New("RoPE position storage is unavailable")
		}
		rotary := attributes.RotaryDimensions
		frequencyBase := attributes.FrequencyBase
		frequencyScale := attributes.FrequencyScale
		originalContext := attributes.OriginalContext
		extFactor := attributes.ExtFactor
		attentionFactor := attributes.AttentionFactor
		betaFast := attributes.BetaFast
		betaSlow := attributes.BetaSlow
		args := []unsafe.Pointer{
			unsafe.Pointer(&input),
			unsafe.Pointer(&positions),
			unsafe.Pointer(&frequencyFactors),
			unsafe.Pointer(&output),
			unsafe.Pointer(&width),
			unsafe.Pointer(&heads),
			unsafe.Pointer(&tokens),
			unsafe.Pointer(&rotary),
			unsafe.Pointer(&frequencyBase),
			unsafe.Pointer(&frequencyScale),
			unsafe.Pointer(&originalContext),
			unsafe.Pointer(&extFactor),
			unsafe.Pointer(&attentionFactor),
			unsafe.Pointer(&betaFast),
			unsafe.Pointer(&betaSlow),
			unsafe.Pointer(&count),
		}
		function := functions.ropeNeoX
		if node.Op == tensor.OpRoPENormal {
			function = functions.ropeNormal
		}
		err = launch1D(state, function, count, args)
		runtime.KeepAlive(input)
		runtime.KeepAlive(positions)
		runtime.KeepAlive(frequencyFactors)
		runtime.KeepAlive(output)
		runtime.KeepAlive(width)
		runtime.KeepAlive(heads)
		runtime.KeepAlive(tokens)
		runtime.KeepAlive(rotary)
		runtime.KeepAlive(frequencyBase)
		runtime.KeepAlive(frequencyScale)
		runtime.KeepAlive(originalContext)
		runtime.KeepAlive(extFactor)
		runtime.KeepAlive(attentionFactor)
		runtime.KeepAlive(betaFast)
		runtime.KeepAlive(betaSlow)
		runtime.KeepAlive(count)
		return err
	case tensor.OpRoPEMulti:
		attributes, ok := node.Attrs.(tensor.RoPEMultiAttributes)
		if !ok {
			return errors.New("invalid rope_multi attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		width, err := uint32Checked(node.Shape.Dims[0], "RoPE multi width")
		if err != nil {
			return err
		}
		heads, err := uint32Checked(node.Shape.Dims[1], "RoPE multi heads")
		if err != nil {
			return err
		}
		tokens, err := uint32Checked(node.Shape.Dims[2], "RoPE multi tokens")
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		positions, ok := attributePointers[node]
		if !ok {
			return errors.New("RoPE multi position storage is unavailable")
		}
		rotary := attributes.RotaryDimensions
		frequencyBase := attributes.FrequencyBase
		var sections [4]uint32
		for index, section := range attributes.Sections {
			if section < 0 {
				return errors.New("RoPE multi section count is negative")
			}
			sections[index] = uint32(section)
		}
		args := []unsafe.Pointer{
			unsafe.Pointer(&input),
			unsafe.Pointer(&positions),
			unsafe.Pointer(&output),
			unsafe.Pointer(&width),
			unsafe.Pointer(&heads),
			unsafe.Pointer(&tokens),
			unsafe.Pointer(&rotary),
			unsafe.Pointer(&frequencyBase),
			unsafe.Pointer(&sections[0]),
			unsafe.Pointer(&sections[1]),
			unsafe.Pointer(&sections[2]),
			unsafe.Pointer(&sections[3]),
			unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.ropeMulti, count, args)
		runtime.KeepAlive(input)
		runtime.KeepAlive(positions)
		runtime.KeepAlive(output)
		runtime.KeepAlive(width)
		runtime.KeepAlive(heads)
		runtime.KeepAlive(tokens)
		runtime.KeepAlive(rotary)
		runtime.KeepAlive(frequencyBase)
		runtime.KeepAlive(sections)
		runtime.KeepAlive(count)
		return err
	case tensor.OpReshape:
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		args := []unsafe.Pointer{
			unsafe.Pointer(&input),
			unsafe.Pointer(&output),
			unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.copy, count, args)
		runtime.KeepAlive(input)
		runtime.KeepAlive(output)
		runtime.KeepAlive(count)
		return err
	case tensor.OpAttention:
		attributes, ok := node.Attrs.(tensor.AttentionAttributes)
		if !ok {
			return errors.New("invalid attention attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		queryNode := node.Inputs[0]
		keyNode := node.Inputs[1]
		valueNode := node.Inputs[2]
		keyWidth, err := uint32Checked(queryNode.Shape.Dims[0], "attention key width")
		if err != nil {
			return err
		}
		valueWidth, err := uint32Checked(valueNode.Shape.Dims[0], "attention value width")
		if err != nil {
			return err
		}
		queryHeads, err := uint32Checked(queryNode.Shape.Dims[1], "attention query heads")
		if err != nil {
			return err
		}
		keyValueHeads, err := uint32Checked(keyNode.Shape.Dims[1], "attention KV heads")
		if err != nil {
			return err
		}
		queryTokens, err := uint32Checked(queryNode.Shape.Dims[2], "attention query tokens")
		if err != nil {
			return err
		}
		keyValueTokens, err := uint32Checked(keyNode.Shape.Dims[2], "attention KV tokens")
		if err != nil {
			return err
		}
		query := pointers[queryNode]
		key := pointers[keyNode]
		value := pointers[valueNode]
		var relativeBias driver.DevicePtr
		if len(node.Inputs) == 4 {
			relativeBias = pointers[node.Inputs[3]]
		}
		relativeBuckets := attributes.RelativeBuckets
		scale := attributes.Scale
		softcap := attributes.Softcap
		maxALiBiBias := attributes.MaxALiBiBias
		var causal uint32
		if attributes.Causal {
			causal = 1
		}
		queryStart := attributes.QueryStart
		window := attributes.Window
		args := []unsafe.Pointer{
			unsafe.Pointer(&query),
			unsafe.Pointer(&key),
			unsafe.Pointer(&value),
			unsafe.Pointer(&relativeBias),
			unsafe.Pointer(&output),
			unsafe.Pointer(&keyWidth),
			unsafe.Pointer(&valueWidth),
			unsafe.Pointer(&queryHeads),
			unsafe.Pointer(&keyValueHeads),
			unsafe.Pointer(&queryTokens),
			unsafe.Pointer(&keyValueTokens),
			unsafe.Pointer(&scale),
			unsafe.Pointer(&softcap),
			unsafe.Pointer(&maxALiBiBias),
			unsafe.Pointer(&causal),
			unsafe.Pointer(&queryStart),
			unsafe.Pointer(&window),
			unsafe.Pointer(&relativeBuckets),
			unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.attention, count, args)
		runtime.KeepAlive(query)
		runtime.KeepAlive(key)
		runtime.KeepAlive(value)
		runtime.KeepAlive(relativeBias)
		runtime.KeepAlive(output)
		runtime.KeepAlive(keyWidth)
		runtime.KeepAlive(valueWidth)
		runtime.KeepAlive(queryHeads)
		runtime.KeepAlive(keyValueHeads)
		runtime.KeepAlive(queryTokens)
		runtime.KeepAlive(keyValueTokens)
		runtime.KeepAlive(scale)
		runtime.KeepAlive(softcap)
		runtime.KeepAlive(maxALiBiBias)
		runtime.KeepAlive(causal)
		runtime.KeepAlive(queryStart)
		runtime.KeepAlive(window)
		runtime.KeepAlive(relativeBuckets)
		runtime.KeepAlive(count)
		return err
	case tensor.OpConcat:
		attributes, ok := node.Attrs.(tensor.ConcatAttributes)
		if !ok || (attributes.Axis != 0 && attributes.Axis != 2) {
			return errors.New("invalid concat attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		leftCount, err := elementCount32(node.Inputs[0].Shape)
		if err != nil {
			return err
		}
		left := pointers[node.Inputs[0]]
		right := pointers[node.Inputs[1]]
		leftWidth, err := uint32Checked(node.Inputs[0].Shape.Dims[0], "concat left width")
		if err != nil {
			return err
		}
		rightWidth, err := uint32Checked(node.Inputs[1].Shape.Dims[0], "concat right width")
		if err != nil {
			return err
		}
		axis := attributes.Axis
		args := []unsafe.Pointer{
			unsafe.Pointer(&left),
			unsafe.Pointer(&right),
			unsafe.Pointer(&output),
			unsafe.Pointer(&leftCount),
			unsafe.Pointer(&leftWidth),
			unsafe.Pointer(&rightWidth),
			unsafe.Pointer(&axis),
			unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.concat, count, args)
		runtime.KeepAlive(left)
		runtime.KeepAlive(right)
		runtime.KeepAlive(output)
		runtime.KeepAlive(leftCount)
		runtime.KeepAlive(leftWidth)
		runtime.KeepAlive(rightWidth)
		runtime.KeepAlive(axis)
		runtime.KeepAlive(count)
		return err
	default:
		return fmt.Errorf("unsupported CUDA operation %s", node.Op)
	}
}

func nativeQuantizedType(value dtype.Type) bool {
	switch value {
	case dtype.Q4_0, dtype.Q4_1, dtype.Q5_0, dtype.Q5_1,
		dtype.Q8_0, dtype.Q8_1, dtype.Q2K, dtype.Q3K, dtype.Q4K, dtype.Q5K, dtype.Q6K, dtype.Q8K,
		dtype.IQ2XXS, dtype.IQ2XS, dtype.IQ2S, dtype.IQ3XXS, dtype.IQ3S, dtype.IQ1S, dtype.IQ1M,
		dtype.IQ4NL, dtype.IQ4XS, dtype.MXFP4, dtype.NVFP4,
		dtype.Q1_0, dtype.Q2_0, dtype.TQ1_0, dtype.TQ2_0:
		return true
	default:
		return false
	}
}

func launch1D(
	state *device.State,
	function driver.Function,
	count uint32,
	arguments []unsafe.Pointer,
) error {
	if count == 0 {
		return nil
	}
	const threads = uint32(256)
	blocks := (count + threads - 1) / threads
	return state.Driver.LaunchKernel(
		function,
		driver.Dim3{X: blocks, Y: 1, Z: 1},
		driver.Dim3{X: threads, Y: 1, Z: 1},
		0,
		state.Stream,
		arguments,
	)
}

func elementCount32(shape tensor.Shape) (uint32, error) {
	elements, err := shape.Elements()
	if err != nil {
		return 0, err
	}
	return uint32Checked(elements, "tensor element count")
}

func rowDimensions32(shape tensor.Shape) (uint32, uint32, error) {
	width, err := uint32Checked(shape.Dims[0], "tensor row width")
	if err != nil {
		return 0, 0, err
	}
	elements, err := shape.Elements()
	if err != nil {
		return 0, 0, err
	}
	return width, uint32(elements / uint64(width)), nil
}

func uint32Checked(value uint64, name string) (uint32, error) {
	if value > math.MaxUint32 {
		return 0, fmt.Errorf("%s %d exceeds uint32", name, value)
	}
	return uint32(value), nil
}

func shapeDimensions32(shape tensor.Shape) ([tensor.MaxDimensions]uint32, error) {
	var result [tensor.MaxDimensions]uint32
	for axis := range tensor.MaxDimensions {
		value, err := uint32Checked(shape.Dims[axis], fmt.Sprintf("tensor dimension %d", axis))
		if err != nil {
			return result, err
		}
		result[axis] = value
	}
	return result, nil
}

func float32Bytes(values []float32) []byte {
	if len(values) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*int(unsafe.Sizeof(values[0])))
}

func uint32Bytes(values []uint32) []byte {
	if len(values) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*int(unsafe.Sizeof(values[0])))
}

func maxInt() int {
	return int(^uint(0) >> 1)
}
