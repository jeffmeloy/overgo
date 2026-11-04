package graphruntime

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/checked"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

type residentBinding struct {
	source string
	name   string
	type_  dtype.Type
	shape  tensor.Shape
}

type residentAllocation struct {
	key     residentBinding
	pointer driver.DevicePtr
	bytes   uint64
	refs    int
	// chunk is the allocation the input is packed in.
	chunk *residentChunk
}

// residentChunk is one card allocation a program's bound inputs are packed in. A chunk is sized to what the card
// charges the input it was made for, so it costs what that input alone would (residentCharge), and the inputs the
// program binds after it fill the rest. A small input the card packs at no loss takes a chunk of its own size. It is
// freed once no input in it is held.
type residentChunk struct {
	pointer     driver.DevicePtr
	bytes, used uint64
	live        int
}

// The card's charge for an allocation, measured as free memory's fall over 256 allocations of each size
// (resident-bytes/measured-granule): up to residentPacked it packs allocations together at no loss; above that, up to
// residentMidPage, it charges residentMidPage (704 KB took 1 MiB); above that, whole residentPages (1.5 MiB took 2,
// 3 MiB took 4, 5 MiB took 6).
const (
	residentPacked  = 512 << 10
	residentMidPage = 1 << 20
	residentPage    = 2 << 20
)

// residentCharge is what the card charges an allocation of bytes, as measured; ok is false on overflow.
func residentCharge(bytes uint64) (uint64, bool) {
	switch {
	case bytes <= residentPacked:
		return bytes, true
	case bytes <= residentMidPage:
		return residentMidPage, true
	}
	return checked.Align(bytes, residentPage)
}

// residentAlignment is where each packed input begins, the alignment a card allocation itself gives.
const residentAlignment = 256

// drop lets go of one input in the chunk and frees the chunk once none is held; freed says it did.
func (c *residentChunk) drop(state *device.State) (freed bool, err error) {
	if c.live--; c.live > 0 {
		return false, nil
	}
	return true, state.Driver.MemFree(c.pointer)
}

// ResidentSession owns one device context and shared immutable inputs.
type ResidentSession struct {
	worker      *device.Worker
	executor    *executor.Executor
	byBinding   map[residentBinding]int
	allocations []residentAllocation
	staticBytes uint64
}

// ResidentProgram owns compiled topology and indexed input slots.
type ResidentProgram struct {
	*executor.IndexedGraph
	dynamic []executor.InputSlot
	weights []int
	// chunk is the allocation the program packs its next new input in.
	chunk *residentChunk
}

// place finds room for a new input of bytes: in the program's chunk when its rest holds them, else in a new chunk of
// what the card charges the input alone. Only a chunk with room past its input becomes the one the next inputs fill.
func (p *ResidentProgram) place(state *device.State, bytes uint64) (driver.DevicePtr, *residentChunk, error) {
	aligned, fits := checked.Align(bytes, residentAlignment)
	charge, charged := residentCharge(aligned)
	if !fits || !charged {
		return 0, nil, errors.New("resident input size overflows")
	}
	chunk := p.chunk
	if chunk == nil || chunk.used+aligned > chunk.bytes {
		pointer, err := state.Driver.MemAlloc(charge)
		if err != nil {
			return 0, nil, err
		}
		chunk = &residentChunk{pointer: pointer, bytes: charge}
		if charge > aligned {
			p.chunk = chunk
		}
	}
	placed := chunk.pointer + driver.DevicePtr(chunk.used)
	chunk.used, chunk.live = chunk.used+aligned, chunk.live+1
	return placed, chunk, nil
}

// ResidentStats reports session-owned device storage and execution.
type ResidentStats struct {
	StaticBytes uint64
	Device      driver.MemoryStats
	Execution   driver.ExecutionStats
}

// NewResidentSession opens one shared resident device session.
func NewResidentSession(ordinal int) (*ResidentSession, error) {
	worker, err := device.New(ordinal)
	if err != nil {
		return nil, err
	}
	runtime, err := executor.NewWithWorker(worker)
	if err != nil {
		_ = worker.Close()
		return nil, err
	}
	return &ResidentSession{
		worker: worker, executor: runtime, byBinding: make(map[residentBinding]int),
	}, nil
}

// Compile fixes one graph and its dynamic input order.
func (s *ResidentSession) Compile(
	ctx context.Context,
	label string,
	dynamic []*tensor.Tensor,
	outputs ...*tensor.Tensor,
) (*ResidentProgram, error) {
	if s == nil || s.worker == nil || s.executor == nil {
		return nil, errors.New("resident session is unavailable")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	program, err := compileResidentProgram(label, dynamic, outputs)
	if err != nil {
		return nil, err
	}
	if err := s.executor.PrepareCompiled(ctx, program.Graph); err != nil {
		return nil, fmt.Errorf("%s: prepare: %w", label, err)
	}
	return program, nil
}

func compileResidentProgram(label string, dynamic, outputs []*tensor.Tensor) (*ResidentProgram, error) {
	indexed, err := executor.CompileIndexed(outputs...)
	if err != nil {
		return nil, fmt.Errorf("%s: compile: %w", label, err)
	}
	program := &ResidentProgram{
		IndexedGraph: indexed,
		dynamic:      make([]executor.InputSlot, len(dynamic)),
	}
	for index, input := range dynamic {
		slot, ok := indexed.Graph.InputSlot(input)
		if !ok {
			return nil, fmt.Errorf("%s: dynamic input is not compiled", label)
		}
		program.dynamic[index] = slot
	}
	return program, nil
}

// Bind uploads or reuses one immutable program input.
func (s *ResidentSession) Bind(
	ctx context.Context,
	program *ResidentProgram,
	source string,
	name string,
	node *tensor.Tensor,
	payload func() ([]byte, error),
) error {
	if s == nil || s.byBinding == nil || program == nil || program.Graph == nil || node == nil {
		return errors.New("resident binding is unavailable")
	}
	slot, ok := program.Graph.InputSlot(node)
	if !ok || checked.Nonzero(program.Inputs.Pointers[slot]) {
		return errors.New("resident static input is invalid")
	}
	bytes, err := node.Shape.Bytes(node.Type)
	if err != nil {
		return err
	}
	key := residentBinding{source: source, name: name, type_: node.Type, shape: node.Shape}
	allocationIndex, found := s.byBinding[key]
	var pointer driver.DevicePtr
	if found {
		pointer = s.allocations[allocationIndex].pointer
	} else {
		data, loadErr := payload()
		if loadErr != nil {
			return loadErr
		}
		if uint64(len(data)) != bytes {
			return fmt.Errorf("resident payload = %d bytes, want %d", len(data), bytes)
		}
		var chunk *residentChunk
		if err := s.worker.Do(ctx, func(state *device.State) error {
			placed, held, placeErr := program.place(state, bytes)
			if placeErr != nil {
				return placeErr
			}
			if copyErr := state.Driver.MemcpyHtoD(placed, data); copyErr != nil {
				freed, freeErr := held.drop(state)
				if freed && held == program.chunk {
					program.chunk = nil
				}
				return errors.Join(copyErr, freeErr)
			}
			pointer, chunk = placed, held
			return nil
		}); err != nil {
			return err
		}
		allocationIndex = len(s.allocations)
		s.byBinding[key] = allocationIndex
		s.allocations = append(s.allocations, residentAllocation{key: key, pointer: pointer, bytes: bytes, chunk: chunk})
		s.staticBytes += bytes
	}
	program.Inputs.Pointers[slot] = pointer
	program.weights = append(program.weights, allocationIndex)
	s.allocations[allocationIndex].refs++
	return nil
}

// BindF32 binds one immutable F32 input.
func (s *ResidentSession) BindF32(
	ctx context.Context,
	program *ResidentProgram,
	source string,
	node *tensor.Tensor,
	values []float32,
) error {
	if node == nil || node.Type != dtype.F32 {
		return errors.New("resident static input must be F32")
	}
	return s.Bind(ctx, program, source, node.Name, node, func() ([]byte, error) {
		return driver.Bytes(values), nil
	})
}

// Execute runs with indexed static inputs and host dynamic inputs.
func (s *ResidentSession) Execute(
	ctx context.Context,
	program *ResidentProgram,
	host map[*tensor.Tensor]reference.Value,
) (map[*tensor.Tensor]reference.Value, error) {
	if s == nil || s.executor == nil || program == nil || program.Graph == nil {
		return nil, errors.New("resident execution is unavailable")
	}
	return s.executor.ExecuteCompiled(ctx, program.Graph, host, program.Inputs)
}

// ExecuteDevice runs with temporary indexed device inputs.
func (s *ResidentSession) ExecuteDevice(
	ctx context.Context,
	program *ResidentProgram,
	host map[*tensor.Tensor]reference.Value,
	pointers []driver.DevicePtr,
) (map[*tensor.Tensor]reference.Value, error) {
	if err := program.bindDynamic(pointers); err != nil {
		return nil, err
	}
	defer program.clearDynamic(len(pointers))
	return s.Execute(ctx, program, host)
}

// Retain runs with temporary device inputs and retains outputs.
func (s *ResidentSession) Retain(
	ctx context.Context,
	program *ResidentProgram,
	host map[*tensor.Tensor]reference.Value,
	pointers []driver.DevicePtr,
) (*executor.RetainedOutputs, error) {
	if s == nil || s.executor == nil || program == nil || program.Graph == nil {
		return nil, errors.New("resident execution is unavailable")
	}
	if err := program.bindDynamic(pointers); err != nil {
		return nil, err
	}
	defer program.clearDynamic(len(pointers))
	return s.executor.ExecuteRetainedCompiled(ctx, program.Graph, host, program.Inputs, nil, nil)
}

// Release drops program references and frees unshared inputs.
func (s *ResidentSession) Release(ctx context.Context, programs ...*ResidentProgram) error {
	if s == nil || s.worker == nil {
		return errors.New("resident session is unavailable")
	}
	return s.worker.Do(ctx, func(state *device.State) error {
		var errs []error
		for _, program := range programs {
			if program == nil {
				continue
			}
			for _, allocationIndex := range program.weights {
				allocation := &s.allocations[allocationIndex]
				allocation.refs--
				if allocation.refs > 0 {
					continue
				}
				if checked.Nonzero(allocation.pointer) {
					// Graphs captured over this program's weights must
					// not replay after the range is freed.
					if s.executor != nil {
						errs = append(errs, s.executor.DropGraphExecs(state))
					}
					_, err := allocation.chunk.drop(state)
					errs = append(errs, err)
				}
				if allocation.bytes <= s.staticBytes {
					s.staticBytes -= allocation.bytes
				}
				delete(s.byBinding, allocation.key)
				*allocation = residentAllocation{}
			}
			program.Graph = nil
			program.Inputs = nil
			program.dynamic = nil
			program.weights = nil
		}
		// A released record is empty and no program names it. Trailing ones leave the
		// list, so a session that binds and releases each request does not grow with
		// them, and a live record keeps its index.
		for count := len(s.allocations); count > 0 && s.allocations[count-1] == (residentAllocation{}); count-- {
			s.allocations = s.allocations[:count-1]
		}
		return errors.Join(errs...)
	})
}

// ReleaseStageMemory frees the execution memory a finished stage leaves for its
// next run, for a session whose next stage places other weights. Bound inputs,
// compiled programs and retained outputs stay.
func (s *ResidentSession) ReleaseStageMemory(ctx context.Context) error {
	if s == nil || s.executor == nil {
		return errors.New("resident session is unavailable")
	}
	return s.executor.ReleaseStageMemory(ctx)
}

// Seal drops deduplication metadata after all programs bind.
func (s *ResidentSession) Seal() {
	if s == nil {
		return
	}
	s.byBinding = nil
	for index := range s.allocations {
		s.allocations[index].key = residentBinding{}
	}
}

// StaticBytes returns unique immutable device storage.
func (s *ResidentSession) StaticBytes() uint64 {
	if s == nil {
		return uint64(tensor.FirstOffset)
	}
	return s.staticBytes
}

// Do runs one operation in the session device context.
func (s *ResidentSession) Do(ctx context.Context, operation func(*device.State) error) error {
	if s == nil || s.worker == nil || operation == nil {
		return errors.New("resident device operation is unavailable")
	}
	return s.worker.Do(ctx, operation)
}

func (p *ResidentProgram) bindDynamic(pointers []driver.DevicePtr) error {
	if p == nil || p.Inputs == nil || len(pointers) != len(p.dynamic) {
		return errors.New("resident dynamic inputs do not match compiled slots")
	}
	for index, pointer := range pointers {
		slot := p.dynamic[index]
		if !checked.Nonzero(pointer) || checked.Nonzero(p.Inputs.Pointers[slot]) {
			p.clearDynamic(index)
			return errors.New("resident dynamic input is invalid")
		}
		p.Inputs.Pointers[slot] = pointer
	}
	return nil
}

func (p *ResidentProgram) clearDynamic(count int) {
	for index := range min(count, len(p.dynamic)) {
		p.Inputs.Pointers[p.dynamic[index]] = driver.DevicePtr(tensor.FirstOffset)
	}
}

// Stats returns session allocation and execution evidence.
func (s *ResidentSession) Stats(ctx context.Context) (ResidentStats, error) {
	if s == nil || s.worker == nil {
		return ResidentStats{}, errors.New("resident session is unavailable")
	}
	deviceStats, err := s.worker.MemoryStats(ctx)
	if err != nil {
		return ResidentStats{}, err
	}
	executionStats, err := s.worker.ExecutionStats(ctx)
	return ResidentStats{StaticBytes: s.staticBytes, Device: deviceStats, Execution: executionStats}, err
}

// Close releases all resident resources.
func (s *ResidentSession) Close(ctx context.Context) error {
	if s == nil {
		return nil
	}
	var errs []error
	if s.executor != nil {
		errs = append(errs, s.executor.Close())
		s.executor = nil
	}
	if s.worker != nil && len(s.allocations) != 0 {
		errs = append(errs, s.worker.Do(ctx, func(state *device.State) error {
			var freeErrors []error
			freed := map[*residentChunk]bool{}
			for index := range s.allocations {
				if chunk := s.allocations[index].chunk; chunk != nil && !freed[chunk] {
					freed[chunk] = true
					freeErrors = append(freeErrors, state.Driver.MemFree(chunk.pointer))
				}
				s.allocations[index] = residentAllocation{}
			}
			return errors.Join(freeErrors...)
		}))
	}
	if s.worker != nil {
		errs = append(errs, s.worker.Close())
		s.worker = nil
	}
	s.allocations = nil
	s.byBinding = nil
	s.staticBytes = uint64(tensor.FirstOffset)
	return errors.Join(errs...)
}
