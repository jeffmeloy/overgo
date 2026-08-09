// Full-CUDA denoiser execution: one session owns device-resident weights,
// retained per-branch cross-attention K/V, and both compiled graphs. The
// step graph executes through the executor's retained path, so after the
// first capture every further step replays one instantiated CUDA graph per
// branch (per-step contents flow through the host-feed uploads and the
// stable retained head target; the launch trace stays byte-identical).
package latentvideo

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// DenoiserCUDASession: device residency plus compiled/retained execution
// state for one DenoiserProgram.
type DenoiserCUDASession struct {
	Program *DenoiserProgram

	worker *device.Worker
	cuda   *executor.Executor

	weightPointers map[string]driver.DevicePtr
	weightAllocs   []driver.DevicePtr
	WeightBytes    uint64

	contextCompiled    *executor.CompiledGraph
	stepCompiled       *executor.CompiledGraph
	contextWeightFeeds map[*tensor.Tensor]driver.DevicePtr
	stepWeightFeeds    map[*tensor.Tensor]driver.DevicePtr

	headBuffer  *executor.DeviceBuffer
	headTargets *executor.RetainedTargets

	branches []*sessionBranchContext

	ctx context.Context
}

// sessionBranchContext: one branch's retained cross-attention K/V plus the
// complete step-graph device feed map (weights + K/V pointers).
type sessionBranchContext struct {
	retained *executor.RetainedOutputs
	feeds    map[*tensor.Tensor]driver.DevicePtr
}

// NewDenoiserCUDASession: compiles both graphs, uploads every weight tensor
// once (BF16 rank-2 projections when the program was compiled BF16), and
// prepares the stable retained head target.
func NewDenoiserCUDASession(program *DenoiserProgram, ordinal int) (session *DenoiserCUDASession, err error) {
	if program == nil {
		return nil, errors.New("denoiser session: program is nil")
	}
	worker, err := device.New(ordinal)
	if err != nil {
		return nil, err
	}
	cuda, err := executor.NewWithWorker(worker)
	if err != nil {
		return nil, errors.Join(err, worker.Close())
	}
	session = &DenoiserCUDASession{
		Program:        program,
		worker:         worker,
		cuda:           cuda,
		weightPointers: make(map[string]driver.DevicePtr),
		ctx:            context.Background(),
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, session.Close())
			session = nil
		}
	}()
	session.contextCompiled, err = executor.Compile(contextGraphOutputs(program)...)
	if err != nil {
		return session, fmt.Errorf("denoiser session context graph: %w", err)
	}
	session.stepCompiled, err = executor.Compile(program.Head)
	if err != nil {
		return session, fmt.Errorf("denoiser session step graph: %w", err)
	}
	if err = session.uploadWeights(); err != nil {
		return session, err
	}
	session.contextWeightFeeds = session.bindWeightFeeds(program.contextWeightInputs)
	session.stepWeightFeeds = session.bindWeightFeeds(program.stepWeightInputs)
	headBytes, err := program.Head.Shape.Bytes(dtype.F32)
	if err != nil {
		return session, err
	}
	session.headBuffer, err = cuda.AllocateDeviceBuffer(session.ctx, headBytes)
	if err != nil {
		return session, err
	}
	headValue, err := session.headBuffer.Value(program.Head.Shape)
	if err != nil {
		return session, err
	}
	session.headTargets = session.stepCompiled.NewRetainedTargets()
	if err = session.headTargets.Set(program.Head, headValue); err != nil {
		return session, err
	}
	return session, nil
}

func contextGraphOutputs(program *DenoiserProgram) []*tensor.Tensor {
	outputs := make([]*tensor.Tensor, 0, 2*len(program.contextKeys))
	outputs = append(outputs, program.contextKeys...)
	outputs = append(outputs, program.contextValues...)
	return outputs
}

// uploadWeights: every unique weight name uploads exactly once, in the
// node's storage type (F32 raw bits; BF16 round-to-nearest-even).
func (s *DenoiserCUDASession) uploadWeights() error {
	type upload struct {
		name string
		node *tensor.Tensor
	}
	var uploads []upload
	seen := make(map[string]*tensor.Tensor)
	for _, inputs := range []map[string]*tensor.Tensor{s.Program.stepWeightInputs, s.Program.contextWeightInputs} {
		for name, node := range inputs {
			if prior, ok := seen[name]; ok {
				if prior.Type != node.Type || !prior.Shape.Equal(node.Shape) {
					return fmt.Errorf("denoiser session weight %s: graphs disagree on storage", name)
				}
				continue
			}
			seen[name] = node
			uploads = append(uploads, upload{name: name, node: node})
		}
	}
	for _, item := range uploads {
		data := s.Program.weights.tensor(item.name)
		elements, err := item.node.Shape.Elements()
		if err != nil || uint64(len(data)) != elements {
			return fmt.Errorf("denoiser session weight %s: have %d elements, need %d", item.name, len(data), elements)
		}
		payload, err := encodeWeightPayload(data, item.node.Type)
		if err != nil {
			return fmt.Errorf("denoiser session weight %s: %w", item.name, err)
		}
		var pointer driver.DevicePtr
		if err := s.worker.Do(s.ctx, func(state *device.State) error {
			var allocErr error
			pointer, allocErr = state.Driver.MemAlloc(uint64(len(payload)))
			if allocErr != nil {
				return allocErr
			}
			if copyErr := state.Driver.MemcpyHtoD(pointer, payload); copyErr != nil {
				freeErr := state.Driver.MemFree(pointer)
				pointer = 0
				return errors.Join(copyErr, freeErr)
			}
			return nil
		}); err != nil {
			return fmt.Errorf("denoiser session upload %s: %w", item.name, err)
		}
		s.weightAllocs = append(s.weightAllocs, pointer)
		s.weightPointers[item.name] = pointer
		s.WeightBytes += uint64(len(payload))
	}
	return nil
}

func encodeWeightPayload(data []float32, storage dtype.Type) ([]byte, error) {
	switch storage {
	case dtype.F32:
		return driver.Bytes(data), nil
	case dtype.BF16:
		payload := make([]byte, 2*len(data))
		for i, value := range data {
			binary.LittleEndian.PutUint16(payload[2*i:], dtype.Float32ToBF16(value))
		}
		return payload, nil
	default:
		return nil, fmt.Errorf("weight storage %s is unsupported", storage)
	}
}

func (s *DenoiserCUDASession) bindWeightFeeds(inputs map[string]*tensor.Tensor) map[*tensor.Tensor]driver.DevicePtr {
	feeds := make(map[*tensor.Tensor]driver.DevicePtr, len(inputs))
	for name, node := range inputs {
		feeds[node] = s.weightPointers[name]
	}
	return feeds
}

// ProjectBranchContext: runs the context graph once and retains every
// per-block K/V on device for the session lifetime.
func (s *DenoiserCUDASession) ProjectBranchContext(context []float32) (any, error) {
	value, err := feedValue(s.Program.contextInput, context, "context")
	if err != nil {
		return nil, err
	}
	hostFeeds := map[*tensor.Tensor]reference.Value{s.Program.contextInput: value}
	retained, err := s.cuda.ExecuteRetainedCompiledWithDeviceFeeds(s.ctx, s.contextCompiled, hostFeeds, s.contextWeightFeeds)
	if err != nil {
		return nil, fmt.Errorf("denoiser session context projection: %w", err)
	}
	branch := &sessionBranchContext{
		retained: retained,
		feeds:    make(map[*tensor.Tensor]driver.DevicePtr, len(s.stepWeightFeeds)+2*len(s.Program.stepCrossKeys)),
	}
	for node, pointer := range s.stepWeightFeeds {
		branch.feeds[node] = pointer
	}
	for layer := range s.Program.stepCrossKeys {
		key, keyOK := retained.Value(s.Program.contextKeys[layer])
		val, valueOK := retained.Value(s.Program.contextValues[layer])
		if !keyOK || !valueOK {
			return nil, errors.Join(fmt.Errorf("denoiser session context layer %d is not retained", layer), retained.Release(s.ctx))
		}
		branch.feeds[s.Program.stepCrossKeys[layer]] = key.Pointer
		branch.feeds[s.Program.stepCrossValues[layer]] = val.Pointer
	}
	s.branches = append(s.branches, branch)
	return branch, nil
}

// ForwardHead: one step-graph execution into the stable retained head
// target; returns the host head patches.
func (s *DenoiserCUDASession) ForwardHead(patchTokens, blockE, headE []float32, branchContext any) ([]float32, error) {
	branch, ok := branchContext.(*sessionBranchContext)
	if !ok || branch == nil {
		return nil, fmt.Errorf("denoiser session forward: branch context is %T", branchContext)
	}
	hostFeeds := make(map[*tensor.Tensor]reference.Value, 3)
	for _, feed := range []struct {
		node *tensor.Tensor
		data []float32
		what string
	}{
		{s.Program.stepPatch, patchTokens, "patch tokens"},
		{s.Program.stepBlockE, blockE, "block conditioning"},
		{s.Program.stepHeadE, headE, "head conditioning"},
	} {
		value, err := feedValue(feed.node, feed.data, feed.what)
		if err != nil {
			return nil, err
		}
		hostFeeds[feed.node] = value
	}
	retained, err := s.cuda.ExecuteRetainedCompiledWithTargets(s.ctx, s.stepCompiled, hostFeeds, branch.feeds, s.headTargets)
	if err != nil {
		return nil, fmt.Errorf("denoiser session step execution: %w", err)
	}
	value, err := retained.CopyToHost(s.ctx, s.Program.Head)
	if releaseErr := retained.Release(s.ctx); releaseErr != nil {
		err = errors.Join(err, releaseErr)
	}
	if err != nil {
		return nil, err
	}
	return value.Data, nil
}

// Denoise: the shared guided UniPC loop over this session.
func (s *DenoiserCUDASession) Denoise(ctx context.Context, request DenoiseRequest) (DenoiseResult, error) {
	if ctx != nil {
		s.ctx = ctx
	}
	return s.Program.DenoiseWithBackend(s, request)
}

// MemoryStats: runtime-owned device allocation accounting.
func (s *DenoiserCUDASession) MemoryStats() (driver.MemoryStats, error) {
	return s.worker.MemoryStats(s.ctx)
}

// ExecutionStats: launch/copy/graph counters for replay evidence.
func (s *DenoiserCUDASession) ExecutionStats() (driver.ExecutionStats, error) {
	return s.worker.ExecutionStats(s.ctx)
}

// ReleaseDenoiseResources: frees the retained branch contexts and the head
// target ahead of decode (pre-decode lifetime release); the session stays
// usable for telemetry until Close.
func (s *DenoiserCUDASession) ReleaseDenoiseResources() error {
	var errs []error
	for _, branch := range s.branches {
		if branch.retained != nil {
			errs = append(errs, branch.retained.Release(s.ctx))
		}
	}
	s.branches = nil
	if s.headBuffer != nil {
		errs = append(errs, s.headBuffer.Release(s.ctx))
		s.headBuffer = nil
	}
	for _, pointer := range s.weightAllocs {
		final := pointer
		errs = append(errs, s.worker.Do(s.ctx, func(state *device.State) error {
			return state.Driver.MemFree(final)
		}))
	}
	s.weightAllocs = nil
	s.weightPointers = map[string]driver.DevicePtr{}
	return errors.Join(errs...)
}

// Close: full teardown (denoise resources, executor pools, worker).
func (s *DenoiserCUDASession) Close() error {
	if s == nil {
		return nil
	}
	var errs []error
	errs = append(errs, s.ReleaseDenoiseResources())
	if s.cuda != nil {
		errs = append(errs, s.cuda.Close())
		s.cuda = nil
	}
	if s.worker != nil {
		errs = append(errs, s.worker.Close())
		s.worker = nil
	}
	return errors.Join(errs...)
}
