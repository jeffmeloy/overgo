// CUDA denoiser: resident weights, branch K/V, compiled graphs.
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

// DenoiserCUDASession: one program's resident execution state.
type DenoiserCUDASession struct {
	Program *DenoiserProgram

	worker *device.Worker
	cuda   *executor.Executor

	weightAllocs device.AllocationSet
	WeightBytes  uint64

	contextCompiled *executor.CompiledGraph
	stepCompiled    *executor.CompiledGraph
	contextInputs   *executor.DeviceInputs
	stepInputs      *executor.DeviceInputs
	stepCrossSlots  []crossInputSlots

	headBuffer  *executor.DeviceBuffer
	headTargets *executor.RetainedTargets

	branches []*sessionBranchContext

	ctx context.Context
}

// sessionBranchContext: retained branch K/V and step feeds.
type sessionBranchContext struct {
	retained *executor.RetainedOutputs
	inputs   *executor.DeviceInputs
}

type crossInputSlots struct {
	key   executor.InputSlot
	value executor.InputSlot
}

// NewDenoiserCUDASession: compile, upload once, bind retained head.
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
		Program:      program,
		worker:       worker,
		cuda:         cuda,
		weightAllocs: device.NewAllocationSet(worker),
		ctx:          context.Background(),
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
	session.contextInputs = session.contextCompiled.NewDeviceInputs()
	session.stepInputs = session.stepCompiled.NewDeviceInputs()
	if err = session.uploadWeights(); err != nil {
		return session, err
	}
	session.stepCrossSlots = make([]crossInputSlots, len(program.stepCrossKeys))
	for layer := range program.stepCrossKeys {
		key, keyOK := session.stepCompiled.InputSlot(program.stepCrossKeys[layer])
		value, valueOK := session.stepCompiled.InputSlot(program.stepCrossValues[layer])
		if !keyOK || !valueOK {
			return session, fmt.Errorf("denoiser session cross input layer %d is not compiled", layer)
		}
		session.stepCrossSlots[layer] = crossInputSlots{key: key, value: value}
	}
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
	program.weights = nil
	program.contextWeightInputs = nil
	program.stepWeightInputs = nil
	return session, nil
}

func contextGraphOutputs(program *DenoiserProgram) []*tensor.Tensor {
	outputs := make([]*tensor.Tensor, 0, 2*len(program.contextKeys))
	outputs = append(outputs, program.contextKeys...)
	outputs = append(outputs, program.contextValues...)
	return outputs
}

// uploadWeights: unique names; declared storage type.
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
		pointer, err := s.weightAllocs.Upload(s.ctx, payload)
		if err != nil {
			return fmt.Errorf("denoiser session upload %s: %w", item.name, err)
		}
		s.WeightBytes += uint64(len(payload))
		for _, binding := range [...]struct {
			compiled *executor.CompiledGraph
			inputs   *executor.DeviceInputs
			node     *tensor.Tensor
		}{
			{s.contextCompiled, s.contextInputs, s.Program.contextWeightInputs[item.name]},
			{s.stepCompiled, s.stepInputs, s.Program.stepWeightInputs[item.name]},
		} {
			if binding.node == nil {
				continue
			}
			slot, ok := binding.compiled.InputSlot(binding.node)
			if !ok {
				return fmt.Errorf("denoiser session weight %s is not compiled", item.name)
			}
			binding.inputs.Pointers[slot] = pointer
		}
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

// ProjectBranchContext: run once; retain per-block K/V.
func (s *DenoiserCUDASession) ProjectBranchContext(context []float32) (any, error) {
	value, err := feedValue(s.Program.contextInput, context, "context")
	if err != nil {
		return nil, err
	}
	hostFeeds := map[*tensor.Tensor]reference.Value{s.Program.contextInput: value}
	retained, err := s.cuda.ExecuteRetainedCompiled(s.ctx, s.contextCompiled, hostFeeds, s.contextInputs, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("denoiser session context projection: %w", err)
	}
	branch := &sessionBranchContext{
		retained: retained,
		inputs:   s.stepCompiled.NewDeviceInputs(),
	}
	copy(branch.inputs.Pointers, s.stepInputs.Pointers)
	for layer := range s.Program.stepCrossKeys {
		key, keyOK := retained.Value(s.Program.contextKeys[layer])
		val, valueOK := retained.Value(s.Program.contextValues[layer])
		if !keyOK || !valueOK {
			return nil, errors.Join(fmt.Errorf("denoiser session context layer %d is not retained", layer), retained.Release(s.ctx))
		}
		slots := s.stepCrossSlots[layer]
		branch.inputs.Pointers[slots.key] = key.Pointer
		branch.inputs.Pointers[slots.value] = val.Pointer
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
	retained, err := s.cuda.ExecuteRetainedCompiled(
		s.ctx, s.stepCompiled, hostFeeds, branch.inputs, s.headTargets, nil,
	)
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

// ReleaseRequestResources: drop prompt projections; retain model and graphs.
func (s *DenoiserCUDASession) ReleaseRequestResources() error {
	var errs []error
	for _, branch := range s.branches {
		if branch.retained != nil {
			errs = append(errs, branch.retained.Release(s.ctx))
		}
	}
	s.branches = nil
	return errors.Join(errs...)
}

// ReleaseDenoiseResources: release branch, head, and weight storage.
func (s *DenoiserCUDASession) ReleaseDenoiseResources() error {
	var errs []error
	errs = append(errs, s.ReleaseRequestResources())
	if s.headBuffer != nil {
		errs = append(errs, s.headBuffer.Release(s.ctx))
		s.headBuffer = nil
	}
	errs = append(errs, s.weightAllocs.Close(s.ctx))
	s.contextInputs = nil
	s.stepInputs = nil
	s.stepCrossSlots = nil
	return errors.Join(errs...)
}

// Close: release resources, executor, worker.
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
