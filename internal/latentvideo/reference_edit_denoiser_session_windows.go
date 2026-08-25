//go:build windows

package latentvideo

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

// ReferenceEditDenoiserStats reports retained one-chunk execution.
type ReferenceEditDenoiserStats struct {
	Layers, Runs, ContextProjections int
	HistoryTokens                    int
	WeightBytes                      uint64
}

// ReferenceEditDenoiserCUDASession owns weights, text K/V, and retained
// per-layer self-attention K/V.
type ReferenceEditDenoiserCUDASession struct {
	worker *device.Worker
	cuda   *executor.Executor
	allocs device.AllocationSet

	cold                   *DenoiserProgram
	warm                   map[int]*DenoiserProgram
	contextExecution       *executor.IndexedGraph
	coldExecution          *executor.IndexedGraph
	warmExecutions         map[int]*executor.IndexedGraph
	targets                map[int]*executor.RetainedTargets
	headBuffer             *executor.DeviceBuffer
	cacheKeys, cacheValues []*executor.DeviceBuffer
	contextRetained        *executor.RetainedOutputs
	historyProgram         *DenoiserProgram
	historyFrames          int
	ctx                    context.Context
	stats                  ReferenceEditDenoiserStats
}

// NewReferenceEditDenoiserCUDASession compiles each bounded prefix shape.
func NewReferenceEditDenoiserCUDASession(
	ctx context.Context,
	checkpoint ReferenceEditCheckpoint,
	geometry LatentGeometry,
	layers, framesPerChunk, localAttentionFrames, totalFrames, ordinal int,
	textContext []float32,
) (session *ReferenceEditDenoiserCUDASession, err error) {
	if ctx == nil {
		ctx = context.Background()
	}
	config := checkpoint.Config
	if !checked.Equal(geometry.Channels, config.InDim) {
		return nil, fmt.Errorf("reference edit denoiser: invalid geometry")
	}
	historyBound, boundOK := checked.AddInt(localAttentionFrames, framesPerChunk)
	_, localAligned := checked.DivExactInt(localAttentionFrames, framesPerChunk)
	_, totalAligned := checked.DivExactInt(totalFrames, framesPerChunk)
	if !checked.PositiveInts(layers, framesPerChunk, localAttentionFrames, totalFrames) || !checked.AtMostInt(layers, config.NumLayers) ||
		!checked.AtLeastInt(localAttentionFrames, framesPerChunk) || !localAligned ||
		!checked.GreaterInt(totalFrames, framesPerChunk) || !boundOK || !checked.AtMostInt(totalFrames, historyBound) || !totalAligned {
		return nil, fmt.Errorf("reference edit denoiser: invalid layers/geometry/history")
	}
	weights, err := checkpoint.LoadDenoiserWeights(layers)
	if err != nil {
		return nil, err
	}
	config.NumLayers = layers
	precision := DenoiserPrecision{MatmulWeights: dtype.BF16, RoundAttentionStorage: true}
	cold, err := CompileDenoiserProgramPrecision(config, weights, geometry, precision)
	if err != nil {
		return nil, err
	}
	worker, err := device.New(ordinal)
	if err != nil {
		return nil, err
	}
	cuda, err := executor.NewWithWorker(worker)
	if err != nil {
		return nil, errors.Join(err, worker.Close())
	}
	session = &ReferenceEditDenoiserCUDASession{
		worker: worker, cuda: cuda, allocs: device.NewAllocationSet(worker),
		cold: cold, warm: make(map[int]*DenoiserProgram), warmExecutions: make(map[int]*executor.IndexedGraph),
		targets: make(map[int]*executor.RetainedTargets), ctx: ctx,
		stats: ReferenceEditDenoiserStats{Layers: layers, HistoryTokens: localAttentionFrames * geometry.Seq / framesPerChunk},
	}
	for startFrame := framesPerChunk; startFrame < totalFrames; startFrame += framesPerChunk {
		session.warm[startFrame], err = CompileDenoiserProgramHistory(config, weights, geometry, precision, DenoiserHistory{
			Tokens: startFrame * geometry.Seq / framesPerChunk, StartFrame: startFrame,
		})
		if err != nil {
			return session, err
		}
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, session.Close())
			session = nil
		}
	}()
	session.contextExecution, err = executor.CompileIndexed(contextGraphOutputs(cold)...)
	if err == nil {
		session.coldExecution, err = executor.CompileExternalIndexed(stepHistoryOutputs(cold)...)
	}
	if err == nil {
		for startFrame, program := range session.warm {
			session.warmExecutions[startFrame], err = executor.CompileExternalIndexed(stepHistoryOutputs(program)...)
			if err != nil {
				break
			}
		}
	}
	if err != nil {
		return session, err
	}
	executions := []*executor.IndexedGraph{session.contextExecution, session.coldExecution}
	for _, execution := range session.warmExecutions {
		executions = append(executions, execution)
	}
	for _, execution := range executions {
		if err = cuda.PrepareCompiled(ctx, execution.Graph); err != nil {
			return session, err
		}
	}
	if err = session.uploadWeights(weights); err != nil {
		return session, err
	}
	if err = session.prepareHistoryCache(totalFrames); err != nil {
		return session, err
	}
	if err = session.projectContext(textContext); err != nil {
		return session, err
	}
	cold.weights = nil
	for _, program := range session.warm {
		program.weights = nil
	}
	return session, nil
}

func (s *ReferenceEditDenoiserCUDASession) prepareHistoryCache(totalFrames int) error {
	programs := map[int]*DenoiserProgram{tensor.FirstOffset: s.cold}
	executions := map[int]*executor.IndexedGraph{tensor.FirstOffset: s.coldExecution}
	for startFrame, program := range s.warm {
		programs[startFrame] = program
		executions[startFrame] = s.warmExecutions[startFrame]
	}
	tokensPerFrame := s.cold.Geometry.Seq / s.cold.Geometry.LatentFrames
	cacheShape := tensor.MustShape(
		uint64(s.cold.Config.Dim/s.cold.Config.NumHeads), uint64(s.cold.Config.NumHeads),
		uint64(totalFrames*tokensPerFrame),
	)
	cacheBytes, err := cacheShape.Bytes(dtype.F32)
	if err != nil {
		return err
	}
	headBytes, err := s.cold.Head.Shape.Bytes(dtype.F32)
	if err != nil {
		return err
	}
	s.headBuffer, err = s.cuda.AllocateDeviceBuffer(s.ctx, headBytes)
	if err != nil {
		return err
	}
	for range s.stats.Layers {
		key, err := s.cuda.AllocateDeviceBuffer(s.ctx, cacheBytes)
		if err != nil {
			return err
		}
		value, err := s.cuda.AllocateDeviceBuffer(s.ctx, cacheBytes)
		if err != nil {
			_ = key.Release(s.ctx)
			return err
		}
		s.cacheKeys = append(s.cacheKeys, key)
		s.cacheValues = append(s.cacheValues, value)
	}
	for startFrame, program := range programs {
		execution := executions[startFrame]
		target := execution.Graph.NewRetainedTargets()
		head, err := s.headBuffer.Value(program.Head.Shape)
		if err != nil {
			return err
		}
		if err := target.Set(program.Head, head); err != nil {
			return err
		}
		for layer := range program.currentSelfKeys {
			key, err := s.cacheKeys[layer].Value(program.currentSelfKeys[layer].Shape)
			if err != nil {
				return err
			}
			value, err := s.cacheValues[layer].Value(program.currentSelfVals[layer].Shape)
			if err != nil {
				return err
			}
			if err := target.Set(program.currentSelfKeys[layer], key); err != nil {
				return err
			}
			if err := target.Set(program.currentSelfVals[layer], value); err != nil {
				return err
			}
			if checked.Equal(startFrame, tensor.FirstOffset) {
				continue
			}
			keySlot, keyOK := execution.Graph.InputSlot(program.stepHistoryKeys[layer])
			valueSlot, valueOK := execution.Graph.InputSlot(program.stepHistoryVals[layer])
			if !keyOK || !valueOK {
				return fmt.Errorf("reference edit denoiser history layer %d is not compiled", layer)
			}
			execution.Inputs.Pointers[keySlot] = key.Pointer
			execution.Inputs.Pointers[valueSlot] = value.Pointer
		}
		s.targets[startFrame] = target
	}
	return nil
}

func stepHistoryOutputs(program *DenoiserProgram) []*tensor.Tensor {
	outputs := []*tensor.Tensor{program.Head}
	outputs = append(outputs, program.currentSelfKeys...)
	outputs = append(outputs, program.currentSelfVals...)
	return outputs
}

func (s *ReferenceEditDenoiserCUDASession) uploadWeights(weights *DenoiserWeights) error {
	type binding struct {
		execution *executor.IndexedGraph
		node      *tensor.Tensor
	}
	graphs := []struct {
		program   *DenoiserProgram
		execution *executor.IndexedGraph
		context   bool
	}{
		{s.cold, s.contextExecution, true},
		{s.cold, s.coldExecution, false},
	}
	for startFrame, program := range s.warm {
		graphs = append(graphs, struct {
			program   *DenoiserProgram
			execution *executor.IndexedGraph
			context   bool
		}{program, s.warmExecutions[startFrame], false})
	}
	for name, values := range weights.values {
		var binds []binding
		var prototype *tensor.Tensor
		for _, item := range graphs {
			node := item.program.stepWeightInputs.Node(name)
			if item.context {
				node = item.program.contextWeightInputs.Node(name)
			}
			if node == nil {
				continue
			}
			if prototype != nil && !tensor.Compatible(prototype, node) {
				return fmt.Errorf("reference edit denoiser weight %s disagrees across graphs", name)
			}
			prototype = node
			binds = append(binds, binding{item.execution, node})
		}
		if len(binds) == 0 {
			continue
		}
		payload, err := encodeWeightPayload(values, prototype.Type)
		if err != nil {
			return err
		}
		pointer, err := s.allocs.Upload(s.ctx, payload)
		if err != nil {
			return err
		}
		s.stats.WeightBytes += uint64(len(payload))
		for _, bind := range binds {
			slot, ok := bind.execution.Graph.InputSlot(bind.node)
			if !ok {
				return fmt.Errorf("reference edit denoiser weight %s is not compiled", name)
			}
			bind.execution.Inputs.Pointers[slot] = pointer
		}
	}
	return nil
}

func (s *ReferenceEditDenoiserCUDASession) projectContext(textContext []float32) error {
	value, err := feedValue(s.cold.contextInput, textContext, "reference edit context")
	if err != nil {
		return err
	}
	s.contextRetained, err = s.cuda.ExecuteRetainedCompiled(
		s.ctx, s.contextExecution.Graph,
		map[*tensor.Tensor]reference.Value{s.cold.contextInput: value},
		s.contextExecution.Inputs, nil, nil,
	)
	if err != nil {
		return err
	}
	targets := []struct {
		program   *DenoiserProgram
		execution *executor.IndexedGraph
	}{{s.cold, s.coldExecution}}
	for startFrame, program := range s.warm {
		targets = append(targets, struct {
			program   *DenoiserProgram
			execution *executor.IndexedGraph
		}{program, s.warmExecutions[startFrame]})
	}
	for _, target := range targets {
		for layer := range s.cold.contextKeys {
			key, keyOK := s.contextRetained.Value(s.cold.contextKeys[layer])
			value, valueOK := s.contextRetained.Value(s.cold.contextValues[layer])
			if !keyOK || !valueOK {
				return fmt.Errorf("reference edit denoiser context layer %d is not retained", layer)
			}
			keySlot, keyOK := target.execution.Graph.InputSlot(target.program.stepCrossKeys[layer])
			valueSlot, valueOK := target.execution.Graph.InputSlot(target.program.stepCrossValues[layer])
			if !keyOK || !valueOK {
				return fmt.Errorf("reference edit denoiser cross layer %d is not compiled", layer)
			}
			target.execution.Inputs.Pointers[keySlot] = key.Pointer
			target.execution.Inputs.Pointers[valueSlot] = value.Pointer
		}
	}
	s.stats.ContextProjections++
	return nil
}

// RunChunk executes one absolute-position chunk. commitHistory advances K/V.
func (s *ReferenceEditDenoiserCUDASession) RunChunk(patchTokens, blockE, headE []float32, startFrame int, commitHistory bool) ([]float32, error) {
	if s == nil || s.cuda == nil {
		return nil, errors.New("reference edit denoiser: closed")
	}
	program, execution := s.cold, s.coldExecution
	if checked.Nonzero(startFrame) {
		var ok bool
		program, ok = s.warm[startFrame]
		if !ok {
			return nil, fmt.Errorf("reference edit denoiser: start frame %d exceeds compiled history", startFrame)
		}
		execution = s.warmExecutions[startFrame]
	}
	if !checked.Equal(startFrame, s.historyFrames) {
		return nil, fmt.Errorf("reference edit denoiser: start frame %d does not continue %d", startFrame, s.historyFrames)
	}
	feeds := make(map[*tensor.Tensor]reference.Value, tensor.TripleExtent)
	for _, feed := range []struct {
		node *tensor.Tensor
		data []float32
		name string
	}{{program.stepPatch, patchTokens, "patch"}, {program.stepBlockE, blockE, "block conditioning"}, {program.stepHeadE, headE, "head conditioning"}} {
		value, err := feedValue(feed.node, feed.data, feed.name)
		if err != nil {
			return nil, err
		}
		feeds[feed.node] = value
	}
	next, err := s.cuda.ExecuteRetainedCompiled(
		s.ctx, execution.Graph, feeds, execution.Inputs, s.targets[startFrame], nil,
	)
	if err != nil {
		return nil, err
	}
	head, err := next.CopyToHost(s.ctx, program.Head)
	if err != nil {
		return nil, errors.Join(err, next.Release(s.ctx))
	}
	if commitHistory {
		s.historyFrames += program.Geometry.LatentFrames
		s.historyProgram = program
	}
	if err := next.Release(s.ctx); err != nil {
		return nil, err
	}
	s.stats.Runs++
	return head.Data, nil
}

func (s *ReferenceEditDenoiserCUDASession) Stats() ReferenceEditDenoiserStats {
	if s == nil {
		return ReferenceEditDenoiserStats{}
	}
	return s.stats
}

func (s *ReferenceEditDenoiserCUDASession) MemoryStats() (driver.MemoryStats, error) {
	if s == nil || s.worker == nil {
		return driver.MemoryStats{}, errors.New("reference edit denoiser: closed")
	}
	return s.worker.MemoryStats(s.ctx)
}

func (s *ReferenceEditDenoiserCUDASession) ExecutionStats() (driver.ExecutionStats, error) {
	if s == nil || s.worker == nil {
		return driver.ExecutionStats{}, errors.New("reference edit denoiser: closed")
	}
	return s.worker.ExecutionStats(s.ctx)
}

// ResetHistory starts a new request while retaining weights and text K/V.
func (s *ReferenceEditDenoiserCUDASession) ResetHistory() error {
	if s == nil || s.cuda == nil {
		return errors.New("reference edit denoiser: closed")
	}
	s.historyFrames, s.historyProgram = tensor.FirstOffset, nil
	return nil
}

func (s *ReferenceEditDenoiserCUDASession) Close() error {
	if s == nil {
		return nil
	}
	var errs []error
	if s.contextRetained != nil {
		errs = append(errs, s.contextRetained.Release(context.WithoutCancel(s.ctx)))
		s.contextRetained = nil
	}
	errs = append(errs, executor.ReleaseDeviceBuffers(context.WithoutCancel(s.ctx), append(s.cacheKeys, s.cacheValues...)...))
	s.cacheKeys, s.cacheValues = nil, nil
	if s.headBuffer != nil {
		errs = append(errs, s.headBuffer.Release(context.WithoutCancel(s.ctx)))
		s.headBuffer = nil
	}
	errs = append(errs, s.allocs.Close(context.WithoutCancel(s.ctx)))
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
