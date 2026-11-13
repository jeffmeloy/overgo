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
	"overgo/internal/sampling"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// referenceEditDenoiserStats reports retained one-chunk execution.
type referenceEditDenoiserStats struct {
	Layers, Runs, ContextProjections int
	HistoryTokens                    int
	WeightBytes                      uint64
}

// referenceEditDenoiserCUDASession owns weights, text K/V, and retained
// per-layer self-attention K/V.
type referenceEditDenoiserCUDASession struct {
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
	// negativeRetained is the negative context's keys and values, bound to every step graph, where the session runs
	// NAG.
	negativeRetained *executor.RetainedOutputs
	historyProgram   *DenoiserProgram
	historyFrames    int
	ctx              context.Context
	stats            referenceEditDenoiserStats
}

// newReferenceEditDenoiserCUDASession compiles each bounded prefix shape, each
// block guided by attention where it is NAG's. The session it gives has no
// text context yet: projectContext gives it one.
func newReferenceEditDenoiserCUDASession(
	ctx context.Context,
	checkpoint referenceEditCheckpoint,
	geometry LatentGeometry,
	layers, framesPerChunk, localAttentionFrames, totalFrames, ordinal int,
	attention sampling.NormalizedAttention,
) (session *referenceEditDenoiserCUDASession, err error) {
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
	weights, err := checkpoint.OpenDenoiserWeights(layers)
	if err != nil {
		return nil, err
	}
	// The session decodes each tensor from the open checkpoint as it uploads it.
	defer func() { err = errors.Join(err, weights.Close()) }()
	config.NumLayers = layers
	precision := DenoiserPrecision{MatmulWeights: dtype.BF16, RoundAttentionStorage: true}
	cold, err := compileDenoiserProgram(config, weights, geometry, precision, denoiserHistory{}, attention)
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
	session = &referenceEditDenoiserCUDASession{
		worker: worker, cuda: cuda, allocs: device.NewAllocationSet(worker),
		cold: cold, warm: make(map[int]*DenoiserProgram), warmExecutions: make(map[int]*executor.IndexedGraph),
		targets: make(map[int]*executor.RetainedTargets), ctx: ctx,
		stats: referenceEditDenoiserStats{Layers: layers, HistoryTokens: localAttentionFrames * geometry.Seq / framesPerChunk},
	}
	for startFrame := framesPerChunk; startFrame < totalFrames; startFrame += framesPerChunk {
		session.warm[startFrame], err = compileDenoiserProgramHistory(config, weights, geometry, precision, denoiserHistory{
			Tokens: startFrame * geometry.Seq / framesPerChunk, StartFrame: startFrame,
		}, attention)
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
	cold.weights = nil
	for _, program := range session.warm {
		program.weights = nil
	}
	return session, nil
}

func (s *referenceEditDenoiserCUDASession) prepareHistoryCache(totalFrames int) error {
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

func (s *referenceEditDenoiserCUDASession) uploadWeights(weights *DenoiserWeights) error {
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
	// An opened set decodes each tensor as it is taken, so the host holds one at a time.
	return weights.each(func(name string, values []float32) error {
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
			return nil
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
		return nil
	})
}

// projectContexts gives the session the text context its chunks attend to:
// each layer's keys and values of it, kept on the device and bound to every
// step graph, and where the session runs NAG, the negative context's beside
// them, which a session that runs none is not given. A session that had
// contexts lets those go once the new ones are bound, so a resident session
// edits under one instruction after another.
func (s *referenceEditDenoiserCUDASession) projectContexts(textContext, negativeContext []float32) (err error) {
	guided := len(s.cold.stepNegativeKeys) > 0
	if guided == (negativeContext == nil) {
		return fmt.Errorf("reference edit denoiser: a session that runs NAG takes a negative context and one that runs none takes none, and this one runs NAG %t", guided)
	}
	project := func(values []float32) (*executor.RetainedOutputs, error) {
		value, err := feedValue(s.cold.contextInput, values, "reference edit context")
		if err != nil {
			return nil, err
		}
		return s.cuda.ExecuteRetainedCompiled(
			s.ctx, s.contextExecution.Graph,
			map[*tensor.Tensor]reference.Value{s.cold.contextInput: value},
			s.contextExecution.Inputs, nil, nil,
		)
	}
	text, err := project(textContext)
	if err != nil {
		return err
	}
	var negative *executor.RetainedOutputs
	if guided {
		if negative, err = project(negativeContext); err != nil {
			return errors.Join(err, text.Release(context.WithoutCancel(s.ctx)))
		}
	}
	previous := []*executor.RetainedOutputs{s.contextRetained, s.negativeRetained}
	s.contextRetained, s.negativeRetained = text, negative
	defer func() {
		for _, held := range previous {
			if held != nil {
				err = errors.Join(err, held.Release(context.WithoutCancel(s.ctx)))
			}
		}
	}()
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
	bind := func(retained *executor.RetainedOutputs, keys, values func(*DenoiserProgram) []*tensor.Tensor) error {
		for _, target := range targets {
			for layer := range s.cold.contextKeys {
				key, keyOK := retained.Value(s.cold.contextKeys[layer])
				value, valueOK := retained.Value(s.cold.contextValues[layer])
				if !keyOK || !valueOK {
					return fmt.Errorf("reference edit denoiser context layer %d is not retained", layer)
				}
				keySlot, keyOK := target.execution.Graph.InputSlot(keys(target.program)[layer])
				valueSlot, valueOK := target.execution.Graph.InputSlot(values(target.program)[layer])
				if !keyOK || !valueOK {
					return fmt.Errorf("reference edit denoiser cross layer %d is not compiled", layer)
				}
				target.execution.Inputs.Pointers[keySlot] = key.Pointer
				target.execution.Inputs.Pointers[valueSlot] = value.Pointer
			}
		}
		return nil
	}
	if err := bind(text, func(p *DenoiserProgram) []*tensor.Tensor { return p.stepCrossKeys },
		func(p *DenoiserProgram) []*tensor.Tensor { return p.stepCrossValues }); err != nil {
		return err
	}
	if guided {
		if err := bind(negative, func(p *DenoiserProgram) []*tensor.Tensor { return p.stepNegativeKeys },
			func(p *DenoiserProgram) []*tensor.Tensor { return p.stepNegativeValues }); err != nil {
			return err
		}
	}
	s.stats.ContextProjections++
	return nil
}

// RunChunk executes one absolute-position chunk. commitHistory advances K/V.
func (s *referenceEditDenoiserCUDASession) RunChunk(ctx context.Context, patchTokens, blockE, headE []float32, startFrame int, commitHistory bool) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if s == nil || s.cuda == nil {
		return nil, errors.New("reference edit denoiser: closed")
	}
	if s.contextRetained == nil {
		return nil, errors.New("reference edit denoiser: no text context is projected")
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
		ctx, execution.Graph, feeds, execution.Inputs, s.targets[startFrame], nil,
	)
	if err != nil {
		return nil, err
	}
	head, err := next.CopyToHost(ctx, program.Head)
	if err != nil {
		return nil, errors.Join(err, next.Release(context.WithoutCancel(ctx)))
	}
	if commitHistory {
		s.historyFrames += program.Geometry.LatentFrames
		s.historyProgram = program
	}
	if err := next.Release(context.WithoutCancel(ctx)); err != nil {
		return nil, err
	}
	s.stats.Runs++
	return head.Data, nil
}

func (s *referenceEditDenoiserCUDASession) Stats() referenceEditDenoiserStats {
	if s == nil {
		return referenceEditDenoiserStats{}
	}
	return s.stats
}

func (s *referenceEditDenoiserCUDASession) MemoryStats() (driver.MemoryStats, error) {
	if s == nil || s.worker == nil {
		return driver.MemoryStats{}, errors.New("reference edit denoiser: closed")
	}
	return s.worker.MemoryStats(s.ctx)
}

func (s *referenceEditDenoiserCUDASession) ExecutionStats() (driver.ExecutionStats, error) {
	if s == nil || s.worker == nil {
		return driver.ExecutionStats{}, errors.New("reference edit denoiser: closed")
	}
	return s.worker.ExecutionStats(s.ctx)
}

// ResetHistory starts a new request while retaining weights and text K/V.
func (s *referenceEditDenoiserCUDASession) ResetHistory() error {
	if s == nil || s.cuda == nil {
		return errors.New("reference edit denoiser: closed")
	}
	s.historyFrames, s.historyProgram = tensor.FirstOffset, nil
	return nil
}

func (s *referenceEditDenoiserCUDASession) Close() error {
	if s == nil {
		return nil
	}
	var errs []error
	for _, held := range []*executor.RetainedOutputs{s.contextRetained, s.negativeRetained} {
		if held != nil {
			errs = append(errs, held.Release(context.WithoutCancel(s.ctx)))
		}
	}
	s.contextRetained, s.negativeRetained = nil, nil
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
