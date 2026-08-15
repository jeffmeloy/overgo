//go:build windows

package latentvideo

import (
	"context"
	"errors"
	"fmt"

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

// ReferenceEditDenoiserCUDASession owns weights, text K/V, and one bounded
// chunk of per-layer self-attention K/V.
type ReferenceEditDenoiserCUDASession struct {
	worker *device.Worker
	cuda   *executor.Executor
	allocs device.AllocationSet

	cold, warm             *DenoiserProgram
	contextGraph           *executor.CompiledGraph
	coldGraph, warmGraph   *executor.CompiledGraph
	contextInputs          *executor.DeviceInputs
	coldInputs, warmInputs *executor.DeviceInputs
	contextRetained        *executor.RetainedOutputs
	historyRetained        *executor.RetainedOutputs
	historyProgram         *DenoiserProgram
	ctx                    context.Context
	stats                  ReferenceEditDenoiserStats
}

// NewReferenceEditDenoiserCUDASession compiles cold and one-chunk-history
// graphs over a checkpoint block prefix.
func NewReferenceEditDenoiserCUDASession(
	ctx context.Context,
	checkpoint ReferenceEditCheckpoint,
	geometry LatentGeometry,
	layers, historyStartFrame, ordinal int,
	textContext []float32,
) (session *ReferenceEditDenoiserCUDASession, err error) {
	if ctx == nil {
		ctx = context.Background()
	}
	config := checkpoint.Config
	if layers <= 0 || layers > config.NumLayers || geometry.Channels != config.InDim || historyStartFrame <= 0 {
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
	warm, err := CompileDenoiserProgramHistory(config, weights, geometry, precision, DenoiserHistory{
		Tokens: geometry.Seq, StartFrame: historyStartFrame,
	})
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
		cold: cold, warm: warm, ctx: ctx,
		stats: ReferenceEditDenoiserStats{Layers: layers, HistoryTokens: geometry.Seq},
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, session.Close())
			session = nil
		}
	}()
	session.contextGraph, err = executor.Compile(contextGraphOutputs(cold)...)
	if err == nil {
		session.coldGraph, err = executor.Compile(stepHistoryOutputs(cold)...)
	}
	if err == nil {
		session.warmGraph, err = executor.Compile(stepHistoryOutputs(warm)...)
	}
	if err != nil {
		return session, err
	}
	for _, graph := range []*executor.CompiledGraph{session.contextGraph, session.coldGraph, session.warmGraph} {
		if err = cuda.PrepareCompiled(ctx, graph); err != nil {
			return session, err
		}
	}
	session.contextInputs = session.contextGraph.NewDeviceInputs()
	session.coldInputs = session.coldGraph.NewDeviceInputs()
	session.warmInputs = session.warmGraph.NewDeviceInputs()
	if err = session.uploadWeights(weights); err != nil {
		return session, err
	}
	if err = session.projectContext(textContext); err != nil {
		return session, err
	}
	cold.weights, warm.weights = nil, nil
	return session, nil
}

func stepHistoryOutputs(program *DenoiserProgram) []*tensor.Tensor {
	outputs := []*tensor.Tensor{program.Head}
	outputs = append(outputs, program.currentSelfKeys...)
	outputs = append(outputs, program.currentSelfVals...)
	return outputs
}

func (s *ReferenceEditDenoiserCUDASession) uploadWeights(weights *DenoiserWeights) error {
	type binding struct {
		graph  *executor.CompiledGraph
		inputs *executor.DeviceInputs
		node   *tensor.Tensor
	}
	graphs := []struct {
		program *DenoiserProgram
		graph   *executor.CompiledGraph
		inputs  *executor.DeviceInputs
		context bool
	}{
		{s.cold, s.contextGraph, s.contextInputs, true},
		{s.cold, s.coldGraph, s.coldInputs, false},
		{s.warm, s.warmGraph, s.warmInputs, false},
	}
	for name, values := range weights.values {
		var binds []binding
		var shape *tensor.Tensor
		for _, item := range graphs {
			node := item.program.stepWeightInputs[name]
			if item.context {
				node = item.program.contextWeightInputs[name]
			}
			if node == nil {
				continue
			}
			if shape != nil && (shape.Type != node.Type || !shape.Shape.Equal(node.Shape)) {
				return fmt.Errorf("reference edit denoiser weight %s disagrees across graphs", name)
			}
			shape = node
			binds = append(binds, binding{item.graph, item.inputs, node})
		}
		if len(binds) == 0 {
			continue
		}
		payload, err := encodeWeightPayload(values, shape.Type)
		if err != nil {
			return err
		}
		pointer, err := s.allocs.Upload(s.ctx, payload)
		if err != nil {
			return err
		}
		s.stats.WeightBytes += uint64(len(payload))
		for _, bind := range binds {
			slot, ok := bind.graph.InputSlot(bind.node)
			if !ok {
				return fmt.Errorf("reference edit denoiser weight %s is not compiled", name)
			}
			bind.inputs.Pointers[slot] = pointer
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
		s.ctx, s.contextGraph,
		map[*tensor.Tensor]reference.Value{s.cold.contextInput: value},
		s.contextInputs, nil, nil,
	)
	if err != nil {
		return err
	}
	for _, target := range []struct {
		program *DenoiserProgram
		graph   *executor.CompiledGraph
		inputs  *executor.DeviceInputs
	}{{s.cold, s.coldGraph, s.coldInputs}, {s.warm, s.warmGraph, s.warmInputs}} {
		for layer := range s.cold.contextKeys {
			key, keyOK := s.contextRetained.Value(s.cold.contextKeys[layer])
			value, valueOK := s.contextRetained.Value(s.cold.contextValues[layer])
			if !keyOK || !valueOK {
				return fmt.Errorf("reference edit denoiser context layer %d is not retained", layer)
			}
			keySlot, keyOK := target.graph.InputSlot(target.program.stepCrossKeys[layer])
			valueSlot, valueOK := target.graph.InputSlot(target.program.stepCrossValues[layer])
			if !keyOK || !valueOK {
				return fmt.Errorf("reference edit denoiser cross layer %d is not compiled", layer)
			}
			target.inputs.Pointers[keySlot] = key.Pointer
			target.inputs.Pointers[valueSlot] = value.Pointer
		}
	}
	s.stats.ContextProjections++
	return nil
}

// RunChunk executes one chunk and replaces the bounded self-attention tail.
func (s *ReferenceEditDenoiserCUDASession) RunChunk(patchTokens, blockE, headE []float32) ([]float32, error) {
	if s == nil || s.cuda == nil {
		return nil, errors.New("reference edit denoiser: closed")
	}
	program, graph, inputs := s.cold, s.coldGraph, s.coldInputs
	if s.historyRetained != nil {
		program, graph, inputs = s.warm, s.warmGraph, s.warmInputs
		for layer := range program.stepHistoryKeys {
			key, keyOK := s.historyRetained.Value(s.historyProgram.currentSelfKeys[layer])
			value, valueOK := s.historyRetained.Value(s.historyProgram.currentSelfVals[layer])
			if !keyOK || !valueOK {
				return nil, fmt.Errorf("reference edit denoiser history layer %d is not retained", layer)
			}
			keySlot, keyOK := graph.InputSlot(program.stepHistoryKeys[layer])
			valueSlot, valueOK := graph.InputSlot(program.stepHistoryVals[layer])
			if !keyOK || !valueOK {
				return nil, fmt.Errorf("reference edit denoiser history layer %d is not compiled", layer)
			}
			inputs.Pointers[keySlot] = key.Pointer
			inputs.Pointers[valueSlot] = value.Pointer
		}
	}
	feeds := make(map[*tensor.Tensor]reference.Value, 3)
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
	next, err := s.cuda.ExecuteRetainedCompiled(s.ctx, graph, feeds, inputs, nil, nil)
	if err != nil {
		return nil, err
	}
	head, err := next.CopyToHost(s.ctx, program.Head)
	if err != nil {
		return nil, errors.Join(err, next.Release(s.ctx))
	}
	prior := s.historyRetained
	s.historyRetained, s.historyProgram = next, program
	if prior != nil {
		if err := prior.Release(s.ctx); err != nil {
			return nil, err
		}
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

func (s *ReferenceEditDenoiserCUDASession) Close() error {
	if s == nil {
		return nil
	}
	var errs []error
	if s.historyRetained != nil {
		errs = append(errs, s.historyRetained.Release(context.WithoutCancel(s.ctx)))
		s.historyRetained = nil
	}
	if s.contextRetained != nil {
		errs = append(errs, s.contextRetained.Release(context.WithoutCancel(s.ctx)))
		s.contextRetained = nil
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
