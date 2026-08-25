//go:build windows

package scratchmodel

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/devicemath"
	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/trainingprogram"
)

// ResidentTrainer owns one scratch model's device training lifecycle.
type ResidentTrainer struct {
	construction Construction
	worker       *device.Worker
	executor     *executor.Executor
	resident     *devicemath.ResidentOpsSession
	muon         *optimizer.ResidentMuonPlan
	weights      driver.DevicePtr
	gradients    driver.DevicePtr
	momentum     driver.DevicePtr
	config       optimizer.Config
	programs     []residentForwardProgram
	training     trainingprogram.Execution[residentTrainingState]
	evaluation   trainingprogram.Execution[residentTrainingState]
	lifecycle    ResidentLifecycle
	closed       bool
}

// ResidentLifecycle separates preparation from execution evidence.
type ResidentLifecycle struct {
	DriverPreparation   time.Duration
	ModelInitialization time.Duration
	ProgramPreparation  time.Duration
}

type residentTrainingState struct {
	trainer   *ResidentTrainer
	tokens    []int
	step      int
	positions int
	graph     ForwardGraph
	retained  *executor.RetainedOutputs
	loss      float64
}

func (s *residentTrainingState) release() {
	if s != nil && s.retained != nil {
		s.retained.Release(context.Background())
		s.retained = nil
	}
}

type residentForwardProgram struct {
	*executor.IndexedGraph
	graph     ForwardGraph
	hostFeeds map[*tensor.Tensor]reference.Value
}

func NewResidentTrainer(construction Construction, totalSteps int) (*ResidentTrainer, error) {
	if totalSteps <= 0 || construction.optimizer.Identity() == "" || len(construction.weights) == 0 {
		return nil, errors.New("scratch model: invalid resident trainer")
	}
	driverStarted := time.Now()
	worker, err := device.New(0)
	if err != nil {
		return nil, err
	}
	trainer := &ResidentTrainer{
		construction: construction, worker: worker,
		lifecycle: ResidentLifecycle{DriverPreparation: time.Since(driverStarted)},
		config: optimizer.Config{
			BaseLearningRate: construction.config.BaseLR,
			Momentum:         construction.config.MuonMomentum,
			Steps:            totalSteps,
			Schedule:         optimizer.ScheduleLinearDecay,
		},
	}
	fail := func(cause error) (*ResidentTrainer, error) {
		return nil, errors.Join(cause, trainer.Close())
	}
	trainer.executor, err = executor.NewWithWorker(worker)
	if err != nil {
		return fail(err)
	}
	residentStarted := time.Now()
	trainer.resident, err = devicemath.NewResidentOpsSession(worker)
	if err != nil {
		return fail(err)
	}
	residentPreparation := time.Since(residentStarted)
	modelStarted := time.Now()
	weightBytes := driver.Bytes(construction.weights)
	trainer.weights, err = trainer.resident.Upload(context.Background(), weightBytes)
	if err != nil {
		return fail(err)
	}
	trainer.gradients, err = trainer.resident.Allocate(context.Background(), uint64(len(weightBytes)))
	if err != nil {
		return fail(err)
	}
	trainer.momentum, err = trainer.resident.Allocate(context.Background(), uint64(len(weightBytes)))
	if err != nil {
		return fail(err)
	}
	for _, documents := range [][]string{construction.split.Train, construction.split.Validation, construction.split.Test} {
		for _, document := range documents {
			tokens, tokenErr := construction.Tokens(document)
			if tokenErr != nil {
				return fail(tokenErr)
			}
			if _, programErr := trainer.forwardProgram(tokens); programErr != nil {
				return fail(programErr)
			}
		}
	}
	trainer.lifecycle.ModelInitialization = time.Since(modelStarted)
	programStarted := time.Now()
	trainer.muon, err = optimizer.NewResidentMuonPlan(worker, construction.optimizer, trainer.config)
	if err != nil {
		return fail(err)
	}
	program, err := bindResidentProgram(trainer)
	if err != nil {
		return fail(err)
	}
	trainer.training, err = program.Select(trainingprogram.PhaseBatch, trainingprogram.PhaseForward, trainingprogram.PhaseBackward, trainingprogram.PhaseOptimize)
	if err != nil {
		return fail(err)
	}
	trainer.evaluation, err = program.Select(trainingprogram.PhaseBatch, trainingprogram.PhaseForward, trainingprogram.PhaseEvaluate)
	if err != nil {
		return fail(err)
	}
	for index := range trainer.programs {
		if err := trainer.executor.PrepareCompiled(context.Background(), trainer.programs[index].Graph); err != nil {
			return fail(err)
		}
	}
	trainer.lifecycle.ProgramPreparation = residentPreparation + time.Since(programStarted)
	return trainer, nil
}

func (t *ResidentTrainer) Lifecycle() (ResidentLifecycle, error) {
	if t == nil || t.closed {
		return ResidentLifecycle{}, errors.New("scratch model: resident trainer unavailable")
	}
	return t.lifecycle, nil
}

func (t *ResidentTrainer) Close() error {
	if t == nil || t.closed {
		return nil
	}
	t.closed = true
	var result error
	if t.muon != nil {
		result = errors.Join(result, t.muon.Close())
	}
	if t.executor != nil {
		result = errors.Join(result, t.executor.Close())
	}
	if t.resident != nil {
		result = errors.Join(result, t.resident.Close())
	}
	if t.worker != nil {
		result = errors.Join(result, t.worker.Close())
	}
	return result
}

func (t *ResidentTrainer) Step(tokens []int, step int) (float64, error) {
	if t == nil || t.closed || step <= 0 {
		return 0, errors.New("scratch model: resident trainer unavailable")
	}
	state := residentTrainingState{trainer: t, tokens: tokens, step: step}
	defer state.release()
	err := t.training.Run(&state)
	return state.loss, err
}

func (t *ResidentTrainer) Evaluate(tokens []int) (float64, error) {
	if t == nil || t.closed {
		return 0, errors.New("scratch model: resident trainer unavailable")
	}
	state := residentTrainingState{trainer: t, tokens: tokens}
	defer state.release()
	err := t.evaluation.Run(&state)
	return state.loss, err
}

func (t *ResidentTrainer) Snapshot() (weights, gradients, momentum []float32, err error) {
	if t == nil || t.closed {
		return nil, nil, nil, errors.New("scratch model: resident trainer unavailable")
	}
	weights = make([]float32, len(t.construction.weights))
	gradients = make([]float32, len(t.construction.weights))
	momentum = make([]float32, len(t.construction.weights))
	if err := devicemath.ReadResident(t.worker, t.weights, devicemath.ResidentSlice{Data: weights}); err != nil {
		return nil, nil, nil, err
	}
	if err := devicemath.ReadResident(t.worker, t.gradients, devicemath.ResidentSlice{Data: gradients}); err != nil {
		return nil, nil, nil, err
	}
	if err := devicemath.ReadResident(t.worker, t.momentum, devicemath.ResidentSlice{Data: momentum}); err != nil {
		return nil, nil, nil, err
	}
	return weights, gradients, momentum, nil
}

func (t *ResidentTrainer) ResetPeakMemory() error {
	if t == nil || t.closed {
		return errors.New("scratch model: resident trainer unavailable")
	}
	return t.worker.Do(context.Background(), func(state *device.State) error {
		state.Driver.ResetPeakBytes()
		return nil
	})
}

func (t *ResidentTrainer) MemoryStats() (driver.MemoryStats, error) {
	if t == nil || t.closed {
		return driver.MemoryStats{}, errors.New("scratch model: resident trainer unavailable")
	}
	return t.worker.MemoryStats(context.Background())
}

func (t *ResidentTrainer) validateTokens(tokens []int) (int, error) {
	if len(tokens) < 2 {
		return 0, errors.New("scratch model: forward tokens absent")
	}
	positions := min(t.construction.config.BlockSize, len(tokens)-1)
	for _, token := range tokens[:positions+1] {
		if token < 0 || token >= t.construction.config.VocabSize {
			return 0, errors.New("scratch model: forward token outside vocabulary")
		}
	}
	return positions, nil
}

func (t *ResidentTrainer) forward(tokens []int, positions int) (ForwardGraph, *executor.RetainedOutputs, error) {
	program, err := t.forwardProgram(tokens)
	if err != nil {
		return ForwardGraph{}, nil, err
	}
	if program.graph.positions != positions {
		return ForwardGraph{}, nil, errors.New("scratch model: compiled batch extent differs")
	}
	rows := make([]uint32, program.graph.positions)
	for index, token := range tokens[:program.graph.positions] {
		rows[index] = uint32(token)
	}
	attributes := program.Graph.NewRuntimeAttributes()
	if err := attributes.Set(program.graph.tokenRows, tensor.GetRowsAttributes{Rows: rows}); err != nil {
		return ForwardGraph{}, nil, err
	}
	retained, err := t.executor.ExecuteRetainedCompiled(
		context.Background(), program.Graph, program.hostFeeds, program.Inputs, nil, attributes,
	)
	return program.graph, retained, err
}

func bindResidentProgram(trainer *ResidentTrainer) (trainingprogram.Execution[residentTrainingState], error) {
	bindings := []trainingprogram.Binding[residentTrainingState]{
		{Operator: scratchOperatorBatch, Execute: func(state *residentTrainingState) error {
			positions, err := state.trainer.validateTokens(state.tokens)
			state.positions = positions
			return err
		}},
		{Operator: scratchOperatorForward, Execute: func(state *residentTrainingState) error {
			var err error
			state.graph, state.retained, err = state.trainer.forward(state.tokens, state.positions)
			return err
		}},
		{Operator: scratchOperatorLossVJP, Execute: func(state *residentTrainingState) error {
			var err error
			state.loss, err = state.trainer.backward(state.graph, state.retained, state.tokens)
			return err
		}},
		{Operator: scratchOperatorMuon, Execute: func(state *residentTrainingState) error {
			return state.trainer.muon.Step(state.trainer.weights, state.trainer.gradients, state.trainer.momentum, state.step)
		}},
		{Operator: scratchOperatorEvaluate, Execute: func(state *residentTrainingState) error {
			value, err := state.retained.CopyToHost(context.Background(), state.graph.Output)
			if err != nil {
				return err
			}
			gradient := make([]float32, len(value.Data))
			state.loss = hostmath.SoftmaxCrossEntropy(
				gradient, value.Data, state.tokens[1:state.graph.positions+1],
				state.graph.positions, state.trainer.construction.config.VocabSize,
			)
			return nil
		}},
	}
	return trainingprogram.Bind(trainer.construction.Program(), bindings)
}

func (t *ResidentTrainer) forwardProgram(tokens []int) (*residentForwardProgram, error) {
	positions := min(t.construction.config.BlockSize, len(tokens)-1)
	for index := range t.programs {
		if t.programs[index].graph.positions == positions {
			return &t.programs[index], nil
		}
	}
	graph, err := t.construction.CompileForwardGraph(tokens)
	if err != nil {
		return nil, err
	}
	indexed, err := executor.CompileIndexed(graph.cacheOutputs()...)
	if err != nil {
		return nil, err
	}
	hostFeeds, err := graph.residentInputs(indexed, t.construction, t.weights)
	if err != nil {
		return nil, err
	}
	t.programs = append(t.programs, residentForwardProgram{
		IndexedGraph: indexed, graph: graph, hostFeeds: hostFeeds,
	})
	return &t.programs[len(t.programs)-1], nil
}

func (g ForwardGraph) residentInputs(
	indexed *executor.IndexedGraph,
	c Construction,
	weightSlab driver.DevicePtr,
) (map[*tensor.Tensor]reference.Value, error) {
	if indexed == nil || weightSlab == 0 || g.Mask == nil || len(g.Parameters) != len(c.parameters) {
		return nil, errors.New("scratch model: resident graph or slab differs")
	}
	mask, err := reference.NewValue(g.Mask.Shape, g.mask)
	if err != nil {
		return nil, err
	}
	hostFeeds := map[*tensor.Tensor]reference.Value{g.Mask: mask}
	for name, input := range g.Parameters {
		binding, ok := c.bindings[name]
		if !ok {
			return nil, fmt.Errorf("scratch model: resident parameter %q absent", name)
		}
		slot, ok := indexed.Graph.InputSlot(input)
		if !ok {
			return nil, fmt.Errorf("scratch model: resident parameter %q is not compiled", name)
		}
		indexed.Inputs.Pointers[slot] = devicemath.ResidentPtr(weightSlab, binding.start)
	}
	return hostFeeds, nil
}

func (t *ResidentTrainer) backward(graph ForwardGraph, retained *executor.RetainedOutputs, tokens []int) (float64, error) {
	var loss float64
	err := t.resident.Run(func(ops *devicemath.ResidentOps) error {
		var err error
		loss, err = t.backwardWithOps(ops, graph, retained, tokens)
		return err
	})
	return loss, err
}

func scratchBackwardArenaElems(config Config, positions int) int {
	hidden := positions * config.Embedding
	base := positions*config.MLPWidth + 7*hidden
	head := 4*positions*config.HeadDim + 2*positions*positions + 1
	return max(hidden, base+max(head, 2*hidden))
}

func (t *ResidentTrainer) backwardWithOps(ops *devicemath.ResidentOps, graph ForwardGraph, retained *executor.RetainedOutputs, tokens []int) (float64, error) {
	c := t.construction
	positions, hidden, vocab := graph.positions, c.config.Embedding, c.config.VocabSize
	pointer := func(node *tensor.Tensor) (driver.DevicePtr, error) {
		value, ok := retained.Value(node)
		if !ok || value.Pointer == 0 {
			return 0, errors.New("scratch model: retained cache absent")
		}
		return value.Pointer, nil
	}
	weight := func(name string) (driver.DevicePtr, error) {
		binding, ok := c.bindings[name]
		if !ok {
			return 0, fmt.Errorf("scratch model: resident weight %q absent", name)
		}
		return devicemath.ResidentPtr(t.weights, binding.start), nil
	}
	gradient := func(name string) (driver.DevicePtr, error) {
		binding, ok := c.bindings[name]
		if !ok {
			return 0, fmt.Errorf("scratch model: resident gradient %q absent", name)
		}
		return devicemath.ResidentPtr(t.gradients, binding.start), nil
	}
	allocatePersistent := func(count int, initial []float32) (driver.DevicePtr, error) {
		if initial != nil {
			return 0, errors.New("scratch model: resident scratch initialization unsupported")
		}
		return ops.AllocF32(count)
	}
	if err := ops.Zero(t.gradients, len(c.weights)); err != nil {
		return 0, err
	}

	logits, err := pointer(graph.Output)
	if err != nil {
		return 0, err
	}
	targets := make([]uint32, positions)
	for position, token := range tokens[1 : positions+1] {
		targets[position] = uint32(token)
	}
	targetsPtr, err := ops.UploadU32(targets)
	if err != nil {
		return 0, err
	}
	lossesPtr, err := allocatePersistent(positions, nil)
	if err != nil {
		return 0, err
	}
	if err := ops.SoftmaxCrossEntropy(logits, targetsPtr, lossesPtr, positions, vocab); err != nil {
		return 0, err
	}
	finalHidden, err := pointer(graph.Hidden)
	if err != nil {
		return 0, err
	}
	dHidden, err := allocatePersistent(positions*hidden, nil)
	if err != nil {
		return 0, err
	}
	dHiddenNext, err := allocatePersistent(positions*hidden, nil)
	if err != nil {
		return 0, err
	}
	head, err := weight("lm_head")
	if err != nil {
		return 0, err
	}
	dHead, err := gradient("lm_head")
	if err != nil {
		return 0, err
	}
	if err := ops.LinearBackwardT(finalHidden, head, logits, dHidden, dHead, positions, hidden, vocab); err != nil {
		return 0, err
	}
	arena, err := ops.NewArena(scratchBackwardArenaElems(c.config, positions))
	if err != nil {
		return 0, err
	}

	for layer := len(graph.layers) - 1; layer >= 0; layer-- {
		layerMark := arena.Mark()
		cache := graph.layers[layer]
		prefix := fmt.Sprintf("l%d.", layer)
		attentionOutput, err := pointer(cache.attentionOutput)
		if err != nil {
			return 0, err
		}
		activation, err := pointer(cache.activation)
		if err != nil {
			return 0, err
		}
		preactivation, err := pointer(cache.preactivation)
		if err != nil {
			return 0, err
		}
		mlpNorm, err := pointer(cache.mlpNorm)
		if err != nil {
			return 0, err
		}
		dActivation, err := arena.AllocF32(positions * c.config.MLPWidth)
		if err != nil {
			return 0, err
		}
		w2, err := weight(prefix + "w2")
		if err != nil {
			return 0, err
		}
		dW2, err := gradient(prefix + "w2")
		if err != nil {
			return 0, err
		}
		if err := ops.LinearBackwardT(activation, w2, dHidden, dActivation, dW2, positions, c.config.MLPWidth, hidden); err != nil {
			return 0, err
		}
		if err := ops.ReLUBackward(dActivation, preactivation, dActivation, positions*c.config.MLPWidth); err != nil {
			return 0, err
		}
		dMLPNorm, err := arena.AllocF32(positions * hidden)
		if err != nil {
			return 0, err
		}
		w1, err := weight(prefix + "w1")
		if err != nil {
			return 0, err
		}
		dW1, err := gradient(prefix + "w1")
		if err != nil {
			return 0, err
		}
		if err := ops.LinearBackwardT(mlpNorm, w1, dActivation, dMLPNorm, dW1, positions, hidden, c.config.MLPWidth); err != nil {
			return 0, err
		}
		dMLPInput, err := arena.AllocF32(positions * hidden)
		if err != nil {
			return 0, err
		}
		if err := ops.MADNormBackward(dMLPNorm, attentionOutput, mlpNorm, dMLPInput, positions, hidden, c.config.Epsilon); err != nil {
			return 0, err
		}
		dAttentionOutput, err := arena.AllocF32(positions * hidden)
		if err != nil {
			return 0, err
		}
		if err := ops.Add(dHidden, dMLPInput, dAttentionOutput, positions*hidden); err != nil {
			return 0, err
		}
		attention, err := pointer(cache.attention)
		if err != nil {
			return 0, err
		}
		dAttention, err := arena.AllocF32(positions * hidden)
		if err != nil {
			return 0, err
		}
		wo, err := weight(prefix + "wo")
		if err != nil {
			return 0, err
		}
		dWO, err := gradient(prefix + "wo")
		if err != nil {
			return 0, err
		}
		if err := ops.LinearBackwardT(attention, wo, dAttentionOutput, dAttention, dWO, positions, hidden, hidden); err != nil {
			return 0, err
		}
		dInput := dHiddenNext
		if err := ops.StridedRowCopy(dAttentionOutput, dInput, positions, hidden, hidden, 0, hidden, 0); err != nil {
			return 0, err
		}
		dQKV, err := arena.AllocF32(positions * 3 * hidden)
		if err != nil {
			return 0, err
		}
		dBias, err := gradient("pos_bias")
		if err != nil {
			return 0, err
		}
		dLT, err := gradient("lt")
		if err != nil {
			return 0, err
		}
		for headIndex, headCache := range cache.heads {
			headMark := arena.Mark()
			q, err := pointer(headCache.query)
			if err != nil {
				return 0, err
			}
			k, err := pointer(headCache.key)
			if err != nil {
				return 0, err
			}
			v, err := pointer(headCache.value)
			if err != nil {
				return 0, err
			}
			probability, err := pointer(headCache.probability)
			if err != nil {
				return 0, err
			}
			dHeadAttention, err := arena.AllocF32(positions * c.config.HeadDim)
			if err != nil {
				return 0, err
			}
			if err := ops.StridedRowCopy(
				dAttention, dHeadAttention,
				positions, c.config.HeadDim, hidden, headIndex*c.config.HeadDim, c.config.HeadDim, 0,
			); err != nil {
				return 0, err
			}
			dQ, err := arena.AllocF32(positions * c.config.HeadDim)
			if err != nil {
				return 0, err
			}
			dK, err := arena.AllocF32(positions * c.config.HeadDim)
			if err != nil {
				return 0, err
			}
			dV, err := arena.AllocF32(positions * c.config.HeadDim)
			if err != nil {
				return 0, err
			}
			dScores, err := arena.AllocF32(positions * positions)
			if err != nil {
				return 0, err
			}
			dProbability, err := arena.AllocF32(positions * positions)
			if err != nil {
				return 0, err
			}
			scaledQuery, err := pointer(headCache.scaledQuery)
			if err != nil {
				return 0, err
			}
			temperatureScale, err := pointer(headCache.temperatureScale)
			if err != nil {
				return 0, err
			}
			if err := ops.AttentionCoreBackwardWithScratch(
				scaledQuery, k, v, probability, dHeadAttention,
				dQ, dK, dV, dScores, dProbability, positions, c.config.HeadDim, 1,
			); err != nil {
				return 0, err
			}
			if err := ops.ScaleByScalar(dQ, temperatureScale, dQ, positions*c.config.HeadDim); err != nil {
				return 0, err
			}
			inverseRoot := float32(1 / math.Sqrt(float64(c.config.HeadDim)))
			if err := ops.Scale(dQ, dQ, inverseRoot, positions*c.config.HeadDim); err != nil {
				return 0, err
			}
			for _, copySpec := range []struct {
				source driver.DevicePtr
				offset int
			}{{dQ, headIndex * c.config.HeadDim}, {dK, hidden + headIndex*c.config.HeadDim}, {dV, 2*hidden + headIndex*c.config.HeadDim}} {
				if err := ops.StridedRowCopy(
					copySpec.source, dQKV,
					positions, c.config.HeadDim, c.config.HeadDim, 0, 3*hidden, copySpec.offset,
				); err != nil {
					return 0, err
				}
			}
			dScale, err := arena.AllocF32(1)
			if err != nil {
				return 0, err
			}
			if err := ops.Zero(dScale, 1); err != nil {
				return 0, err
			}
			if err := ops.AttentionScoreAffineBackward(
				dScores, q, k, dBias, dScale,
				positions, c.config.HeadDim, c.config.BlockSize,
			); err != nil {
				return 0, err
			}
			if err := ops.ScaleByScalar(dScale, temperatureScale, dScale, 1); err != nil {
				return 0, err
			}
			if err := ops.Scale(
				dScale, devicemath.ResidentPtr(dLT, layer*c.config.HeadCount+headIndex), -inverseRoot, 1,
			); err != nil {
				return 0, err
			}
			if err := arena.Reset(headMark); err != nil {
				return 0, err
			}
		}
		qkvNorm, err := pointer(cache.qkvNorm)
		if err != nil {
			return 0, err
		}
		dQKVNorm, err := arena.AllocF32(positions * hidden)
		if err != nil {
			return 0, err
		}
		wqkv, err := weight(prefix + "wqkv")
		if err != nil {
			return 0, err
		}
		dWQKV, err := gradient(prefix + "wqkv")
		if err != nil {
			return 0, err
		}
		if err := ops.LinearBackwardT(qkvNorm, wqkv, dQKV, dQKVNorm, dWQKV, positions, hidden, 3*hidden); err != nil {
			return 0, err
		}
		input, err := pointer(cache.input)
		if err != nil {
			return 0, err
		}
		dQKVInput, err := arena.AllocF32(positions * hidden)
		if err != nil {
			return 0, err
		}
		if err := ops.MADNormBackward(dQKVNorm, input, qkvNorm, dQKVInput, positions, hidden, c.config.Epsilon); err != nil {
			return 0, err
		}
		if err := ops.Add(dInput, dQKVInput, dInput, positions*hidden); err != nil {
			return 0, err
		}
		dHidden, dHiddenNext = dInput, dHidden
		if err := arena.Reset(layerMark); err != nil {
			return 0, err
		}
	}

	embedding, err := pointer(graph.embedding)
	if err != nil {
		return 0, err
	}
	initialNormalized, err := pointer(graph.layers[0].input)
	if err != nil {
		return 0, err
	}
	dEmbedding, err := arena.AllocF32(positions * hidden)
	if err != nil {
		return 0, err
	}
	if err := ops.MADNormBackward(dHidden, embedding, initialNormalized, dEmbedding, positions, hidden, c.config.Epsilon); err != nil {
		return 0, err
	}
	tokenRows, positionRows := make([]uint32, positions), make([]uint32, positions)
	for position := range positions {
		tokenRows[position], positionRows[position] = uint32(tokens[position]), uint32(position)
	}
	tokenRowsPtr, err := ops.UploadU32(tokenRows)
	if err != nil {
		return 0, err
	}
	positionRowsPtr, err := ops.UploadU32(positionRows)
	if err != nil {
		return 0, err
	}
	dWTE, err := gradient("wte")
	if err != nil {
		return 0, err
	}
	dWPE, err := gradient("wpe")
	if err != nil {
		return 0, err
	}
	if err := ops.IndexedRowScatterAdd(dEmbedding, tokenRowsPtr, dWTE, positions, hidden); err != nil {
		return 0, err
	}
	if err := ops.IndexedRowScatterAdd(dEmbedding, positionRowsPtr, dWPE, positions, hidden); err != nil {
		return 0, err
	}
	losses := make([]float32, positions)
	if err := ops.DownloadF32(lossesPtr, losses); err != nil {
		return 0, err
	}
	var loss float64
	for _, value := range losses {
		loss += float64(value)
	}
	return loss, nil
}

func (c Construction) TrainResident(totalSteps int) (TrainingResult, error) {
	if totalSteps <= 0 || len(c.split.Train) == 0 || len(c.split.Validation) == 0 {
		return TrainingResult{}, errors.New("scratch model: invalid resident training run")
	}
	trainer, err := NewResidentTrainer(c, totalSteps)
	if err != nil {
		return TrainingResult{}, err
	}
	defer trainer.Close()
	result := TrainingResult{Losses: make([]float64, totalSteps)}
	materialized, batcher, err := c.documentBatcher(c.split.Train)
	if err != nil {
		return TrainingResult{}, err
	}
	defer materialized.Close()
	for step := range totalSteps {
		document, err := nextDocument(context.Background(), batcher)
		if err != nil {
			return TrainingResult{}, err
		}
		tokens, err := c.Tokens(document)
		if err != nil {
			return TrainingResult{}, err
		}
		result.Losses[step], err = trainer.Step(tokens, step+1)
		if err != nil {
			return TrainingResult{}, err
		}
	}
	validation, err := c.Tokens(c.split.Validation[0])
	if err != nil {
		return TrainingResult{}, err
	}
	result.ValidationLoss, err = trainer.Evaluate(validation)
	return result, err
}
