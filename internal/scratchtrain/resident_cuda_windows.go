//go:build windows

// Package scratchtrain trains a scratch model resident on the device: the
// compiled forward runs on the executor and the loss VJP and Muon step on
// resident buffers. It holds the device path scratchmodel's host trainer
// does not need, so the model builder the server links never reaches
// devicemath; only the trainers import it.
package scratchtrain

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/devicemath"
	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/scratchmodel"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/trainingprogram"
)

// ResidentTrainer owns one scratch model's device training lifecycle.
type ResidentTrainer struct {
	construction scratchmodel.Construction
	worker       *device.Worker
	executor     *executor.Executor
	resident     *devicemath.ResidentOpsSession
	muon         *optimizer.ResidentMuonPlan
	weights      driver.DevicePtr
	gradients    driver.DevicePtr
	momentum     driver.DevicePtr
	config       optimizer.Config
	step         int // the last update Step committed
	programs     []residentForwardProgram
	training     trainingprogram.Execution[residentTrainingState]
	evaluation   trainingprogram.Execution[residentTrainingState]
	lifecycle    ResidentLifecycle
	closed       bool
}

// TrainedState is what a resident trainer trained, copied to the host so
// it outlives the device: the trained model's identity, its weights, and
// the optimizer state the next step continues from.
type TrainedState struct {
	model   artifact.ID
	plan    optimizer.Plan
	weights []float32
	state   optimizer.State
}

// Model is the trained model's identity.
func (s TrainedState) Model() artifact.ID { return s.model }

// Snapshot copies the optimizer state the next step continues from.
func (s TrainedState) Snapshot() optimizer.State {
	state := s.state
	state.Momentum = slices.Clone(state.Momentum)
	return state
}

// SaveWeights writes the trained weights at path, one tensor per parameter
// of the construction's program: the weights Snapshot pairs with.
func (s TrainedState) SaveWeights(path string) error {
	return optimizer.SavePlanWeights(path, s.plan, s.weights)
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
	graph     scratchmodel.ForwardGraph
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
	graph     scratchmodel.ForwardGraph
	hostFeeds map[*tensor.Tensor]reference.Value
}

// NewResidentTrainer uploads the construction's initial weights to the
// first device and compiles a forward program for each document extent its
// split holds, ready for totalSteps Muon steps under policy.
func NewResidentTrainer(construction scratchmodel.Construction, totalSteps int, policy trainingprogram.OptimizerPolicy) (*ResidentTrainer, error) {
	plan := construction.OptimizerPlan()
	if totalSteps <= 0 || plan.Identity() == "" || plan.ParameterCount() == 0 {
		return nil, errors.New("scratch model: invalid resident trainer")
	}
	driverStarted := time.Now()
	worker, err := device.New(0)
	if err != nil {
		return nil, err
	}
	config, err := construction.OptimizerConfig(policy, totalSteps)
	if err != nil {
		_ = worker.Close()
		return nil, err
	}
	trainer := &ResidentTrainer{
		construction: construction, worker: worker,
		lifecycle: ResidentLifecycle{DriverPreparation: time.Since(driverStarted)},
		config:    config,
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
	weightBytes := driver.Bytes(construction.InitialWeights())
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
	split := construction.Split()
	for _, documents := range [][]string{split.Train, split.Validation, split.Test} {
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
	trainer.muon, err = optimizer.NewResidentMuonPlan(worker, plan, trainer.config)
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

// Lifecycle reports how long the trainer's preparation phases took.
func (t *ResidentTrainer) Lifecycle() (ResidentLifecycle, error) {
	if t == nil || t.closed {
		return ResidentLifecycle{}, errors.New("scratch model: resident trainer unavailable")
	}
	return t.lifecycle, nil
}

// Close releases the trainer's device programs, buffers and worker.
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

// Step trains one Muon update, the step'th, on a token batch and returns
// its loss.
func (t *ResidentTrainer) Step(tokens []int, step int) (float64, error) {
	if t == nil || t.closed || step <= 0 {
		return 0, errors.New("scratch model: resident trainer unavailable")
	}
	state := residentTrainingState{trainer: t, tokens: tokens, step: step}
	defer state.release()
	err := t.training.Run(&state)
	if err == nil {
		t.step = step
	}
	return state.loss, err
}

// Evaluate scores a token batch's cross-entropy without updating.
func (t *ResidentTrainer) Evaluate(tokens []int) (float64, error) {
	if t == nil || t.closed {
		return 0, errors.New("scratch model: resident trainer unavailable")
	}
	state := residentTrainingState{trainer: t, tokens: tokens}
	defer state.release()
	err := t.evaluation.Run(&state)
	return state.loss, err
}

// Snapshot copies the resident weights, gradients and momentum to the host.
func (t *ResidentTrainer) Snapshot() (weights, gradients, momentum []float32, err error) {
	if t == nil || t.closed {
		return nil, nil, nil, errors.New("scratch model: resident trainer unavailable")
	}
	count := t.construction.OptimizerPlan().ParameterCount()
	weights, gradients, momentum = make([]float32, count), make([]float32, count), make([]float32, count)
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

// Trained copies what the trainer trained to the host: the weights under
// the construction's optimizer plan and the Muon state its last step left,
// the f32 device momentum widened exactly.
func (t *ResidentTrainer) Trained() (TrainedState, error) {
	weights, _, momentum, err := t.Snapshot()
	if err != nil {
		return TrainedState{}, err
	}
	model, err := t.construction.IdentifyTrainedModel(weights)
	if err != nil {
		return TrainedState{}, err
	}
	plan := t.construction.OptimizerPlan()
	return TrainedState{model: model, plan: plan, weights: weights, state: optimizer.State{
		PlanIdentity: plan.Identity(), Config: t.config, Step: t.step, Momentum: dtype.ConvertSlice[float64](momentum),
	}}, nil
}

// ResetPeakMemory restarts the device's peak memory measurement.
func (t *ResidentTrainer) ResetPeakMemory() error {
	if t == nil || t.closed {
		return errors.New("scratch model: resident trainer unavailable")
	}
	return t.worker.Do(context.Background(), func(state *device.State) error {
		state.Driver.ResetPeakBytes()
		return nil
	})
}

// MemoryStats reports the device memory the trainer's worker measured.
func (t *ResidentTrainer) MemoryStats() (driver.MemoryStats, error) {
	if t == nil || t.closed {
		return driver.MemoryStats{}, errors.New("scratch model: resident trainer unavailable")
	}
	return t.worker.MemoryStats(context.Background())
}

func (t *ResidentTrainer) forward(tokens []int, positions int) (scratchmodel.ForwardGraph, *executor.RetainedOutputs, error) {
	program, err := t.forwardProgram(tokens)
	if err != nil {
		return scratchmodel.ForwardGraph{}, nil, err
	}
	if program.graph.Positions != positions {
		return scratchmodel.ForwardGraph{}, nil, errors.New("scratch model: compiled batch extent differs")
	}
	rows := make([]uint32, program.graph.Positions)
	for index, token := range tokens[:program.graph.Positions] {
		rows[index] = uint32(token)
	}
	attributes := program.Graph.NewRuntimeAttributes()
	if err := attributes.Set(program.graph.TokenRows, tensor.GetRowsAttributes{Rows: rows}); err != nil {
		return scratchmodel.ForwardGraph{}, nil, err
	}
	retained, err := t.executor.ExecuteRetainedCompiled(
		context.Background(), program.Graph, program.hostFeeds, program.Inputs, nil, attributes,
	)
	return program.graph, retained, err
}

func bindResidentProgram(trainer *ResidentTrainer) (trainingprogram.Execution[residentTrainingState], error) {
	bindings := []trainingprogram.Binding[residentTrainingState]{
		{Operator: scratchmodel.ScratchOperatorBatch, Execute: func(state *residentTrainingState) error {
			positions, err := state.trainer.construction.ValidateTokens(state.tokens)
			state.positions = positions
			return err
		}},
		{Operator: scratchmodel.ScratchOperatorForward, Execute: func(state *residentTrainingState) error {
			var err error
			state.graph, state.retained, err = state.trainer.forward(state.tokens, state.positions)
			return err
		}},
		{Operator: scratchmodel.ScratchOperatorLossVJP, Execute: func(state *residentTrainingState) error {
			var err error
			state.loss, err = state.trainer.backward(state.graph, state.retained, state.tokens)
			return err
		}},
		{Operator: scratchmodel.ScratchOperatorMuon, Execute: func(state *residentTrainingState) error {
			return state.trainer.muon.Step(state.trainer.weights, state.trainer.gradients, state.trainer.momentum, state.step)
		}},
		{Operator: scratchmodel.ScratchOperatorEvaluate, Execute: func(state *residentTrainingState) error {
			value, err := state.retained.CopyToHost(context.Background(), state.graph.Output)
			if err != nil {
				return err
			}
			gradient := make([]float32, len(value.Data))
			state.loss = hostmath.SoftmaxCrossEntropy(
				gradient, value.Data, state.tokens[1:state.graph.Positions+1],
				state.graph.Positions, state.trainer.construction.Config().VocabSize,
			)
			return nil
		}},
	}
	return trainingprogram.Bind(trainer.construction.Program(), bindings)
}

func (t *ResidentTrainer) forwardProgram(tokens []int) (*residentForwardProgram, error) {
	positions, err := t.construction.ValidateTokens(tokens)
	if err != nil {
		return nil, err
	}
	for index := range t.programs {
		if t.programs[index].graph.Positions == positions {
			return &t.programs[index], nil
		}
	}
	graph, err := t.construction.CompileForwardGraph(tokens)
	if err != nil {
		return nil, err
	}
	indexed, err := executor.CompileIndexed(graph.CacheOutputs()...)
	if err != nil {
		return nil, err
	}
	hostFeeds, err := residentInputs(graph, indexed, t.construction, t.weights)
	if err != nil {
		return nil, err
	}
	t.programs = append(t.programs, residentForwardProgram{
		IndexedGraph: indexed, graph: graph, hostFeeds: hostFeeds,
	})
	return &t.programs[len(t.programs)-1], nil
}

func residentInputs(
	g scratchmodel.ForwardGraph,
	indexed *executor.IndexedGraph,
	c scratchmodel.Construction,
	weightSlab driver.DevicePtr,
) (map[*tensor.Tensor]reference.Value, error) {
	if indexed == nil || weightSlab == 0 || g.Mask == nil || len(g.Parameters) != c.OptimizerPlan().GroupCount() {
		return nil, errors.New("scratch model: resident graph or slab differs")
	}
	mask, err := reference.NewValue(g.Mask.Shape, g.MaskValues)
	if err != nil {
		return nil, err
	}
	hostFeeds := map[*tensor.Tensor]reference.Value{g.Mask: mask}
	for name, input := range g.Parameters {
		start, ok := c.Offset(name)
		if !ok {
			return nil, fmt.Errorf("scratch model: resident parameter %q absent", name)
		}
		slot, ok := indexed.Graph.InputSlot(input)
		if !ok {
			return nil, fmt.Errorf("scratch model: resident parameter %q is not compiled", name)
		}
		indexed.Inputs.Pointers[slot] = devicemath.ResidentPtr(weightSlab, start)
	}
	return hostFeeds, nil
}

func (t *ResidentTrainer) backward(graph scratchmodel.ForwardGraph, retained *executor.RetainedOutputs, tokens []int) (float64, error) {
	var loss float64
	err := t.resident.Run(func(ops *devicemath.ResidentOps) error {
		var err error
		loss, err = t.backwardWithOps(ops, graph, retained, tokens)
		return err
	})
	return loss, err
}

// The fused QKV projection packs the query, key and value blocks, each
// hidden wide, in that order.
const (
	valueBlock = 2
	qkvBlocks  = 3
)

// The layer backward's arena buffers: the MLP norm, MLP input, attention
// output and attention gradients and the fused QKV gradient live across the
// layer; each head's attention, query, key and value gradients and its
// score and probability gradients live across the head; the QKV norm and
// input gradients follow the heads.
const (
	layerHiddenBuffers = 4 + qkvBlocks
	headBuffers        = 4
	headScoreBuffers   = 2
	qkvInputBuffers    = 2
)

func scratchBackwardArenaElems(config scratchmodel.Config, positions int) int {
	hidden := positions * config.Embedding
	base := positions*config.MLPWidth + layerHiddenBuffers*hidden
	head := headBuffers*positions*config.HeadDim + headScoreBuffers*positions*positions + 1
	return max(hidden, base+max(head, qkvInputBuffers*hidden))
}

func (t *ResidentTrainer) backwardWithOps(ops *devicemath.ResidentOps, graph scratchmodel.ForwardGraph, retained *executor.RetainedOutputs, tokens []int) (float64, error) {
	c, config := t.construction, t.construction.Config()
	positions, hidden, vocab := graph.Positions, config.Embedding, config.VocabSize
	pointer := func(node *tensor.Tensor) (driver.DevicePtr, error) {
		value, ok := retained.Value(node)
		if !ok || value.Pointer == 0 {
			return 0, errors.New("scratch model: retained cache absent")
		}
		return value.Pointer, nil
	}
	weight := func(name string) (driver.DevicePtr, error) {
		start, ok := c.Offset(name)
		if !ok {
			return 0, fmt.Errorf("scratch model: resident weight %q absent", name)
		}
		return devicemath.ResidentPtr(t.weights, start), nil
	}
	gradient := func(name string) (driver.DevicePtr, error) {
		start, ok := c.Offset(name)
		if !ok {
			return 0, fmt.Errorf("scratch model: resident gradient %q absent", name)
		}
		return devicemath.ResidentPtr(t.gradients, start), nil
	}
	allocatePersistent := func(count int, initial []float32) (driver.DevicePtr, error) {
		if initial != nil {
			return 0, errors.New("scratch model: resident scratch initialization unsupported")
		}
		return ops.AllocF32(count)
	}
	if err := ops.Zero(t.gradients, c.OptimizerPlan().ParameterCount()); err != nil {
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
	arena, err := ops.NewArena(scratchBackwardArenaElems(config, positions))
	if err != nil {
		return 0, err
	}

	for layer := len(graph.Layers) - 1; layer >= 0; layer-- {
		layerMark := arena.Mark()
		cache := graph.Layers[layer]
		prefix := fmt.Sprintf("l%d.", layer)
		attentionOutput, err := pointer(cache.AttentionOutput)
		if err != nil {
			return 0, err
		}
		activation, err := pointer(cache.Activation)
		if err != nil {
			return 0, err
		}
		preactivation, err := pointer(cache.Preactivation)
		if err != nil {
			return 0, err
		}
		mlpNorm, err := pointer(cache.MLPNorm)
		if err != nil {
			return 0, err
		}
		dActivation, err := arena.AllocF32(positions * config.MLPWidth)
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
		if err := ops.LinearBackwardT(activation, w2, dHidden, dActivation, dW2, positions, config.MLPWidth, hidden); err != nil {
			return 0, err
		}
		if err := ops.ReLUBackward(dActivation, preactivation, dActivation, positions*config.MLPWidth); err != nil {
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
		if err := ops.LinearBackwardT(mlpNorm, w1, dActivation, dMLPNorm, dW1, positions, hidden, config.MLPWidth); err != nil {
			return 0, err
		}
		dMLPInput, err := arena.AllocF32(positions * hidden)
		if err != nil {
			return 0, err
		}
		if err := ops.MADNormBackward(dMLPNorm, attentionOutput, mlpNorm, dMLPInput, positions, hidden, config.Epsilon); err != nil {
			return 0, err
		}
		dAttentionOutput, err := arena.AllocF32(positions * hidden)
		if err != nil {
			return 0, err
		}
		if err := ops.Add(dHidden, dMLPInput, dAttentionOutput, positions*hidden); err != nil {
			return 0, err
		}
		attention, err := pointer(cache.Attention)
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
		dQKV, err := arena.AllocF32(positions * qkvBlocks * hidden)
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
		for headIndex, headCache := range cache.Heads {
			headMark := arena.Mark()
			q, err := pointer(headCache.Query)
			if err != nil {
				return 0, err
			}
			k, err := pointer(headCache.Key)
			if err != nil {
				return 0, err
			}
			v, err := pointer(headCache.Value)
			if err != nil {
				return 0, err
			}
			probability, err := pointer(headCache.Probability)
			if err != nil {
				return 0, err
			}
			dHeadAttention, err := arena.AllocF32(positions * config.HeadDim)
			if err != nil {
				return 0, err
			}
			if err := ops.StridedRowCopy(
				dAttention, dHeadAttention,
				positions, config.HeadDim, hidden, headIndex*config.HeadDim, config.HeadDim, 0,
			); err != nil {
				return 0, err
			}
			dQ, err := arena.AllocF32(positions * config.HeadDim)
			if err != nil {
				return 0, err
			}
			dK, err := arena.AllocF32(positions * config.HeadDim)
			if err != nil {
				return 0, err
			}
			dV, err := arena.AllocF32(positions * config.HeadDim)
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
			scaledQuery, err := pointer(headCache.ScaledQuery)
			if err != nil {
				return 0, err
			}
			temperatureScale, err := pointer(headCache.TemperatureScale)
			if err != nil {
				return 0, err
			}
			if err := ops.AttentionCoreBackwardWithScratch(
				scaledQuery, k, v, probability, dHeadAttention,
				dQ, dK, dV, dScores, dProbability, positions, config.HeadDim, 1,
			); err != nil {
				return 0, err
			}
			if err := ops.ScaleByScalar(dQ, temperatureScale, dQ, positions*config.HeadDim); err != nil {
				return 0, err
			}
			inverseRoot := float32(1 / math.Sqrt(float64(config.HeadDim)))
			if err := ops.Scale(dQ, dQ, inverseRoot, positions*config.HeadDim); err != nil {
				return 0, err
			}
			for _, copySpec := range []struct {
				source driver.DevicePtr
				offset int
			}{{dQ, headIndex * config.HeadDim}, {dK, hidden + headIndex*config.HeadDim}, {dV, valueBlock*hidden + headIndex*config.HeadDim}} {
				if err := ops.StridedRowCopy(
					copySpec.source, dQKV,
					positions, config.HeadDim, config.HeadDim, 0, qkvBlocks*hidden, copySpec.offset,
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
				positions, config.HeadDim, config.BlockSize,
			); err != nil {
				return 0, err
			}
			if err := ops.ScaleByScalar(dScale, temperatureScale, dScale, 1); err != nil {
				return 0, err
			}
			if err := ops.Scale(
				dScale, devicemath.ResidentPtr(dLT, layer*config.HeadCount+headIndex), -inverseRoot, 1,
			); err != nil {
				return 0, err
			}
			if err := arena.Reset(headMark); err != nil {
				return 0, err
			}
		}
		qkvNorm, err := pointer(cache.QKVNorm)
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
		if err := ops.LinearBackwardT(qkvNorm, wqkv, dQKV, dQKVNorm, dWQKV, positions, hidden, qkvBlocks*hidden); err != nil {
			return 0, err
		}
		input, err := pointer(cache.Input)
		if err != nil {
			return 0, err
		}
		dQKVInput, err := arena.AllocF32(positions * hidden)
		if err != nil {
			return 0, err
		}
		if err := ops.MADNormBackward(dQKVNorm, input, qkvNorm, dQKVInput, positions, hidden, config.Epsilon); err != nil {
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

	embedding, err := pointer(graph.Embedding)
	if err != nil {
		return 0, err
	}
	initialNormalized, err := pointer(graph.Layers[0].Input)
	if err != nil {
		return 0, err
	}
	dEmbedding, err := arena.AllocF32(positions * hidden)
	if err != nil {
		return 0, err
	}
	if err := ops.MADNormBackward(dHidden, embedding, initialNormalized, dEmbedding, positions, hidden, config.Epsilon); err != nil {
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
