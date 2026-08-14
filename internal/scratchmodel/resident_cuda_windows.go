//go:build windows

package scratchmodel

import (
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/devicemath"
	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

// ResidentTrainer owns one scratch model's device training lifecycle.
type ResidentTrainer struct {
	construction Construction
	worker       *device.Worker
	executor     *executor.Executor
	weights      driver.DevicePtr
	gradients    driver.DevicePtr
	momentum     driver.DevicePtr
	config       optimizer.Config
	closed       bool
}

func NewResidentTrainer(construction Construction, totalSteps int) (*ResidentTrainer, error) {
	if totalSteps <= 0 || construction.optimizer.Identity() == "" || len(construction.weights) == 0 {
		return nil, errors.New("scratch model: invalid resident trainer")
	}
	worker, err := device.New(0)
	if err != nil {
		return nil, err
	}
	trainer := &ResidentTrainer{
		construction: construction, worker: worker,
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
	trainer.weights, err = devicemath.AllocResidentF32(worker, len(construction.weights), construction.weights)
	if err != nil {
		return fail(err)
	}
	trainer.gradients, err = devicemath.AllocResidentF32(worker, len(construction.weights), nil)
	if err != nil {
		return fail(err)
	}
	trainer.momentum, err = devicemath.AllocResidentF32(worker, len(construction.weights), nil)
	if err != nil {
		return fail(err)
	}
	return trainer, nil
}

func (t *ResidentTrainer) Close() error {
	if t == nil || t.closed {
		return nil
	}
	t.closed = true
	var result error
	if t.executor != nil {
		result = errors.Join(result, t.executor.Close())
	}
	if t.worker != nil {
		result = errors.Join(result, devicemath.FreeResident(t.worker, t.weights, t.gradients, t.momentum))
		result = errors.Join(result, t.worker.Close())
	}
	return result
}

func (t *ResidentTrainer) Step(tokens []int, step int) (float64, error) {
	if t == nil || t.closed || step <= 0 {
		return 0, errors.New("scratch model: resident trainer unavailable")
	}
	graph, retained, err := t.forward(tokens)
	if err != nil {
		return 0, err
	}
	defer retained.Release(context.Background())
	loss, err := t.backward(graph, retained, tokens)
	if err != nil {
		return 0, err
	}
	if err := optimizer.DeviceMuonPlanResident(
		t.worker, t.weights, t.gradients, t.momentum,
		t.construction.optimizer, step, t.config,
	); err != nil {
		return 0, err
	}
	return loss, nil
}

func (t *ResidentTrainer) Evaluate(tokens []int) (float64, error) {
	if t == nil || t.closed {
		return 0, errors.New("scratch model: resident trainer unavailable")
	}
	graph, retained, err := t.forward(tokens)
	if err != nil {
		return 0, err
	}
	defer retained.Release(context.Background())
	value, err := retained.CopyToHost(context.Background(), graph.Output)
	if err != nil {
		return 0, err
	}
	gradient := make([]float32, len(value.Data))
	return hostmath.SoftmaxCrossEntropy(
		gradient, value.Data, tokens[1:graph.positions+1], graph.positions, t.construction.config.VocabSize,
	), nil
}

func (t *ResidentTrainer) Snapshot() (weights, momentum []float32, err error) {
	if t == nil || t.closed {
		return nil, nil, errors.New("scratch model: resident trainer unavailable")
	}
	weights, momentum = make([]float32, len(t.construction.weights)), make([]float32, len(t.construction.weights))
	if err := devicemath.ReadResident(t.worker, t.weights, devicemath.ResidentSlice{Data: weights}); err != nil {
		return nil, nil, err
	}
	if err := devicemath.ReadResident(t.worker, t.momentum, devicemath.ResidentSlice{Data: momentum}); err != nil {
		return nil, nil, err
	}
	return weights, momentum, nil
}

func (t *ResidentTrainer) forward(tokens []int) (ForwardGraph, *executor.RetainedOutputs, error) {
	graph, err := t.construction.CompileForwardGraph(tokens)
	if err != nil {
		return ForwardGraph{}, nil, err
	}
	hostFeeds, deviceFeeds, err := graph.residentFeeds(t.construction, t.weights)
	if err != nil {
		return ForwardGraph{}, nil, err
	}
	compiled, err := executor.Compile(graph.cacheOutputs()...)
	if err != nil {
		return ForwardGraph{}, nil, err
	}
	retained, err := t.executor.ExecuteRetainedCompiledWithDeviceFeeds(
		context.Background(), compiled, hostFeeds, deviceFeeds,
	)
	return graph, retained, err
}

func (g ForwardGraph) residentFeeds(
	c Construction,
	weightSlab driver.DevicePtr,
) (map[*tensor.Tensor]reference.Value, map[*tensor.Tensor]driver.DevicePtr, error) {
	if weightSlab == 0 || g.Mask == nil || len(g.Parameters) != len(c.parameters) {
		return nil, nil, errors.New("scratch model: resident graph or slab differs")
	}
	mask, err := reference.NewValue(g.Mask.Shape, g.mask)
	if err != nil {
		return nil, nil, err
	}
	hostFeeds := map[*tensor.Tensor]reference.Value{g.Mask: mask}
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr, len(g.Parameters))
	for name, input := range g.Parameters {
		binding, ok := c.bindings[name]
		if !ok {
			return nil, nil, fmt.Errorf("scratch model: resident parameter %q absent", name)
		}
		deviceFeeds[input] = devicemath.ResidentPtr(weightSlab, binding.start)
	}
	return hostFeeds, deviceFeeds, nil
}

func (t *ResidentTrainer) backward(graph ForwardGraph, retained *executor.RetainedOutputs, tokens []int) (float64, error) {
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
	var temporary []driver.DevicePtr
	allocate := func(count int, initial []float32) (driver.DevicePtr, error) {
		value, err := devicemath.AllocResidentF32(t.worker, count, initial)
		if err == nil {
			temporary = append(temporary, value)
		}
		return value, err
	}
	defer func() { _ = devicemath.FreeResident(t.worker, temporary...) }()
	if err := devicemath.ZeroResidentF32(t.worker, t.gradients, len(c.weights)); err != nil {
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
	targetsPtr, err := devicemath.AllocResidentU32(t.worker, targets)
	if err != nil {
		return 0, err
	}
	temporary = append(temporary, targetsPtr)
	lossesPtr, err := allocate(positions, nil)
	if err != nil {
		return 0, err
	}
	if err := devicemath.SoftmaxCrossEntropyBackwardResident(t.worker, logits, targetsPtr, lossesPtr, positions, vocab); err != nil {
		return 0, err
	}
	finalHidden, err := pointer(graph.Hidden)
	if err != nil {
		return 0, err
	}
	dHidden, err := allocate(positions*hidden, nil)
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
	if err := devicemath.LinearBackwardTResident(t.worker, finalHidden, head, logits, dHidden, dHead, positions, hidden, vocab); err != nil {
		return 0, err
	}

	for layer := len(graph.layers) - 1; layer >= 0; layer-- {
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
		dActivation, err := allocate(positions*c.config.MLPWidth, nil)
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
		if err := devicemath.LinearBackwardTResident(t.worker, activation, w2, dHidden, dActivation, dW2, positions, c.config.MLPWidth, hidden); err != nil {
			return 0, err
		}
		if err := devicemath.ReLUBackwardResident(t.worker, dActivation, preactivation, dActivation, positions*c.config.MLPWidth); err != nil {
			return 0, err
		}
		dMLPNorm, err := allocate(positions*hidden, nil)
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
		if err := devicemath.LinearBackwardTResident(t.worker, mlpNorm, w1, dActivation, dMLPNorm, dW1, positions, hidden, c.config.MLPWidth); err != nil {
			return 0, err
		}
		dMLPInput, err := allocate(positions*hidden, nil)
		if err != nil {
			return 0, err
		}
		if err := devicemath.MADNormBackwardResident(t.worker, dMLPNorm, attentionOutput, mlpNorm, dMLPInput, positions, hidden, c.config.Epsilon); err != nil {
			return 0, err
		}
		dAttentionOutput, err := allocate(positions*hidden, nil)
		if err != nil {
			return 0, err
		}
		if err := devicemath.AddResident(t.worker, dHidden, dMLPInput, dAttentionOutput, positions*hidden); err != nil {
			return 0, err
		}
		attention, err := pointer(cache.attention)
		if err != nil {
			return 0, err
		}
		dAttention, err := allocate(positions*hidden, nil)
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
		if err := devicemath.LinearBackwardTResident(t.worker, attention, wo, dAttentionOutput, dAttention, dWO, positions, hidden, hidden); err != nil {
			return 0, err
		}
		dInput, err := allocate(positions*hidden, nil)
		if err != nil {
			return 0, err
		}
		if err := devicemath.StridedRowCopyResident(t.worker, dAttentionOutput, dInput, positions, hidden, hidden, 0, hidden, 0); err != nil {
			return 0, err
		}
		dQKV, err := allocate(positions*3*hidden, nil)
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
			dHeadAttention, err := allocate(positions*c.config.HeadDim, nil)
			if err != nil {
				return 0, err
			}
			if err := devicemath.StridedRowCopyResident(
				t.worker, dAttention, dHeadAttention,
				positions, c.config.HeadDim, hidden, headIndex*c.config.HeadDim, c.config.HeadDim, 0,
			); err != nil {
				return 0, err
			}
			dQ, err := allocate(positions*c.config.HeadDim, nil)
			if err != nil {
				return 0, err
			}
			dK, err := allocate(positions*c.config.HeadDim, nil)
			if err != nil {
				return 0, err
			}
			dV, err := allocate(positions*c.config.HeadDim, nil)
			if err != nil {
				return 0, err
			}
			dScores, err := allocate(positions*positions, nil)
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
			if err := devicemath.AttentionCoreBackwardResident(
				t.worker, scaledQuery, k, v, probability, dHeadAttention,
				dQ, dK, dV, dScores, positions, c.config.HeadDim, 1,
			); err != nil {
				return 0, err
			}
			if err := devicemath.ScaleByResidentScalar(t.worker, dQ, temperatureScale, dQ, positions*c.config.HeadDim); err != nil {
				return 0, err
			}
			inverseRoot := float32(1 / math.Sqrt(float64(c.config.HeadDim)))
			if err := devicemath.ScaleResident(t.worker, dQ, dQ, inverseRoot, positions*c.config.HeadDim); err != nil {
				return 0, err
			}
			for _, copySpec := range []struct {
				source driver.DevicePtr
				offset int
			}{{dQ, headIndex * c.config.HeadDim}, {dK, hidden + headIndex*c.config.HeadDim}, {dV, 2*hidden + headIndex*c.config.HeadDim}} {
				if err := devicemath.StridedRowCopyResident(
					t.worker, copySpec.source, dQKV,
					positions, c.config.HeadDim, c.config.HeadDim, 0, 3*hidden, copySpec.offset,
				); err != nil {
					return 0, err
				}
			}
			dScale, err := allocate(1, nil)
			if err != nil {
				return 0, err
			}
			if err := devicemath.AttentionScoreAffineBackwardResident(
				t.worker, dScores, q, k, dBias, dScale,
				positions, c.config.HeadDim, c.config.BlockSize,
			); err != nil {
				return 0, err
			}
			if err := devicemath.ScaleByResidentScalar(t.worker, dScale, temperatureScale, dScale, 1); err != nil {
				return 0, err
			}
			if err := devicemath.ScaleResident(
				t.worker, dScale, devicemath.ResidentPtr(dLT, layer*c.config.HeadCount+headIndex), -inverseRoot, 1,
			); err != nil {
				return 0, err
			}
		}
		qkvNorm, err := pointer(cache.qkvNorm)
		if err != nil {
			return 0, err
		}
		dQKVNorm, err := allocate(positions*hidden, nil)
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
		if err := devicemath.LinearBackwardTResident(t.worker, qkvNorm, wqkv, dQKV, dQKVNorm, dWQKV, positions, hidden, 3*hidden); err != nil {
			return 0, err
		}
		input, err := pointer(cache.input)
		if err != nil {
			return 0, err
		}
		dQKVInput, err := allocate(positions*hidden, nil)
		if err != nil {
			return 0, err
		}
		if err := devicemath.MADNormBackwardResident(t.worker, dQKVNorm, input, qkvNorm, dQKVInput, positions, hidden, c.config.Epsilon); err != nil {
			return 0, err
		}
		if err := devicemath.AddResident(t.worker, dInput, dQKVInput, dInput, positions*hidden); err != nil {
			return 0, err
		}
		dHidden = dInput
	}

	embedding, err := pointer(graph.embedding)
	if err != nil {
		return 0, err
	}
	initialNormalized, err := pointer(graph.layers[0].input)
	if err != nil {
		return 0, err
	}
	dEmbedding, err := allocate(positions*hidden, nil)
	if err != nil {
		return 0, err
	}
	if err := devicemath.MADNormBackwardResident(t.worker, dHidden, embedding, initialNormalized, dEmbedding, positions, hidden, c.config.Epsilon); err != nil {
		return 0, err
	}
	tokenRows, positionRows := make([]uint32, positions), make([]uint32, positions)
	for position := range positions {
		tokenRows[position], positionRows[position] = uint32(tokens[position]), uint32(position)
	}
	tokenRowsPtr, err := devicemath.AllocResidentU32(t.worker, tokenRows)
	if err != nil {
		return 0, err
	}
	temporary = append(temporary, tokenRowsPtr)
	positionRowsPtr, err := devicemath.AllocResidentU32(t.worker, positionRows)
	if err != nil {
		return 0, err
	}
	temporary = append(temporary, positionRowsPtr)
	dWTE, err := gradient("wte")
	if err != nil {
		return 0, err
	}
	dWPE, err := gradient("wpe")
	if err != nil {
		return 0, err
	}
	if err := devicemath.IndexedRowScatterAddResident(t.worker, dEmbedding, tokenRowsPtr, dWTE, positions, hidden); err != nil {
		return 0, err
	}
	if err := devicemath.IndexedRowScatterAddResident(t.worker, dEmbedding, positionRowsPtr, dWPE, positions, hidden); err != nil {
		return 0, err
	}
	losses := make([]float32, positions)
	if err := devicemath.ReadResident(t.worker, lossesPtr, devicemath.ResidentSlice{Data: losses}); err != nil {
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
	for step := range totalSteps {
		tokens, err := c.Tokens(c.split.Train[step%len(c.split.Train)])
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
