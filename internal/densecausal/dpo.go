package densecausal

import (
	"errors"
	"slices"

	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

type DPOState struct {
	Optimizer TrainState
	Stream    trainingdata.StreamState
}

func (m *Model) TrainDPOBatchesResume(
	reference *Model,
	batches []trainingdata.PreferenceBatch,
	baseLR, momentum, scale float64,
	resume *DPOState,
	observe func(trainingprogram.DPOObservation),
) ([]trainingprogram.DPOObservation, DPOState, error) {
	if reference == nil || len(batches) == 0 {
		return nil, DPOState{}, errors.New("densecausal: DPO reference or batches absent")
	}
	if resume != nil && !resume.Stream.Identity.Valid() {
		return nil, DPOState{}, errors.New("densecausal: DPO resume stream authority absent")
	}
	names, weights, gradients, plan, resolvedLR, err := m.trainSetup(baseLR)
	if err != nil {
		return nil, DPOState{}, err
	}
	update, err := optimizer.New(weights, gradients, plan, optimizer.Config{
		BaseLearningRate: resolvedLR, Momentum: momentum, Schedule: optimizer.ScheduleConstant,
	})
	if err != nil {
		return nil, DPOState{}, err
	}
	if resume != nil {
		if err := update.Restore(resume.Optimizer); err != nil {
			return nil, DPOState{}, err
		}
	}
	stream := trainingdata.StreamState{}
	if resume != nil {
		stream = resume.Stream
	}
	observations := make([]trainingprogram.DPOObservation, 0)
	for _, batch := range batches {
		if len(batch.Pairs) == 0 || !batch.State.Identity.Valid() ||
			stream.Identity.Valid() && (batch.State.Identity != stream.Identity || batch.State.Position <= stream.Position) {
			return nil, DPOState{}, errors.New("densecausal: DPO stream authority differs")
		}
		for _, pair := range batch.Pairs {
			if pair.SharedPrefix <= 0 {
				return nil, DPOState{}, errors.New("densecausal: DPO pair lacks exact prefix")
			}
			scatter(m, names, weights)
			observation, grads, err := m.dpoLossAndGrads(reference, pair, scale)
			if err != nil {
				return nil, DPOState{}, err
			}
			m.gatherGrads(names, gradients, grads)
			step := update.Step()
			observation.Step = uint64(step.Step)
			observation.LearningRate = step.LearningRate
			observation.GradientL2 = step.GradientL2
			observation.UpdateL2 = step.UpdateL2
			if observe != nil {
				observe(observation)
			} else {
				observations = append(observations, observation)
			}
		}
		stream = batch.State
	}
	scatter(m, names, weights)
	return observations, DPOState{Optimizer: update.Snapshot(), Stream: stream}, nil
}

func (m *Model) dpoLossAndGrads(reference *Model, pair trainingdata.PreferencePair, scale float64) (trainingprogram.DPOObservation, Grads, error) {
	policyChosen, err := m.sequenceTrace(pair.Chosen)
	if err != nil {
		return trainingprogram.DPOObservation{}, nil, err
	}
	policyRejected, err := m.sequenceTrace(pair.Rejected)
	if err != nil {
		return trainingprogram.DPOObservation{}, nil, err
	}
	referenceChosen, err := reference.sequenceLogProb(pair.Chosen)
	if err != nil {
		return trainingprogram.DPOObservation{}, nil, err
	}
	referenceRejected, err := reference.sequenceLogProb(pair.Rejected)
	if err != nil {
		return trainingprogram.DPOObservation{}, nil, err
	}
	scores := trainingprogram.PreferenceScores{
		PolicyChosen: policyChosen.score, PolicyRejected: policyRejected.score,
		ReferenceChosen: referenceChosen, ReferenceRejected: referenceRejected,
	}
	objective, err := trainingprogram.DPOLoss(scores, scale)
	if err != nil {
		return trainingprogram.DPOObservation{}, nil, err
	}
	chosen, err := m.sequenceLogProbGrads(policyChosen, objective.ChosenGradient)
	if err != nil {
		return trainingprogram.DPOObservation{}, nil, err
	}
	rejected, err := m.sequenceLogProbGrads(policyRejected, objective.RejectedGradient)
	if err != nil {
		return trainingprogram.DPOObservation{}, nil, err
	}
	for name, values := range rejected {
		addInPlace(chosen[name], values)
	}
	policyMargin := scores.PolicyChosen - scores.PolicyRejected
	referenceMargin := scores.ReferenceChosen - scores.ReferenceRejected
	return trainingprogram.DPOObservation{
		Loss:         objective.Loss,
		PolicyChosen: scores.PolicyChosen, PolicyRejected: scores.PolicyRejected,
		ReferenceChosen: scores.ReferenceChosen, ReferenceRejected: scores.ReferenceRejected,
		PolicyMargin: policyMargin, ReferenceMargin: referenceMargin, RelativeMargin: policyMargin - referenceMargin,
		ChosenTokens: completionCount(pair.Chosen.Completion), RejectedTokens: completionCount(pair.Rejected.Completion),
	}, chosen, nil
}

func completionCount(mask []bool) uint64 {
	var count uint64
	for _, selected := range mask {
		if selected {
			count++
		}
	}
	return count
}

func (m *Model) sequenceLogProb(sequence trainingdata.PreferenceSequence) (float64, error) {
	trace, err := m.sequenceTrace(sequence)
	if err != nil {
		return 0, err
	}
	return trace.score, nil
}

type preferenceTrace struct {
	sequence trainingdata.PreferenceSequence
	states   [][]float32
	normed   []float32
	score    float64
}

func (m *Model) sequenceTrace(sequence trainingdata.PreferenceSequence) (preferenceTrace, error) {
	rows := len(sequence.Tokens) - 1
	if rows <= 0 || len(sequence.Completion) != len(sequence.Tokens) || sequence.Completion[0] {
		return preferenceTrace{}, errors.New("densecausal: invalid preference sequence")
	}
	states, err := m.forwardStates(sequence.Tokens)
	if err != nil {
		return preferenceTrace{}, err
	}
	final := states[m.Dims.Layers]
	normed := make([]float32, len(final))
	hostmath.RMSNormInto(normed, final, m.tensors.finalNorm.values, len(sequence.Tokens), m.Dims.Hidden, m.Dims.RMSEps)
	scores := make([]float64, rows)
	if err := hostmath.SelectedLogProbInto(
		scores, normed[:rows*m.Dims.Hidden], m.tensors.head.values, nil, sequence.Tokens[1:], sequence.Completion[1:],
		rows, m.Dims.Hidden, m.Dims.Vocab, make([]float32, m.Dims.Vocab),
	); err != nil {
		return preferenceTrace{}, err
	}
	var total float64
	for _, score := range scores {
		total += score
	}
	return preferenceTrace{sequence: sequence, states: states, normed: normed, score: total}, nil
}

func (m *Model) sequenceLogProbGrads(trace preferenceTrace, coefficient float64) (Grads, error) {
	sequence, states, normed := trace.sequence, trace.states, trace.normed
	d := m.Dims
	rows := len(sequence.Tokens) - 1
	final := states[d.Layers]
	g := Grads{}
	dNormed := make([]float32, len(normed))
	coefficients := slices.Repeat([]float64{coefficient}, rows)
	if err := hostmath.SelectedLogProbBackward(
		dNormed[:rows*d.Hidden], hostmath.GradientSlot(g, m.tensors.head.name, len(m.tensors.head.values)), nil,
		normed[:rows*d.Hidden], m.tensors.head.values, nil, sequence.Tokens[1:], sequence.Completion[1:], coefficients,
		rows, d.Hidden, d.Vocab, make([]float32, d.Vocab), false,
	); err != nil {
		return nil, err
	}
	dx := make([]float32, len(final))
	hostmath.RMSNormBackward(dx, hostmath.GradientSlot(g, m.tensors.finalNorm.name, d.Hidden), final, m.tensors.finalNorm.values, dNormed, len(sequence.Tokens), d.Hidden, d.RMSEps, false)
	invFreq := hostmath.RopeInvFreq(d.RopeTheta, d.HeadDim)
	for index := d.Layers - 1; index >= 0; index-- {
		var err error
		dx, err = m.layerBackward(index, states[index], dx, invFreq, len(sequence.Tokens), g)
		if err != nil {
			return nil, err
		}
	}
	scatterEmbeddingGradient(hostmath.GradientSlot(g, m.tensors.embedding.name, len(m.tensors.embedding.values)), dx, sequence.Tokens, d.Hidden)
	return g, nil
}
