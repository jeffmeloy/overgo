package densecausal

import (
	"errors"
	"slices"

	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

func (m *Model) TrainDPOExamples(
	reference *Model,
	examples []trainingdata.Example,
	decode trainingdata.TokenDecoder,
	baseLR, momentum, scale float64,
) ([]float64, error) {
	pairs, err := trainingdata.CompilePreferenceBatch(examples, decode)
	if err != nil {
		return nil, err
	}
	losses, _, err := m.TrainDPOPairsResume(reference, pairs, baseLR, momentum, scale, nil)
	return losses, err
}

func (m *Model) TrainDPOPairsResume(
	reference *Model,
	pairs []trainingdata.PreferencePair,
	baseLR, momentum, scale float64,
	resume *TrainState,
) ([]float64, TrainState, error) {
	if reference == nil || len(pairs) == 0 {
		return nil, TrainState{}, errors.New("densecausal: DPO reference or pairs absent")
	}
	names, weights, gradients, plan, resolvedLR, err := m.trainSetup(baseLR)
	if err != nil {
		return nil, TrainState{}, err
	}
	update, err := optimizer.New(weights, gradients, plan, optimizer.Config{
		BaseLearningRate: resolvedLR, Momentum: momentum, Schedule: optimizer.ScheduleConstant,
	})
	if err != nil {
		return nil, TrainState{}, err
	}
	if resume != nil {
		if err := update.Restore(*resume); err != nil {
			return nil, TrainState{}, err
		}
	}
	losses := make([]float64, len(pairs))
	for index, pair := range pairs {
		scatter(m, names, weights)
		loss, grads, err := m.dpoLossAndGrads(reference, pair, scale)
		if err != nil {
			return nil, TrainState{}, err
		}
		losses[index] = loss
		m.gatherGrads(names, gradients, grads)
		update.Step()
	}
	scatter(m, names, weights)
	return losses, update.Snapshot(), nil
}

func (m *Model) dpoLossAndGrads(reference *Model, pair trainingdata.PreferencePair, scale float64) (float64, Grads, error) {
	policyChosen, err := m.sequenceTrace(pair.Chosen)
	if err != nil {
		return 0, nil, err
	}
	policyRejected, err := m.sequenceTrace(pair.Rejected)
	if err != nil {
		return 0, nil, err
	}
	referenceChosen, err := reference.sequenceLogProb(pair.Chosen)
	if err != nil {
		return 0, nil, err
	}
	referenceRejected, err := reference.sequenceLogProb(pair.Rejected)
	if err != nil {
		return 0, nil, err
	}
	objective, err := trainingprogram.DPOLoss(trainingprogram.PreferenceScores{
		PolicyChosen: policyChosen.score, PolicyRejected: policyRejected.score,
		ReferenceChosen: referenceChosen, ReferenceRejected: referenceRejected,
	}, scale)
	if err != nil {
		return 0, nil, err
	}
	chosen, err := m.sequenceLogProbGrads(policyChosen, objective.ChosenGradient)
	if err != nil {
		return 0, nil, err
	}
	rejected, err := m.sequenceLogProbGrads(policyRejected, objective.RejectedGradient)
	if err != nil {
		return 0, nil, err
	}
	for name, values := range rejected {
		addInPlace(chosen[name], values)
	}
	return objective.Loss, chosen, nil
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
	hostmath.RMSNormInto(normed, final, m.Weights["model.norm.weight"], len(sequence.Tokens), m.Dims.Hidden, m.Dims.RMSEps)
	scores := make([]float64, rows)
	if err := hostmath.SelectedLogProbInto(
		scores, normed[:rows*m.Dims.Hidden], m.head(), nil, sequence.Tokens[1:], sequence.Completion[1:],
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
		dNormed[:rows*d.Hidden], hostmath.GradientSlot(g, m.headName(), len(m.head())), nil,
		normed[:rows*d.Hidden], m.head(), nil, sequence.Tokens[1:], sequence.Completion[1:], coefficients,
		rows, d.Hidden, d.Vocab, make([]float32, d.Vocab), false,
	); err != nil {
		return nil, err
	}
	dx := make([]float32, len(final))
	hostmath.RMSNormBackward(dx, hostmath.GradientSlot(g, "model.norm.weight", d.Hidden), final, m.Weights["model.norm.weight"], dNormed, len(sequence.Tokens), d.Hidden, d.RMSEps, false)
	invFreq := hostmath.RopeInvFreq(d.RopeTheta, d.HeadDim)
	for index := d.Layers - 1; index >= 0; index-- {
		var err error
		dx, err = m.layerBackward(index, states[index], dx, invFreq, len(sequence.Tokens), g)
		if err != nil {
			return nil, err
		}
	}
	scatterEmbeddingGradient(hostmath.GradientSlot(g, "model.embed_tokens.weight", len(m.Weights["model.embed_tokens.weight"])), dx, sequence.Tokens, d.Hidden)
	return g, nil
}
