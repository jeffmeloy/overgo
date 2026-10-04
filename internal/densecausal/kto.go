package densecausal

import (
	"errors"

	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

// TrainKTOBatchesResume takes one update per unpaired microbatch under
// KTO's loss against the frozen reference: every completion is scored
// after its own prompt by the policy and the reference, its mismatched
// sequence is scored by both for the KL estimate, and the gradients of the
// completions after their own prompts are summed. The mismatched sequences
// take no backward pass, so the KL estimate carries no gradient.
func (m *Model) TrainKTOBatchesResume(
	reference *Model,
	batches []trainingdata.UnpairedBatch,
	baseLR, momentum float64,
	preference trainingprogram.PreferencePolicy,
	resume *RLState,
) ([]trainingprogram.KTOObservation, RLState, error) {
	if reference == nil || len(batches) == 0 {
		return nil, RLState{}, errors.New("densecausal: KTO reference or batches absent")
	}
	run, err := m.newRLRun("KTO", baseLR, momentum, resume)
	if err != nil {
		return nil, RLState{}, err
	}
	observations := make([]trainingprogram.KTOObservation, 0, len(batches))
	for _, batch := range batches {
		if err := run.admit(batch.State); err != nil {
			return nil, RLState{}, err
		}
		observation, grads, err := m.ktoLossAndGrads(reference, batch.Examples, preference)
		if err != nil {
			return nil, RLState{}, err
		}
		observation.Step, observation.LearningRate, observation.GradientL2, observation.UpdateL2, err = run.stepObserved(grads)
		if err != nil {
			return nil, RLState{}, err
		}
		observations = append(observations, observation)
	}
	return observations, run.state(), nil
}

// ktoLossAndGrads scores one microbatch and returns its observation and the
// summed gradients of its completions after their own prompts.
func (m *Model) ktoLossAndGrads(reference *Model, examples []trainingdata.UnpairedExample, preference trainingprogram.PreferencePolicy) (trainingprogram.KTOObservation, Grads, error) {
	traces := make([]preferenceTrace, len(examples))
	scores := make([]trainingprogram.UnpairedScores, len(examples))
	for index, example := range examples {
		trace, err := m.sequenceTrace(example.Sequence)
		if err != nil {
			return trainingprogram.KTOObservation{}, nil, err
		}
		mismatched, err := m.SequenceLogProb(example.Mismatched)
		if err != nil {
			return trainingprogram.KTOObservation{}, nil, err
		}
		referenceScore, err := reference.SequenceLogProb(example.Sequence)
		if err != nil {
			return trainingprogram.KTOObservation{}, nil, err
		}
		referenceMismatched, err := reference.SequenceLogProb(example.Mismatched)
		if err != nil {
			return trainingprogram.KTOObservation{}, nil, err
		}
		traces[index] = trace
		scores[index] = trainingprogram.UnpairedScores{
			Desirable: example.Desirable, Policy: trace.score, Reference: referenceScore,
			MismatchedPolicy: mismatched, MismatchedReference: referenceMismatched,
		}
	}
	objective, err := preference.UnpairedLoss(scores)
	if err != nil {
		return trainingprogram.KTOObservation{}, nil, err
	}
	grads, err := m.sequenceGradientSum(traces, objective.Gradients)
	if err != nil {
		return trainingprogram.KTOObservation{}, nil, err
	}
	return trainingprogram.KTOObservation{KTOSummary: objective.KTOSummary}, grads, nil
}
