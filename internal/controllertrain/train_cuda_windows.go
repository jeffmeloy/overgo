//go:build windows

package controllertrain

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/processmeasure"
	"overgo/internal/runrecord"
	"overgo/internal/scratchmodel"
	"overgo/internal/scratchtrain"
	"overgo/internal/trainingprogram"
)

// SeedRun: one shared-runtime controller training result.
type SeedRun struct {
	Evidence          SeedEvidence
	InitialModel      artifact.ID
	InitialRun        runrecord.Run
	InitialEvaluation runrecord.Evaluation
	Run               runrecord.Run
	Evaluation        runrecord.Evaluation
	Recipe            artifact.ID
	ScratchDataset    artifact.ID
	ScratchSplit      artifact.ID
	// Trained is what the run trained, the checkpoint a route publishes.
	Trained scratchtrain.TrainedState
}

// TrainSeed trains construction, the scratch model the corpus's training
// documents derive for one seed, for steps updates on the device, one
// training document of its split per update in order, and judges it on the
// corpus's held-out records before and after.
func TrainSeed(corpus Corpus, construction scratchmodel.Construction, policy trainingprogram.OptimizerPolicy, steps int) (SeedRun, error) {
	started := processmeasure.NewStopwatch()
	if steps <= 0 {
		return SeedRun{}, errors.New("controller training: steps must be positive")
	}
	inputs, err := prepareEvaluationInputs(construction, corpus)
	if err != nil {
		return SeedRun{}, err
	}
	trainer, err := scratchtrain.NewResidentTrainer(construction, steps, policy)
	if err != nil {
		return SeedRun{}, err
	}
	defer trainer.Close()
	if err := trainer.ResetPeakMemory(); err != nil {
		return SeedRun{}, err
	}
	initial, err := evaluateSuite(trainer, inputs)
	if err != nil {
		return SeedRun{}, err
	}
	initialWeights, _, _, err := trainer.Snapshot()
	if err != nil {
		return SeedRun{}, err
	}
	initialModel, err := construction.IdentifyTrainedModel(initialWeights)
	if err != nil {
		return SeedRun{}, err
	}
	train := construction.Split().Train
	for step := 1; step <= steps; step++ {
		tokens, tokenErr := construction.Tokens(train[(step-1)%len(train)])
		if tokenErr != nil {
			return SeedRun{}, tokenErr
		}
		if _, stepErr := trainer.Step(tokens, step); stepErr != nil {
			return SeedRun{}, stepErr
		}
	}
	final, err := evaluateSuite(trainer, inputs)
	if err != nil {
		return SeedRun{}, err
	}
	trained, err := trainer.Trained()
	if err != nil {
		return SeedRun{}, err
	}
	model := trained.Model()
	memory, err := trainer.MemoryStats()
	if err != nil {
		return SeedRun{}, err
	}
	recipeID := construction.Program().ID()
	initialRun, initialEvaluation, err := evaluationRecords(recipeID, corpus.Dataset(), initialModel, initial)
	if err != nil {
		return SeedRun{}, err
	}
	run, evaluation, err := evaluationRecords(recipeID, corpus.Dataset(), model, final)
	if err != nil {
		return SeedRun{}, err
	}
	wall, err := started.Elapsed()
	if err != nil {
		return SeedRun{}, err
	}
	return SeedRun{
		Evidence: SeedEvidence{
			Seed: construction.Seed(), Model: model, Run: run.ID, Evaluation: evaluation.ID,
			Initial: initial, Final: final,
			Cost: ResourceCost{WallNS: wall, PeakDeviceBytes: memory.PeakBytes},
		},
		InitialModel: initialModel, InitialRun: initialRun, InitialEvaluation: initialEvaluation,
		Run: run, Evaluation: evaluation, Recipe: recipeID,
		ScratchDataset: construction.Dataset(), ScratchSplit: construction.SplitID(), Trained: trained,
	}, nil
}

type evaluationInputs struct {
	actions  []Action
	expected []Action
	tokens   [][][]int
}

// Prepare the complete denominator before device allocation; both evaluations
// consume the same token sequences rather than rebuilding their inputs.
func prepareEvaluationInputs(construction scratchmodel.Construction, corpus Corpus) (evaluationInputs, error) {
	actions, holdout := corpus.Actions(), corpus.Holdout()
	inputs := evaluationInputs{actions: actions}
	for _, record := range holdout {
		var candidates [][]int
		for _, action := range actions {
			document, err := corpus.CandidateDocument(record.Prompt, action)
			if err != nil {
				return evaluationInputs{}, err
			}
			tokens, err := construction.Tokens(document)
			if err != nil {
				return evaluationInputs{}, fmt.Errorf("controller training: held-out record %s: %w", record.ID, err)
			}
			candidates = append(candidates, tokens)
		}
		inputs.expected = append(inputs.expected, record.Action)
		inputs.tokens = append(inputs.tokens, candidates)
	}
	return inputs, nil
}

func evaluateSuite(trainer *scratchtrain.ResidentTrainer, inputs evaluationInputs) (SuiteMetrics, error) {
	var expectedLoss float64
	correctAction, correctModality := 0, 0
	for recordIndex, candidates := range inputs.tokens {
		bestLoss, bestIndex, expected := math.Inf(1), -1, 0.0
		for actionIndex, tokens := range candidates {
			loss, err := trainer.Evaluate(tokens)
			if err != nil {
				return SuiteMetrics{}, err
			}
			if inputs.actions[actionIndex] == inputs.expected[recordIndex] {
				expected = loss
			}
			if loss < bestLoss {
				bestLoss, bestIndex = loss, actionIndex
			}
		}
		expectedLoss += expected
		if bestIndex >= 0 {
			selected := inputs.actions[bestIndex]
			if selected == inputs.expected[recordIndex] {
				correctAction++
			}
			if selected.Modality == inputs.expected[recordIndex].Modality {
				correctModality++
			}
		}
	}
	count := float64(len(inputs.expected))
	return SuiteMetrics{
		Loss: expectedLoss / count, ActionAccuracy: float64(correctAction) / count,
		ModalityAccuracy: float64(correctModality) / count, ValidActionRate: 1,
	}, nil
}

func evaluationRecords(recipeID, datasetID, model artifact.ID, metrics SuiteMetrics) (runrecord.Run, runrecord.Evaluation, error) {
	run, err := runrecord.NewRun(recipeID, runrecord.OutcomeSucceeded, []artifact.ID{datasetID}, []artifact.ID{model}, "")
	if err != nil {
		return runrecord.Run{}, runrecord.Evaluation{}, err
	}
	evaluation, err := runrecord.NewEvaluation(recipeID, run.ID, datasetID, []runrecord.Metric{
		{Name: "action-accuracy", Value: metrics.ActionAccuracy, Direction: runrecord.DirectionMaximize},
		{Name: "causal-loss", Value: metrics.Loss, Direction: runrecord.DirectionMinimize},
		{Name: "modality-accuracy", Value: metrics.ModalityAccuracy, Direction: runrecord.DirectionMaximize},
		{Name: "valid-action-rate", Value: metrics.ValidActionRate, Direction: runrecord.DirectionMaximize},
	})
	return run, evaluation, err
}

func SeedEvidenceFrom(runs []SeedRun) []SeedEvidence {
	result := make([]SeedEvidence, len(runs))
	for index := range runs {
		result[index] = runs[index].Evidence
	}
	return slices.Clone(result)
}
