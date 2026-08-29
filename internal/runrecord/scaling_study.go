package runrecord

import (
	"cmp"
	"errors"
	"slices"

	"overgo/internal/artifact"
)

const (
	scalingStudyMediaType = "application/vnd.overgo.scaling-study+json"
	scalingStudySchema    = "overgo/scaling-study/v1"
)

type scalingRungAttempt struct {
	Ordinal    uint32      `json:"ordinal"`
	Run        artifact.ID `json:"run"`
	PriorRun   artifact.ID `json:"prior_run,omitzero"`
	Outcome    Outcome     `json:"outcome"`
	Terminal   artifact.ID `json:"terminal"`
	Accounting artifact.ID `json:"accounting,omitzero"`
}

type scalingRungFacts struct {
	Planned   bool `json:"planned"`
	Attempted bool `json:"attempted"`
	Failed    bool `json:"failed"`
	Retried   bool `json:"retried"`
	Evaluated bool `json:"evaluated"`
}

type scalingStudyRung struct {
	Ordinal            uint32               `json:"ordinal"`
	Config             artifact.ID          `json:"config"`
	Model              artifact.ID          `json:"model"`
	Seed               artifact.ID          `json:"seed"`
	Dataset            artifact.ID          `json:"dataset"`
	Split              artifact.ID          `json:"split"`
	DataOrder          artifact.ID          `json:"data_order"`
	StartingCheckpoint artifact.ID          `json:"starting_checkpoint"`
	Hardware           artifact.ID          `json:"hardware"`
	Environment        artifact.ID          `json:"environment"`
	Code               artifact.ID          `json:"code"`
	Evaluator          artifact.ID          `json:"evaluator"`
	Attempts           []scalingRungAttempt `json:"attempts,omitempty"`
	EvaluationRun      artifact.ID          `json:"evaluation_run,omitzero"`
	Evaluation         artifact.ID          `json:"evaluation,omitzero"`
	scalingRungFacts
}

type scalingStudy struct {
	ID       artifact.ID        `json:"-"`
	Version  uint16             `json:"version"`
	Protocol artifact.ID        `json:"protocol"`
	Rungs    []scalingStudyRung `json:"rungs"`
}

var scalingStudyCodec = artifact.JSONDocumentCodec(
	"scaling study", artifact.KindEvidence, scalingStudyMediaType, scalingStudySchema,
	canonicalizeScalingStudy,
	func(value scalingStudy) artifact.ID { return value.ID },
	func(value *scalingStudy, id artifact.ID) { value.ID = id },
	func(value scalingStudy) scalingStudy {
		value.Rungs = cloneScalingStudyRungs(value.Rungs)
		return value
	},
)

var scalingStudyLineage = func(value scalingStudy) []artifact.Lineage {
	parents := []artifact.ID{value.Protocol}
	for _, rung := range value.Rungs {
		parents = append(parents,
			rung.Config, rung.Model, rung.Seed, rung.Dataset, rung.Split, rung.DataOrder, rung.StartingCheckpoint,
			rung.Hardware, rung.Environment, rung.Code, rung.Evaluator,
		)
		for _, attempt := range rung.Attempts {
			parents = append(parents, attempt.Run, attempt.PriorRun, attempt.Terminal, attempt.Accounting)
		}
		parents = append(parents, rung.EvaluationRun, rung.Evaluation)
	}
	valid := parents[:0]
	for _, parent := range parents {
		if parent.Valid() {
			valid = append(valid, parent)
		}
	}
	return artifact.DependencyLineage(value.ID, uniqueScalingAuthorities(valid)...)
}

func canonicalizeScalingStudy(value *scalingStudy) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Protocol.Kind() != artifact.KindProfile || len(value.Rungs) < 2 {
		return errors.New("run record: scaling study requires a protocol and multiple explicit rungs")
	}
	value.Rungs = cloneScalingStudyRungs(value.Rungs)
	slices.SortFunc(value.Rungs, func(left, right scalingStudyRung) int {
		return cmp.Compare(left.Ordinal, right.Ordinal)
	})
	type rungIdentity struct {
		config, model, seed, dataset, split, dataOrder, checkpoint artifact.ID
		hardware, environment, code, evaluator                     artifact.ID
	}
	seenRungs := make(map[rungIdentity]bool, len(value.Rungs))
	seenRuns := make(map[artifact.ID]bool)
	seenTerminals := make(map[artifact.ID]bool)
	seenAccounting := make(map[artifact.ID]bool)
	seenEvaluations := make(map[artifact.ID]bool)
	for index := range value.Rungs {
		rung := &value.Rungs[index]
		identity := rungIdentity{
			config: rung.Config, model: rung.Model, seed: rung.Seed, dataset: rung.Dataset, split: rung.Split,
			dataOrder: rung.DataOrder, checkpoint: rung.StartingCheckpoint, hardware: rung.Hardware,
			environment: rung.Environment, code: rung.Code, evaluator: rung.Evaluator,
		}
		if rung.Ordinal != uint32(index) || rung.Config.Kind() != artifact.KindRecipe || rung.Model.Kind() != artifact.KindModel ||
			rung.Seed.Kind() != artifact.KindEvidence || rung.Dataset.Kind() != artifact.KindDataset ||
			rung.Split.Kind() != artifact.KindDatasetShard || rung.DataOrder.Kind() != artifact.KindEvidence ||
			rung.StartingCheckpoint.Kind() != artifact.KindCheckpoint || rung.Hardware.Kind() != artifact.KindEvidence ||
			rung.Environment.Kind() != artifact.KindEvidence || rung.Code.Kind() != artifact.KindEvidence ||
			rung.Evaluator.Kind() != artifact.KindProfile || seenRungs[identity] {
			return errors.New("run record: invalid or duplicate scaling rung authority")
		}
		seenRungs[identity] = true
		slices.SortFunc(rung.Attempts, func(left, right scalingRungAttempt) int {
			return cmp.Compare(left.Ordinal, right.Ordinal)
		})
		failed := false
		for attemptIndex, attempt := range rung.Attempts {
			first := attemptIndex == 0
			if attempt.Ordinal != uint32(attemptIndex) || attempt.Run.Kind() != artifact.KindRun || seenRuns[attempt.Run] ||
				!ValidOutcome(attempt.Outcome) || attempt.Terminal.Kind() != artifact.KindEvidence || seenTerminals[attempt.Terminal] ||
				first && attempt.PriorRun.Valid() || !first && attempt.PriorRun != rung.Attempts[attemptIndex-1].Run ||
				!first && rung.Attempts[attemptIndex-1].Outcome == OutcomeSucceeded ||
				attempt.Accounting.Valid() && (attempt.Accounting.Kind() != artifact.KindEvidence || seenAccounting[attempt.Accounting]) {
				return errors.New("run record: invalid scaling rung attempt or retry chain")
			}
			seenRuns[attempt.Run] = true
			seenTerminals[attempt.Terminal] = true
			if attempt.Accounting.Valid() {
				seenAccounting[attempt.Accounting] = true
			}
			failed = failed || attempt.Outcome == OutcomeFailed
		}
		evaluated := rung.Evaluation.Valid() || rung.EvaluationRun.Valid()
		if evaluated && (rung.Evaluation.Kind() != artifact.KindEvidence || rung.EvaluationRun.Kind() != artifact.KindRun ||
			len(rung.Attempts) == 0 || rung.EvaluationRun != rung.Attempts[len(rung.Attempts)-1].Run ||
			rung.Attempts[len(rung.Attempts)-1].Outcome != OutcomeSucceeded ||
			!rung.Attempts[len(rung.Attempts)-1].Accounting.Valid() || seenEvaluations[rung.Evaluation]) {
			return errors.New("run record: scaling rung evaluation differs from its successful accounted attempt")
		}
		if evaluated {
			seenEvaluations[rung.Evaluation] = true
		}
		facts := scalingRungFacts{
			Planned: true, Attempted: len(rung.Attempts) != 0, Failed: failed,
			Retried: len(rung.Attempts) > 1, Evaluated: evaluated,
		}
		if rung.scalingRungFacts != facts {
			return errors.New("run record: scaling rung facts differ from retained attempts and evaluation")
		}
	}
	return nil
}

func cloneScalingStudyRungs(rungs []scalingStudyRung) []scalingStudyRung {
	copy := slices.Clone(rungs)
	for index := range copy {
		copy[index].Attempts = slices.Clone(copy[index].Attempts)
	}
	return copy
}
