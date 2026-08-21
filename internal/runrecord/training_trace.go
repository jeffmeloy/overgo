package runrecord

import (
	"encoding/json"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
	"overgo/internal/trainingprogram"
)

const (
	trainingTraceMediaType = "application/vnd.overgo.training-trace+json"
	trainingTraceSchema    = "overgo/training-trace/v2"
)

var trainingTraceCodec = artifact.DocumentCodec[TrainingTrace]{
	Name: "training trace",
	Contract: artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: trainingTraceMediaType, Schema: trainingTraceSchema,
	},
	Decode: func(data []byte, value *TrainingTrace) error {
		var body trainingTraceBody
		if err := strictjson.DecodeBytes(data, &body); err != nil {
			return err
		}
		*value = TrainingTrace{
			Version: body.Version, Run: body.Run, Recipe: body.Recipe, Dataset: body.Dataset,
			Objective: body.Objective, Policy: body.Policy,
			Evaluators: body.Evaluators, DPO: body.DPO, GRPO: body.GRPO,
		}
		if body.Reference != nil {
			value.Reference = *body.Reference
		}
		return nil
	},
	Encode: func(value TrainingTrace) ([]byte, error) {
		var reference *artifact.ID
		if value.Reference.Valid() {
			reference = &value.Reference
		}
		return json.Marshal(trainingTraceBody{
			Version: value.Version, Run: value.Run, Recipe: value.Recipe, Dataset: value.Dataset,
			Objective: value.Objective, Policy: value.Policy, Reference: reference,
			Evaluators: value.Evaluators, DPO: value.DPO, GRPO: value.GRPO,
		})
	},
	Canonicalize: canonicalizeTrainingTrace,
	Clone: func(value TrainingTrace) TrainingTrace {
		value.Evaluators = slices.Clone(value.Evaluators)
		value.DPO = slices.Clone(value.DPO)
		value.GRPO = slices.Clone(value.GRPO)
		return value
	},
	Identity:    func(value TrainingTrace) artifact.ID { return value.ID },
	SetIdentity: func(value *TrainingTrace, id artifact.ID) { value.ID = id },
}

type TrainingTrace struct {
	Version    uint16
	ID         artifact.ID
	Run        artifact.ID
	Recipe     artifact.ID
	Dataset    artifact.ID
	Objective  trainingprogram.ObjectiveKind
	Policy     artifact.ID
	Reference  artifact.ID
	Evaluators []artifact.ID
	DPO        []trainingprogram.DPOObservation
	GRPO       []trainingprogram.GRPOObservation
}

type trainingTraceBody struct {
	Version    uint16                            `json:"version"`
	Run        artifact.ID                       `json:"run"`
	Recipe     artifact.ID                       `json:"recipe"`
	Dataset    artifact.ID                       `json:"dataset"`
	Objective  trainingprogram.ObjectiveKind     `json:"objective"`
	Policy     artifact.ID                       `json:"policy"`
	Reference  *artifact.ID                      `json:"reference,omitempty"`
	Evaluators []artifact.ID                     `json:"evaluators,omitempty"`
	DPO        []trainingprogram.DPOObservation  `json:"dpo,omitempty"`
	GRPO       []trainingprogram.GRPOObservation `json:"grpo,omitempty"`
}

func NewTrainingTrace(run, recipeID, dataset, policy, reference artifact.ID, evaluators []artifact.ID, dpo []trainingprogram.DPOObservation, grpo []trainingprogram.GRPOObservation) (TrainingTrace, error) {
	objective := trainingprogram.ObjectiveDPO
	if len(grpo) > 0 {
		objective = trainingprogram.ObjectiveGRPO
	}
	return trainingTraceCodec.New(TrainingTrace{
		Version: artifact.SecondDocumentVersion, Run: run, Recipe: recipeID, Dataset: dataset,
		Objective: objective, Policy: policy, Reference: reference, Evaluators: slices.Clone(evaluators),
		DPO: slices.Clone(dpo), GRPO: slices.Clone(grpo),
	})
}

func (trace TrainingTrace) Content() (artifact.Content, error) {
	return trainingTraceCodec.Content(trace)
}

func (trace TrainingTrace) Lineage() []artifact.Lineage {
	lineage := []artifact.Lineage{
		{Child: trace.ID, Parent: trace.Run, Relation: artifact.RelationProducedBy},
		{Child: trace.ID, Parent: trace.Recipe, Relation: artifact.RelationDependsOn},
		{Child: trace.ID, Parent: trace.Dataset, Relation: artifact.RelationDependsOn},
		{Child: trace.ID, Parent: trace.Policy, Relation: artifact.RelationDependsOn},
	}
	for _, parent := range append([]artifact.ID{trace.Reference}, trace.Evaluators...) {
		if parent.Valid() {
			lineage = append(lineage, artifact.Lineage{Child: trace.ID, Parent: parent, Relation: artifact.RelationDependsOn})
		}
	}
	return lineage
}

func canonicalizeTrainingTrace(trace *TrainingTrace) error {
	if trace == nil || trace.Version != artifact.SecondDocumentVersion || trace.Run.Kind() != artifact.KindRun ||
		trace.Recipe.Kind() != artifact.KindRecipe || trace.Dataset.Kind() != artifact.KindDataset ||
		trace.Policy.Kind() != artifact.KindModel {
		return errors.New("training trace: invalid authority")
	}
	var prior uint64
	switch trace.Objective {
	case trainingprogram.ObjectiveDPO:
		if trace.Reference.Kind() != artifact.KindModel || trace.Policy == trace.Reference || len(trace.DPO) == 0 || len(trace.GRPO) != 0 || len(trace.Evaluators) != 0 {
			return errors.New("training trace: invalid DPO authority")
		}
		for _, observation := range trace.DPO {
			if !observation.Valid() || observation.Step <= prior {
				return errors.New("training trace: invalid or unordered DPO observation")
			}
			prior = observation.Step
		}
	case trainingprogram.ObjectiveGRPO:
		if trace.Reference.Valid() || len(trace.DPO) != 0 || len(trace.GRPO) == 0 || len(trace.Evaluators) == 0 {
			return errors.New("training trace: invalid GRPO authority")
		}
		evaluatorSet := make(map[artifact.ID]struct{}, len(trace.Evaluators))
		for _, evaluator := range trace.Evaluators {
			if evaluator.Kind() != artifact.KindEvidence {
				return errors.New("training trace: invalid GRPO evaluator")
			}
			evaluatorSet[evaluator] = struct{}{}
		}
		if len(evaluatorSet) != len(trace.Evaluators) {
			return errors.New("training trace: duplicate GRPO evaluator")
		}
		for _, observation := range trace.GRPO {
			if !observation.Valid() || observation.Step <= prior {
				return errors.New("training trace: invalid or unordered GRPO observation")
			}
			if _, ok := evaluatorSet[observation.Evaluator]; !ok {
				return errors.New("training trace: GRPO observation evaluator differs")
			}
			prior = observation.Step
		}
	default:
		return errors.New("training trace: invalid objective")
	}
	return nil
}
