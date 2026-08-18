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
	trainingTraceVersion   uint16 = 1
	trainingTraceMediaType        = "application/vnd.overgo.training-trace+json"
	trainingTraceSchema           = "overgo/training-trace/v1"
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
			Policy: body.Policy, Reference: body.Reference, Observations: body.Observations,
		}
		return nil
	},
	Encode: func(value TrainingTrace) ([]byte, error) {
		return json.Marshal(trainingTraceBody{
			Version: value.Version, Run: value.Run, Recipe: value.Recipe, Dataset: value.Dataset,
			Policy: value.Policy, Reference: value.Reference, Observations: value.Observations,
		})
	},
	Canonicalize: canonicalizeTrainingTrace,
	Clone: func(value TrainingTrace) TrainingTrace {
		value.Observations = slices.Clone(value.Observations)
		return value
	},
	Identity:    func(value TrainingTrace) artifact.ID { return value.ID },
	SetIdentity: func(value *TrainingTrace, id artifact.ID) { value.ID = id },
}

type TrainingTrace struct {
	Version      uint16
	ID           artifact.ID
	Run          artifact.ID
	Recipe       artifact.ID
	Dataset      artifact.ID
	Policy       artifact.ID
	Reference    artifact.ID
	Observations []trainingprogram.DPOObservation
}

type trainingTraceBody struct {
	Version      uint16                           `json:"version"`
	Run          artifact.ID                      `json:"run"`
	Recipe       artifact.ID                      `json:"recipe"`
	Dataset      artifact.ID                      `json:"dataset"`
	Policy       artifact.ID                      `json:"policy"`
	Reference    artifact.ID                      `json:"reference"`
	Observations []trainingprogram.DPOObservation `json:"observations"`
}

func NewDPOTrace(run, recipeID, dataset, policy, reference artifact.ID, observations []trainingprogram.DPOObservation) (TrainingTrace, error) {
	return trainingTraceCodec.New(TrainingTrace{
		Version: trainingTraceVersion, Run: run, Recipe: recipeID, Dataset: dataset,
		Policy: policy, Reference: reference, Observations: slices.Clone(observations),
	})
}

func (trace TrainingTrace) Content() (artifact.Content, error) {
	return trainingTraceCodec.Content(trace)
}

func (trace TrainingTrace) Lineage() []artifact.Lineage {
	return []artifact.Lineage{
		{Child: trace.ID, Parent: trace.Run, Relation: artifact.RelationProducedBy},
		{Child: trace.ID, Parent: trace.Recipe, Relation: artifact.RelationDependsOn},
		{Child: trace.ID, Parent: trace.Dataset, Relation: artifact.RelationDependsOn},
		{Child: trace.ID, Parent: trace.Policy, Relation: artifact.RelationDependsOn},
		{Child: trace.ID, Parent: trace.Reference, Relation: artifact.RelationDependsOn},
	}
}

func canonicalizeTrainingTrace(trace *TrainingTrace) error {
	if trace == nil || trace.Version != trainingTraceVersion || trace.Run.Kind() != artifact.KindRun ||
		trace.Recipe.Kind() != artifact.KindRecipe || trace.Dataset.Kind() != artifact.KindDataset ||
		trace.Policy.Kind() != artifact.KindModel || trace.Reference.Kind() != artifact.KindModel ||
		trace.Policy == trace.Reference || len(trace.Observations) == 0 {
		return errors.New("training trace: invalid authority")
	}
	var prior uint64
	for _, observation := range trace.Observations {
		if !observation.Valid() || observation.Step <= prior {
			return errors.New("training trace: invalid or unordered observation")
		}
		prior = observation.Step
	}
	return nil
}
