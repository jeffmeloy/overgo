package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestRouterObservationCoverageAndRetentionFailClosed(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	run := id(artifact.KindRun, "coverage run")
	base := MoERouterObservation{
		Run: run, Model: id(artifact.KindModel, "coverage model"),
		Dataset: id(artifact.KindDataset, "coverage dataset"), Split: id(artifact.KindDatasetShard, "coverage split"),
		Recipe: id(artifact.KindRecipe, "coverage recipe"), Code: id(artifact.KindEvidence, "coverage code"),
		Checkpoint: id(artifact.KindCheckpoint, "coverage checkpoint"), Policy: id(artifact.KindRecipe, "coverage policy"),
		Rows: 1, Experts: 2, TopK: 1, Selections: []uint32{0}, CombineWeights: []float32{1},
		Accepted: []bool{true}, Margins: []MoERouterMargin{{Observed: true, Value: 1}},
	}
	var observations []MoERouterObservation
	for step := range uint64(2) {
		for _, layer := range []uint32{1, 3} {
			value := base
			value.Step, value.Layer = step+5, layer
			identified, err := NewMoERouterObservation(value)
			if err != nil {
				t.Fatal(err)
			}
			observations = append(observations, identified)
		}
	}
	coverage, err := NewMoERouterObservationCoverage(observations, 5, 2, []uint32{3, 1})
	if err != nil {
		t.Fatal(err)
	}
	if coverage.Run != run || coverage.FirstStep != 5 || coverage.Steps != 2 ||
		len(coverage.Observations) != len(observations) || coverage.ObservationBytes == 0 {
		t.Fatalf("coverage = %+v", coverage)
	}
	if _, err := coverage.Content(); err != nil {
		t.Fatal(err)
	}

	if _, err := NewMoERouterObservationCoverage(observations[:len(observations)-1], 5, 2, []uint32{1, 3}); err == nil {
		t.Fatal("missing layer observation accepted")
	}
	duplicate := append([]MoERouterObservation(nil), observations...)
	duplicate[len(duplicate)-1] = duplicate[0]
	if _, err := NewMoERouterObservationCoverage(duplicate, 5, 2, []uint32{1, 3}); err == nil {
		t.Fatal("duplicate observation accepted")
	}
	scopeChanges := []struct {
		name   string
		mutate func(*MoERouterObservation)
	}{
		{name: "run", mutate: func(value *MoERouterObservation) { value.Run = id(artifact.KindRun, "other run") }},
		{name: "model", mutate: func(value *MoERouterObservation) { value.Model = id(artifact.KindModel, "other model") }},
		{name: "dataset", mutate: func(value *MoERouterObservation) { value.Dataset = id(artifact.KindDataset, "other dataset") }},
		{name: "split", mutate: func(value *MoERouterObservation) { value.Split = id(artifact.KindDatasetShard, "other split") }},
		{name: "recipe", mutate: func(value *MoERouterObservation) { value.Recipe = id(artifact.KindRecipe, "other recipe") }},
		{name: "code", mutate: func(value *MoERouterObservation) { value.Code = id(artifact.KindEvidence, "other code") }},
		{name: "checkpoint", mutate: func(value *MoERouterObservation) { value.Checkpoint = id(artifact.KindCheckpoint, "other checkpoint") }},
		{name: "policy", mutate: func(value *MoERouterObservation) { value.Policy = id(artifact.KindRecipe, "other policy") }},
		{name: "rows", mutate: func(value *MoERouterObservation) {
			value.Selections = []uint32{0, 1}
			value.CombineWeights = []float32{1, 1}
			value.Accepted = []bool{true, true}
			value.Margins = []MoERouterMargin{{Observed: true, Value: 1}, {Observed: true, Value: 1}}
		}},
		{name: "experts", mutate: func(value *MoERouterObservation) { value.Experts = 3 }},
		{name: "top-k", mutate: func(value *MoERouterObservation) {
			value.TopK = 2
			value.Selections = []uint32{0, 1}
			value.CombineWeights = []float32{0.5, 0.5}
			value.Accepted = []bool{true, true}
		}},
	}
	for _, testCase := range scopeChanges {
		t.Run("mixed-"+testCase.name, func(t *testing.T) {
			changed := append([]MoERouterObservation(nil), observations...)
			value := base
			value.Step, value.Layer = changed[1].Step, changed[1].Layer
			testCase.mutate(&value)
			identified, err := NewMoERouterObservation(value)
			if err != nil {
				t.Fatal(err)
			}
			changed[1] = identified
			if _, err := NewMoERouterObservationCoverage(changed, 5, 2, []uint32{1, 3}); err == nil {
				t.Fatal("mixed observation scope accepted")
			}
		})
	}
	if err := validateMoERouterRetentionBytes(artifact.MaxContentBytes, 1); err == nil {
		t.Fatal("retention beyond the repository content bound accepted")
	}
}
