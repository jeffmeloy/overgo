package inference

import (
	"testing"

	"overgo/internal/model"
)

func TestCompiledMultiHeadMTPRequiresCompleteCatalog(t *testing.T) {
	for _, fixture := range []struct {
		architecture string
		weights      model.Weights
		truncate     func(*model.Weights)
	}{
		{architecture: "step35", weights: model.Weights{AppendedMultiCarryDraft: make([]model.AppendedDraftWeights, 2)},
			truncate: func(weights *model.Weights) { weights.AppendedMultiCarryDraft = weights.AppendedMultiCarryDraft[:1] }},
		{architecture: "hy_v3", weights: model.Weights{AppendedMultiDraft: make([]model.AppendedDraftWeights, 2)},
			truncate: func(weights *model.Weights) { weights.AppendedMultiDraft = weights.AppendedMultiDraft[:1] }},
	} {
		runner := attachFixtureProgram(&Runner{preparedModel: preparedModel{
			spec: model.Spec{CommonSpec: model.CommonSpec{
				Architecture: fixture.architecture, NextNPredictLayers: 2,
			}},
			weights: fixture.weights,
		}})
		if _, err := runner.multiHeadMTP(); err != nil {
			t.Fatalf("%s complete catalog: %v", fixture.architecture, err)
		}
		fixture.truncate(&runner.weights)
		if _, err := runner.multiHeadMTP(); err == nil {
			t.Fatalf("%s incomplete catalog was accepted", fixture.architecture)
		}
	}
}
