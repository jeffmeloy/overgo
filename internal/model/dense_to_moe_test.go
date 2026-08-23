package model

import (
	"math"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestDenseToMoERouterInitialization(t *testing.T) {
	first := testutil.ArtifactID(t, artifact.KindModel, "router first expert")
	second := testutil.ArtifactID(t, artifact.KindModel, "router second expert")
	evidence := denseToMoERouterEvidenceFixture(t, first, second)
	plan, err := CompileDenseToMoERouter(evidence)
	if err != nil {
		t.Fatal(err)
	}
	want := float32(1 / math.Sqrt2)
	if plan.TopK != 1 || plan.Width != 2 || len(plan.Layers) != 1 ||
		plan.Layers[0].Weights[0] != want || plan.Layers[0].Weights[1] != -want ||
		plan.Layers[0].Weights[2] != -want || plan.Layers[0].Weights[3] != want {
		t.Fatalf("derived router = %+v", plan)
	}
	if err := plan.ValidateIdentity(); err != nil {
		t.Fatal(err)
	}
	bad := evidence
	bad.HeldOut = cloneDenseToMoESamples(evidence.HeldOut)
	bad.HeldOut[0].Hidden = []float32{0, 3}
	bad, err = NewDenseToMoERouterEvidence(bad)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := CompileDenseToMoERouter(bad); err == nil {
		t.Fatal("router screen accepted incorrect held-out routing")
	}
}

func denseToMoERouterEvidenceFixture(t *testing.T, first, second artifact.ID) DenseToMoERouterEvidence {
	t.Helper()
	value, err := NewDenseToMoERouterEvidence(DenseToMoERouterEvidence{
		TrainingDataset: testutil.ArtifactID(t, artifact.KindDataset, "router training dataset"),
		HeldOutDataset:  testutil.ArtifactID(t, artifact.KindDataset, "router held-out dataset"),
		TrainingRun:     testutil.ArtifactID(t, artifact.KindRun, "router training run"),
		HeldOutRun:      testutil.ArtifactID(t, artifact.KindRun, "router held-out run"),
		Experts:         []artifact.ID{first, second},
		Training: []DenseToMoERouterSample{
			{Layer: 0, Hidden: []float32{2, 0}, Experts: []artifact.ID{first}},
			{Layer: 0, Hidden: []float32{0, 2}, Experts: []artifact.ID{second}},
		},
		HeldOut: []DenseToMoERouterSample{
			{Layer: 0, Hidden: []float32{3, 0}, Experts: []artifact.ID{first}},
			{Layer: 0, Hidden: []float32{0, 3}, Experts: []artifact.ID{second}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return value
}
