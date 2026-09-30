package trainingworkflow

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/bridgetrain"
	"overgo/internal/composition"
	"overgo/internal/hostoptimizer"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// seamSamples generates deterministic pseudo-activations wide enough to
// determine a linear map, using a fixed linear congruential stream so the
// fixture is reproducible without process randomness.
func seamSamples(samples, width int, seed uint64) [][]float64 {
	state := seed
	next := func() float64 {
		state = state*6364136223846793005 + 1442695040888963407
		return float64(int64(state>>33))/float64(1<<30) - 1
	}
	rows := make([][]float64, samples)
	for row := range rows {
		rows[row] = make([]float64, width)
		for column := range rows[row] {
			rows[row][column] = next()
		}
	}
	return rows
}

func applyLinearMap(source [][]float64, adapter [][]float64, noise func(row, column int) float64) [][]float64 {
	width := len(adapter[0])
	target := make([][]float64, len(source))
	for row := range source {
		target[row] = make([]float64, width)
		for column := range width {
			total := 0.0
			for k := range source[row] {
				total += source[row][k] * adapter[k][column]
			}
			target[row][column] = total + noise(row, column)
		}
	}
	return target
}

// TestCandidateRealizationFreezesDonorsTrainsAdapter pins the realization
// contract: the interface adapter initializes from the seam alignment's
// fitted linear map, only the adapter trains — the bridgetrain owner
// proves both frozen model descriptors stayed byte-identical while the
// adapter identity moved — the assembled composite references the frozen
// donor and target models in place as parents with no tensor duplicated,
// an identity-rung candidate assembles with no adapter and no training,
// and activations that no longer admit the enumerated rung refuse.
func TestCandidateRealizationFreezesDonorsTrainsAdapter(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	donorID := testutil.ArtifactID(t, artifact.KindModel, "realization-donor")
	targetID := testutil.ArtifactID(t, artifact.KindModel, "realization-target")
	trainingPolicy := testutil.ArtifactID(t, artifact.KindProfile, "realization-training-policy")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "realization-recipe")
	examples := []bridgetrain.Example{
		{SourceInput: []float32{1, 0}, TargetInput: []float32{1, 1}},
		{SourceInput: []float32{0, 1}, TargetInput: []float32{1, -1}},
	}
	datasetID, err := artifact.JSONID(artifact.KindDataset, examples)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "composition/realization-fixture",
		Artifacts: []artifact.Descriptor{
			{ID: donorID}, {ID: targetID}, {ID: trainingPolicy}, {ID: datasetID},
		},
	}); err != nil {
		t.Fatal(err)
	}
	source := seamSamples(24, 2, 41)
	linearTarget := applyLinearMap(source, [][]float64{{1, 0.5}, {-0.25, 1}}, func(row, column int) float64 {
		return 0.2 * seamSamples(24, 2, 733)[row][column]
	})
	candidate := composition.CompositionCandidate{
		Donor: donorID, Component: "blk.10.ffn_gate.weight",
		Seam: composition.SeamLayerBoundary, Adapter: composition.AdapterLinear, Residual: 0.2, Fitness: 0.8,
	}
	sourceForward := &testutil.FrozenForwardStub{Model: donorID}
	targetForward := &testutil.FrozenForwardStub{Model: targetID}
	training := RealizationTraining{
		Dataset: datasetID, Examples: examples,
		SourceForward: sourceForward, TargetForward: targetForward,
		TrainingPolicy: trainingPolicy,
		Config: hostoptimizer.Config{
			BaseLearningRate: 0.05, Momentum: 0.9, Steps: 4,
			Schedule: hostoptimizer.ScheduleConstant,
		},
		Epochs: 2,
	}
	assembly := RealizationAssembly{Architecture: "llama", Recipe: recipeID}

	realized, err := RealizeCompositionCandidate(
		ctx, store, candidate, targetID,
		RealizationSeamActivations{Source: source, Target: linearTarget},
		training, assembly,
	)
	if err != nil {
		t.Fatal(err)
	}
	if realized.Adapter.BridgeBefore == realized.Adapter.BridgeAfter {
		t.Fatal("the adapter did not train")
	}
	if realized.Adapter.Source.Before != realized.Adapter.Source.After ||
		realized.Adapter.Target.Before != realized.Adapter.Target.After {
		t.Fatalf("a frozen model descriptor moved: %+v", realized.Adapter)
	}
	if sourceForward.Calls != len(examples)*training.Epochs || targetForward.Calls != sourceForward.Calls {
		t.Fatalf("frozen forwards ran %d/%d times", sourceForward.Calls, targetForward.Calls)
	}
	composite := realized.Composite
	if !composite.ID.Valid() || len(composite.Parents) != 2 ||
		composite.Parents[0] != targetID || composite.Parents[1] != donorID {
		t.Fatalf("composite parents = %+v", composite)
	}
	if composite.Adapter != realized.Adapter.BridgeAfter || len(composite.Components) != 0 {
		t.Fatalf("composite duplicated tensors or lost its adapter: %+v", composite)
	}

	identical := applyLinearMap(source, [][]float64{{1, 0.5}, {-0.25, 1}}, func(int, int) float64 { return 0 })
	graft := candidate
	graft.Adapter = composition.AdapterIdentity
	direct, err := RealizeCompositionCandidate(
		ctx, store, graft, targetID,
		RealizationSeamActivations{Source: source, Target: identical},
		RealizationTraining{}, assembly,
	)
	if err != nil {
		t.Fatal(err)
	}
	if direct.Composite.Adapter.Valid() || direct.Adapter.BridgeAfter.Valid() {
		t.Fatalf("identity graft trained an adapter: %+v", direct)
	}
	if sourceForward.Calls != len(examples)*training.Epochs {
		t.Fatal("identity graft executed frozen forwards")
	}

	if _, err := RealizeCompositionCandidate(
		ctx, store, graft, targetID,
		RealizationSeamActivations{Source: source, Target: linearTarget},
		RealizationTraining{}, assembly,
	); err == nil || !strings.Contains(err.Error(), "measurement drifted") {
		t.Fatalf("drifted activations realized: %v", err)
	}
}
