package main

import (
	"math"
	"os"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/longform"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestSelectedGuardAdmissionScope(t *testing.T) {
	repository := t.TempDir()
	store := overgodbtest.Create(t, repository)
	r := guardResult(t)
	manifest := testcheck.Must(artifact.NewManifest(artifact.KindModel, []artifact.Component{{Role: artifact.ComponentWeights, Name: "weights", Artifact: r.Inputs.Model}}))(t)
	info := testcheck.Must(os.Stat(r.ModelPath))(t)
	testcheck.Must(store.Commit(t.Context(), artifact.Batch{Key: "guard/selected-model",
		Artifacts: []artifact.Descriptor{{ID: r.Inputs.Model, Size: uint64(info.Size())}}, Manifests: []artifact.Manifest{manifest},
		Locations: []artifact.LocationEvent{{Action: artifact.LocationAdd, Location: artifact.Location{Artifact: r.Inputs.Model, Kind: artifact.LocationFile, Value: r.ModelPath}}},
	}))(t)
	testutil.PublishArtifact(t, store, r.Program.Profile)
	testutil.PublishArtifact(t, store, r.Program.Definition)
	definition := testcheck.Must(modelrecipe.InferenceWithModelDefinition(manifest.ID, r.Program.Profile, r.Program.Definition,
		recipe.PlacementHost, modelrecipe.DecodeSessionRequest, recipe.ResidencyHostReference))(t)
	testcheck.NoError(t, modelrecipetest.PublishActivation(t.Context(), store, "guard/activation", definition))
	r.Program.Model, r.Program.Recipe = manifest.ID, definition.ID
	r.Program.Placement, r.Program.Residency = recipe.PlacementHost, recipe.ResidencyHostReference
	accepted := testcheck.Must(longform.Publish(t.Context(), store, r.Inputs.Model, r))(t)
	opts := options{Repository: repository, Models: []string{r.ModelPath}}
	historyReads := 0
	readHistory := func() map[string]evaluation.BenchmarkSummary {
		historyReads++
		return map[string]evaluation.BenchmarkSummary{r.ModelPath: {Record: accepted, PromptTokensPerSecond: 999, DecodeTokensPerSecond: 999}}
	}
	if targets, err := readTargets(t.Context(), store, opts, readHistory); err != nil || historyReads != 1 || len(targets) != 1 || targets[0].shortBenchmark != accepted {
		t.Fatalf("measurement lost benchmark input: reads=%d error=%v", historyReads, err)
	}
	for _, rates := range []evaluation.BenchmarkSummary{
		{}, {Record: accepted, PromptTokensPerSecond: 1, DecodeTokensPerSecond: math.NaN()},
		{Record: accepted, PromptTokensPerSecond: math.Inf(1), DecodeTokensPerSecond: 1},
		{Record: accepted, PromptTokensPerSecond: 1, DecodeTokensPerSecond: -1},
		{PromptTokensPerSecond: 1, DecodeTokensPerSecond: 1},
	} {
		if targets, err := readTargets(t.Context(), store, opts, func() map[string]evaluation.BenchmarkSummary {
			return map[string]evaluation.BenchmarkSummary{r.ModelPath: rates}
		}); err == nil || len(targets) != 0 || !strings.Contains(err.Error(), "short-prompt benchmark") {
			t.Fatalf("invalid calibration reached measurement targets: %+v, %v", targets, err)
		}
	}
	testcheck.NoError(t, os.WriteFile(r.ModelPath, []byte("changed selected bytes"), 0o600))
	if _, err := readTargets(t.Context(), store, opts, readHistory); err == nil {
		t.Fatal("changed selected bytes accepted")
	}
	testcheck.NoError(t, os.WriteFile(r.ModelPath, []byte("guard"), 0o600))
	policy, found, err := modelrecipe.CatalogRuntimePolicy(recipe.TaskInference)
	if err != nil || !found {
		t.Fatalf("runtime policy = %t, %v", found, err)
	}
	wrong := testutil.ArtifactID(t, artifact.KindProfile, "wrong-selected-policy")
	testcheck.Must(store.Commit(t.Context(), artifact.Batch{Key: "guard/drifted-policy", Artifacts: []artifact.Descriptor{{ID: wrong}},
		Aliases: []artifact.AliasBinding{{Name: "runtime-policy/v2/" + definition.ID.String(), Target: wrong, Previous: &policy.ID}},
	}))(t)
	if _, err := readTargets(t.Context(), store, opts, readHistory); err == nil {
		t.Fatal("changed selected active policy accepted")
	}
}
