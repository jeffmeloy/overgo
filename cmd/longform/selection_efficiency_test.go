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
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestSelectedGuardAdmissionScope(t *testing.T) {
	repository := t.TempDir()
	store, err := overgodb.Open(repository)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	r := guardResult(t)
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{Role: artifact.ComponentWeights, Name: "weights", Artifact: r.Inputs.Model}})
	if err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(r.ModelPath)
	if err != nil {
		t.Fatal(err)
	}
	_, err = store.Commit(t.Context(), artifact.Batch{Key: "guard/selected-model",
		Artifacts: []artifact.Descriptor{{ID: r.Inputs.Model, Size: uint64(info.Size())}}, Manifests: []artifact.Manifest{manifest},
		Locations: []artifact.LocationEvent{{Action: artifact.LocationAdd, Location: artifact.Location{Artifact: r.Inputs.Model, Kind: artifact.LocationFile, Value: r.ModelPath}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	testutil.PublishArtifact(t, store, r.Program.Profile)
	testutil.PublishArtifact(t, store, r.Program.Definition)
	definition, err := modelrecipe.InferenceWithModelDefinition(manifest.ID, r.Program.Profile, r.Program.Definition,
		recipe.PlacementHost, modelrecipe.DecodeSessionRequest, recipe.ResidencyHostReference)
	if err != nil {
		t.Fatal(err)
	}
	if err := modelrecipetest.PublishActivation(t.Context(), store, "guard/activation", definition); err != nil {
		t.Fatal(err)
	}
	r.Program.Model, r.Program.Recipe = manifest.ID, definition.ID
	r.Program.Placement, r.Program.Residency = recipe.PlacementHost, recipe.ResidencyHostReference
	accepted, err := longform.Publish(t.Context(), store, r.Inputs.Model, r)
	if err != nil {
		t.Fatal(err)
	}
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
	if err := os.WriteFile(r.ModelPath, []byte("changed selected bytes"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := readTargets(t.Context(), store, opts, readHistory); err == nil {
		t.Fatal("changed selected bytes accepted")
	}
	if err := os.WriteFile(r.ModelPath, []byte("guard"), 0o600); err != nil {
		t.Fatal(err)
	}
	policy, found, err := modelrecipe.CatalogRuntimePolicy(recipe.TaskInference)
	if err != nil || !found {
		t.Fatalf("runtime policy = %t, %v", found, err)
	}
	wrong := testutil.ArtifactID(t, artifact.KindProfile, "wrong-selected-policy")
	_, err = store.Commit(t.Context(), artifact.Batch{Key: "guard/drifted-policy", Artifacts: []artifact.Descriptor{{ID: wrong}},
		Aliases: []artifact.AliasBinding{{Name: "runtime-policy/v2/" + definition.ID.String(), Target: wrong, Previous: &policy.ID}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := readTargets(t.Context(), store, opts, readHistory); err == nil {
		t.Fatal("changed selected active policy accepted")
	}
}
