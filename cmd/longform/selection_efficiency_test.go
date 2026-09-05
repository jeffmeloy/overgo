package main

import (
	"io"
	"os"
	"path/filepath"
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
	corpus := filepath.Join(t.TempDir(), "corpus.txt")
	if err := os.WriteFile(corpus, []byte("fixed corpus"), 0o600); err != nil {
		t.Fatal(err)
	}
	opts := options{Repository: repository, Models: []string{r.ModelPath}, ValidateBaselines: true, Corpus: corpus, Baselines: []string{accepted.String()}}
	head, sequence := store.Head()
	historyReads := 0
	readHistory := func() map[string]evaluation.BenchmarkSummary {
		historyReads++
		return map[string]evaluation.BenchmarkSummary{r.ModelPath: {DecodeTokensPerSecond: 999}}
	}
	targets, err := readTargets(t.Context(), store, opts, readHistory)
	if err != nil || len(targets) != 1 {
		t.Fatalf("selected targets = %+v, %v", targets, err)
	}
	if historyReads != 0 || targets[0].short != (longform.ShortRates{}) {
		t.Errorf("read-only admission read benchmark history %d times: %+v", historyReads, targets[0].short)
	}
	coverageOptions := opts
	coverageOptions.ValidateBaselines, coverageOptions.GuardCoverage = false, true
	if inventory, err := readTargets(t.Context(), store, coverageOptions, readHistory); err != nil || len(inventory) != 1 || historyReads != 0 {
		t.Fatalf("coverage inventory read benchmark history: models=%d reads=%d error=%v", len(inventory), historyReads, err)
	}
	if err := bindBaselines(t.Context(), opts, targets); err != nil {
		t.Fatal(err)
	}
	if err := validateSelectedBaselines(io.Discard, targets, r.Surface); err != nil {
		t.Fatal(err)
	}
	if _, err := listTargets(t.Context(), opts); err != nil {
		t.Fatalf("read-only command store opening failed: %v", err)
	}
	if after, current := store.Head(); after != head || current != sequence {
		t.Fatal("selected admission published state")
	}
	all := opts
	all.All, all.Models = true, nil
	if full, err := readTargets(t.Context(), store, all, readHistory); err != nil || len(full) != 1 || historyReads != 0 {
		t.Fatalf("full read-only catalog = %+v, history reads=%d, error=%v", full, historyReads, err)
	}
	opts.ValidateBaselines = false
	if _, err := readTargets(t.Context(), store, opts, readHistory); err != nil || historyReads != 1 {
		t.Fatalf("measurement lost benchmark input: reads=%d error=%v", historyReads, err)
	}
	opts.ValidateBaselines = true
	if err := os.WriteFile(r.ModelPath, []byte("changed selected bytes"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := listTargets(t.Context(), opts); err == nil {
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
	if _, err := listTargets(t.Context(), opts); err == nil {
		t.Fatal("changed selected active policy accepted")
	}
}
