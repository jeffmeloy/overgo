//go:build windows && modeltest

package inference_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/servingtest"
	"overgo/internal/testutil"
)

const (
	carbonServingModelIdentity  = "model:sha256:08d12f90b68fe93f3c39da1bb9f4592a13085c2740716166d2baab708793d3e8"
	carbonServingGoldenIdentity = "evidence:sha256:964ca1d714233666e6f7ca66223036057ced4dce704b0d391b2d1fdac8c96e59"
)

func TestCarbonServingGolden(t *testing.T) {
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	modelPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "Carbon-500M-f16-ropefix.gguf")
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("UNAVAILABLE: converted Carbon artifact absent at %s", modelPath)
	}
	var golden evaluation.ExactSuite
	fixture := testutil.FixturePath(t, "carbon_serving_golden.json")
	requireServingFileIdentity(t, modelPath, artifact.KindModel, carbonServingModelIdentity)
	requireServingFileIdentity(t, fixture, artifact.KindEvidence, carbonServingGoldenIdentity)
	if err := jsonfile.Decode(fixture, &golden); err != nil {
		t.Fatal(err)
	}
	if golden.Schema != "carbon_serving_golden/v1" || len(golden.Cases) == 0 {
		t.Fatalf("invalid Carbon serving evidence: schema=%q cases=%d", golden.Schema, len(golden.Cases))
	}
	plan, err := evaluation.CompileExact(golden)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := servingtest.ResolveActiveGGUFWithPolicy(
		modelPath, recipe.PlacementHybrid, modelrecipe.DecodeSessionCapacity, recipe.ResidencyDeviceNative,
	)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := inference.OpenWithProgram(context.Background(), &loaded, inference.OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()

	if _, err := evaluation.EvaluateExact(context.Background(), runner, plan); err != nil {
		t.Fatal(err)
	}
	t.Logf("Carbon real serving: %d cases match %s", len(golden.Cases), golden.Source)
}

func requireServingFileIdentity(t testing.TB, path string, kind artifact.Kind, want string) {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	identity, _, err := artifact.Identify(kind, file)
	if err != nil {
		t.Fatal(err)
	}
	if identity.String() != want {
		t.Fatalf("%s identity = %s, want %s", path, identity, want)
	}
}
