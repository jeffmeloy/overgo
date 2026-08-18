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
	miniCPM5ServingModelIdentity  = "model:sha256:e37577c45aec2255ac437ae6e86518ea99745d6616dd8c59b1fe7e26a04606f7"
	miniCPM5ServingGoldenIdentity = "evidence:sha256:1b177a279b58dfd98940dffce3c8381bda816a1bfcadd5411c6a645b398753ef"
)

func TestMiniCPM5ServingGolden(t *testing.T) {
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	modelPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "MiniCPM5-1B-f16.gguf")
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("UNAVAILABLE: converted MiniCPM5 artifact absent at %s", modelPath)
	}
	fixture := testutil.FixturePath(t, "minicpm5_serving_golden.json")
	requireServingFileIdentity(t, modelPath, artifact.KindModel, miniCPM5ServingModelIdentity)
	requireServingFileIdentity(t, fixture, artifact.KindEvidence, miniCPM5ServingGoldenIdentity)
	var golden evaluation.ExactSuite
	if err := jsonfile.Decode(fixture, &golden); err != nil {
		t.Fatal(err)
	}
	if golden.Schema != "minicpm5_serving_golden/v1" || len(golden.Cases) == 0 {
		t.Fatalf("invalid MiniCPM5 serving evidence: schema=%q cases=%d", golden.Schema, len(golden.Cases))
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
	err = evaluation.EvaluateExact(context.Background(), runner, plan, func(result evaluation.ExactResult) error {
		t.Logf("MiniCPM5 %s: prompt=%d generated=%d wall=%.3fms",
			result.Name, result.PromptTokens, result.GeneratedTokens, float64(result.WallNS)/1e6)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
