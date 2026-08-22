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
	"overgo/internal/jsonfile"
	"overgo/internal/testutil"
)

const (
	miniCPM5ServingModelIdentity  = "model:sha256:e37577c45aec2255ac437ae6e86518ea99745d6616dd8c59b1fe7e26a04606f7"
	miniCPM5ServingGoldenIdentity = "evidence:sha256:d03ab478b299e2a6ef6c985ca7bcb747b3b5ebca4e75ddea6f542a17391c46ce"
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
	suite, err := os.ReadFile(fixture)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := evaluateExactGGUF(context.Background(), modelPath, suite); err != nil {
		t.Fatal(err)
	}
	t.Logf("MiniCPM5 real serving: %d cases match %s", len(golden.Cases), golden.Source)
}
