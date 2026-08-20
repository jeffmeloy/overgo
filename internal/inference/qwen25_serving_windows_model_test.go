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
	qwen25ServingModelIdentity  = "model:sha256:764daf929f9d93ae9b2ffe886e6bb5871dd5cb85ebb5e0d94700c2158931e471"
	qwen25ServingGoldenIdentity = "evidence:sha256:5de6027bafa11e209e20b8f758b1d77946e034b4af14e815f18164cf992d90f1"
)

func TestQwen25ServingGolden(t *testing.T) {
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	modelPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "Qwen2.5-0.5B-f16.gguf")
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("UNAVAILABLE: converted Qwen2.5 artifact absent at %s", modelPath)
	}
	fixture := testutil.FixturePath(t, "qwen25_serving_golden.json")
	requireServingFileIdentity(t, modelPath, artifact.KindModel, qwen25ServingModelIdentity)
	requireServingFileIdentity(t, fixture, artifact.KindEvidence, qwen25ServingGoldenIdentity)
	var golden evaluation.ExactSuite
	if err := jsonfile.Decode(fixture, &golden); err != nil {
		t.Fatal(err)
	}
	if golden.Schema != "qwen25_serving_golden/v1" || len(golden.Cases) == 0 {
		t.Fatalf("invalid Qwen2.5 serving evidence: schema=%q cases=%d", golden.Schema, len(golden.Cases))
	}
	suite, err := os.ReadFile(fixture)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := evaluateExactGGUF(context.Background(), modelPath, suite); err != nil {
		t.Fatal(err)
	}
	t.Logf("Qwen2.5 real serving: %d cases match %s", len(golden.Cases), golden.Source)
}
