//go:build windows && modeltest

// Package inference_test verifies compiled serving against external evidence.
package inference_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/servingtest"
	"overgo/internal/strictjson"
	"overgo/internal/testutil"
)

type servingGoldenCase struct {
	Name, Model, Fixture, Schema string
	ModelIdentity, Evidence      string
}

var servingGoldenCases = []servingGoldenCase{
	{Name: "Carbon", Model: "Carbon-500M-f16-ropefix.gguf", Fixture: "carbon_serving_golden.json", Schema: "carbon_serving_golden/v1",
		ModelIdentity: "model:sha256:08d12f90b68fe93f3c39da1bb9f4592a13085c2740716166d2baab708793d3e8", Evidence: "evidence:sha256:2a7b3a0e163f79b6ffc9cd5d7ea36801caf196118ede870dff6220e6be12fb41"},
	{Name: "MiniCPM5", Model: "MiniCPM5-1B-f16.gguf", Fixture: "minicpm5_serving_golden.json", Schema: "minicpm5_serving_golden/v1",
		ModelIdentity: "model:sha256:e37577c45aec2255ac437ae6e86518ea99745d6616dd8c59b1fe7e26a04606f7", Evidence: "evidence:sha256:d03ab478b299e2a6ef6c985ca7bcb747b3b5ebca4e75ddea6f542a17391c46ce"},
	{Name: "Qwen2.5", Model: "Qwen2.5-0.5B-f16.gguf", Fixture: "qwen25_serving_golden.json", Schema: "qwen25_serving_golden/v1",
		ModelIdentity: "model:sha256:764daf929f9d93ae9b2ffe886e6bb5871dd5cb85ebb5e0d94700c2158931e471", Evidence: "evidence:sha256:b6d7d8c8f35aa9dce6ada81aca2ddafbdeefba5ff25305312c57d1fac9f02a1e"},
}

func TestServingGoldenEvidenceCatalog(t *testing.T) {
	for _, test := range servingGoldenCases {
		t.Run(test.Name, func(t *testing.T) {
			readServingGolden(t, test)
		})
	}
}

func TestServingGoldenEvidence(t *testing.T) {
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range servingGoldenCases {
		t.Run(test.Name, func(t *testing.T) {
			modelPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", test.Model)
			if _, err := os.Stat(modelPath); err != nil {
				t.Skipf("UNAVAILABLE: converted artifact absent at %s", modelPath)
			}
			requireServingFileIdentity(t, modelPath, artifact.KindModel, test.ModelIdentity)
			golden, suite := readServingGolden(t, test)
			if _, err := evaluateExactGGUF(context.Background(), modelPath, suite); err != nil {
				t.Fatal(err)
			}
			t.Logf("%s real serving: %d cases match %s", test.Name, len(golden.Cases), golden.Source)
		})
	}
}

func readServingGolden(t testing.TB, test servingGoldenCase) (evaluation.ExactSuite, []byte) {
	t.Helper()
	path := testutil.FixturePath(t, test.Fixture)
	requireServingFileIdentity(t, path, artifact.KindEvidence, test.Evidence)
	suite, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var golden evaluation.ExactSuite
	if err := strictjson.DecodeBytes(suite, &golden); err != nil {
		t.Fatal(err)
	}
	if golden.Schema != test.Schema || len(golden.Cases) == 0 {
		t.Fatalf("invalid serving evidence: schema=%q cases=%d", golden.Schema, len(golden.Cases))
	}
	return golden, suite
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

// evaluateExactGGUF: exact suite through compiled serving authority.
func evaluateExactGGUF(ctx context.Context, path string, suite []byte) (evaluation.CampaignResult, error) {
	if ctx == nil {
		return evaluation.CampaignResult{}, errors.New("serving test: evaluation context is absent")
	}
	root, err := os.MkdirTemp("", "overgo-evaluation-fixture-")
	if err != nil {
		return evaluation.CampaignResult{}, err
	}
	defer os.RemoveAll(root)
	store, err := overgodb.Open(root)
	if err != nil {
		return evaluation.CampaignResult{}, err
	}
	fail := func(cause error) (evaluation.CampaignResult, error) {
		return evaluation.CampaignResult{}, errors.Join(cause, store.Close())
	}
	if err := servingtest.PublishActiveGGUFWithPolicy(
		ctx, store, path, recipe.PlacementHybrid,
		modelrecipe.DecodeSessionCapacity, recipe.ResidencyDeviceNative,
	); err != nil {
		return fail(err)
	}
	loaded, err := modelrecipe.ResolveActiveGGUF(ctx, store, path)
	if err != nil {
		return fail(err)
	}
	identity, err := loaded.Identity()
	if err != nil {
		_ = loaded.Close()
		return fail(err)
	}
	runner, err := inference.OpenWithProgram(ctx, &loaded, inference.OpenOptions{})
	if err != nil {
		_ = loaded.Close()
		return fail(err)
	}
	environment, err := runrecord.CurrentEnvironment("cuda:0", "cuda")
	if err != nil {
		_ = runner.Close()
		return fail(err)
	}
	campaign, err := evaluation.NewCampaign(store, runner, identity, environment, strings.Repeat("0", 40))
	if err != nil {
		_ = runner.Close()
		return fail(err)
	}
	compiled, err := evaluation.CompileSuite(suite, campaign.Authorities())
	if err != nil {
		_ = runner.Close()
		return fail(fmt.Errorf("serving test: compile exact suite: %w", err))
	}
	result, evaluateErr := campaign.Evaluate(ctx, compiled)
	return result, errors.Join(evaluateErr, runner.Close(), store.Close())
}
