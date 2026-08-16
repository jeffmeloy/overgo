//go:build windows && modeltest

package inference

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/servingtest"
	"overgo/internal/testutil"
)

type carbonServingCase struct {
	Name            string `json:"name"`
	Prompt          string `json:"prompt"`
	MaxTokens       int    `json:"max_tokens"`
	Text            string `json:"text"`
	PromptTokens    int    `json:"prompt_tokens"`
	GeneratedTokens int    `json:"generated_tokens"`
}

type carbonServingGolden struct {
	Schema string              `json:"schema"`
	Source string              `json:"source"`
	Cases  []carbonServingCase `json:"cases"`
}

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
	var golden carbonServingGolden
	fixture := testutil.FixturePath(t, "carbon_serving_golden.json")
	requireCarbonFileIdentity(t, modelPath, artifact.KindModel, carbonServingModelIdentity)
	requireCarbonFileIdentity(t, fixture, artifact.KindEvidence, carbonServingGoldenIdentity)
	if err := jsonfile.Decode(fixture, &golden); err != nil {
		t.Fatal(err)
	}
	if golden.Schema != "carbon_serving_golden/v1" || len(golden.Cases) == 0 {
		t.Fatalf("invalid Carbon serving evidence: schema=%q cases=%d", golden.Schema, len(golden.Cases))
	}
	loaded, err := servingtest.ResolveActiveGGUFWithPolicy(
		modelPath, recipe.PlacementHybrid, modelrecipe.DecodeSessionCapacity, recipe.ResidencyDeviceNative,
	)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := OpenWithProgram(context.Background(), &loaded, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()

	for _, testCase := range golden.Cases {
		t.Run(testCase.Name, func(t *testing.T) {
			greedy, err := sampling.New(sampling.Config{Temperature: 0})
			if err != nil {
				t.Fatal(err)
			}
			var generated strings.Builder
			promptTokens := 0
			ids, _, err := runner.Generate(context.Background(), testCase.Prompt, GenerateOptions{
				MaxNewTokens: testCase.MaxTokens,
				Sampler:      greedy,
				DeviceGreedy: true,
				OnToken: func(event TokenEvent) error {
					generated.WriteString(event.Piece)
					return nil
				},
				OnPromptEvaluated: func(evaluation PromptEvaluation) {
					promptTokens = evaluation.Tokens
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			generatedTokens := len(ids) - promptTokens
			if promptTokens != testCase.PromptTokens || generatedTokens != testCase.GeneratedTokens {
				t.Fatalf("token counts prompt/generated=%d/%d, want %d/%d",
					promptTokens, generatedTokens, testCase.PromptTokens, testCase.GeneratedTokens)
			}
			if text := generated.String(); text != testCase.Text {
				pieces := make([]string, 0, generatedTokens)
				for _, id := range ids[promptTokens:] {
					piece, pieceErr := runner.TokenPiece(id)
					if pieceErr != nil {
						t.Fatal(pieceErr)
					}
					pieces = append(pieces, piece)
				}
				t.Fatalf("generated text = %q, ids=%v pieces=%q, want %q", text, ids[promptTokens:], pieces, testCase.Text)
			}
		})
	}
	t.Logf("Carbon real serving: %d cases match %s", len(golden.Cases), golden.Source)
}

func requireCarbonFileIdentity(t testing.TB, path string, kind artifact.Kind, want string) {
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
