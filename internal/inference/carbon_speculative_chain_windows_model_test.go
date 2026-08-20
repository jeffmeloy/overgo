//go:build windows && modeltest

package inference_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/dataroot"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/servingtest"
	"overgo/internal/testutil"
)

// TestCarbonSpeculativeChain pins the property speculative pairing rests on,
// with the draft model solo until the owner downloads the target: greedy
// decode is prefix-consistent across the text boundary. Drafting a k-token
// window and then continuing from the re-encoded extended prompt must equal
// drafting straight through -- that equality is what lets a verifier accept
// drafted windows, and it is also the Tier-0 chain contract, where models
// hand context across the boundary as text. The protocol constraint this
// measures: the crossing is bijective only for k-aligned sequences; a padded
// trailing token decodes its padding into the text and breaks re-encoding,
// so chain windows must stay k-aligned. Drafter wall is measured per token
// to price the future pair.
func TestCarbonSpeculativeChain(t *testing.T) {
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	modelPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "Carbon-500M-f16-ropefix.gguf")
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("UNAVAILABLE: converted Carbon artifact absent at %s", modelPath)
	}
	var golden evaluation.ExactSuite
	if err := jsonfile.Decode(testutil.FixturePath(t, "carbon_serving_golden.json"), &golden); err != nil {
		t.Fatal(err)
	}
	// The 96-base corpus prompt: sixteen full 6-mers, k-aligned by
	// construction, so the text boundary crossing is exact.
	prompt := ""
	for _, testCase := range golden.Cases {
		if testCase.Name == "mrna_atg_start" {
			prompt = testCase.Prompt
		}
	}
	if prompt == "" {
		t.Fatal("golden suite lacks the k-aligned corpus case")
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

	generate := func(from string, tokens int) (string, time.Duration) {
		t.Helper()
		greedy, err := sampling.New(sampling.Config{Temperature: 0})
		if err != nil {
			t.Fatal(err)
		}
		text := ""
		started := time.Now()
		_, _, err = runner.Generate(context.Background(), from, inference.GenerateOptions{
			MaxNewTokens: tokens, Sampler: greedy, DeviceGreedy: true,
			OnToken: func(event inference.TokenEvent) error {
				text += event.Piece
				return nil
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		return text, time.Since(started)
	}

	const window, total = 8, 16
	straight, straightWall := generate(prompt, total)
	drafted, draftWall := generate(prompt, window)
	continued, _ := generate(prompt+drafted, total-window)
	if drafted+continued != straight {
		t.Fatalf("prefix consistency broken across the text boundary:\n straight  %q\n chained   %q",
			straight, drafted+continued)
	}
	replay, _ := generate(prompt, total)
	if replay != straight {
		t.Fatalf("greedy decode not reproducible across sessions: %q vs %q", replay, straight)
	}
	if len(drafted) != window*6 {
		t.Fatalf("drafted %d bases from %d tokens, want k-aligned 6-mer windows", len(drafted), window)
	}
	t.Logf("solo chain verified: %d-token windows prefix-consistent; drafter wall %.1f ms/token (straight %.1f ms/token); target pair pending Carbon-3B download",
		window, float64(draftWall.Milliseconds())/float64(window), float64(straightWall.Milliseconds())/float64(total))
}
