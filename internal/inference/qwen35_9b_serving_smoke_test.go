package inference

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

// Qwen3.5-9B (Q8_0 GGUF) device serving smoke. The 9B is the same qwen35
// hybrid-recurrent architecture as the served 4B (rung-12), scaled up with one
// structural delta: an UNTIED lm_head (output.weight distinct from token_embd).
// The structural read path is proven by TestQwen35_9BUntiedHeadGGUFStructuralOracle
// (internal/model); this test closes the remaining gap -- a real one-token
// decode on device through the same hybrid_recurrent serving entry point the 4B
// uses (openFixtureRunnerWithOptions -> ResolveActiveGGUF PlacementHybrid).
//
// No 9B GOLDEN exists (reconstruction; the artifact ships no HF config/tokenizer
// and there is no independent Python/host oracle for this checkpoint). This
// asserts the reconstruction's plausibility bar -- deterministic, non-constant,
// coherent greedy output ("a 9B that answers Paris") -- confirms the untied head
// is LIVE (the runner's mapped head is distinct from the embedding), and pins
// the device output as a serving-smoke REGRESSION anchor (not an oracle). The
// fixture (fixtures/qwen35_9b_serving_smoke.json) is regenerated on device with
// OVERGO_WRITE_9B_SMOKE_FIXTURE=1 and asserted byte-for-byte otherwise.
//
// Skips when the artifact is absent (worktree-safe). Override with
// OVERGO_QWEN35_9B_GGUF.
const qwen35_9BDefaultPath = `C:\Users\jeffm\adaptive_new\models\Qwen3.5-9B-Q8_0.gguf`

// qwen35_9BArtifactSHA256 pins the exact GGUF this smoke fixture was captured
// against (provenance; the fixture's generated ids are only meaningful for this
// artifact). Recomputed with sha256sum on the 8.9GiB container.
const qwen35_9BArtifactSHA256 = "809626574d0cb43d4becfa56169980da2bb448f2299270f7be443cb89d0a6ae4"

type qwen35_9BSmokeCase struct {
	Name         string              `json:"name"`
	Prompt       string              `json:"prompt"`
	PromptIDs    []tokenizer.TokenID `json:"prompt_ids"`
	GeneratedIDs []tokenizer.TokenID `json:"generated_ids"`
	Text         string              `json:"text"`
}

type qwen35_9BSmokeFixture struct {
	Schema      string               `json:"schema"`
	Status      string               `json:"status"`
	Provisional bool                 `json:"provisional"`
	GoldenGap   string               `json:"golden_gap"`
	Source      string               `json:"source"`
	Model       string               `json:"model"`
	ModelSHA256 string               `json:"model_sha256"`
	Device      string               `json:"device"`
	DecodeMode  string               `json:"decode_mode"`
	Cases       []qwen35_9BSmokeCase `json:"cases"`
}

func qwen35_9BServingPath(t *testing.T) string {
	t.Helper()
	path := os.Getenv("OVERGO_QWEN35_9B_GGUF")
	if path == "" {
		path = qwen35_9BDefaultPath
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("Qwen3.5-9B GGUF absent (%s); set OVERGO_QWEN35_9B_GGUF to run", path)
	}
	return path
}

func TestQwen35_9BServingSmoke(t *testing.T) {
	path := qwen35_9BServingPath(t)

	loadStart := time.Now()
	runner, err := openFixtureRunnerWithOptions(path, OpenOptions{
		DeviceOrdinal:           0,
		PreloadQuantizedWeights: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	loadElapsed := time.Since(loadStart)

	// Untied head is LIVE: the runner mapped a distinct output.weight, not the
	// tied embedding. A silent tied fallback (Weights.Output == nil) would still
	// produce plausible text from the WRONG matrix, so assert the live mapping.
	if runner.weights.Output == nil {
		t.Fatal("runner.weights.Output is nil: untied 9B head fell back to the tied embedding")
	}
	if runner.weights.Output.Name != "output.weight" {
		t.Errorf("runner.weights.Output.Name = %q, want output.weight", runner.weights.Output.Name)
	}
	if runner.weights.Output.Offset == runner.weights.TokenEmbedding.Offset {
		t.Errorf("mapped head shares embedding offset %d: not distinct (tied)", runner.weights.Output.Offset)
	}
	t.Logf("untied head LIVE: output.weight offset=%d distinct from token_embd offset=%d",
		runner.weights.Output.Offset, runner.weights.TokenEmbedding.Offset)

	ctx := context.Background()
	newGreedy := func() *sampling.Sampler {
		s, err := sampling.New(sampling.Config{})
		if err != nil {
			t.Fatal(err)
		}
		if !s.IsRawGreedy() {
			t.Fatal("default sampler is not raw greedy")
		}
		return s
	}

	const newTokens = 8
	generate := func(prompt string) (promptIDs, generatedIDs []tokenizer.TokenID, text string) {
		ids, text, err := runner.Generate(ctx, prompt, GenerateOptions{
			MaxNewTokens: newTokens,
			Sampler:      newGreedy(),
		})
		if err != nil {
			t.Fatal(err)
		}
		promptIDs, err = runner.promptTokenIDs(prompt, GenerateOptions{})
		if err != nil {
			t.Fatal(err)
		}
		return promptIDs, ids[len(promptIDs):], text
	}

	prompts := []struct{ name, prompt string }{
		{"capital", "The capital of France is"},
		{"count", "Count from one to five: one, two,"},
	}

	var cases []qwen35_9BSmokeCase
	var firstGenStart time.Time
	var firstPerToken time.Duration
	for i, p := range prompts {
		if i == 0 {
			firstGenStart = time.Now()
		}
		promptIDs, generatedIDs, text := generate(p.prompt)
		if i == 0 {
			elapsed := time.Since(firstGenStart)
			if len(generatedIDs) > 0 {
				firstPerToken = elapsed / time.Duration(len(generatedIDs))
			}
		}
		if len(generatedIDs) == 0 {
			t.Fatalf("%s: no tokens generated", p.name)
		}
		// Non-constant: a degenerate/NaN-logit path collapses to one repeated
		// token. Coherent varied output is the CPU-cheap finiteness proxy.
		constant := true
		for _, id := range generatedIDs[1:] {
			if id != generatedIDs[0] {
				constant = false
				break
			}
		}
		if constant {
			t.Errorf("%s: generated tokens constant (%d repeated): degenerate/non-finite logits", p.name, generatedIDs[0])
		}
		// Deterministic on repeat.
		_, generatedIDs2, text2 := generate(p.prompt)
		if !slices.Equal(generatedIDs, generatedIDs2) || text != text2 {
			t.Errorf("%s: non-deterministic greedy: %v %q vs %v %q", p.name, generatedIDs, text, generatedIDs2, text2)
		}
		t.Logf("SERVE %s prompt=%q -> %q generated=%v", p.name, p.prompt, text, generatedIDs)
		cases = append(cases, qwen35_9BSmokeCase{
			Name: p.name, Prompt: p.prompt,
			PromptIDs: promptIDs, GeneratedIDs: generatedIDs, Text: text,
		})
	}

	// Plausibility bar: a 9B that answers Paris.
	if !strings.Contains(strings.ToLower(cases[0].Text), "paris") {
		t.Errorf("plausibility bar unmet: greedy output %q does not mention Paris", cases[0].Text)
	}

	// Peak device memory (Q8_0 stays resident; BF16 native-dtype prefill lever
	// keeps the compute path from upcasting the resident weights).
	peakMiB := 0.0
	if stats, err := runner.DeviceMemoryStats(ctx); err == nil {
		peakMiB = float64(stats.PeakBytes) / (1024 * 1024)
	} else {
		t.Logf("DeviceMemoryStats unavailable: %v", err)
	}
	t.Logf("load=%s ms/token=%.2f peakMiB=%.0f", loadElapsed, float64(firstPerToken)/float64(time.Millisecond), peakMiB)

	fixturePath := filepath.Join("..", "..", "fixtures", "qwen35_9b_serving_smoke.json")
	if os.Getenv("OVERGO_WRITE_9B_SMOKE_FIXTURE") != "" {
		fixture := qwen35_9BSmokeFixture{
			Schema:      "qwen35_9b_serving_smoke/v1",
			Status:      "LIVE",
			Provisional: true,
			GoldenGap: "no independent oracle: the 9B ships no HF config/tokenizer and no Python/host " +
				"reference logits exist for this checkpoint. These ids are a device-serving REGRESSION " +
				"anchor, plausibility-validated (Paris + coherent counting), NOT a numeric golden. The " +
				"Q8_0 quant decoder + hybrid decode conventions are cross-referenced by " +
				"TestQwen35_9BUntiedHeadGGUFStructuralOracle and shared with the golden-validated 4B path.",
			Source: "overgo internal/inference device serving (hybrid_recurrent, PlacementHybrid, " +
				"PreloadQuantizedWeights), greedy temp=0, 8 new tokens/case; deterministic across repeat runs " +
				"(id-identical); untied lm_head live (output.weight distinct from token_embd)",
			Model:       "models/Qwen3.5-9B-Q8_0.gguf (Unsloth bare GGUF; no HF sidecar)",
			ModelSHA256: qwen35_9BArtifactSHA256,
			Device:      "NVIDIA GeForce RTX 4090 D",
			DecodeMode:  "greedy (temp 0), 8 new tokens per case, ID-level",
			Cases:       cases,
		}
		blob, err := json.MarshalIndent(fixture, "", " ")
		if err != nil {
			t.Fatal(err)
		}
		blob = append(blob, '\n')
		if err := os.WriteFile(fixturePath, blob, 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote serving-smoke fixture %s", fixturePath)
		return
	}

	blob, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("read fixture (regenerate with OVERGO_WRITE_9B_SMOKE_FIXTURE=1): %v", err)
	}
	var want qwen35_9BSmokeFixture
	if err := json.Unmarshal(blob, &want); err != nil {
		t.Fatal(err)
	}
	if want.ModelSHA256 != qwen35_9BArtifactSHA256 {
		t.Errorf("fixture model_sha256 %s != pinned artifact %s", want.ModelSHA256, qwen35_9BArtifactSHA256)
	}
	if len(want.Cases) != len(cases) {
		t.Fatalf("fixture has %d cases, device produced %d", len(want.Cases), len(cases))
	}
	for i, wc := range want.Cases {
		got := cases[i]
		if wc.Prompt != got.Prompt ||
			!slices.Equal(wc.PromptIDs, got.PromptIDs) ||
			!slices.Equal(wc.GeneratedIDs, got.GeneratedIDs) ||
			wc.Text != got.Text {
			t.Errorf("case %q regression:\n fixture: %v %q\n device:  %v %q",
				wc.Name, wc.GeneratedIDs, wc.Text, got.GeneratedIDs, got.Text)
		}
	}
}
