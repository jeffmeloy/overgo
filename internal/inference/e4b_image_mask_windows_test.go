//go:build modeltest

package inference

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodbtest"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// TestE4BImageMaskCausality compares a future-image intervention with the
// pinned native hidden-state capture, including a deliberately wrong mask.
// It establishes influence parity, not full hidden-value or task-quality parity.
func TestE4BImageMaskCausality(t *testing.T) {
	cudatest.Require(t)
	started := time.Now()
	publish := os.Getenv("OVERGO_E4B_PUBLISH_MASK") == "1"
	var revision string
	if publish {
		revision = testcheck.Must(runrecord.VerifyingCommit(testutil.RepoRoot(t)))(t)
	}
	var golden struct {
		ConfigSHA string              `json:"config_sha256"`
		ImageSHA  string              `json:"image_sha256"`
		PromptIDs []tokenizer.TokenID `json:"prompt_ids"`
		Query     int                 `json:"query_position"`
		Future    int                 `json:"future_position"`
		Results   []struct {
			Declaration *string              `json:"declaration"`
			Deltas      map[string]float64   `json:"hidden_max_abs_delta"`
			Baseline    map[string][]float64 `json:"baseline"`
			Perturbed   map[string][]float64 `json:"perturbed"`
		} `json:"results"`
	}
	data := testcheck.Must(os.ReadFile(testutil.FixturePath(t, "e4b_vision", "hidden_mask_golden.json")))(t)
	testcheck.NoError(t, json.Unmarshal(data, &golden))
	if len(golden.Results) != 2 || len(golden.Results[0].Deltas) != 2 || len(golden.Results[1].Deltas) != 2 || golden.Results[0].Declaration != nil || golden.Results[1].Declaration == nil || *golden.Results[1].Declaration != "vision" ||
		golden.Results[0].Deltas["0"] != 0 || golden.Results[0].Deltas["5"] != 0 || golden.Results[1].Deltas["0"] <= 0 {
		t.Fatal("native capture does not distinguish causal and wrong vision masks")
	}
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	store := overgodbtest.OpenReadOnly(t, roots.Store)
	modelPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "gemma-4-E4B-it-bf16.gguf")
	projectorPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "gemma-4-E4B-it-mmproj-bf16.gguf")
	loaded := testcheck.Must(modelrecipe.ResolveActiveGGUF(t.Context(), store, modelPath))(t)
	runner := testcheck.Must(OpenWithProgram(t.Context(), &loaded, OpenOptions{}))(t)
	defer runner.Close()
	if runner.ModelID().String() != "model:sha256:fb09299dd00edd7ffdcf8cb48e475d2a9c9e30a22c51f79f6d4d793e983c557b" {
		t.Fatal("model differs from native checkpoint binding")
	}
	inventory, media, processor, err := projector.InspectProjection(t.Context(), projectorPath)
	if err != nil || processor == nil {
		t.Fatalf("processor: %v", err)
	}
	config, found, err := modelrecipe.ResolveModelConfig(t.Context(), store, runner.ModelID())
	if err != nil || !found {
		t.Fatalf("model config: %v", err)
	}
	if !slices.ContainsFunc(config.Sources, func(source modelartifact.ConfigSource) bool {
		return source.Name == "config.json" && source.SHA256 == golden.ConfigSHA
	}) {
		t.Fatal("model config differs from native capture")
	}
	bound, err := processor.BindModelConfig(config)
	if err != nil || bound.ImageAttention != "causal" {
		t.Fatalf("E4B image attention: %v", err)
	}
	projection := testcheck.Must(projector.OpenSession(t.Context(), projectorPath, projector.OpenOptions{CUDA: true, MediaPreprocess: &bound}))(t)
	defer projection.Close()
	imageBytes := testcheck.Must(os.ReadFile(testutil.FixturePath(t, "e4b_vision", "gemma4_mm_image.png")))(t)
	if fmt.Sprintf("%x", sha256.Sum256(imageBytes)) != golden.ImageSHA {
		t.Fatal("image differs from native capture")
	}
	picture := testcheck.Must(png.Decode(bytes.NewReader(imageBytes)))(t)
	prompt := testcheck.Must(projection.BuildImagePrompt(t.Context(), runner, picture, "", "What is in this image? One word.", false))(t)
	ids, projected, err := CompileProjectedInputs(prompt, runner.spec.EmbeddingLength)
	testcheck.NoError(t, err)
	if !slices.Equal(ids, golden.PromptIDs) || len(projected.BidirectionalAttentionBlocks) != 0 ||
		int(prompt.EmbeddingTokenIndices[0]) != golden.Query || int(prompt.EmbeddingTokenIndices[len(prompt.EmbeddingTokenIndices)-1]) != golden.Future {
		t.Fatal("probe prompt or image positions differ from native capture")
	}
	width := int(runner.spec.EmbeddingLength)
	for _, result := range golden.Results {
		for _, layer := range []string{"0", "5"} {
			before, after := result.Baseline[layer], result.Perturbed[layer]
			if len(before) != width || len(after) != width {
				t.Fatal("native hidden-state vector denominator differs")
			}
			maximum := 0.0
			for i, value := range before {
				delta := math.Abs(after[i] - value)
				if math.IsNaN(delta) || math.IsInf(delta, 0) {
					t.Fatal("native capture contains nonfinite hidden state")
				}
				maximum = max(maximum, delta)
			}
			if maximum != result.Deltas[layer] {
				t.Fatal("native hidden-state summary differs from captured vectors")
			}
		}
	}
	for _, wrong := range []bool{false, true} {
		var baseline map[int32][]float32
		for _, perturb := range []bool{false, true} {
			input := projected
			input.EmbeddingOverrides = slices.Clone(projected.EmbeddingOverrides)
			if wrong {
				input.BidirectionalAttentionBlocks = []AttentionBlock{{Start: uint32(golden.Query), End: uint32(golden.Future + 1)}}
			}
			if perturb {
				last := len(input.EmbeddingOverrides) - 1
				input.EmbeddingOverrides[last].Embedding = slices.Clone(input.EmbeddingOverrides[last].Embedding)
				for i := range input.EmbeddingOverrides[last].Embedding {
					input.EmbeddingOverrides[last].Embedding[i] *= -1
				}
			}
			capture, err := newLayerInputCapture([]int32{1, 6}, len(runner.weights.Layers))
			testcheck.NoError(t, err)
			runner.mu.Lock()
			_, _, err = runner.forwardCachedProjectedChunkModeLocked(t.Context(), ids, nil, input, false, capture)
			runner.mu.Unlock()
			testcheck.NoError(t, err)
			rows := make(map[int32][]float32)
			for _, layer := range []int32{1, 6} {
				value, ok := capture.values[layer]
				if !ok || len(value.Data) != len(ids)*width {
					t.Fatal("hidden capture missing")
				}
				rows[layer] = slices.Clone(value.Data[golden.Query*width : (golden.Query+1)*width])
			}
			if !perturb {
				baseline = rows
				continue
			}
			for _, layer := range []int32{1, 6} {
				maximum := 0.0
				for i, value := range rows[layer] {
					delta := math.Abs(float64(value - baseline[layer][i]))
					if math.IsNaN(delta) || math.IsInf(delta, 0) {
						t.Fatal("nonfinite hidden state")
					}
					maximum = max(maximum, delta)
				}
				if !wrong && maximum != 0 {
					t.Fatalf("future image leaked to layer %d: %g", layer-1, maximum)
				}
				if wrong && layer == 1 && maximum == 0 {
					t.Fatal("wrong-mask control was insensitive")
				}
				t.Logf("wrong_mask=%t layer=%d future-image hidden delta=%g", wrong, layer-1, maximum)
			}
		}
	}
	t.Log("native influence parity: 2 policies x 2 interventions x 2 captured layers; no complete hidden-value parity or model-quality claim")
	if publish {
		current, err := runrecord.VerifyingCommit(testutil.RepoRoot(t))
		if err != nil || current != revision {
			t.Fatalf("mask producer source changed: %v", err)
		}
		definition := testcheck.Must(modelrecipe.ProjectionDefinition(runner.ModelID(), inventory.Manifest.ID, bound.ID, media...))(t)
		if _, err := recipe.RequireDefinition(t.Context(), store, definition.ID); err != nil {
			t.Fatalf("publish the config-bound projection candidate through the protocol producer first: %v", err)
		}
		environment := testcheck.Must(runrecord.CurrentEnvironment("cuda:0", "cuda"))(t)
		duration := uint64(time.Since(started).Nanoseconds())
		record := testcheck.Must(runrecord.NewGateRecord(definition.ID, environment.ID, revision, runrecord.OutcomeSucceeded, "", duration,
			[]runrecord.GateStep{{Name: "image-mask-causality", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: duration,
				Evidence: fmt.Sprintf("native_hidden_sha256=%x;policies=2;interventions=2;layers=2", sha256.Sum256(data))}}))(t)
		batch := testcheck.Must(record.Batch("validation/image-mask/" + record.Result.ID.String()))(t)
		content := testcheck.Must(environment.Content())(t)
		batch.Contents = append(batch.Contents, content)
		writer := overgodbtest.Create(t, roots.Store)
		if _, err := artifact.CommitBatch(t.Context(), writer, batch); err != nil {
			t.Fatal(err)
		}
		t.Logf("published mask influence proof: gate=%s run=%s recipe=%s", record.Result.ID, record.Run.ID, definition.ID)
	}
}
