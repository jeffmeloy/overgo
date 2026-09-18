package main

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/inference"
	"overgo/internal/modelintake"
	"overgo/internal/overgodb"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/testutil"
)

// TestProjectionExecutionE4B exercises the exact verifier's real generator.
// It is a producer check; the full declared-mode suite must still pass before
// canonical projection promotion. No evidence is published by this test.
// Oracle: native checkpoint ee0ef6023621cff504d758262d4e04895a5af4a2,
// Transformers 5.14.1 / Torch 2.12.0+cu126, SDPA, TF32 disabled, both
// float32 and bfloat16; capture source SHA256
// 44a5922da8725064fe6028c157d53c003b1fbc2147f88b29d0762c5b0dcc556c.
func TestProjectionExecutionE4B(t *testing.T) {
	cudatest.Require(t)
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	var paths []string
	for _, reference := range []string{
		"tensor-set:sha256:cd4ada4703c2b76a84a10da94f09b9199b6d4dad7e3dcabbe79d8cee745f4501",
		"tensor-set:sha256:185786ec6d77c31f87e6ebdcf8a0d095dbb7175999122229f8e82dcdad25004e",
	} {
		id, err := artifact.ParseID(reference)
		if err != nil {
			t.Fatal(err)
		}
		path, err := artifact.AvailablePath(t.Context(), store, id, artifact.LocationFile)
		if err != nil {
			t.Fatal(err)
		}
		paths = append(paths, path)
	}
	candidate, err := modelintake.PrepareProjectionCandidate(t.Context(), store, paths[0], paths[1])
	if err != nil {
		t.Fatal(err)
	}
	language, err := modelintake.PrepareInferenceCandidate(t.Context(), store, paths[0], modelintake.SessionOverride{}, recipe.ResidencyHybridNative)
	if err != nil {
		t.Fatal(err)
	}
	projection, err := projector.OpenSession(t.Context(), paths[1], projector.OpenOptions{CUDA: true, MediaPreprocess: candidate.Processor})
	if err != nil {
		t.Fatal(err)
	}
	runtime := &deferredExactRuntime{release: projection.Close, open: func(ctx context.Context) (exactRuntime, error) {
		runner, err := openExactCandidate(ctx, paths[0], language)
		if err != nil {
			return nil, err
		}
		if err := runner.RuntimePolicy().ValidateIdentity(); err != nil {
			runner.Close()
			return nil, err
		}
		return &projectedExactRuntime{runner: runner, projection: projection}, nil
	}}
	defer func() {
		if err := runtime.Close(); err != nil {
			t.Error(err)
		}
	}()
	file := projectionFile{Kind: "image", Path: testutil.FixturePath(t, "e4b_vision", "gemma4_mm_image.png")}
	file.Artifact, err = artifact.ParseID("file:sha256:996fea5cea787f3abb6d1377fc88642ade6bdb8dc9a7b9e6ad909e58687b776e")
	if err != nil {
		t.Fatal(err)
	}
	input := projectionInput{Kind: "image", Files: []projectionFile{file}, Text: []string{"", "What is in this image? One word."}}
	raw, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	var output strings.Builder
	promptTokens := 0
	ids, _, err := runtime.Generate(t.Context(), string(raw), inference.GenerateOptions{MaxNewTokens: 16, Sampler: greedy, DeviceGreedy: true,
		OnToken:           func(event inference.TokenEvent) error { output.WriteString(event.Piece); return nil },
		OnPromptEvaluated: func(result inference.PromptEvaluation) { promptTokens = result.Tokens },
	})
	if err != nil {
		t.Fatal(err)
	}
	// Independent native float32 and bfloat16 SDPA agree on all three values.
	if output.String() != "Pattern" || promptTokens != 300 || len(ids)-promptTokens != 2 {
		t.Fatalf("complete image output: %q prompt/generated=%d/%d; want Pattern 300/2", output.String(), promptTokens, len(ids)-promptTokens)
	}
	t.Log("executed E4B projection-to-language: complete native-reference output Pattern, 300 prompt and 2 generated tokens; other declared modes, HTTP, datasets and resources NOT accepted")
}
