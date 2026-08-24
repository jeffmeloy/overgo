//go:build windows

package sensenovarecipe

import (
	"bytes"
	"context"
	"crypto/sha256"
	"fmt"
	"testing"
	"time"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/jsonfile"
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

func TestSenseNovaProductionImageGeneration(t *testing.T) {
	cudatest.Require(t)
	var oracle generationLeadershipOracle
	if err := jsonfile.Decode(senseNovaGenerationGold, &oracle); err != nil {
		t.Fatal(err)
	}
	request := GenerationRequest{
		Prompt: oracle.Request.Prompt, Width: oracle.Request.Width, Height: oracle.Request.Height,
		Steps: oracle.Request.Steps, Seed: oracle.Request.Seed, CFGScale: oracle.Request.CFGScale,
		TimestepShift: oracle.Request.TimestepShift,
	}
	result := runProductionImageGeneration(t, request)
	hash := fmt.Sprintf("%x", sha256.Sum256(result.image.Data))
	const productionPNG = "6ef4065018d328c2945e6de0a8b87fe3eb2aec44039b43207398969712446de0"
	if hash != productionPNG || result.image.Width != request.Width || result.image.Height != request.Height || result.image.Channels != 3 {
		t.Fatalf("production image=%dx%dx%d sha256=%s", result.image.Width, result.image.Height, result.image.Channels, hash)
	}
	t.Logf("SenseNova production recipe: wall=%.3fs peak=%.3fGiB png=%s", result.wall.Seconds(), float64(result.peakBytes)/(1<<30), hash)
}

type productionGenerationResult struct {
	image     latentimage.EncodedImage
	wall      time.Duration
	peakBytes uint64
}

func runProductionImageGeneration(t testing.TB, request GenerationRequest) productionGenerationResult {
	t.Helper()
	ctx := context.Background()
	started := time.Now()
	generator, err := LoadGenerator(senseNovaModelDir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := generator.Close(ctx); err != nil {
			t.Errorf("close: %v", err)
		}
	}()

	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "sensenova-production")
	testutil.PublishArtifact(t, store, modelID)
	definition, err := modelrecipe.RoutedImageDefinition(modelID)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	runtime, err := workflowruntime.NewForProgram(store, program)
	if err != nil {
		t.Fatal(err)
	}
	if err := RegisterRuntime(runtime, modelID, generator); err != nil {
		t.Fatal(err)
	}
	input := definition.Inputs[0]
	operation, err := workflowruntime.ExecutionID(definition.ID, "sensenova/production")
	if err != nil {
		t.Fatal(err)
	}
	result, err := runtime.ExecuteProgram(ctx, "sensenova/production", operation, nil, program, map[recipe.PortName]workflowruntime.Value{
		input.Name: {Kind: input.Data, Items: []workflowruntime.Datum{{Value: request}}},
	})
	if err != nil {
		stats := generator.Stats()
		t.Logf("SenseNova failed phases: steps=%d prepare=%.3fs vision=%.3fs condition=%.3fs body=%.3fs flow=%.3fs guidance=%.3fs decode=%.3fs",
			stats.Steps, stats.Prepare.Seconds(), stats.Vision.Seconds(), stats.Condition.Seconds(), stats.Body.Seconds(),
			stats.Flow.Seconds(), stats.Guidance.Seconds(), stats.Decode.Seconds())
		t.Fatal(err)
	}
	output := definition.Outputs[0]
	datum, one := result.Outputs[output.Name].Single()
	image, typed := datum.Value.(latentimage.EncodedImage)
	if !one || !typed || !result.Commit.Valid() {
		t.Fatalf("production output = (%T,%v,%v), commit=%s", datum.Value, one, typed, result.Commit)
	}
	if datum.Content == nil || datum.Content.Descriptor.MediaType != image.MediaType || !bytes.Equal(datum.Content.Data, image.Data) {
		t.Fatal("production artifact is not the encoded PNG")
	}
	memory, err := generator.worker.MemoryStats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	stats := generator.Stats()
	t.Logf("SenseNova production phases: steps=%d prepare=%.3fs vision=%.3fs condition=%.3fs body=%.3fs flow=%.3fs guidance=%.3fs decode=%.3fs",
		stats.Steps, stats.Prepare.Seconds(), stats.Vision.Seconds(), stats.Condition.Seconds(), stats.Body.Seconds(),
		stats.Flow.Seconds(), stats.Guidance.Seconds(), stats.Decode.Seconds())
	return productionGenerationResult{image: image, wall: time.Since(started), peakBytes: memory.PeakBytes}
}
