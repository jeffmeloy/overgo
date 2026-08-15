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
	"overgo/internal/recipe"
	"overgo/internal/repodb"
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

	store, err := repodb.Open(t.TempDir())
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
	result, err := runtime.ExecuteProgram(ctx, "sensenova/production", program, map[recipe.PortName]workflowruntime.Value{
		input.Name: {Kind: input.Data, Items: []workflowruntime.Datum{{Value: request}}},
	})
	if err != nil {
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
	hash := fmt.Sprintf("%x", sha256.Sum256(image.Data))
	if hash != senseNovaTerminalPNG || image.Width != request.Width || image.Height != request.Height || image.Channels != 3 {
		t.Fatalf("production image=%dx%dx%d sha256=%s", image.Width, image.Height, image.Channels, hash)
	}
	memory, err := generator.worker.MemoryStats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("SenseNova production recipe: wall=%.3fs peak=%.3fGiB png=%s", time.Since(started).Seconds(), float64(memory.PeakBytes)/(1<<30), hash)
}
