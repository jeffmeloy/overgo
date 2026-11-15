//go:build modeltest && windows

package sensenovarecipe

import (
	"bytes"
	"context"
	"crypto/sha256"
	"fmt"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/jsonfile"
	"overgo/internal/latentimage"
	"overgo/internal/recipe"
	"overgo/internal/routedlm"
	"overgo/internal/testcheck"
	"overgo/internal/workflowruntime"
)

// Run both fixed-output references under one exclusive measurement owner.
// The references cover the production device peripherals and the independent
// host-peripheral trajectory, including its native intermediate comparisons.
func TestSenseNovaOutputReferenceAcceptance(t *testing.T) {
	cudatest.Require(t)
	if cudatest.MeasurementProcess(t, 0) {
		return
	}
	t.Run("production", TestSenseNovaProductionImageGeneration)
	t.Run("native", TestSenseNovaGenerationLeadership)
}

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
	if hash != senseNovaProductionPNG || result.image.Width != request.Width || result.image.Height != request.Height || result.image.Channels != 3 {
		t.Fatalf("production image=%dx%dx%d sha256=%s", result.image.Width, result.image.Height, result.image.Channels, hash)
	}
	t.Logf("SenseNova production recipe: wall=%.3fs peak=%.3fGiB png=%s", result.wall.Seconds(), float64(result.peakBytes)/(1<<30), hash)
}

type productionGenerationResult struct {
	image     latentimage.EncodedImage
	wall      time.Duration
	peakBytes uint64
	// admitted is the card memory the generator admits where it keeps both branches resident (holdBranches): their
	// weights and the headroom beside them; none where it keeps neither.
	admitted uint64
}

func runProductionImageGeneration(t testing.TB, request GenerationRequest) productionGenerationResult {
	t.Helper()
	ctx := context.Background()
	started := time.Now()
	store, program, generator := newProductionGenerationFixture(t)
	definition := program.Definition()
	modelID := definition.Model
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
	produced := productionGenerationResult{image: image, wall: time.Since(started), peakBytes: memory.PeakBytes}
	if generator.resident[0] != nil {
		produced.admitted = residentBranchHeadroomBytes
		for branch := range generator.resident {
			produced.admitted += testcheck.Must(routedlm.BranchWeightBytes(generator.source, generator.config, generator.binding, branch))(t)
		}
	}
	return produced
}
