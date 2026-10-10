//go:build windows

package mediacapability

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/latentvideo"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/modeltest"
	"overgo/internal/overgodb"
	"overgo/internal/processmeasure"
	"overgo/internal/testutil"
)

// TestVideoPromptIsEncodedWhileItsModelLoads holds where the text encoder's
// pass of a first prompt-form request goes: beside the loading of the
// generator, and not after it. The request names a prompt its process has
// encoded for no request before, and the model's shared negative prompt with
// it. Its parts, timed as the scorecard's cell times them, hold the lease of
// the model's session and no pass over the encoder: the pass was made while
// the session loaded, and the request waited at most for the rest of it. The
// clip is the one the same request gives when the prompts are encoded after
// the load, as they were: a second session, whose prompts the process holds
// by then, draws it byte for byte.
func TestVideoPromptIsEncodedWhileItsModelLoads(t *testing.T) {
	cudatest.Require(t)
	wan := modeltest.Directory(t, "Wan2.1-T2V-1.3B")
	profile, err := latentvideo.ResolveProfile(wan)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(latentvideo.WanTextConditioningSpec(wan, 1).EncoderCheckpoint); err != nil {
		t.Skipf("UNAVAILABLE: encoder checkpoint absent: %v", err)
	}
	profileContent, err := profile.Content()
	if err != nil {
		t.Fatal(err)
	}
	modelID := testutil.ArtifactID(t, artifact.KindModel, "wan-prompt-beside-load")
	definition, err := modelrecipe.GenerationDefinition(modelrecipe.ModuleLatentVideoPrepare, modelID, profile.ID)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(latentvideo.WanRequest{
		Prompt: "a grey heron lifts off over still water at first light, held for the load test",
		Seed:   47, Frames: 5, Width: 32, Height: 16, Steps: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	// serve gives the request to a capability and a store of its own, as a process that has served nothing does.
	serve := func(name string) ([]byte, string) {
		t.Helper()
		store, err := overgodb.Open(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		publishArtifactExtent(t, store, modelID, wan)
		if _, err := store.Commit(t.Context(), artifact.Batch{Key: "test/video-prompt-load-profile", Contents: []artifact.Content{profileContent}}); err != nil {
			t.Fatal(err)
		}
		ctx, stages := processmeasure.WithStages(t.Context())
		started := time.Now()
		output, err := videoCapability().Execute(ctx, store, wan, modelrecipetest.CandidateExecution(t, store, program), string(raw))
		wall := time.Since(started)
		if err != nil {
			t.Fatal(err)
		}
		video, ok := capabilityruntime.Unwrap(output).(latentvideo.EncodedVideo)
		if !ok {
			t.Fatalf("video output type=%T", output)
		}
		var parts strings.Builder
		if _, err := stages.Report(&parts, wall); err != nil {
			t.Fatalf("the %s request's parts are no division of it: %v", name, err)
		}
		t.Logf("the %s request took %.2f s:\n%s", name, wall.Seconds(), parts.String())
		return video.Data, parts.String()
	}
	first, parts := serve("first")
	for _, absent := range []string{"text encoder run", "text encoder weights read"} {
		if strings.Contains(parts, absent) {
			t.Errorf("the first request's own parts hold %q: its prompts were encoded after its model had loaded", absent)
		}
	}
	for _, present := range []string{"model session leased", "prompt rows awaited"} {
		if !strings.Contains(parts, present) {
			t.Errorf("the first request's parts hold no %q", present)
		}
	}
	if again, _ := serve("second"); string(again) != string(first) {
		t.Error("a second session, whose prompts were encoded before it loaded, drew another clip")
	}
}
