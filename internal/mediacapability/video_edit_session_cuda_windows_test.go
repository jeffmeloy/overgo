//go:build windows

package mediacapability

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/latentvideo"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/modeltest"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// servedEdits is a store an edit by the installed LiveEdit model is served
// from: the model's artifact, the base model's profile and one source clip.
type servedEdits struct {
	store     *overgodb.Store
	directory string
	execution modelrecipe.CapabilityEvidenceSelection
	clip      artifact.ID
}

// stripedClip is a clip of moving stripes, its values from 0 to 1.
func stripedClip(frames, height, width int) latentvideo.SourceVideo {
	plane := height * width
	source := latentvideo.SourceVideo{
		Pixels: make([]float32, media.RGBChannels*frames*plane), Channels: media.RGBChannels, Frames: frames, Height: height, Width: width,
	}
	for channel := range media.RGBChannels {
		for frame := range frames {
			for at := range plane {
				x, y := at%width, at/width
				source.Pixels[(channel*frames+frame)*plane+at] = float32((x+2*y+7*frame+40*channel)%width) / float32(width)
			}
		}
	}
	return source
}

// serveEdits opens a store that holds what a served edit of source stands on.
// It skips where the model or its text encoder is not installed.
func serveEdits(t *testing.T, name string, source latentvideo.SourceVideo) servedEdits {
	t.Helper()
	cudatest.Require(t)
	edit := modeltest.Directory(t, "LiveEdit")
	wan, checkpoint := latentvideo.LiveEditArtifacts(edit)
	for _, required := range []string{checkpoint, latentvideo.WanTextConditioningSpec(wan, 1).EncoderCheckpoint} {
		if _, err := os.Stat(required); err != nil {
			t.Skipf("UNAVAILABLE: %v", err)
		}
	}
	profile, err := latentvideo.ResolveProfile(wan)
	if err != nil {
		t.Fatal(err)
	}
	profileContent, err := profile.Content()
	if err != nil {
		t.Fatal(err)
	}
	modelID := testutil.ArtifactID(t, artifact.KindModel, name)
	definition, err := modelrecipe.ReferenceVideoEditDefinition(modelID, profile.ID)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	publishArtifactExtent(t, store, modelID, edit)
	clip, err := latentvideo.SourceFramesContent(source)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "test/" + name, Contents: []artifact.Content{profileContent, clip}}); err != nil {
		t.Fatal(err)
	}
	return servedEdits{store: store, directory: edit, execution: modelrecipetest.CandidateExecution(t, store, program), clip: clip.Descriptor.ID}
}

// editService serves LiveEdit requests from a director of its own, closed
// when the test ends: the device memory of a session a test loads is the
// test's to give back.
func editService(t *testing.T) capabilityruntime.Executor {
	t.Helper()
	director, err := editSessions()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := director.Close(context.Background()); err != nil {
			t.Error(err)
		}
	})
	return editExecutor(director, nil)
}

// serve serves one edit of the clip and gives what the request measured and
// the clip it made.
func (edits servedEdits) serve(ctx context.Context, t *testing.T, service capabilityruntime.Executor, prompt string, seed int64) (capabilityruntime.Measured, latentvideo.EncodedVideo) {
	t.Helper()
	raw, err := json.Marshal(latentvideo.ReferenceEditRequest{Prompt: prompt, Seed: seed, SourceArtifact: edits.clip})
	if err != nil {
		t.Fatal(err)
	}
	output, err := service(ctx, edits.store, edits.directory, edits.execution, string(raw))
	if err != nil {
		t.Fatal(err)
	}
	measured, _ := output.(capabilityruntime.Measured)
	video, ok := measured.Output.(latentvideo.EncodedVideo)
	if !ok {
		t.Fatalf("a served edit is %T", output)
	}
	return measured, video
}

// TestServedEditKeepsItsSession serves edits of one clip under one
// instruction, then another, then another seed, from one loaded session. An
// instruction's context and a seed are things a request gives a session, not
// things a session is made of: the second and third edits load nothing, and
// each is the edit a session loaded for it alone makes, byte for byte, as the
// first is again after the others.
func TestServedEditKeepsItsSession(t *testing.T) {
	edits := serveEdits(t, "liveedit-session", stripedClip(21, 16, 32))
	requests := []struct {
		prompt string
		seed   int64
	}{
		{"turn the stripes green", 3},
		{"turn the stripes into falling snow", 3},
		{"turn the stripes into falling snow", 4},
	}
	resident := editService(t)
	kept := make([]latentvideo.EncodedVideo, len(requests))
	for index, request := range requests {
		measured, video := edits.serve(t.Context(), t, resident, request.prompt, request.seed)
		if loaded := measured.Load > 0; loaded != (index == 0) || video.ResidentRun != index+1 {
			t.Errorf("edit %d (%q, seed %d) loaded for %v and is its session's run %d: only the first loads, and one session serves them in turn",
				index, request.prompt, request.seed, measured.Load, video.ResidentRun)
		}
		kept[index] = video
	}
	for index := 1; index < len(requests); index++ {
		if bytes.Equal(kept[index].Data, kept[index-1].Data) {
			t.Errorf("edit %d is the clip edit %d is: its instruction or its seed did not reach the session", index, index-1)
		}
	}
	// The first again, after the others: nothing of them stays in the session.
	if _, again := edits.serve(capabilityruntime.FreshRun(t.Context()), t, resident, requests[0].prompt, requests[0].seed); !bytes.Equal(again.Data, kept[0].Data) {
		t.Error("the first edit, served again after the others, is another clip")
	}
	// The others from a session loaded for the last of them.
	alone := editService(t)
	for index := len(requests) - 1; index > 0; index-- {
		if _, video := edits.serve(capabilityruntime.FreshRun(t.Context()), t, alone, requests[index].prompt, requests[index].seed); !bytes.Equal(video.Data, kept[index].Data) {
			t.Errorf("edit %d from the session that served other edits before it is not the clip a session new to it makes", index)
		}
	}
}
