//go:build modeltest && windows

package mediacapability

import (
	"bytes"
	"context"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/modeltest"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// servePictures opens a store that holds what a served picture stands on. It
// skips where the model is not installed.
func servePictures(t *testing.T, name string) servedPictures {
	t.Helper()
	cudatest.Require(t)
	directory := modeltest.Directory(t, "Krea-2-Turbo")
	profile := testcheck.Must(latentimage.ResolveProfile(directory))(t)
	profileContent := testcheck.Must(profile.Content())(t)
	modelID := testutil.ArtifactID(t, artifact.KindModel, name)
	definition := testcheck.Must(modelrecipe.GenerationDefinition(modelrecipe.ModuleLatentImagePrepare, modelID, profile.ID))(t)
	program := testcheck.Must(modelrecipe.CompileCapability(definition))(t)
	store := overgodbtest.Open(t, t.TempDir())
	publishArtifactExtent(t, store, modelID, directory)
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "test/" + name, Contents: []artifact.Content{profileContent}}); err != nil {
		t.Fatal(err)
	}
	return servedPictures{store: store, directory: directory, execution: modelrecipetest.CandidateExecution(t, store, program)}
}

// TestServedPictureKeepsItsSession serves pictures of one extent under one
// prompt, then another, then another seed, from one loaded session. A prompt
// and a seed are things a request gives a session, not things a session is
// made of: the second and third pictures load nothing, and each is the picture
// a session loaded for it alone makes, byte for byte, as the first is again
// after the others. A request cancelled while it places another prompt's text
// encoder on the device gives back what the session held: the request after
// it loads a session of its own, which would not fit beside what a session
// left behind, and makes the same picture.
func TestServedPictureKeepsItsSession(t *testing.T) {
	pictures := servePictures(t, "picture-session")
	requests := []struct {
		prompt string
		seed   int64
	}{
		{"a red fox licking a vanilla ice cream cone in snow", 3},
		{"a lighthouse on a cliff in a storm at night, waves breaking below it", 3},
		{"a lighthouse on a cliff in a storm at night, waves breaking below it", 4},
	}
	director := pictureService(t)
	resident := director.Executor()
	kept := make([]latentimage.EncodedImage, len(requests))
	for index, request := range requests {
		began := time.Now()
		measured, picture := pictures.serve(t.Context(), t, resident, request.prompt, request.seed)
		t.Logf("picture %d (%q, seed %d): served in %v, loaded for %v, stages %v", index, request.prompt, request.seed, time.Since(began).Round(time.Millisecond), measured.Load, measured.Phases)
		if loaded := measured.Load > 0; loaded != (index == 0) {
			t.Errorf("picture %d (%q, seed %d) loaded for %v: only the first loads, and one session serves them in turn", index, request.prompt, request.seed, measured.Load)
		}
		kept[index] = picture
	}
	for index := 1; index < len(requests); index++ {
		if bytes.Equal(kept[index].Data, kept[index-1].Data) {
			t.Errorf("picture %d is the picture %d is: its prompt or its seed did not reach the session", index, index-1)
		}
	}
	// The first again, after the others: nothing of them stays in the session.
	if measured, again := pictures.serve(capabilityruntime.FreshRun(t.Context()), t, resident, requests[0].prompt, requests[0].seed); !bytes.Equal(again.Data, kept[0].Data) || measured.Load > 0 {
		t.Errorf("the first picture, served again after the others, loaded for %v or is another picture", measured.Load)
	}
	// A request for another prompt, cancelled in the middle of placing that
	// prompt's text encoder: it fails, and what the session held is given back.
	cancelled, cancel := context.WithCancel(capabilityruntime.FreshRun(t.Context()))
	defer cancel()
	left := 100
	if _, err := resident(checkedContext{cancelled, cancel, &left}, pictures.store, pictures.directory, pictures.execution, pictures.request(t, requests[1].prompt, requests[1].seed)); err == nil || left > 0 {
		t.Fatalf("a request cancelled in the middle of its work ended with %v and %d checks to go; want it cancelled", err, left)
	}
	if measured, after := pictures.serve(capabilityruntime.FreshRun(t.Context()), t, resident, requests[0].prompt, requests[0].seed); !bytes.Equal(after.Data, kept[0].Data) || measured.Load == 0 {
		t.Errorf("after a cancelled request the next loaded for %v: it was served by the session the cancelled request left, or made another picture", measured.Load)
	}
	// The others from a session loaded for the last of them, once the first
	// director has given the device back.
	testcheck.NoError(t, director.Close(t.Context()))
	alone := pictureService(t).Executor()
	for index := len(requests) - 1; index > 0; index-- {
		if _, picture := pictures.serve(capabilityruntime.FreshRun(t.Context()), t, alone, requests[index].prompt, requests[index].seed); !bytes.Equal(picture.Data, kept[index].Data) {
			t.Errorf("picture %d from the session that served other pictures before it is not the picture a session new to it makes", index)
		}
	}
}
