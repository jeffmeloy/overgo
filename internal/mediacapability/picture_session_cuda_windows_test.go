//go:build windows

package mediacapability

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/modeltest"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// servedPictures is a store a picture by the installed Krea model is served
// from: the model's artifact and its profile.
type servedPictures struct {
	store     *overgodb.Store
	directory string
	execution modelrecipe.CapabilityEvidenceSelection
}

// servePictures opens a store that holds what a served picture stands on. It
// skips where the model is not installed.
func servePictures(t *testing.T, name string) servedPictures {
	t.Helper()
	cudatest.Require(t)
	directory := modeltest.Directory(t, "Krea-2-Turbo")
	profile, err := latentimage.ResolveProfile(directory)
	if err != nil {
		t.Fatal(err)
	}
	profileContent, err := profile.Content()
	if err != nil {
		t.Fatal(err)
	}
	modelID := testutil.ArtifactID(t, artifact.KindModel, name)
	definition, err := modelrecipe.GenerationDefinition(modelrecipe.ModuleLatentImagePrepare, modelID, profile.ID)
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
	publishArtifactExtent(t, store, modelID, directory)
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "test/" + name, Contents: []artifact.Content{profileContent}}); err != nil {
		t.Fatal(err)
	}
	return servedPictures{store: store, directory: directory, execution: modelrecipetest.CandidateExecution(t, store, program)}
}

// pictureService makes a director of picture sessions of its own, closed
// when the test ends at the latest: the device memory of a session a test
// loads is the test's to give back, and two sessions of the model do not fit
// the device together.
func pictureService(t *testing.T) *pictureDirector {
	t.Helper()
	director, err := pictureSessions()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := director.Close(context.Background()); err != nil {
			t.Error(err)
		}
	})
	return director
}

// request is one small picture's request: the prompt and the seed are what
// the test varies.
func (servedPictures) request(t *testing.T, prompt string, seed int64) string {
	t.Helper()
	raw, err := json.Marshal(latentimage.Request{Prompt: prompt, Width: 256, Height: 256, Steps: 8, Seed: seed, DynamicShiftMu: 1.15})
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

// serve serves one picture and gives what the request measured and the
// picture it made.
func (pictures servedPictures) serve(ctx context.Context, t *testing.T, service capabilityruntime.Executor, prompt string, seed int64) (capabilityruntime.Measured, latentimage.EncodedImage) {
	t.Helper()
	output, err := service(ctx, pictures.store, pictures.directory, pictures.execution, pictures.request(t, prompt, seed))
	if err != nil {
		t.Fatal(err)
	}
	measured, _ := output.(capabilityruntime.Measured)
	picture, ok := measured.Output.(latentimage.EncodedImage)
	if !ok {
		t.Fatalf("a served picture is %T", output)
	}
	return measured, picture
}

// checkedContext is cancelled when its error has been asked for a number of
// times: a request's work asks at each tensor it places on the device, so the
// cancellation falls in the middle of that work, at the same place each run.
type checkedContext struct {
	context.Context
	cancel context.CancelFunc
	left   *int
}

func (ctx checkedContext) Err() error {
	if *ctx.left--; *ctx.left == 0 {
		ctx.cancel()
	}
	return ctx.Context.Err()
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
	if err := director.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	alone := pictureService(t).Executor()
	for index := len(requests) - 1; index > 0; index-- {
		if _, picture := pictures.serve(capabilityruntime.FreshRun(t.Context()), t, alone, requests[index].prompt, requests[index].seed); !bytes.Equal(picture.Data, kept[index].Data) {
			t.Errorf("picture %d from the session that served other pictures before it is not the picture a session new to it makes", index)
		}
	}
}
