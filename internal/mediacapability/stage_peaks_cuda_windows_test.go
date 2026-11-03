//go:build windows && modeltest

package mediacapability

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/latentimage"
	"overgo/internal/latentvideo"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/modeltest"
	"overgo/internal/overgodb"
	"overgo/internal/processmeasure"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// The writer's requests: the prompts and sizes stagePeakRequests names.
const (
	stagePeaksPrompt      = "A small red sailboat on a quiet lake at sunrise."
	stagePeaksOtherPrompt = "A lighthouse on a rocky cliff under a starry sky."
	stagePeaksEdit        = "turn the stripes into falling snow"
	stagePeaksSquare      = 1024
	stagePeaksLargest     = 2048
	stagePeaksClipFrames  = 81
	// wholeRequest names the part a request is read as: what the process held at most during it is its peak.
	wholeRequest = "the whole request"
)

// stagePeaksProtocol says how the record was read.
const stagePeaksProtocol = "One test process. Each family's requests are served from a session director of its own, closed before the next " +
	"family loads, so the card holds one family at a time, the first request of each a fresh load. Each request is read under a " +
	"ledger of its parts that reads the process's device high-water mark at each part's start and end; a part's peak is the most " +
	"the process held while it was open, a part begun inside it included, and the whole request's the most it held during it. " +
	"Krea at 1024 square on its own steps; MOSS on its own steps; Wan at its own frames, size and steps; LiveEdit on a striped clip " +
	"at its source size; the autoencoder on the frame the reconstruction evaluation cuts. Answers are never served from the store."

// TestMediaStagePeaksAreRecorded serves each request stagePeakRequests names on the installed models and writes, to the
// record TestStagePeaksCoverEveryMediaFamily reads, the device memory each named part of each request held. It takes
// about half an hour of the card, so it runs only as a probe.
func TestMediaStagePeaksAreRecorded(t *testing.T) {
	cudatest.Require(t)
	cudatest.RequireProbe(t)
	root := testutil.RepoRoot(t)
	commit, err := gitauthority.Query(t.Context(), root, "rev-parse", "HEAD")
	if err != nil {
		t.Fatal(err)
	}
	library, err := driver.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer library.Close()
	if err := library.Init(); err != nil {
		t.Fatal(err)
	}
	device, err := library.DeviceInfo(0)
	if err != nil {
		t.Fatal(err)
	}
	record := stagePeaksRecord{Commit: strings.TrimSpace(string(commit)), Device: device.Name, Protocol: stagePeaksProtocol}
	// read serves one request under a ledger that reads the device, and keeps what each part held.
	read := func(family, label string, request any, served servedPictures, service capabilityruntime.Executor) {
		t.Helper()
		raw, err := json.Marshal(request)
		if err != nil {
			t.Fatal(err)
		}
		ctx, stages := processmeasure.WithDeviceStages(capabilityruntime.FreshRun(t.Context()), driver.StagePeakBytes, driver.ResetStagePeakBytes)
		clock := processmeasure.NewStopwatch()
		whole := processmeasure.Stage(ctx, wholeRequest)
		_, err = service(ctx, served.store, served.directory, served.execution, string(raw))
		whole()
		elapsed, clockErr := clock.Elapsed()
		parts, partsErr := stages.Parts()
		if err := errors.Join(err, clockErr, partsErr); err != nil {
			t.Fatalf("%s at %s: %v", family, label, err)
		}
		at := slices.IndexFunc(parts, func(part processmeasure.Part) bool { return part.Name == wholeRequest })
		// The whole request's part is read before it is taken out: deleting it clears the slice's last element.
		peak := parts[at].PeakDeviceBytes
		reading := stagePeaksRequest{
			Family: family, Label: label, Request: raw, Wall: time.Duration(elapsed),
			PeakDeviceBytes: peak, Stages: slices.Delete(parts, at, at+1),
		}
		var report strings.Builder
		if _, err := stages.Report(&report, reading.Wall); err != nil {
			t.Fatal(err)
		}
		t.Logf("%s at %s: %.1f s, %.2f GiB at most\n%s", family, label, reading.Wall.Seconds(), float64(reading.PeakDeviceBytes)/(1<<30), report.String())
		record.Requests = append(record.Requests, reading)
	}
	// closed gives back a family's sessions before the next family loads.
	closed := func(director interface{ Close(context.Context) error }) {
		t.Helper()
		if err := director.Close(t.Context()); err != nil {
			t.Fatal(err)
		}
	}

	krea, pictures := servePictures(t, "krea-stage-peaks"), pictureService(t)
	for _, request := range []struct {
		label, prompt string
		seed          int64
	}{{"first", stagePeaksPrompt, 1}, {"repeat", stagePeaksPrompt, 2}, {"new prompt", stagePeaksOtherPrompt, 2}} {
		read("krea", request.label, latentimage.Request{Prompt: request.prompt, Width: stagePeaksSquare, Height: stagePeaksSquare, Seed: request.seed}, krea, pictures.Executor())
	}
	closed(pictures)

	moss, pictures := catalogPictures(t, modeltest.Directory(t, "MOSS-Image-SFT-797fa8e678b4f")), pictureService(t)
	for _, size := range []int{stagePeaksSquare, stagePeaksLargest} {
		read("moss", strconv.Itoa(size), latentimage.Request{Prompt: stagePeaksPrompt, Width: size, Height: size, Seed: 1}, moss, pictures.Executor())
	}
	closed(pictures)

	wan, clips := servedClips(t), wanService(t)
	read("wan", "user size", latentvideo.WanRequest{Prompt: stagePeaksPrompt, Seed: 1}, wan, sessionExecutor(clips, nil))
	closed(clips)

	width, height, err := latentvideo.EditSourceSize(modeltest.Directory(t, "LiveEdit"))
	if err != nil {
		t.Fatal(err)
	}
	edits := serveEdits(t, "liveedit-stage-peaks", stripedClip(stagePeaksClipFrames, height, width))
	editor, err := editSessions()
	if err != nil {
		t.Fatal(err)
	}
	read("liveedit", "81 frames", latentvideo.ReferenceEditRequest{Prompt: stagePeaksEdit, Seed: 1, SourceArtifact: edits.clip},
		servedPictures{store: edits.store, directory: edits.directory, execution: edits.execution}, editExecutor(editor, nil))
	closed(editor)

	directory := modeltest.Directory(t, "Wan2.1-T2V-1.3B")
	read("wan-vae", "round trip", latentvideo.ReconstructionRequest{Image: roundTripFrame(t)}, servedReconstruction(t, directory), Catalog[recipe.TaskReconstruction].Execute)

	if err := jsonfile.Write(filepath.Join(root, stagePeaksRecordPath), record, 0o644); err != nil {
		t.Fatal(err)
	}
}

// servedClips is a store a clip by the installed Wan model is served from: the model's artifact and its profile.
func servedClips(t *testing.T) servedPictures {
	t.Helper()
	wan := modeltest.Directory(t, "Wan2.1-T2V-1.3B")
	if _, err := os.Stat(latentvideo.WanTextConditioningSpec(wan, 1).EncoderCheckpoint); err != nil {
		t.Skipf("UNAVAILABLE: encoder checkpoint absent: %v", err)
	}
	profile, err := latentvideo.ResolveProfile(wan)
	if err != nil {
		t.Fatal(err)
	}
	profileContent, err := profile.Content()
	if err != nil {
		t.Fatal(err)
	}
	modelID := testutil.ArtifactID(t, artifact.KindModel, "wan-stage-peaks")
	definition, err := modelrecipe.GenerationDefinition(modelrecipe.ModuleLatentVideoPrepare, modelID, profile.ID)
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
	publishArtifactExtent(t, store, modelID, wan)
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "test/wan-stage-peaks", Contents: []artifact.Content{profileContent}}); err != nil {
		t.Fatal(err)
	}
	return servedPictures{store: store, directory: wan, execution: modelrecipetest.CandidateExecution(t, store, program)}
}

// wanService makes a director of Wan sessions of its own, closed when the test ends at the latest.
func wanService(t *testing.T) *wanDirector {
	t.Helper()
	director, err := wanSessions()
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

// roundTripFrame is the frame the reconstruction evaluation cuts from its first clip, as the PNG a request carries.
func roundTripFrame(t *testing.T) []byte {
	t.Helper()
	fixtures := filepath.Join(testutil.RepoRoot(t), "fixtures", "wan")
	var reference videoAutoencoderReference
	if err := jsonfile.DecodeStrict(filepath.Join(fixtures, "vae_round_trip.json"), &reference); err != nil {
		t.Fatalf("the reference round trip is absent (%v); write it with %s", err, videoAutoencoderScript)
	}
	frame, err := os.ReadFile(filepath.Join(fixtures, filepath.FromSlash(reference.Picture.File)))
	if err != nil {
		t.Fatal(err)
	}
	return frame
}
