package capabilityruntime_test

import (
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/speechrecognition"
	"overgo/internal/speechrecognitiontest"
	"overgo/internal/strictjson"
	"overgo/internal/testskip"
	"overgo/internal/workflowruntime"
)

type nativeStreamRestart struct {
	Store    string                              `json:"store"`
	Recipe   artifact.ID                         `json:"recipe"`
	Source   recipecontract.AudioReference       `json:"source"`
	Resume   artifact.ID                         `json:"resume"`
	Chunks   []workflowruntime.AudioStreamChunk  `json:"chunks"`
	Expected []workflowruntime.AudioStreamResult `json:"expected"`
}

func TestNativeAudioStreamIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration + ": registered native model and fresh-process restart required")
	}
	if path := os.Getenv("OVERGO_NATIVE_AUDIO_RESTART_CHILD"); path != "" {
		verifyNativeRestartChild(t, path)
		return
	}
	directory := t.TempDir()
	storePath := filepath.Join(directory, "store")
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	fixture := speechrecognitiontest.PublishNative(t, store)
	// The longest capture crosses the bounded attention-history rollover.
	clip := slices.MaxFunc(fixture.Clips, func(a, b speechrecognitiontest.NativeClip) int { return len(a.PCM) - len(b.PCM) })
	source, chunks := fixture.PublishStream(t, store, clip, fixture.Profile.Frontend.SampleRate+1)
	session, err := speechrecognition.LoadSession(t.Context(), store, fixture.Definition.ID, 4<<30)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(t.Context())
	stream, err := session.OpenStream(t.Context(), speechrecognition.StreamRequest{Source: source, Resume: artifact.ID{}})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close(t.Context())
	request := nativeStreamRestart{Store: storePath, Recipe: fixture.Definition.ID, Source: source, Chunks: chunks[1:]}
	var text string
	for i, chunk := range chunks {
		result, err := stream.Process(t.Context(), chunk)
		if err != nil {
			t.Fatal(err)
		}
		output, err := speechrecognition.RequireTranscriptionChunk(t.Context(), store, result.Output)
		if err != nil {
			t.Fatal(err)
		}
		text += output.Text
		if i == 0 {
			batch, err := stream.CheckpointBatch("native-stream/fresh-process")
			if err == nil {
				_, err = artifact.CommitBatch(t.Context(), store, batch)
			}
			if err != nil {
				t.Fatal(err)
			}
			request.Resume = batch.Contents[0].Descriptor.ID
		} else {
			request.Expected = append(request.Expected, result)
		}
	}
	if text != clip.Streaming {
		t.Fatalf("native transcript=%q want=%q", text, clip.Streaming)
	}
	if err := errors.Join(stream.Close(t.Context()), session.Close(t.Context()), store.Close()); err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(directory, "restart.json")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	command := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestNativeAudioStreamIntegration$", "-test.count=1", "-test.v")
	command.Env = append(os.Environ(), "OVERGO_NATIVE_AUDIO_RESTART_CHILD="+path)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("fresh-process recurrent restart: %v\n%s", err, output)
	}
	t.Logf("%s", output)
	t.Logf("native CPU waveform stream: %d chunks; fresh-process continuation after chunk 1; exact output and restart artifact identities through finalization; capture=%s; no GPU or word-timestamp claim", len(chunks), clip.Capture)
}

func verifyNativeRestartChild(t *testing.T, path string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var request nativeStreamRestart
	if err := strictjson.DecodeBytes(data, &request); err != nil {
		t.Fatal(err)
	}
	if len(request.Chunks) == 0 || len(request.Chunks) != len(request.Expected) {
		t.Fatal("restart child has no exact continuation denominator")
	}
	store, err := overgodb.Open(request.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	session, err := speechrecognition.LoadSession(t.Context(), store, request.Recipe, 4<<30)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(t.Context())
	wrong := request.Source
	wrong.Audio = request.Chunks[0].Audio
	if stream, err := session.OpenStream(t.Context(), speechrecognition.StreamRequest{Source: wrong, Resume: request.Resume}); err == nil || stream != nil || session.Snapshot().Active != 0 {
		t.Fatal("wrong source restart acquired or retained a lease")
	}
	stream, err := session.OpenStream(t.Context(), speechrecognition.StreamRequest{Source: request.Source, Resume: request.Resume})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close(t.Context())
	for i, chunk := range request.Chunks {
		result, err := stream.Process(t.Context(), chunk)
		if err != nil || result != request.Expected[i] {
			t.Fatalf("child chunk %d changed output or state: %v", chunk.Sequence, err)
		}
	}
	if session.Snapshot().Active != 0 {
		t.Fatal("child finalization retained model lease")
	}
	t.Logf("fresh process reproduced %d completed output/state pairs bit-for-bit", len(request.Chunks))
}
