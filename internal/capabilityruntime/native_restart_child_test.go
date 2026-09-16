package capabilityruntime_test

import (
	"os"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/speechrecognition"
	"overgo/internal/strictjson"
	"overgo/internal/workflowruntime"
)

// nativeRestartChildEnvironment carries the continuation request to the
// fresh process the session acceptance re-executes as itself.
const nativeRestartChildEnvironment = "OVERGO_NATIVE_AUDIO_RESTART_CHILD"

// nativeStreamRestart is the continuation a fresh process reproduces: the
// stream's source and checkpoint, the chunks after it, and the outputs the
// uninterrupted pass produced for them.
type nativeStreamRestart struct {
	Store    string                              `json:"store"`
	Recipe   artifact.ID                         `json:"recipe"`
	Source   recipecontract.AudioReference       `json:"source"`
	Resume   artifact.ID                         `json:"resume"`
	Chunks   []workflowruntime.AudioStreamChunk  `json:"chunks"`
	Expected []workflowruntime.AudioStreamResult `json:"expected"`
}

// verifyNativeRestartChild reloads the registered model and the persisted
// stream cursor in this fresh process, refuses a mismatched source before
// residency, and reproduces every continuation output and state identity.
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
