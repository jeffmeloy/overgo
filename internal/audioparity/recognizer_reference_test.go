package audioparity

import (
	"encoding/json"
	"os"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/speechrecognition"
	"overgo/internal/strictjson"
	"overgo/internal/testskip"
)

// TestRecognizerReproducesReferenceTranscripts holds the packaged recognizer
// to the same transcripts the boundary-by-boundary acceptance checks: given
// the checkpoint, its execution declaration, its processor declaration and
// its frontend, one call per recording returns exactly the reference text.
// The acceptance beside this one proves every intermediate value; this one
// proves that what the onboarding path will load and call is that same
// pipeline, assembled once and kept, and that an empty request is refused
// before a model is read.
func TestRecognizerReproducesReferenceTranscripts(t *testing.T) {
	t.Parallel()
	if testing.Short() {
		t.Skip(testskip.ShortIntegration)
	}
	if err := speechrecognition.ValidateTranscriptionRequest(speechrecognition.TranscriptionRequest{}); err == nil {
		t.Fatal("an empty transcription request was accepted")
	}
	storeRoot := resolveReferenceRoots(t).store
	store, err := overgodb.OpenReadOnly(storeRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	encoded, err := os.ReadFile("testdata/granite_speech_5_election.json")
	if err != nil {
		t.Fatal(err)
	}
	election, err := NormalizeElection(encoded)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err = os.ReadFile("testdata/encoder_capture.json")
	if err != nil {
		t.Fatal(err)
	}
	var capture encoderCapture
	if err := strictjson.DecodeBytes(encoded, &capture); err != nil {
		t.Fatal(err)
	}
	if err := capture.validate(election); err != nil {
		t.Fatal(err)
	}
	modelRoot, _ := loadASRModel(t, store, election)
	spec, err := graniteRecognizerSpec()
	if err != nil {
		t.Fatal(err)
	}
	// Explicit integration-test host ceiling, not a production default.
	const referenceMemoryBytes = 4 << 30
	recognizer, err := speechrecognition.LoadRecognizer(t.Context(), modelRoot, spec, referenceMemoryBytes)
	if err != nil {
		t.Fatal(err)
	}
	corpusPath, err := artifact.AvailablePath(t.Context(), store, election.Dataset.Artifact, artifact.LocationFile)
	if err != nil {
		t.Fatal(err)
	}
	completed := 0
	err = dataset.ReadParquetTextRows(t.Context(), corpusPath, "bytes", len(capture.Cases), func(index uint64, payload string) error {
		if index >= uint64(len(capture.Cases)) {
			t.Fatal("extra corpus row")
		}
		expected := capture.Cases[index]
		text, transcribeErr := recognizer.Transcribe(t.Context(), speechrecognition.TranscriptionRequest{
			Audio: []byte(payload), MaximumSamples: expected.Samples, MemoryBytes: referenceMemoryBytes,
		})
		if transcribeErr != nil {
			t.Fatalf("%s: %v", expected.Fixture, transcribeErr)
		}
		if text != expected.Text {
			t.Fatalf("%s transcript %q, want %q", expected.Fixture, text, expected.Text)
		}
		completed++
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if completed != len(capture.Cases) {
		t.Fatalf("transcribed %d of %d recordings", completed, len(capture.Cases))
	}
	t.Logf("packaged recognizer: %d recordings transcribed exactly; no training, streaming or word-error claim", completed)
}

// graniteRecognizerSpec reads the declarations this checkpoint carries: how
// its encoder is bound, how its waveform becomes features, and which
// vocabulary entry the classifier reads as blank.
func graniteRecognizerSpec() (speechrecognition.RecognizerSpec, error) {
	encoded, err := os.ReadFile("recipes/granite5asr_execution.json")
	if err != nil {
		return speechrecognition.RecognizerSpec{}, err
	}
	var declaration speechrecognition.Declaration
	if err := strictjson.DecodeBytes(encoded, &declaration); err != nil {
		return speechrecognition.RecognizerSpec{}, err
	}
	encoded, err = os.ReadFile("../audiodsp/testdata/power_logmel_frontend.json")
	if err != nil {
		return speechrecognition.RecognizerSpec{}, err
	}
	var frontend audiodsp.FrontendConfig
	if err := json.Unmarshal(encoded, &frontend); err != nil {
		return speechrecognition.RecognizerSpec{}, err
	}
	recipe, _, err := graniteRecipe()
	if err != nil {
		return speechrecognition.RecognizerSpec{}, err
	}
	return speechrecognition.RecognizerSpec{
		Declaration: declaration, Frontend: frontend,
		Grouping: audiodsp.GroupedFeatureConfig{
			StackFrames: int(recipe.Preprocessor.StackFactor),
			DeltaRadius: int(recipe.Preprocessor.DeltaWinLength-1) / 2,
			// The reference selects every grouped frame; the declaration states
			// no stride of its own.
			FinalFrameSamples: 1,
		},
		Blank: int(recipe.Config.PadTokenID),
	}, nil
}
