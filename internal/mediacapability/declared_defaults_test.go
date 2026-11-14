package mediacapability

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAClipsSizeAndASpeechsVoiceAreDeclaredWithDefaults holds the two requests a person sent unanswerable from a page
// left as it opened to what the model declares (declared-defaults/do): a clip's frames and scale are offered with
// defaults, one second at the clip's rate and the model's own size, where the page had offered both empty and a clip
// asked for with neither was refused; and a speech request that names no voice, no prepared voice and no reference
// recording is read in the reference's default voice for the model's language, alba for English and juergen for
// German, and names it, where it was refused. A request that names a recording is passed on as it came.
func TestAClipsSizeAndASpeechsVoiceAreDeclaredWithDefaults(t *testing.T) {
	t.Parallel()
	controls, refusal := Controls(modelrecipe.ModuleOscillatorVideoPrepare, t.TempDir())
	bounds := map[string]Bounds{}
	for _, control := range controls {
		if control.Bounds != nil {
			bounds[control.Name] = *control.Bounds
		}
	}
	if refusal != "" || bounds["frames"] != (Bounds{Default: 8, Rate: 8}) || bounds["scale"] != (Bounds{Default: 1}) {
		t.Fatalf("a clip's controls declare %+v (%s); want frames 8 at 8 a second and scale 1", bounds, refusal)
	}

	model := func(language string) string {
		directory := t.TempDir()
		config := `{"weights_path_without_voice_cloning": "hf://kyutai/pocket-tts-without-voice-cloning/languages/` + language + `/model.safetensors"}`
		testcheck.NoError(t, os.WriteFile(filepath.Join(directory, "pockettts_config.json"), []byte(config), 0o600))
		return directory
	}
	var given []string
	run := func(_ context.Context, _ artifact.Repository, _ string, _ modelrecipe.CapabilityEvidenceSelection, raw string) (any, error) {
		given = append(given, raw)
		return nil, nil
	}
	speak := speechExecutor(run, run)
	recording := testutil.ArtifactID(t, artifact.KindFile, "a recording")
	asked := `{"text": "hello", "reference_audio": "` + recording.String() + `", "reference_end_sample": 10, "voice_memory_bytes": 64}`
	for _, request := range []struct{ language, raw, voice string }{
		{"english_2026-01", `{"text": "hello"}`, "alba"},
		{"german_24l", `{"text": "hello"}`, "juergen"},
		{"english_2026-01", asked, ""},
	} {
		given = nil
		testcheck.Must(speak(t.Context(), nil, model(request.language), modelrecipe.CapabilityEvidenceSelection{}, request.raw))(t)
		var named struct{ Voice string }
		testcheck.NoError(t, json.Unmarshal([]byte(given[0]), &named))
		if named.Voice != request.voice || request.voice == "" && !strings.Contains(given[0], recording.String()) {
			t.Errorf("a %s model reads %s as %s; want the voice %q", request.language, request.raw, given[0], request.voice)
		}
	}
}
