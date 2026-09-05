package mediacapability

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/modelrecipe"
)

// TestControlsOfferExportedVoices pins the choices a request field takes
// from the model's artifact: the speech request's voice control offers
// every voice the directory exports, newest generation first, and a
// directory exporting none leaves the control a free input.
func TestControlsOfferExportedVoices(t *testing.T) {
	directory := t.TempDir()
	for _, path := range []string{"embeddings_v2/alba.safetensors", "embeddings_v3/anna.safetensors", "embeddings_v3/alba.safetensors", "embeddings_v3/notes.txt"} {
		if err := os.MkdirAll(filepath.Dir(filepath.Join(directory, path)), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(directory, path), []byte("voice"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	controls, refusal := Controls(modelrecipe.ModuleSpeechTokenize, directory)
	if refusal != "" {
		t.Fatal(refusal)
	}
	var voice *Control
	for index := range controls {
		if controls[index].Name == "voice" {
			voice = &controls[index]
		}
	}
	if voice == nil || voice.Type != ControlText || !voice.Required || len(voice.Choices) != 2 || voice.Choices[0] != "alba" || voice.Choices[1] != "anna" {
		t.Fatalf("voice control = %+v", voice)
	}
	plain, refusal := Controls(modelrecipe.ModuleSpeechTokenize, "")
	if refusal != "" || len(plain) != len(controls) {
		t.Fatalf("controls without a directory = %+v, %q", plain, refusal)
	}
	for _, control := range plain {
		if len(control.Choices) != 0 {
			t.Fatalf("a directory-less control carries choices: %+v", control)
		}
	}
	if _, refusal := Controls(modelrecipe.ModuleOscillatorImagePrepare, directory); refusal != "" {
		t.Fatalf("a request without exported fields was refused: %q", refusal)
	}
}
