package speechsynth

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestPocketTTSLanguageContract(t *testing.T) {
	fixture := loadFixture[struct {
		Config     json.RawMessage
		Shapes     map[string][]int
		Tokenizer  []byte
		Identities map[string]string
		Cases      []struct {
			Text, Spoken   string
			IDs            []int
			FramesAfterEnd int `json:"frames_after_end"`
		}
	}](t, "german_contract.json")
	var config artifactConfig
	if err := json.Unmarshal(fixture.Config, &config); err != nil {
		t.Fatal(err)
	}
	dims, err := deriveDims(fixture.Shapes, config)
	if err != nil {
		t.Fatal(err)
	}
	directory := t.TempDir()
	path := filepath.Join(directory, "tokenizer.model")
	if err := os.WriteFile(path, fixture.Tokenizer, 0o600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(fixture.Tokenizer)
	if hex.EncodeToString(sum[:]) != fixture.Identities["tokenizer.model"] {
		t.Fatal("tokenizer differs from independent package")
	}
	tokenizer, err := LoadTokenizer(path)
	if err != nil {
		t.Fatal(err)
	}
	var policy struct {
		RemoveSemicolons bool `json:"remove_semicolons"`
	}
	if err := json.Unmarshal(fixture.Config, &policy); err != nil || !policy.RemoveSemicolons || len(fixture.Cases) == 0 {
		t.Fatalf("language text contract absent: %v", err)
	}
	synth := &Synthesizer{model: &Model{Dims: dims, Codec: &CodecDecoder{}}, tokenizer: tokenizer, removeSemicolons: policy.RemoveSemicolons}
	for _, reference := range fixture.Cases {
		spoken, frames, err := spokenText(reference.Text, policy.RemoveSemicolons)
		if err != nil || spoken != reference.Spoken || frames != reference.FramesAfterEnd {
			t.Fatalf("%q: preparation=%q frames=%d err=%v", reference.Text, spoken, frames, err)
		}
		plan, err := synth.tokenize(SynthesisRequest{Text: reference.Text, Voice: "anna"})
		if err != nil || !slices.Equal(plan.tokens, reference.IDs) || plan.framesAfterEnd != reference.FramesAfterEnd {
			t.Fatalf("%q: request tokenization=%v err=%v", reference.Text, plan.tokens, err)
		}
	}
}
