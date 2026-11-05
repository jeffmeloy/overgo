package hfbpe

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/modeltest"
)

// IDs were captured with Hugging Face tokenizers 0.21.1 from MiniCPM5-1B's own
// tokenizer.json by scripts/python_golden_references/minicpm5_tokenizer_ids.py.
// The texts reach each stage MiniCPM5 declares: number runs of at most three
// split out, the Llama-3 expression with number runs of any length, byte-level
// BPE, and its added vocabulary, the raw tokens matched before the 478 it
// declares normalized.
func TestMiniCPM5DeclaredPipelineReference(t *testing.T) {
	t.Parallel()
	const sourceSHA = "32427a1cdc6803fe2ebb730c4fa867cef086d764323a4dbf7199e029ee66ca55"
	const referenceSHA = "ab6a1d29964d4411e1832122486b7e0d4b3c4f963db4bc45f52caa0ef353642a"
	var reference struct {
		SourceSHA string `json:"source_sha256"`
		Version   string `json:"reference_version"`
		Cases     []struct {
			Text string `json:"text"`
			IDs  []int  `json:"ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(readRxFixture(t, "minicpm5-reference.json", referenceSHA), &reference); err != nil {
		t.Fatal(err)
	}
	if reference.SourceSHA != sourceSHA || reference.Version != "0.21.1" || len(reference.Cases) != 12 {
		t.Fatal("MiniCPM5 reference provenance or cardinality changed")
	}
	directory := modeltest.Directory(t, "MiniCPM5-1B")
	full, err := os.ReadFile(filepath.Join(directory, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(full)); got != sourceSHA {
		t.Fatalf("MiniCPM5 tokenizer source drifted: %s", got)
	}
	encoder, err := Load(directory)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range reference.Cases {
		got, err := encoder.Encode(tc.Text)
		if err != nil || !slices.Equal(got, tc.IDs) {
			t.Errorf("%q encodes to %v (%v), the tokenizers library to %v", tc.Text, got, err, tc.IDs)
		}
		if decoded := encoder.Decode(tc.IDs); decoded != tc.Text {
			t.Errorf("%v decodes to %q, want %q", tc.IDs, decoded, tc.Text)
		}
	}
}
