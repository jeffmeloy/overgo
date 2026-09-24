package hfbpe

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/modeltest"
	"overgo/internal/tokenizer"
)

type rxBrainSample struct {
	Text   string     `json:"text"`
	IDs    []int      `json:"ids"`
	Pieces []string   `json:"pieces"`
	Stages [][]string `json:"stages"`
}

// IDs and intermediate pieces were captured with Hugging Face tokenizers
// 0.22.2 before this compiler was changed. The projected vocabulary retains
// the source IDs and every merge reachable from these byte-level inputs.
func TestRxBrainDeclaredPipelineReference(t *testing.T) {
	const sourceSHA = "ae5ca95eeb8e9a8774513a996e4e820a05c27db32966aa8e071c10828402cb73"
	const referenceSHA = "f7dcb5718d7db84575e3d9e208abcc451bcafac737c84c2f5d9f746a85f7c468"
	const projectionSHA = "133d9763670437cf8c36a6507392f50577d65568f007a0fcc1e7fc11ca5ca61a"
	var reference struct {
		SourceSHA string          `json:"source_sha256"`
		Version   string          `json:"reference_version"`
		Cases     []rxBrainSample `json:"cases"`
	}
	raw := readRxFixture(t, "rxbrain-reference.json", referenceSHA)
	if err := json.Unmarshal(raw, &reference); err != nil {
		t.Fatal(err)
	}
	if reference.SourceSHA != sourceSHA || reference.Version != "0.22.2" || len(reference.Cases) != 153 {
		t.Fatal("RxBrain reference provenance or cardinality changed")
	}
	projection := readRxFixture(t, "rxbrain-tokenizer-projection.json", projectionSHA)
	modelDir := modeltest.Directory(t, "Hy-Embodied-RxBrain-1.0")
	full, err := os.ReadFile(filepath.Join(modelDir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(full)); got != sourceSHA {
		t.Fatalf("RxBrain tokenizer source drifted: %s", got)
	}
	for _, artifact := range []struct {
		name string
		raw  []byte
	}{
		{"projection", projection}, {"full", full},
	} {
		t.Run(artifact.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), artifact.raw, 0600); err != nil {
				t.Fatal(err)
			}
			encoder, err := Load(dir)
			if err != nil {
				t.Fatal(err)
			}
			for index, tc := range reference.Cases {
				if artifact.name == "projection" {
					parts := encoder.preTokenize(tc.Text)
					var encoded []string
					for _, part := range parts {
						var b strings.Builder
						for i := range len(part) {
							b.WriteRune(encoder.b2u[part[i]])
						}
						encoded = append(encoded, b.String())
					}
					if !slices.Equal(encoded, tc.Pieces) {
						t.Fatalf("case %d pre-tokenized pieces: got=%q want=%q", index, encoded, tc.Pieces)
					}
				}
				got, err := encoder.Encode(tc.Text)
				if err != nil || !slices.Equal(got, tc.IDs) {
					t.Fatalf("case %d input=%q IDs=%v want=%v err=%v", index, tc.Text, got, tc.IDs, err)
				}
			}
			if artifact.name == "projection" {
				checkRxBrainStages(t, artifact.raw, reference.Cases)
			}
		})
	}
}

func readRxFixture(t *testing.T, name, digest string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(raw)); got != digest {
		t.Fatalf("%s digest changed: %s", name, got)
	}
	return raw
}

func checkRxBrainStages(t *testing.T, artifact []byte, cases []rxBrainSample) {
	t.Helper()
	var declaration struct {
		PreTokenizer struct {
			Stages []struct {
				Pattern struct {
					Regex string `json:"Regex"`
				} `json:"pattern"`
			} `json:"pretokenizers"`
		} `json:"pre_tokenizer"`
	}
	if err := json.Unmarshal(artifact, &declaration); err != nil {
		t.Fatal(err)
	}
	if len(declaration.PreTokenizer.Stages) != 4 {
		t.Fatal("RxBrain stage count changed")
	}
	for index, tc := range cases {
		if len(tc.Stages) == 0 {
			continue
		}
		parts := []string{tc.Text}
		for stage := range tc.Stages {
			split, _, err := tokenizer.CompileBPESplit(declaration.PreTokenizer.Stages[stage].Pattern.Regex)
			if err != nil {
				t.Fatal(err)
			}
			var next []string
			for _, part := range parts {
				next = append(next, split(part)...)
			}
			parts = next
			if !slices.Equal(parts, tc.Stages[stage]) {
				t.Fatalf("case %d stage %d got=%q want=%q", index, stage, parts, tc.Stages[stage])
			}
		}
	}
}
