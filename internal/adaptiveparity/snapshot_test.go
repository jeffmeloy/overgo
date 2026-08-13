package adaptiveparity

import (
	"bytes"
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
)

func TestSnapshotImportRoundTrip(t *testing.T) {
	model := identify(t, artifact.KindModel, "fixture-model")
	dataset := identify(t, artifact.KindDataset, "fixture-corpus")
	golden := identify(t, artifact.KindEvidence, "fixture-golden")
	measurement := identify(t, artifact.KindEvidence, "fixture-measurement")

	snapshot, err := New(
		[]Source{{Name: "adaptive", Repository: "local/adaptive_new", Commit: "214950b3b0316bcdcab38a3b95127a927c0ab5be"}},
		[]Capability{{
			ID: "image-to-text", Source: "adaptive", EntryPoint: "go/extmodel/vqa.go:Run",
			Signature: Signature{Inputs: []Modality{ModalityImage, ModalityText}, Outputs: []Modality{ModalityText}},
			Artifacts: []Reference{{Name: "model", Identity: model}},
			Corpora:   []Reference{{Name: "heldout", Identity: dataset}},
			Goldens:   []Reference{{Name: "answers", Identity: golden}},
			Measurements: []Measurement{{
				Runtime: "adaptive", Protocol: "seed=42; warm; exact-answer", WallNanos: 18_400_000_000,
				PeakBytes: 11_970_000_000, Evidence: measurement,
			}},
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	content, err := snapshot.Content()
	if err != nil {
		t.Fatal(err)
	}

	// Import accepts noncanonical producer field order, then owns canonical bytes.
	var unordered map[string]any
	if err := json.Unmarshal(content.Data, &unordered); err != nil {
		t.Fatal(err)
	}
	producer, err := json.MarshalIndent(unordered, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	imported, canonical, err := Import(bytes.NewReader(producer))
	if err != nil {
		t.Fatal(err)
	}
	if imported.ID != snapshot.ID || !bytes.Equal(canonical, content.Data) {
		t.Fatalf("import identity/bytes differ: %s vs %s", imported.ID, snapshot.ID)
	}
	parsed, err := Parse(canonical)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.ID != snapshot.ID || parsed.Capabilities[0].Signature.Inputs[0] != ModalityImage {
		t.Fatalf("round trip changed identity or ordered signature: %+v", parsed)
	}
}

func identify(t *testing.T, kind artifact.Kind, value string) artifact.ID {
	t.Helper()
	id, err := artifact.IdentifyBytes(kind, []byte(value))
	if err != nil {
		t.Fatal(err)
	}
	return id
}
