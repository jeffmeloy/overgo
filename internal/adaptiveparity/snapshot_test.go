package adaptiveparity

import (
	"bytes"
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestSnapshotImportRoundTrip(t *testing.T) {
	model := testutil.ArtifactID(t, artifact.KindModel, "fixture-model")
	dataset := testutil.ArtifactID(t, artifact.KindDataset, "fixture-corpus")
	golden := testutil.ArtifactID(t, artifact.KindEvidence, "fixture-golden")
	measurement := testutil.ArtifactID(t, artifact.KindEvidence, "fixture-measurement")

	snapshot := testcheck.Must(New(
		[]Source{{Name: "adaptive", Repository: "local/adaptive_new", Commit: "214950b3b0316bcdcab38a3b95127a927c0ab5be"}},
		[]Capability{{
			ID: "image-to-text", Source: "adaptive", EntryPoint: "go/extmodel/vqa.go:Run",
			Signature: Signature{Inputs: []Modality{ModalityImage, ModalityText}, Outputs: []Modality{ModalityText}},
			State:     PromotionLead,
			Artifacts: []Reference{{Name: "model", Identity: model}},
			Corpora:   []Reference{{Name: "heldout", Identity: dataset}},
			Goldens:   []Reference{{Name: "answers", Identity: golden}},
			Measurements: []Measurement{{
				Role: MeasurementReference, Runtime: "adaptive", Protocol: "seed=42; warm; exact-answer", WallNanos: 18_400_000_000,
				PeakBytes: 11_970_000_000, Evidence: measurement,
			}, {
				Role: MeasurementCandidate, Runtime: "overgo", Protocol: "seed=42; warm; exact-answer", WallNanos: 9_500_000_000,
				PeakBytes: 10_590_000_000, Evidence: measurement,
			}},
		}},
	))(t)
	content := testcheck.Must(snapshot.Content())(t)

	// Import accepts noncanonical producer field order, then owns canonical bytes.
	var unordered map[string]any
	testcheck.NoError(t, json.Unmarshal(content.Data, &unordered))
	producer := testcheck.Must(json.MarshalIndent(unordered, "", "  "))(t)
	imported, canonical, err := Import(bytes.NewReader(producer))
	testcheck.NoError(t, err)
	if imported.ID != snapshot.ID || !bytes.Equal(canonical, content.Data) {
		t.Fatalf("import identity/bytes differ: %s vs %s", imported.ID, snapshot.ID)
	}
	parsed := testcheck.Must(Parse(canonical))(t)
	if parsed.ID != snapshot.ID || parsed.Capabilities[0].Signature.Inputs[0] != ModalityImage {
		t.Fatalf("round trip changed identity or ordered signature: %+v", parsed)
	}
}

func TestSnapshotRejectsUnprovedPromotionVerdicts(t *testing.T) {
	model := testutil.ArtifactID(t, artifact.KindModel, "fixture-model")
	dataset := testutil.ArtifactID(t, artifact.KindDataset, "fixture-corpus")
	golden := testutil.ArtifactID(t, artifact.KindEvidence, "fixture-golden")
	measurement := testutil.ArtifactID(t, artifact.KindEvidence, "fixture-measurement")
	base := Capability{
		ID: "text", Source: "adaptive", EntryPoint: "inference.Run",
		Signature: Signature{Inputs: []Modality{ModalityText}, Outputs: []Modality{ModalityText}},
		Artifacts: []Reference{{Name: "model", Identity: model}},
		Corpora:   []Reference{{Name: "corpus", Identity: dataset}},
		Goldens:   []Reference{{Name: "tokens", Identity: golden}},
	}
	tests := []struct {
		name string
		edit func(*Capability)
	}{
		{"lead-without-peak", func(capability *Capability) {
			capability.State = PromotionLead
			capability.Measurements = []Measurement{
				{Role: MeasurementReference, Runtime: "adaptive", Protocol: "same", WallNanos: 2, Evidence: measurement},
				{Role: MeasurementCandidate, Runtime: "overgo", Protocol: "same", WallNanos: 1, Evidence: measurement},
			}
		}},
		{"tradeoff-without-loss", func(capability *Capability) {
			capability.State = PromotionTradeoff
			capability.Measurements = []Measurement{
				{Role: MeasurementReference, Runtime: "adaptive", Protocol: "same", WallNanos: 2, PeakBytes: 2, Evidence: measurement},
				{Role: MeasurementCandidate, Runtime: "overgo", Protocol: "same", WallNanos: 1, PeakBytes: 1, Evidence: measurement},
			}
		}},
		{"refusal-with-artifact", func(capability *Capability) {
			capability.State = PromotionRefused
			capability.Refusal = "artifact absent"
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			capability := base
			test.edit(&capability)
			if _, err := New(
				[]Source{{Name: "adaptive", Repository: "local/adaptive_new", Commit: "214950b3b0316bcdcab38a3b95127a927c0ab5be"}},
				[]Capability{capability},
			); err == nil {
				t.Fatal("unproved promotion verdict accepted")
			}
		})
	}
}
