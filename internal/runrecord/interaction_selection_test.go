package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/testutil"
)

// TestBoundedInteractionSelection proves complete canonical transcript arcs
// can be selected through the shared aggregate budget without losing their
// interaction citation or causal root.
func TestBoundedInteractionSelection(t *testing.T) {
	root := testutil.ArtifactID(t, artifact.KindEvidence, "interaction-selection-root")
	head := artifact.CommitID{1}
	var sources []dataset.InteractionSelectionSource
	for index, text := range []string{"recent request", "older request"} {
		transcript, err := NewInteractionTranscript([]InteractionMessage{{Role: "user", Content: text}})
		if err != nil {
			t.Fatal(err)
		}
		content, err := interactionTranscriptCodec.Content(transcript)
		if err != nil {
			t.Fatal(err)
		}
		sources = append(sources, dataset.InteractionSelectionSource{
			Source: transcript.ID, CausalRoot: root, Tokens: uint64(index + 2), Bytes: content.Descriptor.Size,
			Documents: 1, Depth: uint32(index + 1),
		})
	}
	selection, err := dataset.SelectInteractions(dataset.InteractionSelectionBounds{
		MaxTokens: 8, MaxBytes: 1 << 10, MaxDocuments: 2, MaxDepth: 2, MaxResults: 2,
	}, head, sources, nil)
	if err != nil || len(selection.Sources) != 2 {
		t.Fatalf("interaction selection = (%+v, %v)", selection, err)
	}
	if err := selection.Validate(); err != nil {
		t.Fatalf("durable interaction selection: %v", err)
	}
	for index, source := range selection.Sources {
		if source.Source != sources[index].Source || source.CausalRoot != root {
			t.Fatalf("selection lost cited arc %d: %+v", index, source)
		}
	}
}
