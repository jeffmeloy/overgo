package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestInteractionEfficiencyBaseline holds the trace document to its
// contract: exact work counts on a named surface, bound to a result
// and evidence, round-tripping canonically; empty work, unknown
// surfaces, and unbound results are refused.
func TestInteractionEfficiencyBaseline(t *testing.T) {
	result := testutil.ArtifactBytesID(t, artifact.KindRun, []byte("trace-result"))
	evidence := testutil.ArtifactBytesID(t, artifact.KindEvidence, []byte("trace-evidence"))
	trace, err := NewEfficiencyTrace(EfficiencyTrace{
		Surface: SurfaceStorage,
		Task:    "scale-corpus session",
		Work: InteractionWork{
			SemanticTransitions: 512, Commits: 512, ArtifactReads: 65,
			ReturnedFacts: 64, Bytes: 4096,
		},
		Result: result, Evidence: evidence,
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := trace.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseEfficiencyTrace(content.Data)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.ID != trace.ID || parsed.Work != trace.Work || parsed.Surface != trace.Surface {
		t.Fatalf("round trip drifted: %+v vs %+v", parsed, trace)
	}
	if lineage := trace.Lineage(); len(lineage) != 2 {
		t.Fatalf("trace lineage must bind result and evidence: %d edges", len(lineage))
	}

	rejects := []EfficiencyTrace{
		{Surface: "board", Task: "t", Work: InteractionWork{Commits: 1}, Result: result, Evidence: evidence},
		{Surface: SurfaceAgent, Task: "", Work: InteractionWork{Commits: 1}, Result: result, Evidence: evidence},
		{Surface: SurfaceAgent, Task: "t", Result: result, Evidence: evidence},
		{Surface: SurfaceAgent, Task: "t", Work: InteractionWork{Commits: 1}, Evidence: evidence},
		{Surface: SurfaceAgent, Task: "t", Work: InteractionWork{Commits: 1}, Result: result, Evidence: result},
	}
	for index, invalid := range rejects {
		if _, err := NewEfficiencyTrace(invalid); err == nil {
			t.Fatalf("invalid trace %d admitted", index)
		}
	}
}
