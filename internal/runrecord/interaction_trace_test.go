package runrecord

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestInteractionTraceToolExchangesRequireBijection(t *testing.T) {
	trace := InteractionTrace{Events: []InteractionTraceEvent{
		{Kind: InteractionEventToolCall, Message: InteractionMessage{
			Role: "assistant", ToolCalls: []InteractionToolCall{{
				ID: "call-1", Type: "function", Name: "inspect", Arguments: `{}`,
			}},
		}},
		{Kind: InteractionEventToolResult, Message: InteractionMessage{
			Role: "tool", ToolCallID: "call-1", ToolResultError: true,
		}},
	}}
	calls, failures, err := trace.ToolExchanges()
	if err != nil || len(calls) != 1 || calls[0].ID != "call-1" || failures != 1 {
		t.Fatalf("tool exchanges = %+v failures=%d err=%v", calls, failures, err)
	}
	clone := func() InteractionTrace {
		result := trace
		result.Events = slices.Clone(trace.Events)
		for index := range result.Events {
			result.Events[index].Message.ToolCalls = slices.Clone(trace.Events[index].Message.ToolCalls)
		}
		return result
	}
	for _, refusal := range []struct {
		name   string
		mutate func(*InteractionTrace)
	}{
		{name: "empty call identity", mutate: func(value *InteractionTrace) {
			value.Events[0].Message.ToolCalls[0].ID = ""
		}},
		{name: "duplicate call", mutate: func(value *InteractionTrace) {
			value.Events[0].Message.ToolCalls = append(value.Events[0].Message.ToolCalls, value.Events[0].Message.ToolCalls[0])
		}},
		{name: "missing result", mutate: func(value *InteractionTrace) {
			value.Events = value.Events[:1]
		}},
		{name: "foreign result", mutate: func(value *InteractionTrace) {
			value.Events[1].Message.ToolCallID = "foreign"
		}},
		{name: "duplicate result", mutate: func(value *InteractionTrace) {
			value.Events = append(value.Events, value.Events[1])
		}},
		{name: "result before call", mutate: func(value *InteractionTrace) {
			value.Events[0], value.Events[1] = value.Events[1], value.Events[0]
		}},
	} {
		mutated := clone()
		refusal.mutate(&mutated)
		if _, _, err := mutated.ToolExchanges(); err == nil {
			t.Fatalf("tool exchanges accepted %s", refusal.name)
		}
	}
}

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
			ReturnedFacts: 64, Bytes: 4096, Failures: 1,
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
