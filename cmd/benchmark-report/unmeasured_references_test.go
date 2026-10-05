package main

import (
	"testing"

	"overgo/internal/evaluation"
)

// TestUnmeasuredReferencesReadTheSuitesMeasured holds the list of published
// references without a committed evaluation to the suites the report holds:
// a suite is a column per prompting protocol, so a reference for a suite
// measured under either is not listed as unmeasured, and one for a suite
// measured under none is.
func TestUnmeasuredReferencesReadTheSuitesMeasured(t *testing.T) {
	t.Parallel()
	measured := evaluation.ReferenceScore{Suite: "store/mmlu", Metric: "accuracy", Value: 0.58, Protocol: "0-shot", Source: "card"}
	absent := evaluation.ReferenceScore{Suite: "store/ifeval", Metric: "prompt_strict", Value: 0.898, Protocol: "instruct", Source: "card"}
	row := modelRow{
		references: []evaluation.ReferenceScore{measured, absent},
		suites: map[string]suiteRow{
			"store/mmlu (chat-template)":   {references: []evaluation.ReferenceScore{measured}},
			"store/mmlu (raw-completion)":  {references: []evaluation.ReferenceScore{measured}},
			"store/gsm8k (raw-completion)": {},
		},
	}
	if got := unmeasuredReferences(row); len(got) != 1 || got[0] != absent {
		t.Fatalf("unmeasured references = %+v, want only %+v", got, absent)
	}
	if got := unmeasuredReferences(modelRow{references: []evaluation.ReferenceScore{measured}}); len(got) != 1 {
		t.Fatalf("a model with no suites lists %d unmeasured references, want its one", len(got))
	}
}
