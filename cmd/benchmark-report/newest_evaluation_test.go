package main

import (
	"testing"

	"overgo/internal/evaluation"
)

// TestReportKeepsTheNewestEvaluationOfALabel holds a report column to the
// latest committed evaluation: the index lists a recipe's evaluations newest
// first, one per comparison, so two records of one suite and protocol scored
// under different comparisons must show the first, not the last, and a
// column of another protocol is kept beside it.
func TestReportKeepsTheNewestEvaluationOfALabel(t *testing.T) {
	t.Parallel()
	raw, template := evaluation.PromptingRawCompletion.Label(), evaluation.PromptingChatTemplate.Label()
	listed := []evaluation.EvalSummary{
		{Suite: "store/mmlu", Prompting: raw, CodeCommit: "newest"},
		{Suite: "store/mmlu", Prompting: template, CodeCommit: "other protocol"},
		{Suite: "store/mmlu", Prompting: raw, CodeCommit: "older, another comparison"},
		{Suite: "store/mmlu", Prompting: "unknown", CodeCommit: "unknown execution"},
		{Suite: "store/mmlu", Prompting: "unknown", CodeCommit: "older unknown execution"},
	}
	var got []string
	for _, kept := range newestPerColumn(listed) {
		got = append(got, kept.CodeCommit)
	}
	want := []string{"newest", "other protocol", "unknown execution"}
	if len(got) != len(want) {
		t.Fatalf("columns kept %q, want %q", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("columns kept %q, want %q", got, want)
		}
	}
}
