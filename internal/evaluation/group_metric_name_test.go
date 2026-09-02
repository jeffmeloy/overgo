package evaluation

import (
	"testing"

	"overgo/internal/textcheck"
)

// Group accuracy metrics ride the run-record label alphabet; a subset name
// from a dataset never decides whether the evaluation can be recorded.
func TestGroupAccuracyMetricNamesAreRecordLabels(t *testing.T) {
	cases := map[string]string{
		"boolean_expressions":   "accuracy.boolean_expressions",
		"murder_mysteries":      "accuracy.murder_mysteries",
		"Web of Lies/Extra":     "accuracy.web-of-lies-extra",
		"date-understanding.v2": "accuracy.date-understanding.v2",
		"tracking objects (7)":  "accuracy.tracking-objects--7-",
		"ünicode":               "accuracy.--nicode",
	}
	for group, want := range cases {
		got := groupAccuracyMetricName(group)
		if got != want {
			t.Errorf("%q -> %q, want %q", group, got, want)
		}
		if !textcheck.LowerIdentifier(got, len(got)) {
			t.Errorf("%q is not a record label", got)
		}
	}
	metrics := accuracyMetrics(0.5, []AccuracyGroup{{Name: "A/B", Accuracy: 1}})
	if len(metrics) != 2 || metrics[1].Name != "accuracy.a-b" {
		t.Fatalf("metrics = %+v", metrics)
	}
	contract := accuracyMetricContract([]string{"A/B", "A/B", ""})
	if len(contract) != 2 || contract[1].Name != "accuracy.a-b" {
		t.Fatalf("contract = %+v", contract)
	}
}
