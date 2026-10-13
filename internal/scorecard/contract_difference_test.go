package scorecard

import (
	"strings"
	"testing"
)

// TestAContractDifferenceIsSaidInWords holds the reason a cell on another
// contract is unknown to the words a reader needs: each part that differs is
// named with both its values, and a part that agrees is not mentioned.
func TestAContractDifferenceIsSaidInWords(t *testing.T) {
	t.Parallel()
	now := Contract{Metric: "token-accuracy", Origin: originEvaluation, Suite: "suite-a"}
	then := Contract{Metric: "token-accuracy", Origin: originEvaluation, Suite: "suite-b"}
	got := ContractDifference(now, then)
	if want := `measured on another contract: the suite is "suite-a", the reference's "suite-b"`; got != want {
		t.Fatalf("difference = %q, want %q", got, want)
	}
	then.Metric, then.Origin = "accuracy", originServing
	got = ContractDifference(now, then)
	for _, part := range []string{`the measure is "token-accuracy", the reference's "accuracy"`, `the origin is "evaluation", the reference's "serving"`} {
		if !strings.Contains(got, part) {
			t.Errorf("difference %q does not say %s", got, part)
		}
	}
	if strings.Contains(got, "{") || strings.Contains(got, "Metric:") {
		t.Errorf("difference %q renders a struct, not words", got)
	}
}
