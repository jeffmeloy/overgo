package evaluation

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/sequencescore"
	"overgo/internal/testcheck"
)

// TestAProtocolRefusalIsHandedToTheRun holds the chat protocol's refusal to the run that awaits it (ocr-domains): a
// suite whose generated answers name no candidate is refused with ErrProtocolRefused, and the refusal's words are
// written to the file RefusalEnvironment names, which validate reads to report the cell as one its model's output
// cannot serve rather than a broken measurement.
func TestAProtocolRefusalIsHandedToTheRun(t *testing.T) {
	// Serial: it names the refusal's file in the process environment.
	path := filepath.Join(t.TempDir(), "refusal.txt")
	t.Setenv(RefusalEnvironment, path)
	cases := make([]MultipleChoiceCase, chatProbeCases)
	for index := range cases {
		cases[index] = MultipleChoiceCase{Name: fmt.Sprintf("mmlu-pro/x/%d", index), Prompt: "Q\nA:", Candidates: []string{" A", " B"}, Answer: 1}
	}
	suite := MultipleChoiceSuite{
		Kind: MultipleChoiceKind, Schema: "test/mmlu-pro/v1", Source: "store/mmlu-pro",
		Normalization: sequencescore.NormalizationMean, Aggregation: AggregationAccuracy, Cases: cases,
	}
	_, _, err := scoreChoicesAlone(t, t.Context(), &chatFixture{answers: []string{" 1. The answer to the question"}}, suite)
	if !errors.Is(err, ErrProtocolRefused) {
		t.Fatalf("a suite whose answers name no candidate ended as %v; want ErrProtocolRefused", err)
	}
	if written := string(testcheck.Must(os.ReadFile(path))(t)); written != err.Error() {
		t.Errorf("the run's refusal file holds %q; want the refusal's words %q", written, err.Error())
	}
}
