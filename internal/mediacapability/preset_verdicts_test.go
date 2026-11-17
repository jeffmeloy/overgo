package mediacapability

import (
	"fmt"
	"strings"
)

// verdictSaid refuses a verdict whose counts are empty or whose words say other counts, or a refusal it did not make.
func verdictSaid(verdict presetVerdict) error {
	counts := fmt.Sprintf("%d wins, %d losses and %d ties over %d media", verdict.Wins, verdict.Losses, verdict.Ties, verdict.Media)
	if verdict.Wins+verdict.Losses+verdict.Ties == 0 || verdict.Media == 0 || !strings.HasSuffix(verdict.Says, counts) ||
		strings.HasPrefix(verdict.Says, "refused") == verdict.Admitted {
		return fmt.Errorf("it holds %s, admitted %t, saying %q", counts, verdict.Admitted, verdict.Says)
	}
	return nil
}
