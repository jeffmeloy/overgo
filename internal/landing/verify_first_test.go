package landing

import (
	"slices"
	"testing"
)

// TestAStepsCheckRunsBeforeTheLanes holds a landing to running its step's check, the check's undo evidence included,
// before every package's tests and the lanes (fail-early/verify-first): a check of seconds that fails or proves nothing
// is refused in seconds, where it ran last, after eight minutes of tests and lanes (42 of the 353 refusals the gate had
// recorded on 2026-11-10 were the verify's). A check that runs on the card, minutes to hours, runs before them too: 13
// of 26 such checks failed only after the tests and lanes had run, 204 minutes, where taking the card's turn twice cost
// 95 minutes in all (SQA-322).
func TestAStepsCheckRunsBeforeTheLanes(t *testing.T) {
	t.Parallel()
	order := func(verifiers ...string) []string {
		var names []string
		for _, check := range (&landing{verifiers: verifiers}).checks() {
			names = append(names, check.name)
		}
		return names
	}
	for _, check := range []string{
		"go test ./internal/devicebudget -count=1 -run '^TestADeclaredDeviceBudgetBoundsWhatACardWould$'",
		"OVERGO_CUDA_TEST=1 OVERGO_DATA_ROOT=C:/data go test -tags modeltest ./cmd/loop -count=1 -run '^TestTheLocalWorkerLandsStepsUnaided$' -timeout 120m",
	} {
		stages := order(check)
		if verify := slices.Index(stages, "verify"); verify < 0 || verify > slices.Index(stages, "test") || verify > slices.Index(stages, "lanes") {
			t.Errorf("a landing of %q orders its stages %v; want the verify before the tests and the lanes", check, stages)
		}
	}
}
