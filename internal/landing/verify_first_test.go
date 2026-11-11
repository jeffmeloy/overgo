package landing

import (
	"slices"
	"testing"
)

// TestAStepsCheckRunsBeforeTheLanes holds a landing to running its step's check, the check's undo evidence included,
// before every package's tests and the lanes (fail-early/verify-first): a check of seconds that fails or proves nothing
// is refused in seconds, where it ran last, after eight minutes of tests and lanes (42 of the 353 refusals the gate had
// recorded on 2026-11-10 were the verify's). A check that runs on the card, minutes to hours, keeps its place after
// them, so a test failure of minutes is not held behind it.
func TestAStepsCheckRunsBeforeTheLanes(t *testing.T) {
	t.Parallel()
	order := func(verifiers ...string) []string {
		var names []string
		for _, check := range (&landing{verifiers: verifiers}).checks() {
			names = append(names, check.name)
		}
		return names
	}
	host := order("go test ./internal/devicebudget -count=1 -run '^TestADeclaredDeviceBudgetBoundsWhatACardWould$'")
	if verify := slices.Index(host, "verify"); verify > slices.Index(host, "test") || verify > slices.Index(host, "lanes") {
		t.Errorf("a landing whose check runs on the host orders its stages %v; want the verify before the tests and the lanes", host)
	}
	for _, device := range []string{
		"OVERGO_CUDA_TEST=1 OVERGO_DATA_ROOT=C:/data go test -tags modeltest ./cmd/loop -count=1 -run '^TestTheLocalWorkerLandsStepsUnaided$' -timeout 120m",
		"OVERGO_DATA_ROOT=C:/data go test -tags modeltest ./cmd/validate -count=1 -run '^TestThreeWorkloadsAreReadUnderSmallerCards$' -timeout 120m",
	} {
		if card := order(device); slices.Index(card, "verify") < slices.Index(card, "lanes") {
			t.Errorf("a landing whose check runs on the card orders its stages %v; want the verify after the lanes", card)
		}
	}
}
