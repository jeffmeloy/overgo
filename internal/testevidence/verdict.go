package testevidence

import (
	"fmt"
	"sort"
	"strings"
)

// VerdictClass names how a claim's evidence is allowed to vary across runs.
// Classification derives from observable command markers, never from a
// hand-maintained claim table.
type VerdictClass string

const (
	// VerdictBitwiseDeterministic: host execution with no stochastic or device
	// marker. Two runs must reach identical per-test verdicts; a claim that
	// cannot repeat is not deterministic evidence.
	VerdictBitwiseDeterministic VerdictClass = "bitwise-deterministic"
	// VerdictToleranceBounded: device execution. CUDA scatter-accumulate
	// (float atomicAdd in gradient paths) is not bitwise reproducible; the
	// test itself owns a measured tolerance, so one run is the evidence.
	VerdictToleranceBounded VerdictClass = "tolerance-bounded"
	// VerdictStochasticMultiSeed: stochastic training outcomes. Two runs
	// agreeing tests reproducibility, not reliability; the test must compare
	// across seeds and the marker declares that contract.
	VerdictStochasticMultiSeed VerdictClass = "stochastic-multi-seed"
)

// deviceMarkers and seedMarkers are the observable command fragments that
// change a claim's verdict class. A marker-free command is host execution and
// defaults to the strictest class.
var (
	seedMarkers   = []string{"multiseed", "multi-seed", "multi_seed", "overgo_seeds"}
	deviceMarkers = []string{"overgo_cuda_test", "device-lane", "compute-sanitizer", "overgo_sealed_authority_test"}
)

// ClassifyVerifyCommand derives the verdict class for one verify command.
func ClassifyVerifyCommand(command string) VerdictClass {
	lower := strings.ToLower(command)
	for _, marker := range seedMarkers {
		if strings.Contains(lower, marker) {
			return VerdictStochasticMultiSeed
		}
	}
	for _, marker := range deviceMarkers {
		if strings.Contains(lower, marker) {
			return VerdictToleranceBounded
		}
	}
	return VerdictBitwiseDeterministic
}

// RepeatAgreement requires two evidence outputs of the same bitwise-
// deterministic go test command to reach identical per-test verdicts. It
// reports the first disagreement by test identity and both observed actions;
// tests appearing in only one run disagree with the absent action.
func RepeatAgreement(first, second string) error {
	firstReport, err := GoTestJSONReport(first)
	if err != nil {
		return fmt.Errorf("first run: %w", err)
	}
	secondReport, err := GoTestJSONReport(second)
	if err != nil {
		return fmt.Errorf("second run: %w", err)
	}
	firstActions := verdictActions(firstReport)
	secondActions := verdictActions(secondReport)
	names := make([]string, 0, len(firstActions)+len(secondActions))
	for name := range firstActions {
		names = append(names, name)
	}
	for name := range secondActions {
		if _, seen := firstActions[name]; !seen {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	for _, name := range names {
		firstAction, inFirst := firstActions[name]
		secondAction, inSecond := secondActions[name]
		if !inFirst {
			firstAction = "absent"
		}
		if !inSecond {
			secondAction = "absent"
		}
		if firstAction != secondAction {
			return fmt.Errorf("repeat disagreement: %s was %s then %s", name, firstAction, secondAction)
		}
	}
	return nil
}

func verdictActions(report GoTestReport) map[string]string {
	actions := make(map[string]string, len(report.tests))
	for _, test := range report.tests {
		if test.Name == "" {
			continue
		}
		actions[test.Package+"."+test.Name] = test.Action
	}
	return actions
}
