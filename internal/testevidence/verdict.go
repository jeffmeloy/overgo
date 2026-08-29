package testevidence

import (
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
)

// VerdictClass names how a claim's evidence is allowed to vary across runs.
// Classification derives from observable command markers, never from a
// hand-maintained claim table.
type VerdictClass string

// VerifyPolicy identifies one immutable verify-command classifier. Existing
// policies are replay contracts: new marker semantics require a new policy,
// never edits to an existing classifier.
type VerifyPolicy string

const (
	// VerifyPolicyV1 freezes the original host/device/multiseed marker rules.
	VerifyPolicyV1 VerifyPolicy = "verify-classifier-v1"
	// CurrentVerifyPolicy is the policy new evidence should record. Historical
	// replay must use the policy embedded in that evidence instead.
	CurrentVerifyPolicy VerifyPolicy = VerifyPolicyV1

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

// deviceMarkersV1 and seedMarkersV1 are frozen observable command fragments
// for VerifyPolicyV1. A marker-free command is host execution and defaults to
// the strictest class.
var (
	seedMarkersV1   = [...]string{"multiseed", "multi-seed", "multi_seed", "overgo_seeds"}
	deviceMarkersV1 = [...]string{"overgo_cuda_test", "device-lane", "compute-sanitizer", "overgo_sealed_authority_test"}
)

// ClassifyVerifyCommand derives the verdict class for newly produced evidence
// under CurrentVerifyPolicy. Durable replay must call
// ClassifyVerifyCommandForPolicy with the recorded policy.
func ClassifyVerifyCommand(command string) VerdictClass {
	verdict, err := ClassifyVerifyCommandForPolicy(CurrentVerifyPolicy, command)
	if err != nil {
		panic(err)
	}
	return verdict
}

// ClassifyVerifyCommandForPolicy derives a verdict using the exact immutable
// policy named by durable evidence. Unknown future policies fail closed.
func ClassifyVerifyCommandForPolicy(policy VerifyPolicy, command string) (VerdictClass, error) {
	switch policy {
	case VerifyPolicyV1:
		return classifyVerifyCommandV1(command), nil
	default:
		return "", fmt.Errorf("test evidence: unknown verify classifier policy %q", policy)
	}
}

func classifyVerifyCommandV1(command string) VerdictClass {
	lower := strings.ToLower(command)
	for _, marker := range seedMarkersV1 {
		if strings.Contains(lower, marker) {
			return VerdictStochasticMultiSeed
		}
	}
	for _, marker := range deviceMarkersV1 {
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
	return repeatAgreement(first, second, false)
}

// RepeatAgreementForCommand permits ordinary output only when the verifier is
// explicitly mixed, matching VerifyGoTestEvidence's evidence boundary.
func RepeatAgreementForCommand(command, first, second string) error {
	return repeatAgreement(first, second, hasNonTestCommand(command))
}

func repeatAgreement(first, second string, allowAuxiliary bool) error {
	firstReport, err := goTestJSONReport(first, false, allowAuxiliary)
	if err != nil {
		return fmt.Errorf("first run: %w", err)
	}
	secondReport, err := goTestJSONReport(second, false, allowAuxiliary)
	if err != nil {
		return fmt.Errorf("second run: %w", err)
	}
	firstActions := verdictActions(firstReport)
	secondActions := verdictActions(secondReport)
	names := slices.Collect(maps.Keys(firstActions))
	for name := range secondActions {
		if _, seen := firstActions[name]; !seen {
			names = append(names, name)
		}
	}
	slices.Sort(names)
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
			return errors.New("repeat disagreement: " + name + " was " + firstAction + " then " + secondAction)
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
