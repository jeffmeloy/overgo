package runrecord

import (
	"fmt"
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

	// VerdictBitwiseDeterministic requires identical host test verdicts across runs.
	VerdictBitwiseDeterministic VerdictClass = "bitwise-deterministic"
	// VerdictToleranceBounded accepts one device run under its test-owned tolerance.
	VerdictToleranceBounded VerdictClass = "tolerance-bounded"
	// VerdictStochasticMultiSeed requires test-owned comparison across seeds.
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
