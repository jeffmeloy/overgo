package runrecord

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"

	"overgo/internal/testevidence"
	"overgo/internal/textcheck"
)

const completionAcceptancePrefix = "completion policy="

// FormatCompletionAcceptanceEvidence binds one classifier policy, plan row,
// and exact verify-command digest into bounded GateStep evidence. The verdict
// is derived by the named immutable policy rather than supplied independently.
func FormatCompletionAcceptanceEvidence(
	policy testevidence.VerifyPolicy,
	planReference, verify string,
) (string, error) {
	item, step, found := strings.Cut(planReference, "/")
	if !found ||
		!textcheck.BoundedToken(item, len(item), "/\\\x00\r\n") ||
		!textcheck.BoundedToken(step, len(step), "/\\\x00\r\n") {
		return "", errors.New("run record: completion evidence requires an item/step plan reference")
	}
	if !validText(verify) {
		return "", errors.New("run record: completion evidence requires the exact bounded verify command")
	}
	verdict, err := testevidence.ClassifyVerifyCommandForPolicy(policy, verify)
	if err != nil {
		return "", fmt.Errorf("run record: completion evidence classifier: %w", err)
	}
	if !validLabel(string(verdict)) {
		return "", errors.New("run record: completion evidence classifier returned an invalid verdict")
	}
	digest := sha256.Sum256([]byte(verify))
	return fmt.Sprintf(
		"%s%s plan=%s verify_sha256=%s verdict=%s",
		completionAcceptancePrefix, policy, planReference, hex.EncodeToString(digest[:]), verdict,
	), nil
}

// VerifyCompletionAcceptanceEvidence selects the immutable classifier named by
// the evidence, then requires the exact canonical plan row, verifier digest,
// and derived verdict. It never consults the mutable current-policy choice.
func VerifyCompletionAcceptanceEvidence(evidence, planReference, verify string) error {
	policy, err := completionAcceptancePolicy(evidence)
	if err != nil {
		return err
	}
	expected, err := FormatCompletionAcceptanceEvidence(policy, planReference, verify)
	if err != nil {
		return err
	}
	if evidence != expected {
		return errors.New("run record: completion acceptance evidence does not match the plan authority")
	}
	return nil
}

func completionAcceptancePolicy(evidence string) (testevidence.VerifyPolicy, error) {
	remainder, found := strings.CutPrefix(evidence, completionAcceptancePrefix)
	if !found {
		return "", errors.New("run record: completion acceptance evidence lacks a classifier policy")
	}
	policy, _, found := strings.Cut(remainder, " ")
	if !found || policy == "" {
		return "", errors.New("run record: completion acceptance evidence has an invalid classifier policy")
	}
	return testevidence.VerifyPolicy(policy), nil
}
