package main

import (
	"fmt"
	"slices"

	"overgo/internal/automationcheck"
	"overgo/internal/testevidence"
)

// Every declared name needs executed evidence: a surviving sibling cannot hide
// a renamed, skipped or missing contract. The whole step output is parsed once
// and every declared name is checked against that one report, not re-parsed per
// contract.
func verifyDeviceContracts(command []string, output string) error {
	run, skip := stepTestPattern(command, "-run"), stepTestPattern(command, "-skip")
	var names []string
	for _, scope := range automationcheck.DeviceScopes(automationcheck.DeviceVerificationPlan{Full: true}) {
		if !slices.Contains(command, scope.Package) {
			continue
		}
		for _, name := range scope.Tests {
			if run != nil && !run.MatchString(name) || skip != nil && skip.MatchString(name) {
				continue
			}
			names = append(names, name)
		}
	}
	if len(names) == 0 {
		return nil
	}
	if err := testevidence.VerifyGoTestNames(output, false, names); err != nil {
		return fmt.Errorf("device contract: %w", err)
	}
	return nil
}
