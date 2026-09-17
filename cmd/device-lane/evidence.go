package main

import (
	"fmt"
	"regexp"
	"slices"

	"overgo/internal/automationcheck"
	"overgo/internal/testevidence"
)

// Every declared name needs executed evidence: a surviving sibling cannot hide
// a renamed, skipped or missing contract. Reuse the common event validator.
func verifyDeviceContracts(command []string, output string) error {
	run, skip := stepTestPattern(command, "-run"), stepTestPattern(command, "-skip")
	for _, scope := range automationcheck.DeviceScopes(automationcheck.DeviceVerificationPlan{Full: true}) {
		if !slices.Contains(command, scope.Package) {
			continue
		}
		for _, name := range scope.Tests {
			if run != nil && !run.MatchString(name) || skip != nil && skip.MatchString(name) {
				continue
			}
			if err := testevidence.VerifyGoTestEvidence("go test -run '^"+regexp.QuoteMeta(name)+"$'", output); err != nil {
				return fmt.Errorf("device contract %s %s: %w", scope.Package, name, err)
			}
		}
	}
	return nil
}
