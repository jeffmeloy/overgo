package testevidence

import (
	"errors"
	"fmt"
	"maps"
	"slices"
)

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
	firstReport, err := goTestInvocationReports(first, false, allowAuxiliary)
	if err != nil {
		return fmt.Errorf("first run: %w", err)
	}
	secondReport, err := goTestInvocationReports(second, false, allowAuxiliary)
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

func verdictActions(reports []GoTestReport) map[string]string {
	actions := map[string]string{}
	occurrences := map[string]int{}
	for _, report := range reports {
		for _, execution := range report.Executions {
			occurrences[execution.Package]++
		}
		for _, test := range report.tests {
			if test.Name == "" {
				continue
			}
			name := test.Package
			if occurrences[test.Package] > 1 {
				name += fmt.Sprintf("#%d", occurrences[test.Package])
			}
			actions[name+"."+test.Name] = test.Action
		}
	}
	return actions
}
