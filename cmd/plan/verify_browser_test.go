package main

import "testing"

// TestBrowserVerifyOutsideLane pins the rule a verify must satisfy: a
// browser test is named only through the lane runner.
func TestBrowserVerifyOutsideLane(t *testing.T) {
	for command, outside := range map[string]bool{
		"go test ./internal/server -run '^TestWebUIBrowserFirstRun$' -count=1":                                 true,
		"go run ./cmd/webui-lane -run '^TestWebUIBrowserFirstRun$' -require 'cold-start leg'":                  false,
		"go test ./internal/server -run '^TestIdleShellAnswersWhileNothingServes$' -count=1":                   false,
		"go test ./internal/webuilane -count=1 && go run ./cmd/webui-lane -run '^TestWebUIBrowserAcceptance$'": false,
	} {
		if browserVerifyOutsideLane(command) != outside {
			t.Fatalf("%q outside the lane = %v, want %v", command, !outside, outside)
		}
	}
}
