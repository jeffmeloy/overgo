package main

import "testing"

// TestVacuousVerify pins the skip-is-not-a-pass rule: a 0-exit verify whose
// output shows an oracle that could not run is refused. The un0 case is the
// concrete regression this guard exists to prevent -- its golden loader skips
// with exactly this message, which advanced through tightening's gate.
func TestVacuousVerify(t *testing.T) {
	cases := []struct {
		name   string
		out    string
		vacant bool
	}{
		{"un0 golden absent", "--- SKIP: TestGeneratorMatchesTorch\n    golden_test.go:43: UNAVAILABLE: reference golden un0_generator_golden.json absent; parity NOT verified\nok  \toscillatorimage\t0.2s\n", true},
		{"parity-not-verified phrasing", "some oracle: parity NOT verified\n", true},
		{"no tests to run", "testing: warning: no tests to run\nPASS\nok  \tpkg\t0.1s [no tests to run]\n", true},
		{"no test files", "ok  \tovergo/pkg\t(cached) [no test files]\n", true},
		{"real pass", "--- PASS: TestReal (0.01s)\nok  \tovergo/pkg\t0.3s\n", false},
		{"plain ok", "ok  \tovergo/pkg\t0.30s\n", false},
		{"cached ok", "ok  \tovergo/pkg\t(cached)\n", false},
	}
	for _, c := range cases {
		got := vacuousVerify(c.out) != ""
		if got != c.vacant {
			t.Errorf("%s: vacuousVerify vacant=%v want %v", c.name, got, c.vacant)
		}
	}
}
