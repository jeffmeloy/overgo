package main

import "testing"

// TestVacuousTestEvidence pins the gate's skip-is-not-a-pass net: a 0-exit
// impacted-test run that skipped its oracle (UNAVAILABLE / parity NOT verified)
// is refused as passing evidence. Environment/real passes are unaffected.
func TestVacuousTestEvidence(t *testing.T) {
	cases := []struct {
		name   string
		out    string
		vacant bool
	}{
		{"unavailable golden", "--- SKIP: TestX\n    UNAVAILABLE: reference golden absent; parity NOT verified\nok\tpkg\t0.2s\n", true},
		{"parity not verified", "oracle: parity NOT verified\n", true},
		{"real pass", "ok\tovergo/internal/oscillatorimage\t1.4s\n", false},
		{"cached pass", "ok\tovergo/pkg\t(cached)\n", false},
	}
	for _, c := range cases {
		if got := vacuousTestEvidence(c.out) != ""; got != c.vacant {
			t.Errorf("%s: vacant=%v want %v", c.name, got, c.vacant)
		}
	}
}
