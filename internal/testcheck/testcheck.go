// Package testcheck holds the one-call checks a test makes of what it ran, so a test says a check in a line. It imports
// testing alone, so a test of any package may use it.
package testcheck

import "testing"

// NoError ends the test at once with err when there is one.
func NoError(t testing.TB, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
