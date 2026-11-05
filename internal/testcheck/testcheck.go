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

// Must gives a function that ends the test it is given at once with err when there is one and returns value otherwise,
// so a test takes a call's value and checks its error in one expression: testcheck.Must(os.ReadFile(path))(t).
func Must[T any](value T, err error) func(testing.TB) T {
	return func(t testing.TB) T {
		t.Helper()
		NoError(t, err)
		return value
	}
}
