// Package overgodbtest opens the throwaway stores tests read and write, each in one call: the test ends at once when
// the store cannot open, and the store closes, its close checked, when the test and its subtests end.
package overgodbtest

import (
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
)

// Open opens the store at directory, as overgodb.Open does, for the rest of the test.
func Open(t testing.TB, directory string) *overgodb.Store {
	t.Helper()
	return held(t, overgodb.Open, directory)
}

// Create creates the store at directory, as overgodb.Create does, for the rest of the test.
func Create(t testing.TB, directory string) *overgodb.Store {
	t.Helper()
	return held(t, overgodb.Create, directory)
}

// OpenReadOnly opens the store at directory for reading, as overgodb.OpenReadOnly does, for the rest of the test.
func OpenReadOnly(t testing.TB, directory string) *overgodb.Store {
	t.Helper()
	return held(t, overgodb.OpenReadOnly, directory)
}

// held opens the store at directory through open and closes it when the test ends.
func held(t testing.TB, open func(string) (*overgodb.Store, error), directory string) *overgodb.Store {
	t.Helper()
	store := testcheck.Must(open(directory))(t)
	t.Cleanup(func() { testcheck.NoError(t, store.Close()) })
	return store
}
