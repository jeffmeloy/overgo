package repoanalysis

import (
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/testevidence"
)

// fellUnder says a count under the ceiling the constant name of this package's file holds, which refuses nothing: a
// ceiling only falls, and the landing whose invariant tests said the fall writes the lower ceiling into its commit
// (testevidence.CeilingFell).
func fellUnder(t *testing.T, file, name string, ceiling, count int) {
	t.Helper()
	if count >= ceiling {
		return
	}
	root := testcheck.Must(filepath.Abs(checkoutRoot))(t)
	here := testcheck.Must(filepath.Abs(file))(t)
	t.Logf(testevidence.CeilingFell, name, filepath.ToSlash(testcheck.Must(filepath.Rel(root, here))(t)), ceiling, count)
}
