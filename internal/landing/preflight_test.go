package landing

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// TestPreflightRunsTheCheapStagesOnly holds the preflight to the stages that
// take seconds and refuse what the change itself got wrong, and the full
// landing's long stages out of it: every package's tests, the lanes and the
// scorecard slice run only when the change lands.
func TestPreflightRunsTheCheapStagesOnly(t *testing.T) {
	t.Parallel()
	var cheap, long []string
	for _, check := range (&landing{}).checks() {
		if check.cheap {
			cheap = append(cheap, check.name)
		} else {
			long = append(long, check.name)
		}
	}
	if want := []string{"format", "fix", "vet", "rules", "invariants", "verify"}; !slices.Equal(cheap, want) {
		t.Errorf("the preflight runs %v, want %v", cheap, want)
	}
	if want := []string{"test", "lanes", "scorecard"}; !slices.Equal(long, want) {
		t.Errorf("the stages it leaves to the landing are %v, want %v", long, want)
	}
}

// TestInvariantPackagesExist holds the invariant stage to packages the tree
// has: a renamed one would let the stage pass over nothing.
func TestInvariantPackagesExist(t *testing.T) {
	t.Parallel()
	for _, pattern := range invariantPackages {
		if info, err := os.Stat(filepath.Join("..", "..", filepath.FromSlash(pattern))); err != nil || !info.IsDir() {
			t.Errorf("invariant package %s is not a directory of the tree: %v", pattern, err)
		}
	}
}
