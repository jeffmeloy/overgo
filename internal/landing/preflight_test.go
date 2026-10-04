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

// TestGeneratorsAreThoseTheRepositoryHas holds the derived-file stage to the
// commands a repository has: the API manifest and the compatibility claims are
// regenerated and rebound when their commands exist, and a tree without them,
// a fixture's, has nothing to derive.
func TestGeneratorsAreThoseTheRepositoryHas(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		commands []string
		want     []string
	}{
		{nil, nil},
		{[]string{"api-manifest"}, []string{"./cmd/api-manifest"}},
		{[]string{"api-manifest", "validate"}, []string{"./cmd/api-manifest", "./cmd/validate"}},
	} {
		root := t.TempDir()
		for _, command := range test.commands {
			if err := os.MkdirAll(filepath.Join(root, "cmd", command), 0o755); err != nil {
				t.Fatal(err)
			}
		}
		var got []string
		for _, generator := range (&landing{Options: Options{Repo: root}}).generators() {
			got = append(got, generator.arguments[1])
		}
		if !slices.Equal(got, test.want) {
			t.Errorf("commands %v derive with %v, want %v", test.commands, got, test.want)
		}
	}
}
