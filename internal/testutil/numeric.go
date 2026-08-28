package testutil

import (
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/dataroot"
)

type Float interface {
	~float32 | ~float64
}

// MaxAbsDiff returns the maximum elementwise absolute difference.
func MaxAbsDiff[L, R Float](left []L, right []R) float64 {
	if len(left) != len(right) {
		return math.Inf(1)
	}
	worst := 0.0
	for index := range left {
		difference := math.Abs(float64(left[index]) - float64(right[index]))
		if math.IsNaN(difference) {
			return math.Inf(1)
		}
		worst = max(worst, difference)
	}
	return worst
}

// RequireSliceClose checks elementwise absolute tolerance.
func RequireSliceClose[L, R Float](t testing.TB, label string, got []L, want []R, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", label, len(got), len(want))
	}
	if difference := MaxAbsDiff(got, want); difference > tolerance {
		t.Fatalf("%s max abs diff = %g, want <= %g", label, difference, tolerance)
	}
}

// RequireFiniteDecrease checks a fixed-length decreasing trajectory.
func RequireFiniteDecrease(t testing.TB, label string, values []float64, count int) {
	t.Helper()
	if len(values) != count {
		t.Fatalf("%s length = %d, want %d", label, len(values), count)
	}
	for index, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			t.Fatalf("%s[%d] is not finite: %g", label, index, value)
		}
		if index > 0 && value >= values[index-1] {
			t.Fatalf("%s did not decrease at %d: %v", label, index, values)
		}
	}
}

// RequireClose checks one absolute evidence bound.
func RequireClose(t testing.TB, label string, got, want, tolerance float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-want) > tolerance {
		t.Fatalf("%s = %g, want %g +/- %g", label, got, want, tolerance)
	}
}

// RequireRange checks one inclusive evidence interval.
func RequireRange(t testing.TB, label string, got, minimum, maximum float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || got < minimum || got > maximum {
		t.Fatalf("%s = %g, want [%g, %g]", label, got, minimum, maximum)
	}
}

// RequireWithin fails the test unless got matches want elementwise
// within tol; the max-abs-diff observation is logged either way so
// golden gates leave a numeric trace.
func RequireWithin[L, R Float](t testing.TB, name string, got []L, want []R, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d != golden %d", name, len(got), len(want))
	}
	diff := MaxAbsDiff(got, want)
	t.Logf("%s: max abs diff %.6e (gate %.0e)", name, diff, tol)
	if diff > tol {
		t.Fatalf("%s diverges: %g > %g", name, diff, tol)
	}
}

// RequireElementsWithin fails at the first element whose difference
// exceeds tol, naming the index for golden triage.
func RequireElementsWithin[L, R Float](t testing.TB, name string, got []L, want []R, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if d := math.Abs(float64(got[i]) - float64(want[i])); d > tol {
			t.Fatalf("%s[%d]: |%g - %g| = %g > %g", name, i, got[i], want[i], d, tol)
		}
	}
}

// ModelArtifactDir resolves one named model directory under the
// repository data roots for artifact-gated golden tests.
func ModelArtifactDir(t testing.TB, name string) string {
	t.Helper()
	roots, err := dataroot.Resolve(RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	return filepath.Join(roots.Models, name)
}

// ForbidVocabularyRedefinition scans one package directory for
// string-typed constant declarations outside the owning package that
// restate any of the given vocabulary values, so a canonical
// vocabulary cannot quietly fork per subsystem. The caller passes its
// own package directory; allowed names exempt exact constant names
// (mappings that intentionally share wire values).
func ForbidVocabularyRedefinition(t testing.TB, packageDir string, vocabulary []string, allowed ...string) {
	t.Helper()
	values := map[string]bool{}
	for _, value := range vocabulary {
		values[value] = true
	}
	exempt := map[string]bool{}
	for _, name := range allowed {
		exempt[name] = true
	}
	entries, err := os.ReadDir(packageDir)
	if err != nil {
		t.Fatal(err)
	}
	fileSet := token.NewFileSet()
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		syntax, err := parser.ParseFile(fileSet, filepath.Join(packageDir, name), nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		ast.Inspect(syntax, func(node ast.Node) bool {
			spec, ok := node.(*ast.ValueSpec)
			if !ok {
				return true
			}
			for index, value := range spec.Values {
				literal, ok := value.(*ast.BasicLit)
				if !ok || literal.Kind != token.STRING {
					continue
				}
				unquoted, err := strconv.Unquote(literal.Value)
				if err != nil || !values[unquoted] || index >= len(spec.Names) || exempt[spec.Names[index].Name] {
					continue
				}
				t.Errorf("%s redefines canonical vocabulary value %q as %s; use the owning type", name, unquoted, spec.Names[index].Name)
			}
			return true
		})
	}
}
