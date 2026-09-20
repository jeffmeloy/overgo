package gate

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/plan"
)

// TestProofHorizonMovesOnlyToTheProvedRevision holds a landing's horizon to
// what its gate proved: a landing that does not ship the receipt, or ships
// its removal, is admitted; one that ships exactly the authority's own seal
// is admitted; and any other commit or count is refused with the command
// that produces the right one.
func TestProofHorizonMovesOnlyToTheProvedRevision(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	path := filepath.Join(repo, filepath.FromSlash(plan.ProofHorizonPath))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	var proved plan.CompletionAuthority
	sealed := `{"commit":"","completions":0}`
	for _, test := range []struct {
		name, shipped string
		paths         []string
		refused       bool
	}{
		{name: "a landing that does not touch the receipt", shipped: `{"commit":"abc","completions":9}`, paths: []string{plan.Path}},
		{name: "a landing that removes the receipt", paths: []string{plan.ProofHorizonPath}},
		{name: "the authority's own seal", shipped: sealed, paths: []string{plan.ProofHorizonPath}},
		{name: "another commit", shipped: `{"commit":"abc","completions":0}`, paths: []string{plan.ProofHorizonPath}, refused: true},
		{name: "another count", shipped: `{"commit":"","completions":1}`, paths: []string{plan.ProofHorizonPath}, refused: true},
	} {
		_ = os.Remove(path)
		if test.shipped != "" {
			if err := os.WriteFile(path, []byte(test.shipped), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		err := (&gateContext{repo: repo, paths: test.paths}).admitProofHorizon(proved)
		if test.refused != (err != nil) || err != nil && !strings.Contains(err.Error(), "-seal-horizon") {
			t.Fatalf("%s = %v, refused want %v", test.name, err, test.refused)
		}
	}
}
