package planverify

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testevidence"
)

func TestMixedVerifierRepeatEvidence(t *testing.T) {
	for _, test := range []struct {
		name, source, auxiliary, wantError string
	}{
		{
			name:      "ordinary auxiliary output",
			source:    "package probe\nimport \"testing\"\nfunc TestBehavior(t *testing.T) {}\n",
			auxiliary: "printf 'smoke: passed=1 failed=0\\n'",
		},
		{
			name:      "malformed evidence",
			source:    "package probe\nimport \"testing\"\nfunc TestBehavior(t *testing.T) {}\n",
			auxiliary: "printf '{malformed\\n'", wantError: "VACUOUS",
		},
		{
			name: "repeat unavailable with unchanged verdict",
			source: `package probe
import ("os"; "testing")
func TestBehavior(t *testing.T) {
 if _, err := os.Stat("seen"); os.IsNotExist(err) {
  if err := os.WriteFile("seen", nil, 0600); err != nil { t.Fatal(err) }
 } else { t.Log("UNAVAILABLE: repeat fixture") }
}
`,
			auxiliary: "printf 'smoke: passed=1 failed=0\n'", wantError: "REPEAT vacuous",
		},
		{
			name: "changed repeat verdict",
			source: `package probe
import ("os"; "testing")
func TestBehavior(t *testing.T) {
 if _, err := os.Stat("seen"); os.IsNotExist(err) {
  t.Run("first-only", func(t *testing.T) {})
  if err := os.WriteFile("seen", nil, 0600); err != nil { t.Fatal(err) }
 }
}
`,
			auxiliary: "printf 'smoke: passed=1 failed=0\\n'", wantError: "NOT deterministic",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			directory := t.TempDir()
			for name, content := range map[string]string{
				"go.mod":        "module example.com/probe\n\ngo 1.26\n",
				"probe_test.go": test.source,
			} {
				if err := os.WriteFile(filepath.Join(directory, name), []byte(content), 0600); err != nil {
					t.Fatal(err)
				}
			}
			verdict, err := Execute(t.Context(), directory, "go test ./... -count=1 && "+test.auxiliary, nil)
			if test.wantError != "" {
				if err == nil || !strings.Contains(err.Error(), test.wantError) {
					t.Fatalf("verdict=%s error=%v; want %s", verdict, err, test.wantError)
				}
				return
			}
			if err != nil || verdict != testevidence.VerdictBitwiseDeterministic {
				t.Fatalf("verdict=%s error=%v", verdict, err)
			}
		})
	}
}
