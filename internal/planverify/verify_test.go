package planverify

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
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
	} {
		t.Run(test.name, func(t *testing.T) {
			directory := t.TempDir()
			for name, content := range map[string]string{
				"go.mod":        "module example.com/probe\n\ngo 1.26\n",
				"probe_test.go": test.source,
			} {
				testcheck.NoError(t, os.WriteFile(filepath.Join(directory, name), []byte(content), 0600))
			}
			verdict, err := Execute(t.Context(), directory, "go test ./... -count=1 && "+test.auxiliary, nil)
			if test.wantError != "" {
				if err == nil || !strings.Contains(err.Error(), test.wantError) {
					t.Fatalf("verdict=%s error=%v; want %s", verdict, err, test.wantError)
				}
				return
			}
			if err != nil || verdict != runrecord.VerdictBitwiseDeterministic {
				t.Fatalf("verdict=%s error=%v", verdict, err)
			}
		})
	}
}
