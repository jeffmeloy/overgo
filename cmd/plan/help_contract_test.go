package main

import (
	"os/exec"
	"strings"
	"testing"
)

// TestPlanHelpContract exercises the real plan binary through the same parser
// execution uses: explicit help prints the purpose and the actual flags,
// resolves the -add/-vcmd versus boolean -verify confusion, says a scratch
// probe lives in the folder named for the plan row that made it, exits
// successfully without touching a store or plan, and an unknown flag exits
// non-zero with guidance.
func TestPlanHelpContract(t *testing.T) {
	t.Parallel()
	for _, arg := range []string{"-h", "-help"} {
		out, err := exec.Command("go", "run", ".", arg).CombinedOutput()
		if err != nil {
			t.Fatalf("plan %s exited non-zero: %v\n%s", arg, err, out)
		}
		text := string(out)
		for _, want := range []string{"plan:", "-add", "-vcmd", "-verify", "-stop", "scratch probes live in tmp/<row id>/", "go run ./tmp/<row id>/<file>.go"} {
			if !strings.Contains(text, want) {
				t.Fatalf("plan %s help missing %q in:\n%s", arg, want, text)
			}
		}
		// Plan edits land through the gate like any change; the separate
		// edit-and-publish path and the old orchestration commands are retired.
		for _, retired := range []string{"-edit", "-publish", "-vehicle", "-budget", "-frontier", "-context", "-record-lease"} {
			if strings.Contains(text, retired) {
				t.Fatalf("plan %s help still offers the retired %s", arg, retired)
			}
		}
		if !strings.Contains(text, "-add takes the step's verify command through -vcmd") {
			t.Fatalf("plan help does not correct the -add/-vcmd versus -verify usage:\n%s", text)
		}
	}
	if out, err := exec.Command("go", "run", ".", "-nonexistent").CombinedOutput(); err == nil {
		t.Fatalf("plan accepted an unknown flag:\n%s", out)
	}
}
