package main

import (
	"strings"
	"testing"

	"overgo/internal/plan"
)

// TestACheckThatCannotRunAsWrittenIsRefusedWhenFiled holds check-form: a check the commit's form stage cannot run as
// written is refused when its step is filed, by -add, -addstep, -setverify or a document's rows, with the form stage's
// reason, and the checks the plan's steps take are not. full-measurement/run's check, which assigned a variable before
// running anything, was refused at its commit on 2026-11-24, after its nine hours of measurement, the sixth refusal of
// its kind.
func TestACheckThatCannotRunAsWrittenIsRefusedWhenFiled(t *testing.T) {
	t.Parallel()
	refused := `s=$(go run ./cmd/validate -scorecard) && for m in Un-0; do printf '%s\n' "$s" | grep -qE "^\| $m \|" || exit 1; done`
	if err := verifyRefusal(refused); err == nil || !strings.Contains(err.Error(), "cannot run as written") || !strings.Contains(err.Error(), "runs nothing") {
		t.Fatalf("a check assigning a variable before running anything is filed: %v", err)
	}
	for _, command := range []string{
		"false",
		"go run ./cmd/validate > /dev/null",
		"git merge-base --is-ancestor 3fe41bfa0 HEAD && go run ./cmd/validate -lanes-passed-at 3fe41bfa0",
		"go run ./cmd/validate -scorecard | grep -oE '^\\| Un-0 \\| image-gen \\| [0-9]' | sort -u | wc -l | grep -qx 1",
	} {
		if err := verifyRefusal(command); err != nil {
			t.Errorf("%q may be a check: %v", command, err)
		}
	}
	rows := []plan.Item{{ID: "measured", Steps: []plan.Step{{ID: "run", Verify: refused}}}}
	if err := checksRefused(rows); err == nil || !strings.HasPrefix(err.Error(), "measured/run: ") {
		t.Fatalf("rows filed whole are refused by the step whose check cannot run as written: %v", err)
	}
}
