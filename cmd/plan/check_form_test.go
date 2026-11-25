package main

import (
	"strings"
	"testing"

	"overgo/internal/plan"
)

// TestACheckThatCannotRunAsWrittenIsRefusedWhenFiled holds check-form: a check the commit's form stage cannot run as
// written is refused when its step is filed, by -add, -addstep, -setverify or a document's rows, with the form stage's
// reason, and the checks the plan's steps take are not. A plain chain whose first segment only assigns runs nothing
// of its own and is refused; full-measurement/run's check, which assigned a command's output in a composite command,
// is a check (check-form-reads, SQA-483).
func TestACheckThatCannotRunAsWrittenIsRefusedWhenFiled(t *testing.T) {
	t.Parallel()
	const refused = "A=1 && go test ./internal/plan -run '^TestX$'"
	if err := verifyRefusal(refused); err == nil || !strings.Contains(err.Error(), "cannot run as written") || !strings.Contains(err.Error(), "runs nothing") {
		t.Fatalf("a plain chain whose first segment only assigns is filed: %v", err)
	}
	for _, command := range []string{
		`s=$(go run ./cmd/validate -scorecard) && for m in Un-0; do printf '%s\n' "$s" | grep -qE "^\| $m \|" || exit 1; done`,
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

// TestACheckNoWorkCanPassIsRefusedWhenFiled holds check-form-reads/do (SQA-482): the check false, which fails at its
// commit whatever the work does, is refused when filed, with the edit that names the test the change will add, and a
// check naming that test is filed. measured-names-first/do, contended-credit/do, user-size-inapplicable/do and
// flash-next-projection/do were each filed with false, retired, and landed under another step.
func TestACheckNoWorkCanPassIsRefusedWhenFiled(t *testing.T) {
	t.Parallel()
	for _, command := range []string{"false", " false "} {
		if err := verifyRefusal(command); err == nil || !strings.Contains(err.Error(), "fails whatever the work does") {
			t.Errorf("the check %q was filed as %v; want it refused", command, err)
		}
	}
	if err := verifyRefusal("go test ./internal/plan -count=1 -run '^TestTheChangesOwnTest$'"); err != nil {
		t.Errorf("a check naming the test the change will add was refused: %v", err)
	}
}
