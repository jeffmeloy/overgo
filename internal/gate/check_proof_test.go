package gate

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/planverify"
)

// Changed verification must refuse before executing even a passing command.
// Preflight uses an on-disk execution witness; the frozen path must refuse
// before asking for candidate preparation or recording acceptance.
func TestChangedCheckRefusesBeforeExecution(t *testing.T) {
	t.Parallel()
	for _, mode := range []string{"preflight", "frozen"} {
		t.Run(mode, func(t *testing.T) {
			g, _, _ := verificationBatchFixture(t, "pass")
			document, err := plan.Load(filepath.Join(g.repo, plan.Path))
			if err != nil {
				t.Fatal(err)
			}
			document.Items[0].Steps[0].Verify = "go test . -run '^TestExecutionWitness$' -count=1 -v"
			if err := plan.Save(filepath.Join(g.repo, plan.Path), document); err != nil {
				t.Fatal(err)
			}
			source := `package batchfixture
import ("os"; "testing")
func TestExecutionWitness(t *testing.T) {
    if err := os.WriteFile("witness", []byte("executed"), 0600); err != nil { t.Fatal(err) }
}
`
			if err := os.WriteFile(filepath.Join(g.repo, "unit_test.go"), []byte(source), 0600); err != nil {
				t.Fatal(err)
			}
			g, _, _ = verificationBatchContext(t, g.repo)
			var output bytes.Buffer
			if mode == "preflight" {
				err = g.preflightAcceptance(t.Context(), &output)
			} else {
				_, err = g.stepAcceptance()
			}
			if err == nil || !strings.Contains(err.Error(), "changes its own check") {
				t.Errorf("changed check was not refused before execution: %v; output=%s", err, output.String())
			}
			if _, err := os.Stat(filepath.Join(g.repo, "witness")); !errors.Is(err, os.ErrNotExist) {
				t.Errorf("changed acceptance command ran: %v", err)
			}
			if g.acceptedTree != "" || g.stepEvidence["acceptance"] != "" {
				t.Fatal("refused check recorded acceptance")
			}
		})
	}
}

// TestGrowingLandingNeedsACheckThatFailsWithoutIt holds the proof rule to its
// cases. A landing that adds production code is refused when its check
// already passes at the base commit, and admitted when the check fails or its
// test is absent there. A landing that removes or holds production code is
// admitted on a passing check, because what it has to show is that nothing
// broke. Any other failure at the base is reported, not waved through.
func TestGrowingLandingNeedsACheckThatFailsWithoutIt(t *testing.T) {
	t.Parallel()
	failed := fmt.Errorf("%w: exit status 1", planverify.ErrFailed)
	absent := fmt.Errorf("%w: no test ran", planverify.ErrVacuous)
	broken := errors.New("candidate worktree unavailable")
	for _, tc := range []struct {
		name    string
		growth  int
		baseErr error
		refused bool
		want    error
	}{
		{name: "grows and already passes", growth: 40, refused: true},
		{name: "grows and fails without it", growth: 40, baseErr: failed},
		{name: "grows and its test is absent", growth: 40, baseErr: absent},
		{name: "holds and passes", growth: 0},
		{name: "shrinks and passes", growth: -120},
		{name: "grows and the base cannot run", growth: 40, baseErr: broken, want: broken},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := requireProvenCheck(tc.growth, tc.baseErr)
			switch {
			case tc.want != nil:
				if !errors.Is(err, tc.want) {
					t.Fatalf("error = %v, want %v", err, tc.want)
				}
			case tc.refused != (err != nil):
				t.Fatalf("refused = %t, want %t: %v", err != nil, tc.refused, err)
			}
		})
	}
}

// TestRowCannotRewriteItsOwnCheck holds a landing to the check its row had at
// the base commit. Changing it in the landing it judges is refused; the same
// check, or a row the landing adds, is admitted.
func TestRowCannotRewriteItsOwnCheck(t *testing.T) {
	t.Parallel()
	base := plan.Plan{Items: []plan.Item{{ID: "work", Steps: []plan.Step{{ID: "do", Verify: "go test ./a -run '^TestA$'"}}}}}
	if err := checkUnchangedSinceBase(base, "work/do", "go test ./a -run '^TestWeaker$'"); err == nil {
		t.Fatal("a landing rewrote its own row's check")
	}
	if err := checkUnchangedSinceBase(base, "work/do", "go test ./a -run '^TestA$'"); err != nil {
		t.Fatalf("an unchanged check was refused: %v", err)
	}
	if err := checkUnchangedSinceBase(base, "added/do", "go test ./b -run '^TestB$'"); err != nil {
		t.Fatalf("a row the landing adds was refused: %v", err)
	}
}
