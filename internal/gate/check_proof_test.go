package gate

import (
	"errors"
	"fmt"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/planverify"
)

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
