package landing

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/processcontrol"
)

// TestAnOnTheCardCheckFailsBeforeTheTestStage lands a change whose step's check runs on the card (verify-first/do,
// SQA-322): a check that fails refuses the landing before any package's tests run, where it ran after the tests and the
// lanes, and 13 of 26 checks on the card failed only after them, 204 minutes; a check that passes gives the card's turn
// up before the test stage, so the tests, which use no card, ride on no turn, a fixture test failing where they do.
func TestAnOnTheCardCheckFailsBeforeTheTestStage(t *testing.T) {
	// Serial: it names the lane turn and the lanes the tests take, not the host's own, and the environment the turn is
	// passed in.
	const name = "overgo-test-verify-first-turn"
	turn, gated := laneTurn, lanes
	laneTurn = func() (string, error) { return name, nil }
	lanes = nil
	t.Cleanup(func() { laneTurn, lanes = turn, gated })
	tests := "package p\n\nimport (\n\t\"os\"\n\t\"strings\"\n\t\"testing\"\n)\n\n" +
		"func TestA(t *testing.T) {\n\tif a() != \"shipped\" {\n\t\tt.Fatal(a())\n\t}\n}\n\n" +
		"func TestB(t *testing.T) {\n\tif held := os.Getenv(\"" + processcontrol.TurnEnvironment + "\"); strings.Contains(held, \"" + name + "\") {\n" +
		"\t\tt.Fatalf(\"a test the test stage runs rides on the card's turn %q\", held)\n\t}\n}\n"
	land := func(t *testing.T, value string) (string, error) {
		t.Helper()
		fixture := newRowFixture(t, map[string]string{"internal/p/p.go": "package p\n\nfunc a() string { return \"base\" }\n"})
		editPlan(t, fixture, func(document *plan.Plan) {
			document.Items[0].Steps[0].Verify = plan.CardCheckSetting + " go test ./internal/p -run '^TestA$' -count=1"
		})
		fixture.git("add", "-A")
		fixture.git("commit", "-q", "-m", "the row's check")
		fixture.write("internal/p/p.go", "package p\n\nfunc a() string { return \""+value+"\" }\n")
		fixture.write("internal/p/p_test.go", tests)
		var said strings.Builder
		_, err := Run(t.Context(), Options{Repo: fixture.repo, Ref: "row/do", Log: &said,
			Message: []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: first-output holds.\n")})
		return said.String(), err
	}
	// A stage is said as it starts: "land: stage 9 of 11: test".
	tested := func(said string) bool {
		return slices.ContainsFunc(strings.Split(said, "\n"), func(line string) bool {
			return strings.HasPrefix(line, "land: stage ") && strings.HasSuffix(line, ": test")
		})
	}
	said, err := land(t, "unshipped")
	if err == nil || !strings.Contains(err.Error(), "verify") || tested(said) {
		t.Errorf("a landing whose check on the card fails was refused with %v; want it refused at the verify, before the test stage:\n%s", err, said)
	}
	if said, err := land(t, "shipped"); err != nil {
		t.Errorf("a landing whose check on the card passed was refused: %v; its tests ride on no turn once the check is done:\n%s", err, said)
	}
}
