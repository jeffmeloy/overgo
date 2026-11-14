package landing

import (
	"strings"
	"sync"
	"testing"

	"overgo/internal/plan"
)

// TestACheckOnTheCardHoldsTheCardsTurn lands a change that reaches no lane on the card but whose step's check runs on
// it, while another process holds the card's turn (check-turn/do): the landing waits for the turn, its cheap stages
// passed, before it runs the check, and before the host's tests (verify-first/do), so the check's processes ride on its
// place. A landing took the turn only for a
// change reaching a card lane, and tabfm-cases/do's check, its trainer stepping on the card, took the card per context,
// each waiting out a render's clip.
func TestACheckOnTheCardHoldsTheCardsTurn(t *testing.T) {
	// Serial: it names the lane turn and the lanes the tests take, not the host's own.
	turn, gated := laneTurn, lanes
	laneTurn = func() (string, error) { return "overgo-test-check-turn", nil }
	lanes = nil
	t.Cleanup(func() { laneTurn, lanes = turn, gated })
	fixture := newRowFixture(t, map[string]string{"internal/p/p.go": "package p\n\nfunc a() string { return \"base\" }\n"})
	editPlan(t, fixture, func(document *plan.Plan) {
		document.Items[0].Steps[0].Verify = cudaTestSetting + " go test ./internal/p -run '^TestA$' -count=1"
	})
	fixture.git("add", "-A")
	fixture.git("commit", "-q", "-m", "the row's check")
	fixture.write("internal/p/p.go", "package p\n\nfunc a() string { return \"shipped\" }\n")
	fixture.write("internal/p/p_test.go", "package p\n\nimport \"testing\"\n\nfunc TestA(t *testing.T) {\n\tif a() != \"shipped\" {\n\t\tt.Fatal(a())\n\t}\n}\n")
	held, let := make(chan struct{}), make(chan struct{})
	var group sync.WaitGroup
	group.Go(heldTurn(t, held, let))
	<-held
	freed := make(chan struct{})
	var said strings.Builder
	var before string
	landed := make(chan error, 1)
	group.Go(func() {
		_, err := Run(t.Context(), Options{Repo: fixture.repo, Ref: "row/do",
			Log:     freedLog{freed: sync.OnceFunc(func() { before = said.String(); close(freed) }), said: &said},
			Message: []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: first-output holds.\n")})
		landed <- err
	})
	select {
	case <-freed:
	case err := <-landed:
		close(let)
		group.Wait()
		t.Fatalf("the landing ended, %v, without waiting for the card's turn before its check on the card:\n%s", err, said.String())
	}
	close(let)
	group.Wait()
	if err := <-landed; err != nil {
		t.Fatalf("the landing refused once the turn was free: %v", err)
	}
	if !strings.Contains(before, "land: invariants ok") || strings.Contains(before, "land: verify ok") || strings.Contains(before, "land: test ok") {
		t.Errorf("the landing waited for the card's turn before its cheap stages passed, or ran its check on the card without it or after the tests:\n%s", before)
	}
}
