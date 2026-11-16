package landing

import (
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"overgo/internal/jsonfile"
	"overgo/internal/plan"
	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// TestACardWaitIsRecordedUnderItsStage refuses a landing while it waits for the card's turn (turn-outcome/do): a
// commit another session makes in the wait refuses it once the turn is its own (takeTurnBack). Its record's outcome is
// the stage on the card it waited for, the step's check, where the stage loop named the stage only once the turn was
// taken and recorded the wait under the stage before it: on 2026-11-14 a wait of 130 minutes sits in
// landing_waits.json under "test".
func TestACardWaitIsRecordedUnderItsStage(t *testing.T) {
	// Serial: it names the lane turn and the lanes the tests take, not the host's own.
	turn, gated := laneTurn, lanes
	laneTurn = func() (string, error) { return "overgo-test-card-wait-outcome", nil }
	lanes = nil
	t.Cleanup(func() { laneTurn, lanes = turn, gated })
	fixture := newRowFixture(t, map[string]string{"internal/p/p.go": "package p\n\nfunc a() string { return \"base\" }\n"})
	editPlan(t, fixture, func(document *plan.Plan) {
		document.Items[0].Steps[0].Verify = plan.CardCheckSetting + " go test ./internal/p -run '^TestA$' -count=1"
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
	landed := make(chan error, 1)
	group.Go(func() {
		_, err := Run(t.Context(), Options{Repo: fixture.repo, Ref: "row/do", Log: freedLog{freed: sync.OnceFunc(func() { close(freed) })},
			Message: []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: first-output holds.\n")})
		landed <- err
	})
	select {
	case <-freed:
	case err := <-landed:
		close(let)
		group.Wait()
		t.Fatalf("the landing ended, %v, without waiting for the card's turn", err)
	}
	fixture.write("docs/other.md", "another session's change\n")
	fixture.git("add", "docs/other.md")
	fixture.git("commit", "-q", "-m", "another session's change")
	close(let)
	group.Wait()
	if err := <-landed; err == nil || !strings.Contains(err.Error(), "landed while this waited") {
		t.Fatalf("the landing ended %v; want it refused for the commit made while it waited", err)
	}
	var runs []spent
	testcheck.NoError(t, jsonfile.Decode(filepath.Join(fixture.repo, processlock.StateDirectory, spentRecord), &runs))
	if len(runs) == 0 || runs[len(runs)-1].Outcome != "verify" {
		t.Fatalf("the refused landing's record reads %+v; want its outcome verify, the stage on the card it waited for", runs)
	}
}
