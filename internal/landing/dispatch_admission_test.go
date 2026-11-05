package landing

import (
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
	"overgo/internal/worklease"
)

// TestALandingRefusesWhatDispatchWouldPass lands as a worker over a plan whose first row is ready, whose second waits
// on it and whose third is another lane's. The second and the third are refused, and the trunk does not move. Under a
// recorded stop the ready row is refused too. Once the stop is resumed, it lands.
func TestALandingRefusesWhatDispatchWouldPass(t *testing.T) {
	t.Parallel()
	fixture := newRowFixture(t, map[string]string{".gitignore": ".overgo-runtime/\n", "a.txt": "base\n", "p/p.go": "// Package p is a fixture.\npackage p\n"})
	step := func(after ...string) []plan.Step {
		return []plan.Step{{ID: "do", Title: "do", Status: plan.StatusOpen, Verify: "true", DependsOn: after}}
	}
	if err := plan.Save(fixture.planPath(), plan.Plan{Campaign: "fixture", Items: []plan.Item{
		{ID: "row", Title: "row", Status: plan.StatusOpen, Steps: step()},
		{ID: "later", Title: "later", Status: plan.StatusOpen, Steps: step("row/do")},
		{ID: "lane", Title: "lane", Status: plan.StatusOpen, Owner: "other-lane", Steps: step()},
	}}); err != nil {
		t.Fatal(err)
	}
	fixture.git("add", "-A")
	fixture.git("commit", "-q", "-m", "plan of three rows")
	fixture.write("a.txt", "changed\n")
	request := plan.DispatchRequest{Role: "fixture-lead", Worker: "fixture-worker", Mode: plan.ExecutionInteractive}
	land := func(reference string) error {
		t.Helper()
		var log strings.Builder
		_, err := Run(t.Context(), Options{Repo: fixture.repo, Ref: reference, Log: &log, Dispatch: &request,
			Message: []byte(reference + ": change\n\nCause: a fixture change.\nPredicted effect: first-output holds.\n")})
		return err
	}
	head := fixture.git("rev-parse", "HEAD")
	for _, refused := range []string{"later/do", "lane/do"} {
		if err := land(refused); err == nil || !strings.Contains(err.Error(), refused) {
			t.Fatalf("landing %s, which dispatch would not give, ended as %v, want it refused naming it", refused, err)
		}
	}
	event := plan.ControlEvent{Kind: "stop", Lane: worklease.UnassignedRole, Worker: "stop-worker", Worktree: fixture.repo,
		Mode: plan.ExecutionAll, ReasonCode: "requested", Detail: "asked to stop", CodeCommit: head}
	stopped, err := plan.RecordStop(t.Context(), event)
	testcheck.NoError(t, err)
	if err := land("row/do"); err == nil || !strings.Contains(err.Error(), "stop") {
		t.Fatalf("landing row/do under a recorded stop ended as %v, want it refused for the stop", err)
	}
	if now := fixture.git("rev-parse", "HEAD"); now != head {
		t.Fatalf("a refused landing moved the trunk to %s from %s", now, head)
	}
	event.Kind, event.ReasonCode, event.Detail, event.Previous = "resume", "resume", "resumed", stopped.ID
	if _, err := plan.RecordStop(t.Context(), event); err != nil {
		t.Fatal(err)
	}
	if err := land("row/do"); err != nil {
		t.Fatalf("landing row/do, the step dispatch gives, once the stop is resumed: %v", err)
	}
}
