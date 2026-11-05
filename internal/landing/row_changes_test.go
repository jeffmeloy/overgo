package landing

import (
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// rowChangeFixture is a checkout whose plan holds row/do, ready to land, and later/do, whose check is "false", with a
// Go package for the stages to read; land lands named under a message that ends with extra.
func rowChangeFixture(t *testing.T) (*sliceFixture, func(named, extra string) error) {
	t.Helper()
	fixture := newRowFixture(t, map[string]string{"a.txt": "base\n", "p/p.go": "// Package p is a fixture.\npackage p\n"})
	if err := plan.Save(fixture.planPath(), plan.Plan{Campaign: "fixture", Items: []plan.Item{
		{ID: "row", Title: "row", Status: plan.StatusOpen, Steps: []plan.Step{{ID: "do", Title: "row", Status: plan.StatusOpen, Verify: "true"}}},
		{ID: "later", Title: "later", Status: plan.StatusOpen, Steps: []plan.Step{{ID: "do", Title: "later", Status: plan.StatusOpen, Verify: "false"}}},
	}}); err != nil {
		t.Fatal(err)
	}
	fixture.git("add", "-A")
	fixture.git("commit", "-q", "-m", "plan of two rows")
	return fixture, func(named, extra string) error {
		t.Helper()
		var log strings.Builder
		_, err := Run(t.Context(), Options{Repo: fixture.repo, Ref: named, Log: &log,
			Message: []byte(named + ": change\n\nCause: a fixture change.\nPredicted effect: first-output holds.\n" + extra)})
		return err
	}
}

// editPlan rewrites the fixture's plan through edit.
func editPlan(t *testing.T, fixture *sliceFixture, edit func(*plan.Plan)) {
	t.Helper()
	document, err := plan.Load(fixture.planPath())
	testcheck.NoError(t, err)
	edit(&document)
	testcheck.NoError(t, plan.Save(fixture.planPath(), document))
}

// TestALandingThatDeletesARowNamesIt lands row/do with later/do removed from the plan: a message that does not name the
// removed step is refused and the trunk stays, and one that names it lands with a trailer naming it.
func TestALandingThatDeletesARowNamesIt(t *testing.T) {
	t.Parallel()
	fixture, land := rowChangeFixture(t)
	fixture.write("a.txt", "changed\n")
	editPlan(t, fixture, func(document *plan.Plan) { document.Items = document.Items[:1] })
	head := fixture.git("rev-parse", "HEAD")
	if err := land("row/do", ""); err == nil || !strings.Contains(err.Error(), "later/do") {
		t.Fatalf("a landing that removes later/do without naming it ended as %v, want it refused naming the step", err)
	}
	if now := fixture.git("rev-parse", "HEAD"); now != head {
		t.Fatalf("the refused landing moved the trunk to %s from %s", now, head)
	}
	if err := land("row/do", "Plan: later/do is retired, since the fixture needs it no more.\n"); err != nil {
		t.Fatalf("the landing that names the step it removes: %v", err)
	}
	if body := fixture.git("log", "-1", "--format=%B"); !strings.Contains(body, retiredRowTrailer+": later/do") {
		t.Fatalf("the commit that removed later/do holds no trailer naming it:\n%s", body)
	}
}

// TestAReplacedCheckMustFailWithoutTheChange lands row/do while replacing later/do's check: a replacement that already
// passes on the commit the change builds on proves nothing of the work and is refused, and one that fails there lands.
func TestAReplacedCheckMustFailWithoutTheChange(t *testing.T) {
	t.Parallel()
	fixture, land := rowChangeFixture(t)
	fixture.write("a.txt", "changed\n")
	replace := func(check string) {
		editPlan(t, fixture, func(document *plan.Plan) { document.Items[1].Steps[0].Verify = check })
	}
	replace("true")
	if err := land("row/do", ""); err == nil || !strings.Contains(err.Error(), "proves nothing") {
		t.Fatalf("a replacing check that passes without the change ended as %v, want it refused", err)
	}
	replace("test -f p/marker.txt")
	fixture.write("p/marker.txt", "made by the change\n")
	if err := land("row/do", ""); err != nil {
		t.Fatalf("a replacing check that fails without the change: %v", err)
	}
}
