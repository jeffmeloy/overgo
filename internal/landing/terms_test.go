package landing

import (
	"fmt"
	"slices"
	"strings"
	"testing"

	"overgo/internal/gosyntax"
	"overgo/internal/plan"
)

// Sources of the fixture package a landing grows: nothing, one function, and
// a second beside it.
const (
	emptyPackage = "package p\n"
	oneFunction  = emptyPackage + "\nfunc a() string { return \"a\" }\n"
	twoFunctions = oneFunction + "\nfunc b() string { return a() + a() }\n"
)

// packageNodes counts the production nodes of the fixture package holding source.
func packageNodes(t *testing.T, source string) int {
	t.Helper()
	snapshot, err := (gosyntax.Snapshot{}).Overlay(map[string][]byte{"internal/p/p.go": []byte(source)})
	if err != nil {
		t.Fatal(err)
	}
	return writtenNodes(snapshot, false)
}

// commitPlan commits document as the fixture's plan.
func (f *sliceFixture) commitPlan(document plan.Plan) {
	f.t.Helper()
	if err := plan.Save(f.planPath(), document); err != nil {
		f.t.Fatal(err)
	}
	f.git("add", "-A")
	f.git("commit", "-q", "-m", "plan")
}

// rowPlan is a plan of one row that may add count nodes, with the steps named,
// each checked by the command beside it.
func rowPlan(count int, steps ...[2]string) plan.Plan {
	row := plan.Item{ID: "row", Title: "row", Status: plan.StatusOpen, Budget: plan.Budget{Nodes: count, Reason: "the fixture's growth"}}
	for _, step := range steps {
		row.Steps = append(row.Steps, plan.Step{ID: step[0], Title: step[0], Status: plan.StatusOpen, Verify: step[1]})
	}
	return plan.Plan{Campaign: "fixture", Doctrine: "fixture", Items: []plan.Item{row}}
}

// TestChangeIsJudgedByTheTermsItStartedWith holds a change to the count and the
// check its row had at the commit it builds on. A change that carries code and
// also raises its own row's count, replaces its own step's check with one that
// passes, and adds a step, is checked by the earlier check, bounded by the
// earlier count, and cannot land the step it added. The same edit to the plan
// alone lands by the plan it holds, which is how a row is filed and its terms
// changed; and the code then lands under the terms that landing left.
func TestChangeIsJudgedByTheTermsItStartedWith(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	fixture := newRowFixture(t, map[string]string{"internal/p/p.go": emptyPackage})
	grown := packageNodes(t, oneFunction) - packageNodes(t, emptyPackage)
	fixture.commitPlan(rowPlan(grown-1, [2]string{"do", "false"}))
	edited := rowPlan(grown, [2]string{"do", "true"}, [2]string{"file", "true"})
	if err := plan.Save(fixture.planPath(), edited); err != nil {
		t.Fatal(err)
	}
	fixture.write("internal/p/p.go", oneFunction)

	l := fixture.candidateOf("row/do")
	if left := l.charge(0); !slices.Equal(l.verifiers, []string{"false"}) || left != grown-1 {
		t.Fatalf("a change that edits its row's terms is judged by the check %q and may add %d, want the check and the %d of the commit it builds on", l.verifiers, left, grown-1)
	}
	if err := l.verify(ctx); err == nil {
		t.Error("a change passed the check it wrote for itself")
	}
	if err := l.rules(ctx); err == nil || !strings.Contains(err.Error(), fmt.Sprintf("may add %d", grown-1)) {
		t.Errorf("a change that raised its own row's count: %v, want the growth refused by the earlier count", err)
	}
	added := &landing{Options: Options{Repo: fixture.repo, Ref: "row/file"}, scratch: t.TempDir()}
	if err := added.parents(ctx); err != nil {
		t.Fatal(err)
	}
	if err := added.shipSet(ctx); err == nil || !strings.Contains(err.Error(), "land the plan's edit alone first") {
		t.Errorf("a change that carries code landed a step it added itself: %v", err)
	}

	// The plan's edit alone is judged by the plan it holds, and files the terms.
	fixture.write("internal/p/p.go", emptyPackage)
	filing := fixture.candidateOf("row/file")
	if !slices.Equal(filing.verifiers, []string{"true"}) {
		t.Fatalf("a change to the plan alone is checked by %q, want the check its plan gives the step", filing.verifiers)
	}
	if _, err := filing.commit(ctx); err != nil {
		t.Fatal(err)
	}
	fixture.write("internal/p/p.go", oneFunction)
	l = fixture.candidateOf("row/do")
	if left := l.charge(0); !slices.Equal(l.verifiers, []string{"true"}) || left != grown {
		t.Fatalf("after its terms landed, the change is judged by the check %q and may add %d, want the filed check and %d", l.verifiers, left, grown)
	}
	if err := l.rules(ctx); err != nil {
		t.Fatalf("the change did not pass under the terms a landing before it filed: %v", err)
	}
}

// TestItemsGrowthIsChargedAcrossItsLandings holds a row's count to the whole of
// what its landings add. A row of two steps may add one node less than the two
// together add: the first landing is admitted and the plan it publishes
// carries what is left, and the second, which the count as first written would
// still admit, is refused the growth the first used. What a landing removes
// comes back to the count, and rows landed together share theirs, the
// remainder staying with the first the plan still holds.
func TestItemsGrowthIsChargedAcrossItsLandings(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	fixture := newRowFixture(t, map[string]string{"internal/p/p.go": emptyPackage})
	first := packageNodes(t, oneFunction) - packageNodes(t, emptyPackage)
	second := packageNodes(t, twoFunctions) - packageNodes(t, oneFunction)
	count := first + second - 1
	if count < second {
		t.Fatalf("the fixture's count %d does not admit the second step's %d nodes alone, so it shows nothing", count, second)
	}
	fixture.commitPlan(rowPlan(count, [2]string{"do", "true"}, [2]string{"later", "true"}, [2]string{"last", "true"}))
	left := func(commit string) int {
		t.Helper()
		published, err := plan.Parse([]byte(fixture.git("cat-file", "blob", commit+":"+plan.Path)))
		if err != nil || len(published.Items) != 1 {
			t.Fatalf("the landed plan = %+v, %v", published.Items, err)
		}
		return published.Items[0].Budget.Nodes
	}

	fixture.write("internal/p/p.go", oneFunction)
	l := fixture.candidateOf("row/do")
	if err := l.rules(ctx); err != nil {
		t.Fatalf("the first step's growth was refused: %v", err)
	}
	commit, err := l.commit(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if got := left(commit); got != count-first {
		t.Fatalf("after a landing that added %d of %d nodes the plan gives the row %d, want %d", first, count, got, count-first)
	}

	fixture.write("internal/p/p.go", twoFunctions)
	l = fixture.candidateOf("row/later")
	if err := l.rules(ctx); err == nil || !strings.Contains(err.Error(), fmt.Sprintf("adds %d production syntax nodes", second)) || !strings.Contains(err.Error(), fmt.Sprintf("may add %d", count-first)) {
		t.Fatalf("the second step's growth of %d with %d left: %v, want it refused the growth the first step used", second, count-first, err)
	}

	// A landing that removes code returns what it removed to the count.
	fixture.write("internal/p/p.go", emptyPackage)
	l = fixture.candidateOf("row/later")
	if err := l.rules(ctx); err != nil {
		t.Fatal(err)
	}
	if commit, err = l.commit(ctx); err != nil {
		t.Fatal(err)
	}
	if got := left(commit); got != count {
		t.Fatalf("after a landing that removed the %d nodes the first added the plan gives the row %d, want %d", first, got, count)
	}

	// Rows landed together are admitted by the sum of their counts, and the
	// first of them the plan still holds is charged: each row's count is the
	// one it started with, whatever the change wrote, and a row not landed is
	// left as the change has it.
	row := func(id string, count int) plan.Item { return plan.Item{ID: id, Budget: plan.Budget{Nodes: count}} }
	together := &landing{Options: Options{Ref: "gone/do,first/do,other/do"},
		terms:    plan.Plan{Items: []plan.Item{row("gone", 4), row("first", 6), row("other", 3), row("aside", 2)}},
		advanced: plan.Plan{Items: []plan.Item{row("first", 100), row("other", 100), row("aside", 7)}}}
	had := together.charge(5)
	var counts []int
	for _, item := range together.advanced.Items {
		counts = append(counts, item.Budget.Nodes)
	}
	if had != 13 || !slices.Equal(counts, []int{1, 3, 7}) {
		t.Fatalf("rows of 4, 6 and 3 nodes landed together with 5 added: %d between them before, and the plan gives %v; want 13, then the first row still open charged to 1, the other its own 3, and the row not landed untouched", had, counts)
	}
}
