package landing

import (
	"fmt"
	"io"
	"slices"
	"strings"
	"testing"

	"overgo/internal/gosyntax"
	"overgo/internal/plan"
	"overgo/internal/testcheck"
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
	snapshot := testcheck.Must((gosyntax.Snapshot{}).Overlay(map[string][]byte{"internal/p/p.go": []byte(source)}))(t)
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
	return plan.Plan{Campaign: "fixture", Items: []plan.Item{row}}
}

// TestChangeIsJudgedByTheTermsItStartedWith holds a change to the count and the
// check its row had at the commit it builds on. A change that carries code and
// also raises its own row's count, replaces its own step's check with one that
// passes, and adds a step, is checked by the earlier check and bounded by the
// earlier count, and the step it added answers to a check that must fail
// without the change. The same edit to the plan
// alone lands by the plan it holds, which is how a row is filed and its terms
// changed; and the code then lands under the terms that landing left.
func TestChangeIsJudgedByTheTermsItStartedWith(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	fixture := newRowFixture(t, map[string]string{"internal/p/p.go": emptyPackage})
	grown := packageNodes(t, oneFunction) - packageNodes(t, emptyPackage)
	fixture.commitPlan(rowPlan(grown-1, [2]string{"do", "false"}))
	edited := rowPlan(grown, [2]string{"do", "true"}, [2]string{"file", "true"})
	testcheck.NoError(t, plan.Save(fixture.planPath(), edited))
	fixture.write("internal/p/p.go", oneFunction)

	l := fixture.candidateOf("row/do")
	if left := l.charge(0); !slices.Equal(l.verifiers, []string{"false"}) || left != grown-1 {
		t.Fatalf("a change that edits its row's terms is judged by the check %q and may add %d, want the check and the %d of the commit it builds on", l.verifiers, left, grown-1)
	}
	if err := l.verify(ctx); err == nil {
		t.Error("a change passed the check it wrote for itself")
	}
	if err := l.rules(ctx); err != nil || !strings.Contains(l.details["rules"], fmt.Sprintf("of the %d its row may add", grown-1)) {
		t.Errorf("a change that raised its own row's count: %v, %q, want its growth recorded against the earlier count", err, l.details["rules"])
	}
	added := fixture.candidateOf("row/file")
	added.Log = io.Discard
	if err := added.verify(ctx); err == nil || !strings.Contains(err.Error(), "proves nothing") {
		t.Errorf("a step added with a check that passes without the change: %v", err)
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

// TestANewRowLandsWithItsWork: a row new in a change lands with it, judged by its own check and count, and its check, which
// fails on the commit the change builds on, passes with the change, so no row need be filed in a commit of its own.
func TestANewRowLandsWithItsWork(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	fixture := newRowFixture(t, map[string]string{"internal/p/p.go": emptyPackage})
	grown := packageNodes(t, oneFunction) - packageNodes(t, emptyPackage)
	fixture.commitPlan(rowPlan(0, [2]string{"later", "true"}))
	document := rowPlan(0, [2]string{"later", "true"})
	document.Items = append(document.Items, plan.Item{
		ID: "new", Title: "new", Status: plan.StatusOpen, Budget: plan.Budget{Nodes: grown, Reason: "the new row's estimate"},
		Steps: []plan.Step{{ID: "do", Title: "new", Status: plan.StatusOpen, Verify: "test -f internal/p/marker"}},
	})
	testcheck.NoError(t, plan.Save(fixture.planPath(), document))
	fixture.write("internal/p/p.go", oneFunction)
	fixture.write("internal/p/marker", "here\n")
	l := fixture.candidateOf("new/do")
	l.Log = io.Discard
	if err := l.rules(ctx); err != nil {
		t.Fatalf("a new row's own count did not admit its work: %v", err)
	}
	if err := l.verify(ctx); err != nil {
		t.Fatalf("a new row's check: %v", err)
	}
	if _, err := l.commit(ctx); err != nil {
		t.Fatal(err)
	}
	landed := testcheck.Must(plan.Load(fixture.planPath()))(t)
	if slices.ContainsFunc(landed.Items, func(item plan.Item) bool { return item.ID == "new" }) {
		t.Fatal("the new row stayed in the plan its landing published")
	}
}

// TestItemsGrowthIsChargedAcrossItsLandings records against a row's count the
// whole of what its landings add. A row whose count is one node less than its
// first two steps add has each landing's growth taken from it: the plan the
// first publishes carries what is left, and the second, past the count, lands
// and leaves it below zero by what it grew past. What a landing removes comes
// back to the count, and rows landed together share theirs, the remainder
// staying with the first the plan still holds.
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
	fixture.commitPlan(rowPlan(count, [2]string{"do", "true"}, [2]string{"later", "true"}, [2]string{"last", "true"}, [2]string{"end", "true"}))
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
	testcheck.NoError(t, err)
	if got := left(commit); got != count-first {
		t.Fatalf("after a landing that added %d of %d nodes the plan gives the row %d, want %d", first, count, got, count-first)
	}

	fixture.write("internal/p/p.go", twoFunctions)
	l = fixture.candidateOf("row/later")
	if err := l.rules(ctx); err != nil {
		t.Fatalf("the second step's growth of %d with %d left was refused: %v, want it recorded", second, count-first, err)
	}
	if commit, err = l.commit(ctx); err != nil {
		t.Fatal(err)
	}
	if got := left(commit); got != count-first-second {
		t.Fatalf("after a landing that grew past the count the plan gives the row %d, want %d", got, count-first-second)
	}

	// A landing that removes code returns what it removed to the count.
	fixture.write("internal/p/p.go", emptyPackage)
	l = fixture.candidateOf("row/last")
	testcheck.NoError(t, l.rules(ctx))
	if commit, err = l.commit(ctx); err != nil {
		t.Fatal(err)
	}
	if got := left(commit); got != count {
		t.Fatalf("after a landing that removed the %d nodes the two added the plan gives the row %d, want %d", first+second, got, count)
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

// TestAFilingBesideTheFindingsRegisterIsAFiling holds a change to the plan and to the findings register a review left
// uncommitted to a filing: it lands a step the plan's edit adds, by the check that edit gives it, and its commit
// carries the register. The same edit with code beside it answers to a check that must fail without the change, as any added step does.
func TestAFilingBesideTheFindingsRegisterIsAFiling(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	fixture := newRowFixture(t, map[string]string{"internal/p/p.go": emptyPackage, findingsRegister: "{}\n"})
	fixture.commitPlan(rowPlan(0, [2]string{"do", "false"}))
	testcheck.NoError(t, plan.Save(fixture.planPath(), rowPlan(0, [2]string{"do", "false"}, [2]string{"file", "true"})))
	fixture.write(findingsRegister, "{\"findings\": [\"SQA-1\"]}\n")
	fixture.write("internal/p/p.go", oneFunction)
	carrying := fixture.candidateOf("row/file")
	carrying.Log = io.Discard
	if err := carrying.verify(ctx); err == nil || !strings.Contains(err.Error(), "proves nothing") {
		t.Errorf("a step that carries code, with a check that passes without it: %v", err)
	}

	fixture.write("internal/p/p.go", emptyPackage)
	filing := fixture.candidateOf("row/file")
	if !slices.Equal(filing.verifiers, []string{"true"}) {
		t.Fatalf("a filing beside the findings register is checked by %q, want the check its plan gives the step", filing.verifiers)
	}
	if _, err := filing.commit(ctx); err != nil {
		t.Fatal(err)
	}
	if held := fixture.git("show", "HEAD:"+findingsRegister); !strings.Contains(held, "SQA-1") {
		t.Errorf("the filing's commit holds the register as %q, want the review's findings", held)
	}
}
