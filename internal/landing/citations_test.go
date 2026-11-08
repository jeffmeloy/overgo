package landing

import (
	"bytes"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestARenamedTestsCitationsFollowIt holds the fix stage to a renamed test's citations: on 2026-11-08
// landing-scope/scoped-commit was refused because skill.md cited a test its own change had renamed. A change renames
// TestOld to TestNew in one test file and, in another, removes TestFirst and TestSecond beside adding TestThird; the
// commit and the checkout hold skill.md with TestOld's citation rewritten, though no step lists skill.md, and the
// landing prints the lines citing TestFirst and TestSecond, which no rename names, with TestThird beside them.
func TestARenamedTestsCitationsFollowIt(t *testing.T) {
	t.Parallel()
	test := func(names ...string) string {
		var source strings.Builder
		source.WriteString("package p\n\nimport \"testing\"\n")
		for _, name := range names {
			source.WriteString("\nfunc " + name + "(t *testing.T) {}\n")
		}
		return source.String()
	}
	fixture := newRowFixture(t, map[string]string{
		"skill.md":             "- a rule (check: internal/p TestOld)\n- two rules (check: internal/q TestFirst, internal/q TestSecond)\n",
		"internal/p/p.go":      "package p\n",
		"internal/p/p_test.go": test("TestOld"),
		"internal/q/p.go":      "package p\n",
		"internal/q/p_test.go": test("TestFirst", "TestSecond"),
	})
	editPlan(t, fixture, func(document *plan.Plan) { document.Items[0].Steps[0].Files = []string{"internal/p", "internal/q"} })
	fixture.git("add", "-A")
	fixture.git("commit", "-q", "-m", "a step that lists its files")
	fixture.write("internal/p/p_test.go", test("TestNew"))
	fixture.write("internal/q/p_test.go", test("TestThird"))
	var said bytes.Buffer
	l := &landing{Options: Options{Repo: fixture.repo, Ref: "row/do", Log: &said}, scratch: t.TempDir(), details: map[string]string{}}
	t.Cleanup(func() {
		if l.treeHeld != nil {
			testcheck.NoError(t, l.treeHeld())
		}
	})
	testcheck.NoError(t, l.parents(t.Context()))
	testcheck.NoError(t, l.shipSet(t.Context()))
	testcheck.NoError(t, l.fix(t.Context()))
	want := "- a rule (check: internal/p TestNew)\n- two rules (check: internal/q TestFirst, internal/q TestSecond)"
	if committed := fixture.git("show", l.candidate+":skill.md"); committed != want || strings.TrimSpace(fixture.read("skill.md")) != want {
		t.Fatalf("the commit holds skill.md\n%s\nand the checkout\n%s\nwant\n%s", committed, fixture.read("skill.md"), want)
	}
	for _, gone := range []string{"TestFirst", "TestSecond"} {
		if !strings.Contains(said.String(), "skill.md cites "+gone+", which the change removed beside adding TestThird") {
			t.Fatalf("the landing said\n%s\nwant the line citing %s printed with TestThird", said.String(), gone)
		}
	}
}
