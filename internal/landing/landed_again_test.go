package landing

import (
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestEveryStageFailureLeavesTheTrunkAlone fails each stage a change can break, format, vet, rules, test and verify,
// and holds the trunk and the checkout's plan where they were: a stage's refusal returns before the commit begins
// (Run). The commit's own failure is the kill between the trunk's move and the plan's save, which leaves the trunk
// moved and the checkout's plan as it was, still holding the step that landed. The next landing would publish that
// plan, filing the landed step again with every stage passing; it is refused, naming the step and its landing, and
// the trunk does not move. With the plan the trunk holds written back it lands, and the landed step stays gone.
func TestEveryStageFailureLeavesTheTrunkAlone(t *testing.T) {
	t.Parallel()
	const source = "// Package p is a fixture.\npackage p\n"
	fixture := newRowFixture(t, map[string]string{"a.txt": "base\n", "b.txt": "base\n", "p/p.go": source})
	if err := plan.Save(fixture.planPath(), plan.Plan{Campaign: "fixture", Items: []plan.Item{
		{ID: "row", Title: "row", Status: plan.StatusOpen, Steps: []plan.Step{{ID: "do", Title: "row", Status: plan.StatusOpen, Verify: "true"}}},
		{ID: "other", Title: "other", Status: plan.StatusOpen, Steps: []plan.Step{{ID: "do", Title: "other", Status: plan.StatusOpen, Verify: "true"}}},
		{ID: "failing", Title: "failing", Status: plan.StatusOpen, Steps: []plan.Step{{ID: "do", Title: "failing", Status: plan.StatusOpen, Verify: "false"}}},
	}}); err != nil {
		t.Fatal(err)
	}
	fixture.git("add", "-A")
	fixture.git("commit", "-q", "-m", "plan of three rows")
	land := func(reference string) (string, error) {
		t.Helper()
		var log strings.Builder
		return Run(t.Context(), Options{Repo: fixture.repo, Ref: reference, Log: &log,
			Message: []byte(reference + ": change\n\nCause: a fixture change.\nPredicted effect: first-output holds.\n")})
	}
	unmoved := func(name, head, planned string) {
		t.Helper()
		if now := fixture.git("rev-parse", "HEAD"); now != head || fixture.read(plan.Path) != planned {
			t.Fatalf("%s: the refused landing moved the trunk to %s from %s, or changed the plan", name, now, head)
		}
	}

	for _, broken := range []struct {
		stage, reference string
		files            map[string]string
	}{
		{"format", "row/do", map[string]string{"p/p.go": source + "\nfunc Unparsed( {\n"}},
		{"vet", "row/do", map[string]string{"p/p.go": source + "\nimport \"fmt\"\n\n// Printed prints.\nfunc Printed() { fmt.Printf(\"%d\", \"text\") }\n"}},
		{"rules", "row/do", map[string]string{"p/p.go": source + "\n// Seven is seven.\nfunc Seven() int { return 7 }\n"}},
		{"test", "row/do", map[string]string{"p/p_test.go": "package p\n\nimport \"testing\"\n\nfunc TestFails(t *testing.T) { t.Fatal(\"broken\") }\n"}},
		{"verify", "failing/do", map[string]string{"a.txt": "verified\n"}},
	} {
		head, planned := fixture.git("rev-parse", "HEAD"), fixture.read(plan.Path)
		for name, content := range broken.files {
			fixture.write(name, content)
		}
		if _, err := land(broken.reference); err == nil || !strings.Contains(err.Error(), stagePrefix+broken.stage+":") {
			t.Fatalf("a change broken at %s ended as %v, want that stage's refusal", broken.stage, err)
		}
		unmoved(broken.stage, head, planned)
		fixture.git("checkout", "-q", "--", "p/p.go", "a.txt")
		fixture.git("clean", "-q", "-f", "--", "p")
	}

	before := fixture.read(plan.Path)
	fixture.write("a.txt", "changed\n")
	first, err := land("row/do")
	testcheck.NoError(t, err)
	landed := fixture.read(plan.Path)
	// What a kill between the trunk's move and the plan's save leaves.
	fixture.write(plan.Path, before)
	fixture.write("b.txt", "changed\n")
	if _, err := land("other/do"); err == nil || !strings.Contains(err.Error(), "row/do") || !strings.Contains(err.Error(), first[:12]) {
		t.Fatalf("a landing whose plan holds row/do again, which landed as %.12s, ended as %v, want it refused naming both", first, err)
	}
	unmoved("the plan from before a landing", first, before)
	fixture.write(plan.Path, landed)
	if _, err := land("other/do"); err != nil {
		t.Fatalf("the landing with the trunk's plan written back: %v", err)
	}
	published, err := plan.Parse([]byte(fixture.git("show", "HEAD:"+plan.Path)))
	testcheck.NoError(t, err)
	for _, item := range published.Items {
		if item.ID == "row" {
			t.Fatalf("the landed step row/do is in the trunk's plan again: %+v", item)
		}
	}
}
