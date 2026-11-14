package landing

import (
	"slices"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestADerivedFileEntersACommitOnlyFromDerive holds a generator's file to the copy derive writes from the candidate,
// whatever the change holds (derive-always/do): a checkout whose API manifest was regenerated for other, uncommitted
// code lands a change to the plan alone as one, the manifest left as the checkout holds it and the candidate holding
// the base's; and a change to a file that is no Go has its manifest derived in the fix stage, where only a change to
// Go was derived. On 2026-11-14 such a manifest entered a change to the plan alone, which then ran every stage as a
// change to code, derived nothing, and was refused twice by the manifest's own test.
func TestADerivedFileEntersACommitOnlyFromDerive(t *testing.T) {
	t.Parallel()
	const generator = "package main\n\nimport \"os\"\n\nfunc main() { os.WriteFile(\"docs/api_manifest.json\", []byte(\"regenerated\\n\"), 0o644) }\n"
	stub := map[string]string{"cmd/api-manifest/main.go": generator, "docs/api_manifest.json": "base\n", "notes.txt": "base\n", "internal/p/p.go": "package p\n"}

	alone := newRowFixture(t, stub)
	alone.write("docs/api_manifest.json", "stale\n")
	editPlan(t, alone, func(document *plan.Plan) { document.Items[0].Title = "row, retitled" })
	l := alone.candidate()
	if !l.planOnly() || slices.Contains(l.paths, "docs/api_manifest.json") || alone.git("cat-file", "blob", l.candidate+":docs/api_manifest.json") != "base" {
		t.Fatalf("a change to the plan alone beside a stale manifest takes %q (plan-only %t); want the plan alone, the candidate's manifest the base's", l.paths, l.planOnly())
	}
	if held := alone.read("docs/api_manifest.json"); held != "stale\n" {
		t.Fatalf("the checkout's manifest is %q after the change to the plan was taken; want it left as it stood", held)
	}

	notes := newRowFixture(t, stub)
	notes.write("docs/api_manifest.json", "stale\n")
	notes.write("notes.txt", "changed\n")
	l = notes.candidate()
	testcheck.NoError(t, l.fix(t.Context()))
	if manifest := notes.git("cat-file", "blob", l.candidate+":docs/api_manifest.json"); manifest != "regenerated" {
		t.Fatalf("a change to no Go beside a stale manifest is taken with manifest %q; want the one derive writes from the candidate", manifest)
	}
}
