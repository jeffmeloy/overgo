package landing

import (
	"testing"

	"overgo/internal/plan"
)

// TestAStepThatListsAGeneratedFileTakesItsEdit holds a generated file a step lists to the step's edit
// (cycle-lanes/claim-edits): compatibility.json declares each claim's check and summary, which the claims generator
// rebinds, and a generated file taken only as derived from the candidate (derive-always/do) shut every such edit out,
// so a claim's check could not be given the switch its test needs. The step that lists the manifest is taken with the
// checkout's edit, where the candidate had held the base's.
func TestAStepThatListsAGeneratedFileTakesItsEdit(t *testing.T) {
	t.Parallel()
	const generator = "package main\n\nimport \"os\"\n\nfunc main() { os.WriteFile(\"docs/api_manifest.json\", []byte(\"regenerated\\n\"), 0o644) }\n"
	listed := newRowFixture(t, map[string]string{"cmd/api-manifest/main.go": generator, "docs/api_manifest.json": "base\n"})
	editPlan(t, listed, func(document *plan.Plan) { document.Items[0].Steps[0].Files = []string{"docs/api_manifest.json"} })
	listed.git("add", "-A")
	listed.git("commit", "-q", "-m", "a step that lists the manifest")
	listed.write("docs/api_manifest.json", "declared\n")
	l := listed.candidate()
	if taken := listed.git("cat-file", "blob", l.candidate+":docs/api_manifest.json"); taken != "declared" {
		t.Fatalf("a step that lists the manifest is taken with %q; want the checkout's edit", taken)
	}
}
