package landing

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/protection"
	"overgo/internal/testcheck"
)

// TestALandingNamesAStrayDocument holds the rules stage to the checkout's ignored documents: a draft JSON and a
// hand-off Markdown kept out of git under docs are strays, which the stage names, outside the commit; a probe's output
// under tmp/<row>/, the system's state, the harness's own runtime files (SQA-334: its scheduled tasks file once refused
// every landing), an ignored file of another kind and the machine files the checkout keeps by name are not, and a
// document the change adds, not ignored, is the change's own.
func TestALandingNamesAStrayDocument(t *testing.T) {
	t.Parallel()
	harnessFile := protection.HarnessConfigDirectory + "scheduled_tasks.json"
	fixture := newSliceFixture(t, map[string]string{
		".gitignore": "docs/draft.json\ndocs/hand-off.md\ntmp/\n.overgo-runtime/\nlocal-models.json\n*.log\n**/" + harnessFile + "\n",
	})
	for name, content := range map[string]string{
		"docs/draft.json": "{}", "docs/hand-off.md": "# do this\n", "tmp/row/probe.json": "{}", harnessFile: "{}",
		".overgo-runtime/stop.json": "{}", "local-models.json": "{}", "run.log": "log", "docs/added.md": "# landed\n",
	} {
		fixture.write(name, content)
	}
	strays := testcheck.Must(strayDocuments(t.Context(), fixture.repo))(t)
	if want := []string{"docs/draft.json", "docs/hand-off.md"}; !slices.Equal(strays, want) {
		t.Fatalf("stray documents = %v, want %v", strays, want)
	}
	if note := strayNote(strays, "row"); !strings.Contains(note, "docs/draft.json, docs/hand-off.md") || !strings.Contains(note, "outside the commit") {
		t.Fatalf("the note %q does not name its strays as outside the commit", note)
	}
}
