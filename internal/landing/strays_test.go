package landing

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestALandingRefusesAStrayDocument holds the rules stage to the checkout's ignored documents: a draft JSON and a
// hand-off Markdown kept out of git under docs are strays, named in the refusal, which the stage's rules know by its
// phrase; a probe's output under tmp/<row>/, the system's state, an ignored file of another kind and the machine files
// the checkout keeps by name are not, and a document the change adds, not ignored, is the change's own.
func TestALandingRefusesAStrayDocument(t *testing.T) {
	t.Parallel()
	fixture := newSliceFixture(t, map[string]string{
		".gitignore": "docs/draft.json\ndocs/hand-off.md\ntmp/\n.overgo-runtime/\nlocal-models.json\n*.log\n",
	})
	for name, content := range map[string]string{
		"docs/draft.json": "{}", "docs/hand-off.md": "# do this\n", "tmp/row/probe.json": "{}",
		".overgo-runtime/stop.json": "{}", "local-models.json": "{}", "run.log": "log", "docs/added.md": "# landed\n",
	} {
		fixture.write(name, content)
	}
	strays := testcheck.Must(strayDocuments(t.Context(), fixture.repo))(t)
	if want := []string{"docs/draft.json", "docs/hand-off.md"}; !slices.Equal(strays, want) {
		t.Fatalf("stray documents = %v, want %v", strays, want)
	}
	refusal := strayRefusal(strays)
	known := slices.ContainsFunc(stageRules, func(rule struct{ name, phrase string }) bool {
		return rule.name == "stray-documents" && strings.Contains(refusal, rule.phrase)
	})
	if !known || !strings.Contains(refusal, "docs/draft.json, docs/hand-off.md") {
		t.Fatalf("the refusal %q names its strays under a rule the stage knows: %v", refusal, known)
	}
	clean := newSliceFixture(t, map[string]string{".gitignore": "tmp/\n"})
	clean.write("tmp/row/probe.json", "{}")
	if strays := testcheck.Must(strayDocuments(t.Context(), clean.repo))(t); len(strays) != 0 {
		t.Fatalf("a probe under tmp/<row>/ is a stray: %v", strays)
	}
}
