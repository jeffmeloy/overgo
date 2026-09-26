package codeprofile

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/gitauthority"
)

// TestPathSurfaceCatalogClassifiesEveryPath holds the one path catalog to its
// classes over every tracked path -- a package's Go file is Go source and a
// build input, a command's is automation, the plan is a document tests do not
// read -- to named examples, and to having no prefix that matches nothing.
func TestPathSurfaceCatalogClassifiesEveryPath(t *testing.T) {
	t.Parallel()
	listed, err := gitauthority.Query(t.Context(), "../..", "ls-files")
	if err != nil {
		t.Fatal(err)
	}
	paths := strings.Fields(string(listed))
	for _, path := range paths {
		surface := Classify(path)
		goFile := strings.HasSuffix(path, ".go")
		packaged := strings.HasPrefix(path, "cmd/") || strings.HasPrefix(path, "internal/")
		if (surface&GoSource != 0) != (goFile && packaged) || surface&GoSource != 0 && surface&GoInput == 0 ||
			(surface&Automation != 0) != (goFile && strings.HasPrefix(path, "cmd/")) ||
			strings.HasPrefix(path, "docs/") && surface&Document == 0 || surface&Plan != 0 && surface&TestData != 0 {
			t.Errorf("%s classified %b", path, surface)
		}
	}
	for _, rule := range surfacePrefixes {
		for _, prefix := range rule.prefixes {
			if !slices.ContainsFunc(paths, func(path string) bool { return strings.HasPrefix(path, prefix) }) {
				t.Errorf("prefix %q of class %b matches no tracked path", prefix, rule.surface)
			}
		}
	}
	for path, want := range map[string]Surface{
		"README.md":                    Document | TestData,
		"docs/plan.json":               Document | Plan,
		"docs/structure_budgets.json":  Document | TestData | Budget,
		"internal/gate/gate.go":        GoSource | GoInput,
		"internal/gate/README.md":      Document,
		"kernels/manifest.json":        GoInput,
		"scripts/guard.sh":             Harness,
		"cmd/server/main.go":           GoSource | GoInput | Automation | ModelJourney,
		"internal/server/webui/app.js": GoInput | WebUI,
	} {
		if got := Classify(path); got != want {
			t.Errorf("Classify(%q) = %b, want %b", path, got, want)
		}
	}
	if !Holds([]string{"docs/x.md", `internal\server\webui\app.js`}, WebUI) || Holds([]string{"docs/x.md"}, GoInput) {
		t.Error("Holds misreads a path list")
	}
}
