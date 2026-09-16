package gate

import (
	"strings"
	"testing"
)

// TestGeneratedDocumentsSelectTheirNamers holds the test scope of a
// regenerated authority document to the packages that name it: on the live
// graph each such document, the repository-root ones included, selects a
// handful of direct packages, never the readers whose reach the source does
// not name. A regenerated document rides nearly every commit, so this bound
// is the short group's floor.
func TestGeneratedDocumentsSelectTheirNamers(t *testing.T) {
	t.Parallel()
	// Measured 2026-09-16: docs/ documents select 5 to 7 direct packages;
	// the root documents selected 160 before bare root names were documents.
	const ceiling = 12
	liveRepositoryFixture(t).use(t, func(g *gateContext) {
		for _, document := range generatedAuthorityPaths {
			g.paths = []string{document}
			scope, err := g.deriveTestScope()
			if err != nil {
				t.Fatal(err)
			}
			if len(scope.direct) > ceiling {
				t.Errorf("%s selects %d direct packages, want at most %d: %s", document, len(scope.direct), ceiling, strings.Join(scope.direct, ","))
			}
			t.Logf("%s: direct=%d uncertain=%d", document, len(scope.direct), len(scope.uncertain))
		}
	})
}
