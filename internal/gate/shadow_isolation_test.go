package gate

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// TestShadowIsolationIsMeasuredNotEnforced holds the isolation rule under
// trial to being a measurement. A reader parses Go from a path it is handed
// and never finds the repository itself. One importer's tests hand it a file
// under a temporary directory; another's name a parent path. A change to an
// unrelated source still selects both importers, as it did before the rule
// existed, and the measure names only the confined one as what the rule would
// leave out.
func TestShadowIsolationIsMeasuredNotEnforced(t *testing.T) {
	t.Parallel()
	g := runtimeReaderFixture(t)
	for name, source := range map[string]string{
		"internal/reader/reader.go": `package reader
import ("go/parser";"go/token";"os")
func Check(path string) error { data, err := os.ReadFile(path); if err != nil { return err }; _, err = parser.ParseFile(token.NewFileSet(), path, data, 0); return err }
`,
		"internal/readerclient/client.go": `package readerclient
import "overgo/internal/reader"
func Check(path string) error { return reader.Check(path) }
`,
		"internal/readerclient/client_test.go": `package readerclient
import ("os";"path/filepath";"testing")
func TestCheck(t *testing.T) { path := filepath.Join(t.TempDir(), "a.go"); if err := os.WriteFile(path, []byte("package a\n"), 0o644); err != nil { t.Fatal(err) }; if err := Check(path); err != nil { t.Fatal(err) } }
`,
		"internal/reaching/reaching.go": `package reaching
import "overgo/internal/reader"
func Check(path string) error { return reader.Check(path) }
`,
		"internal/reaching/reaching_test.go": `package reaching
import ("path/filepath";"testing")
func TestCheck(t *testing.T) { if err := Check(filepath.Join("..", "unrelated", "unrelated.go")); err != nil { t.Fatal(err) } }
`,
	} {
		path := filepath.Join(g.repo, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(source), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	g.paths = []string{"internal/unrelated/unrelated.go"}
	scope, err := g.deriveTestScope()
	if err != nil {
		t.Fatal(err)
	}
	for _, importer := range []string{"overgo/internal/readerclient", "overgo/internal/reaching"} {
		if !slices.Contains(scope.selected(), importer) {
			t.Fatalf("the measure changed selection: %s is no longer selected; uncertain=%v", importer, scope.uncertain)
		}
	}
	if want := []string{"overgo/internal/readerclient"}; !slices.Equal(scope.shadowIsolated, want) {
		t.Fatalf("shadow isolated = %v, want %v: the confined importer alone", scope.shadowIsolated, want)
	}
}
