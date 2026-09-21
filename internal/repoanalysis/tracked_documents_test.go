package repoanalysis

import (
	"slices"
	"strings"
	"testing"
)

// TestTrackedDocumentsAreClassified holds every document at the repository
// root and beneath docs -- tracked, or untracked and not ignored -- to one
// reviewed family, and each family's kind to what the source says: a document
// declared unread is named by no production package, a generated one is handed
// to a file writer by the tool that names it, and a baseline, a receipt or a
// fixture is read by something. A stray document fails here until a plan row
// classifies it.
func TestTrackedDocumentsAreClassified(t *testing.T) {
	t.Parallel()
	root := "../.."
	snapshot, err := DiscoverGo(root, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	documents, err := TrackedDocuments(t.Context(), root)
	if err != nil {
		t.Fatal(err)
	}
	census, err := TrackedDocumentCensus(snapshot, documents, TrackedDocumentFamilies)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateTrackedDocuments(census, TrackedDocumentFamilies); err != nil {
		t.Fatal(err)
	}
	kinds := map[DocumentKind]int{}
	for _, document := range census {
		kinds[document.Family.Kind]++
	}
	t.Logf("documents=%d by kind: %v", len(census), kinds)
}

// TestTrackedDocumentCensusFollowsTheName pins how a document's name reaches
// the census: a writer is a call named as a file writer is whose arguments
// carry the name -- as a literal, through a package-level constant declared in
// another file, through an imported package's constant, or through a local
// variable -- and every other naming is a read. A name inside a longer file
// name, a variable of another function, an import path and a classification of
// documents name nothing.
func TestTrackedDocumentCensusFollowsTheName(t *testing.T) {
	t.Parallel()
	snapshot, err := SourceSnapshot{}.Overlay(map[string][]byte{
		"cmd/report/paths.go": []byte("package main\nconst reportPath = \"docs/REPORT.md\"\n"),
		"cmd/report/main.go": []byte(`package main
import (
	"overgo/internal/atomicfile"
	"overgo/internal/clioptions"
	"overgo/internal/owner"
)
func run() error {
	if err := clioptions.OutputGenerated(nil, reportPath, false, true, "stale", nil); err != nil { return err }
	target := filepath.Join(root, owner.LedgerPath)
	return atomicfile.Write(target, nil, 0)
}
func other() { target := "docs/baseline.json"; _ = target }
func stream(w io.Writer) { w.Write([]byte("see docs/notes.md")) ; load("docs/model_report.json") }
`),
		"internal/owner/owner.go":        []byte("package owner\nconst LedgerPath = \"docs/ledger.json\"\nfunc Load() { read(LedgerPath) }\n"),
		"internal/reader/reader.go":      []byte("package reader\nimport _ \"docs/imported.json\"\nvar pattern = \"docs/receipts/*.json\"\nvar samples = \"docs/samples\"\nvar families = []DocumentFamily{{\"docs/unread.json\", \"unread\"}}\n"),
		"internal/reader/reader_test.go": []byte("package reader\nvar testOnly = \"docs/unread.json\"\n"),
	})
	if err != nil {
		t.Fatal(err)
	}
	documents := []string{
		"docs/REPORT.md", "docs/baseline.json", "docs/imported.json", "docs/ledger.json", "docs/model_report.json",
		"docs/notes.md", "docs/receipts/run.json", "docs/samples/a.png", "docs/unread.json", "report.json",
	}
	families := []DocumentFamily{{"docs/REPORT.md", DocumentGenerated}, {"docs/", DocumentUnread}, {"stale/", DocumentFixture}}
	census, err := TrackedDocumentCensus(snapshot, documents, families)
	if err != nil {
		t.Fatal(err)
	}
	var measured []string
	for _, document := range census {
		measured = append(measured, document.Path+" W="+strings.Join(document.Writers, ",")+" R="+strings.Join(document.Readers, ","))
	}
	want := []string{
		"docs/REPORT.md W=cmd/report R=",
		"docs/baseline.json W= R=cmd/report",
		"docs/imported.json W= R=",
		"docs/ledger.json W=cmd/report R=internal/owner",
		"docs/model_report.json W= R=cmd/report",
		"docs/notes.md W= R=cmd/report",
		"docs/receipts/run.json W= R=internal/reader",
		"docs/samples/a.png W= R=internal/reader",
		"docs/unread.json W= R=",
		"report.json W= R=",
	}
	if !slices.Equal(measured, want) {
		t.Fatalf("census:\n%s\nwant:\n%s", strings.Join(measured, "\n"), strings.Join(want, "\n"))
	}
	err = ValidateTrackedDocuments(census, families)
	for _, finding := range []string{
		"report.json belongs to no family",
		"docs/ledger.json is declared unread and is named by [cmd/report] [internal/owner]",
		"document family stale/ matches no tracked document",
	} {
		if err == nil || !strings.Contains(err.Error(), finding) {
			t.Errorf("missing finding %q in: %v", finding, err)
		}
	}
	if err != nil && (strings.Contains(err.Error(), "docs/REPORT.md") || strings.Contains(err.Error(), "docs/unread.json")) {
		t.Errorf("a consistent document was refused: %v", err)
	}
	unwritten, err := TrackedDocumentCensus(snapshot, []string{"docs/baseline.json"}, []DocumentFamily{{"docs/baseline.json", DocumentGenerated}})
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateTrackedDocuments(unwritten, nil); err == nil || !strings.Contains(err.Error(), "declared generated and no package that names it") {
		t.Errorf("a generated document no tool writes passed: %v", err)
	}
}
