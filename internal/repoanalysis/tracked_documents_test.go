package repoanalysis

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestTrackedDocumentsAreClassified holds every document at the repository
// root and beneath docs -- tracked, or untracked and not ignored -- to one
// reviewed family, and each family's kind to what was measured: a generated
// document is handed to a file writer by the tool that names it, and every
// document a person does not author is read by something -- a package, a test,
// or a document that is itself read. A stray document fails here until a plan
// row classifies it, and so does one that loses its last reader.
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
	err = DocumentReferences(census, func(document string) ([]byte, error) {
		return os.ReadFile(filepath.Join(root, filepath.FromSlash(document)))
	})
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
// the census from source. A writer is a call named as a file writer is -- a
// function, never a method on a stream -- whose arguments carry the name: as a
// literal, through a package-level constant declared in another file, through
// an imported package's constant, or through a local variable. Every other
// naming is a read, a test's included: a test that opens a document reads it,
// and what a test writes makes no writer. A name inside a longer file name, a
// variable of another function, an import path and a classification of
// documents name nothing, and a directory names what is beneath it only in
// production source, which may walk it -- a test joins a directory to a file
// name of its own, and the directory alone would pass off a stray as read.
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
		"internal/owner/owner.go":   []byte("package owner\nconst LedgerPath = \"docs/ledger.json\"\nfunc Load() { read(LedgerPath) }\n"),
		"internal/reader/reader.go": []byte("package reader\nimport _ \"docs/imported.json\"\nvar pattern = \"docs/receipts/*.json\"\nvar samples = \"docs/samples\"\nvar families = []DocumentFamily{{\"docs/classified.json\", \"receipt\"}}\n"),
		"internal/reader/reader_test.go": []byte(`package reader
import "os"
func TestGolden(t *testing.T) { os.WriteFile("docs/golden.json", nil, 0); open("docs/tested.json"); join("docs/receipts", row.Name) }
`),
	})
	if err != nil {
		t.Fatal(err)
	}
	documents := []string{
		"docs/REPORT.md", "docs/baseline.json", "docs/classified.json", "docs/golden.json", "docs/imported.json",
		"docs/ledger.json", "docs/model_report.json", "docs/notes.md", "docs/receipts/run.json", "docs/receipts/stray.txt", "docs/samples/a.png",
		"docs/tested.json", "report.json",
	}
	census, err := TrackedDocumentCensus(snapshot, documents, nil)
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
		"docs/classified.json W= R=",
		"docs/golden.json W= R=internal/reader",
		"docs/imported.json W= R=",
		"docs/ledger.json W=cmd/report R=internal/owner",
		"docs/model_report.json W= R=cmd/report",
		"docs/notes.md W= R=cmd/report",
		"docs/receipts/run.json W= R=internal/reader",
		"docs/receipts/stray.txt W= R=",
		"docs/samples/a.png W= R=internal/reader",
		"docs/tested.json W= R=internal/reader",
		"report.json W= R=",
	}
	if !slices.Equal(measured, want) {
		t.Fatalf("census:\n%s\nwant:\n%s", strings.Join(measured, "\n"), strings.Join(want, "\n"))
	}
}

// TestNoTrackedDocumentIsUnread holds the census to refusing a document that
// nothing reads, whatever kind it is declared, and to seeing the reads source
// syntax cannot: a specification a tool reads lists its evidence files, and
// the tool then opens each one, so a document is read when a document that is
// read names it -- by its repository path, or by its path from the naming
// document's own directory, as a report writes a link. Reading is followed to
// a fixed point and starts only from what is read: two documents that name
// only each other are both unread. A name that is the tail of a longer file
// name is no reference, a document a person authors needs no reader, and an
// unclassified document, a generated one no tool writes and a family that
// matches nothing are each refused.
func TestNoTrackedDocumentIsUnread(t *testing.T) {
	t.Parallel()
	texts := map[string]string{
		"docs/spec.json":             `{"evidence_files": ["docs/runs/evidence.txt"]}`,
		"docs/runs/evidence.txt":     "continued in nested.txt; unrelated: other_orphan.txt",
		"docs/runs/nested.txt":       "end of the chain",
		"docs/REPORT.md":             "see the [protocol](protocol.json)",
		"docs/protocol.json":         "{}",
		"docs/runs/orphan.txt":       "see docs/runs/twin.txt",
		"docs/runs/twin.txt":         "see docs/runs/orphan.txt",
		"docs/GUIDE.md":              "a person reads this",
		"docs/unwritten_report.json": "{}",
		"stray.json":                 "{}",
	}
	families := []DocumentFamily{
		{"docs/spec.json", DocumentFixture}, {"docs/REPORT.md", DocumentGenerated}, {"docs/GUIDE.md", DocumentAuthored},
		{"docs/unwritten_report.json", DocumentGenerated}, {"docs/", DocumentReceipt}, {"stale/", DocumentFixture},
	}
	census := make([]TrackedDocument, 0, len(texts))
	for document := range texts {
		tracked := TrackedDocument{Path: document}
		for _, family := range families {
			if family.matches(document) {
				tracked.Family = family
				break
			}
		}
		census = append(census, tracked)
	}
	slices.SortFunc(census, func(left, right TrackedDocument) int { return strings.Compare(left.Path, right.Path) })
	for index := range census {
		switch census[index].Path {
		case "docs/spec.json", "docs/unwritten_report.json":
			census[index].Readers = []string{"cmd/tool"}
		case "docs/REPORT.md":
			census[index].Writers = []string{"cmd/tool"}
		}
	}
	err := DocumentReferences(census, func(document string) ([]byte, error) { return []byte(texts[document]), nil })
	if err != nil {
		t.Fatal(err)
	}
	var refused []string
	for _, finding := range unwrapFindings(ValidateTrackedDocuments(census, families)) {
		refused = append(refused, finding.Error())
	}
	slices.Sort(refused)
	want := []string{
		"document family stale/ matches no tracked document",
		"tracked document docs/runs/orphan.txt is declared receipt and nothing reads it",
		"tracked document docs/runs/twin.txt is declared receipt and nothing reads it",
		"tracked document docs/unwritten_report.json is declared generated and no package that names it writes a file",
		"tracked document stray.json belongs to no family",
	}
	if len(refused) != len(want) {
		t.Fatalf("findings:\n%s\nwant:\n%s", strings.Join(refused, "\n"), strings.Join(want, "\n"))
	}
	for index, prefix := range want {
		if !strings.HasPrefix(refused[index], prefix) {
			t.Errorf("finding %d = %q, want it to start %q", index, refused[index], prefix)
		}
	}
	if err := DocumentReferences(census, func(string) ([]byte, error) { return nil, os.ErrNotExist }); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("a document that cannot be read passed: %v", err)
	}
}

// unwrapFindings returns the findings one joined error carries.
func unwrapFindings(err error) []error {
	joined, isJoined := err.(interface{ Unwrap() []error })
	if !isJoined {
		return nil
	}
	return joined.Unwrap()
}
