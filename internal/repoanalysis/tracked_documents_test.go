package repoanalysis

import (
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"

	"overgo/internal/gosyntax"
)

// TestEveryTrackedFileHasAReader holds every document in the checkout --
// tracked, or untracked and not ignored, wherever it is -- to a reader: a
// package or test that names or embeds it, the tool that writes it, a read
// document that names it, or being an entry document or license notice a
// person opens. A stray document fails here, and so does one that loses its
// last reader.
func TestEveryTrackedFileHasAReader(t *testing.T) {
	t.Parallel()
	census, err := CheckoutDocuments(t.Context(), "../..")
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateTrackedDocuments(census); err != nil {
		t.Fatal(err)
	}
	t.Logf("documents=%d all read", len(census))
}

// TestEveryFixtureHasAReader holds every file beneath the fixtures folder,
// whatever its type, to a reader, as a document is held: a golden tensor that
// lost its test is megabytes nobody opens, and nothing said so while only the JSON
// beside it was measured. In a census of invented files, a tensor a test names
// by its file name is read, and so is one that only a read manifest names; one
// that nothing names is refused; and so is one that only a directory covers,
// whether a test joins that directory to a name from its table or a package
// writes the folder's own bare name, as a list of the repository's roots does.
// A file that is no text is never opened to look for the names it carries.
// Then the real checkout: the census covers every fixture git lists there,
// and each has a reader.
func TestEveryFixtureHasAReader(t *testing.T) {
	t.Parallel()
	const fixtures = "fixtures/"
	snapshot, err := gosyntax.Snapshot{}.Overlay(map[string][]byte{
		"internal/golden/golden_test.go": []byte(`package golden
func TestGolden(t *testing.T) { load("latent.f32le"); open("fixtures/model/manifest.json"); join("fixtures/joined", row.Name) }
`),
		"internal/prose/prose.go": []byte("package prose\nvar roots = []string{\"cmd/\", \"fixtures/\"}\n"),
	})
	if err != nil {
		t.Fatal(err)
	}
	texts := map[string]string{"fixtures/model/manifest.json": `{"weights": "weights.bin"}`}
	census, err := TrackedDocumentCensus(snapshot, []string{
		"fixtures/joined/tensor.bin", "fixtures/model/latent.f32le", "fixtures/model/manifest.json", "fixtures/model/weights.bin", "fixtures/stray.sum",
	})
	if err == nil {
		err = DocumentReferences(census, func(document string) ([]byte, error) {
			text, isText := texts[document]
			if !isText {
				return nil, fmt.Errorf("%s is no text, and was opened to look for the names it carries", document)
			}
			return []byte(text), nil
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	var refused []string
	for _, finding := range unwrapFindings(ValidateTrackedDocuments(census)) {
		refused = append(refused, strings.Fields(finding.Error())[2])
	}
	if want := []string{"fixtures/joined/tensor.bin", "fixtures/stray.sum"}; !slices.Equal(refused, want) {
		t.Errorf("the census refuses %q, want %q: the fixture nothing names and the one only a directory covers, and neither the one a test names nor the one a read manifest names", refused, want)
	}

	listed, err := checkoutFiles(t.Context(), "../..", []string{fixtures})
	if err != nil {
		t.Fatal(err)
	}
	checkout, err := CheckoutDocuments(t.Context(), "../..")
	if err != nil {
		t.Fatal(err)
	}
	live, covered := LiveDocuments(checkout), 0
	for _, document := range checkout {
		if !strings.HasPrefix(document.Path, fixtures) {
			continue
		}
		covered++
		if !live[document.Path] {
			t.Errorf("the fixture %s has no reader: no package, test, embed or read document names it; give it a reader or delete it", document.Path)
		}
	}
	if covered != len(listed) || covered == 0 {
		t.Fatalf("the census covers %d of the %d files the checkout holds beneath %s, want every one of them", covered, len(listed), fixtures)
	}
	if !t.Failed() {
		t.Logf("fixtures=%d all read", covered)
	}
}

// TestTrackedDocumentCensusFollowsTheName pins how a document's name reaches
// the census from source. A writer is a call named as a file writer is -- a
// function, never a method on a stream -- whose arguments carry the name: as a
// literal, through a package-level constant declared in another file, through
// an imported package's constant, or through a local variable. Every other
// naming is a read, a test's included: a test that opens a document reads it,
// and what a test writes makes no writer. A go:embed pattern reads what it
// matches, and a test's bare word reads the document whose file name it is
// without its extension. A name inside a longer file name, a variable of
// another function and an import path name nothing, and a directory names
// what is beneath it only in production source, which may walk it -- a test
// joins a directory to a file name of its own, and the directory alone would
// pass off a stray as read.
func TestTrackedDocumentCensusFollowsTheName(t *testing.T) {
	t.Parallel()
	snapshot, err := gosyntax.Snapshot{}.Overlay(map[string][]byte{
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
		"internal/reader/reader.go": []byte("package reader\nimport _ \"docs/imported.json\"\nvar pattern = \"docs/receipts/*.json\"\nvar samples = \"docs/samples\"\n"),
		"internal/reader/reader_test.go": []byte(`package reader
import "os"
func TestGolden(t *testing.T) { os.WriteFile("docs/golden.json", nil, 0); open("docs/tested.json"); join("docs/receipts", row.Name) }
`),
		"internal/policy/policy.go":            []byte("package policy\nimport _ \"embed\"\n//go:embed rules.json\nvar rules []byte\n"),
		"internal/fixtures/fixtures_test.go":   []byte("package fixtures\nfunc TestBlocks(t *testing.T) { for _, name := range []string{\"resblock\", \"rope\"} { load(name + \".json\") } }\n"),
		"internal/fixtures/stray_name_test.go": []byte("package fixtures\nvar _ = \"a sentence mentioning unused\"\n"),
		"internal/production/production.go":    []byte("package production\nvar word = \"orphan\"\n"),
	})
	if err != nil {
		t.Fatal(err)
	}
	documents := []string{
		"docs/REPORT.md", "docs/baseline.json", "docs/golden.json", "docs/imported.json",
		"docs/ledger.json", "docs/model_report.json", "docs/notes.md", "docs/receipts/run.json", "docs/receipts/stray.txt", "docs/samples/a.png",
		"docs/tested.json", "fixtures/blocks/resblock.json", "fixtures/blocks/unused.json", "fixtures/orphan.json",
		"internal/policy/rules.json", "report.json",
	}
	census, err := TrackedDocumentCensus(snapshot, documents)
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
		"docs/golden.json W= R=internal/reader",
		"docs/imported.json W= R=",
		"docs/ledger.json W=cmd/report R=internal/owner",
		"docs/model_report.json W= R=cmd/report",
		"docs/notes.md W= R=cmd/report",
		"docs/receipts/run.json W= R=internal/reader",
		"docs/receipts/stray.txt W= R=",
		"docs/samples/a.png W= R=internal/reader",
		"docs/tested.json W= R=internal/reader",
		"fixtures/blocks/resblock.json W= R=internal/fixtures",
		"fixtures/blocks/unused.json W= R=",
		"fixtures/orphan.json W= R=",
		"internal/policy/rules.json W= R=internal/policy",
		"report.json W= R=",
	}
	if !slices.Equal(measured, want) {
		t.Fatalf("census:\n%s\nwant:\n%s", strings.Join(measured, "\n"), strings.Join(want, "\n"))
	}
}

// TestNoTrackedDocumentIsUnread holds the census to refusing a document that
// nothing reads and to seeing the reads source syntax cannot: a specification
// a tool reads lists its evidence files, and the tool then opens each one, so a
// document is read when a document that is read names it -- by its repository
// path, or by its path from the naming document's own directory, as a report
// writes a link. Reading is followed to a fixed point and starts only from
// what is read: two documents that name only each other are both unread. The
// entry documents and the license notices need no reader; a work record such
// as the plan names what a row will delete and keeps nothing; a name that is
// the tail of a longer file name is no reference.
func TestNoTrackedDocumentIsUnread(t *testing.T) {
	t.Parallel()
	texts := map[string]string{
		"README.md":              "see the [guide](docs/GUIDE.md)",
		"licenses/tool.txt":      "license terms",
		"docs/plan.json":         `{"rationale": "delete docs/stale.md"}`,
		"docs/stale.md":          "old notes",
		"docs/spec.json":         `{"evidence_files": ["docs/runs/evidence.txt"]}`,
		"docs/runs/evidence.txt": "continued in nested.txt; unrelated: other_orphan.txt",
		"docs/runs/nested.txt":   "end of the chain",
		"docs/REPORT.md":         "see the [protocol](protocol.json)",
		"docs/protocol.json":     "{}",
		"docs/runs/orphan.txt":   "see docs/runs/twin.txt",
		"docs/runs/twin.txt":     "see docs/runs/orphan.txt",
		"docs/GUIDE.md":          "a person reads this",
		"stray.json":             "{}",
	}
	census := make([]TrackedDocument, 0, len(texts))
	for document := range texts {
		census = append(census, TrackedDocument{Path: document})
	}
	slices.SortFunc(census, func(left, right TrackedDocument) int { return strings.Compare(left.Path, right.Path) })
	for index := range census {
		switch census[index].Path {
		case "docs/spec.json", "docs/plan.json":
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
	for _, finding := range unwrapFindings(ValidateTrackedDocuments(census)) {
		refused = append(refused, finding.Error())
	}
	slices.Sort(refused)
	want := []string{
		"tracked document docs/runs/orphan.txt has no reader",
		"tracked document docs/runs/twin.txt has no reader",
		"tracked document docs/stale.md has no reader",
		"tracked document stray.json has no reader",
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
