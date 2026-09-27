package repoanalysis

import (
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/gosyntax"
)

func TestRepositoryGoStyleNamesAndComments(t *testing.T) {
	t.Parallel()
	requireNoRepositoryStyleFindings(t, map[goStyleKind]bool{
		goStyleReceiverConsistency: true, goStyleInitialism: true,
		goStyleGetter: true, goStyleImportAlias: true,
	})
}

func TestRepositoryGoStyleFlowAndErrors(t *testing.T) {
	t.Parallel()
	requireNoRepositoryStyleFindings(t, map[goStyleKind]bool{
		goStyleNamedResult: true, goStyleNakedReturn: true,
	})
}

func TestRepositoryGoStyleComplete(t *testing.T) {
	t.Parallel()
	report := repositoryGoStyleReport(t)
	justified := map[goStyleKind]string{
		goStyleFormat:     "untouched CRLF files normalize when edited",
		goStylePackageDoc: "package documentation follows ownership migration",
		goStyleExportDoc:  "export documentation follows consumer-led surface reduction",
		goStyleErrorText:  "capitalized model and protocol names remain proper nouns",
	}
	var findings []string
	for _, finding := range report.Findings {
		if justified[finding.Kind] == "" {
			findings = append(findings, fmt.Sprintf("%s:%d:%s:%s", finding.File, finding.Line, finding.Kind, finding.Symbol))
		}
	}
	if len(findings) != 0 {
		t.Fatalf("%d unclassified style findings:\n%s", len(findings), strings.Join(findings, "\n"))
	}
}

func requireNoRepositoryStyleFindings(t *testing.T, selected map[goStyleKind]bool) {
	t.Helper()
	report := repositoryGoStyleReport(t)
	var findings []string
	for _, finding := range report.Findings {
		if selected[finding.Kind] {
			findings = append(findings, fmt.Sprintf("%s:%d:%s:%s", finding.File, finding.Line, finding.Kind, finding.Symbol))
		}
	}
	if len(findings) != 0 {
		t.Fatalf("%d naming/comment findings:\n%s", len(findings), strings.Join(findings, "\n"))
	}
}

func repositoryGoStyleReport(t *testing.T) goStyleReport {
	return liveRepositoryCensus(t, goStyleCensus)
}

func liveRepositoryCensus[T any](t *testing.T, census func(gosyntax.Snapshot) (T, error)) T {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := gosyntax.DiscoverGo(root, "cmd", "internal")
	if err != nil {
		t.Fatal(err)
	}
	report, err := census(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	return report
}

func TestGoStyleCensus(t *testing.T) {
	t.Parallel()
	snapshot, err := (gosyntax.Snapshot{}).Overlay(map[string][]byte{
		"good/good.go": []byte("// Package good provides fixtures.\npackage good\n\n// Item represents a value.\ntype Item struct{}\n\n// Value returns the item value.\nfunc (i Item) Value() int { return 1 }\n"),
		"bad/bad.go": []byte(`package bad
import "fmt"
type Public struct{}
func (receiver Public) GetUrl()(result int) {
	if true {
		return
	}
	_ = fmt.Errorf("Bad value")
	_ = receiver
	_ = receiver
	_ = receiver
	_ = receiver
	_ = receiver
	return
}
`),
	})
	if err != nil {
		t.Fatal(err)
	}
	report, err := goStyleCensus(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	for _, kind := range []goStyleKind{
		goStyleFormat, goStylePackageDoc, goStyleExportDoc, goStyleInitialism,
		goStyleGetter, goStyleNamedResult, goStyleNakedReturn, goStyleErrorText,
	} {
		if !hasStyleFinding(report, kind) {
			t.Fatalf("%s finding is absent: %+v", kind, report.Findings)
		}
	}
}

// TestGoStyleDeltaKeysPackageDocByPackage passes a file added to a package
// that already lacked its comment, though the new file sorts first and the
// finding is shown at it, and still refuses a change that removes a
// package's comment.
func TestGoStyleDeltaKeysPackageDocByPackage(t *testing.T) {
	t.Parallel()
	snapshot := func(files map[string]string) gosyntax.Snapshot {
		t.Helper()
		overlay := map[string][]byte{}
		for name, source := range files {
			overlay[name] = []byte(source)
		}
		result, err := (gosyntax.Snapshot{}).Overlay(overlay)
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	undocumented := snapshot(map[string]string{"p/b.go": "package p\n"})
	added := snapshot(map[string]string{"p/b.go": "package p\n", "p/a_test.go": "package p\n"})
	if err := ValidateGoStyleDelta(added, undocumented); err != nil {
		t.Fatalf("a file added to an undocumented package was refused: %v", err)
	}
	documented := snapshot(map[string]string{"p/b.go": "// Package p is a fixture.\npackage p\n"})
	if err := ValidateGoStyleDelta(undocumented, documented); err == nil || !strings.Contains(err.Error(), string(goStylePackageDoc)) {
		t.Fatalf("removing a package comment was not refused: %v", err)
	}
}

func TestGoStyleApplicability(t *testing.T) {
	t.Parallel()
	snapshot, err := (gosyntax.Snapshot{}).Overlay(map[string][]byte{
		"p/a.go":      []byte("// Package p owns values.\npackage p\ntype local struct{}\nfunc (long local) one() {}\nfunc (long local) Identity() {}\nfunc (long local) GetRows(rows []int) []int { return rows }\n"),
		"p/b.go":      []byte("package p\nfunc (other local) two() {}\n"),
		"p/a_test.go": []byte("package p\nfunc ExportedFixture() {}\n"),
	})
	if err != nil {
		t.Fatal(err)
	}
	report, err := goStyleCensus(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if !hasStyleFinding(report, goStyleReceiverConsistency) {
		t.Fatalf("receiver inconsistency is absent: %+v", report.Findings)
	}
	for _, finding := range report.Findings {
		if finding.Kind == goStyleExportDoc && finding.Symbol == "ExportedFixture" {
			t.Fatalf("test helper required production Godoc: %+v", finding)
		}
		if finding.Kind == goStyleInitialism && finding.Symbol == "Identity" {
			t.Fatalf("word was treated as an initialism: %+v", finding)
		}
		if finding.Kind == goStyleGetter && finding.Symbol == "GetRows" {
			t.Fatalf("operation was treated as an accessor: %+v", finding)
		}
	}
}

func TestGoStyleGeneratedExclusion(t *testing.T) {
	t.Parallel()
	snapshot, err := (gosyntax.Snapshot{}).Overlay(map[string][]byte{
		"generated/value.go": []byte("// Code generated by fixture. DO NOT EDIT.\npackage generated\ntype Public struct{}\n"),
	})
	if err != nil {
		t.Fatal(err)
	}
	report, err := goStyleCensus(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if report.Excluded != 1 || len(report.Findings) != 0 {
		t.Fatalf("generated census = %+v", report)
	}
}

func hasStyleFinding(report goStyleReport, kind goStyleKind) bool {
	for _, finding := range report.Findings {
		if finding.Kind == kind {
			return true
		}
	}
	return false
}
