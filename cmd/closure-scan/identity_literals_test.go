package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/overgodb"
	"overgo/internal/repoanalysis"
)

// TestIdentityLiteralsClassifyAsMathematicalFacts pins: the decimal radix
// and a numeric width in strconv calls, zero reset into a variable, zero
// returned alone, zero and one bounding min and max, and zero handed to a
// math function are classified by a named identity rule as mathematical
// facts and are never policy candidates; a policy count, a non-decimal
// radix and a zero handed to an ordinary call stay with review; the check
// audits every identity site by rule; a triage or proposal naming an
// identity literal is refused as already classified.
func TestIdentityLiteralsClassifyAsMathematicalFacts(t *testing.T) {
	root := t.TempDir()
	source := `package sample

import (
	"math"
	"strconv"
)

var count int

func Parse(text string) (uint64, error) { return strconv.ParseUint(text, 10, 32) }
func Render(value int64) string       { return strconv.FormatInt(value, 10) }
func Reset()                          { count = 0 }
func Empty() int                      { return 0 }
func Clamp(unit float64) float64      { return min(max(unit, 0), 1) }
func Infinite(value float64) bool     { return math.IsInf(value, 0) }
func Retries() int                    { return 3 }
func Hex(value uint64) string         { return strconv.FormatUint(value, 16) }
func Seek(offset int64) int64         { return Position(offset, 0) }
func Position(offset, whence int64) int64 { return offset + whence }
`
	path := filepath.Join(root, "internal", "sample", "facts.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture\n\ngo 1.24\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	snapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	sites, err := closurescan.CensusLiterals(snapshot, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]closurescan.IdentityRule{
		"Parse:10": closurescan.IdentityDecimalRadix, "Parse:32": closurescan.IdentityBitSize,
		"Render:10": closurescan.IdentityDecimalRadix, "Reset:0": closurescan.IdentityZeroReset,
		"Empty:0": closurescan.IdentityZeroResult, "Clamp:0": closurescan.IdentityUnitBound,
		"Clamp:1": closurescan.IdentityUnitBound, "Infinite:0": closurescan.IdentityMathArgument,
		"Retries:3": "", "Hex:16": "", "Seek:0": "",
	}
	seen := map[string]bool{}
	for _, site := range sites {
		key := site.Scope + ":" + site.Value
		rule, known := want[key]
		if !known {
			t.Fatalf("unexpected literal site %s at line %d", key, site.Line)
		}
		seen[key] = true
		if site.Identity != rule {
			t.Fatalf("%s identity = %q, want %q", key, site.Identity, rule)
		}
		if rule != "" && (site.Class != closurescan.LiteralMathematical || site.Policy) {
			t.Fatalf("%s class=%s policy=%t under rule %s", key, site.Class, site.Policy, rule)
		}
		if rule == "" && !site.Policy {
			t.Fatalf("%s left review without a rule", key)
		}
	}
	for key := range want {
		if !seen[key] {
			t.Fatalf("literal site %s was not scanned", key)
		}
	}

	candidates, err := closurescan.ScanSnapshot(snapshot, nil, closurescan.CandidateLiterals)
	if err != nil {
		t.Fatal(err)
	}
	var identity closurescan.Candidate
	for _, candidate := range candidates {
		if candidate.Identity == closurescan.IdentityDecimalRadix && candidate.Scope == "Parse" {
			identity = candidate
		}
	}
	if identity.Name == "" {
		t.Fatal("identity literal is not a scan candidate")
	}
	var audit bytes.Buffer
	if audited := auditIdentityLiterals(&audit, candidates); audited != 8 {
		t.Fatalf("audited %d identity sites:\n%s", audited, audit.String())
	}
	if !strings.Contains(audit.String(), "closure-scan: identity decimal-radix internal/sample/facts.go:") {
		t.Fatalf("audit lacks the rule line:\n%s", audit.String())
	}

	triage := triageFile{Rows: []triageRow{{
		Kind: identity.Kind, Name: identity.Name, File: identity.File, Scope: identity.Scope, Line: identity.Line,
		Tier: string(closureledger.TierMathematicalFact), Status: string(closureledger.StatusClosed),
		Understanding: "Decimal.", ClosurePath: "None.", RerankTrigger: "None.",
	}}}
	encoded, err := json.Marshal(triage)
	if err != nil {
		t.Fatal(err)
	}
	triagePath := filepath.Join(root, "triage.json")
	if err := os.WriteFile(triagePath, encoded, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := emit(root, "store", triagePath, candidates); err == nil || !strings.Contains(err.Error(), "classified by rule decimal-radix") {
		t.Fatalf("triage of an identity literal = %v", err)
	}
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := proposeTriageRows(t.Context(), snapshot, store, []string{identity.Name}); err == nil || !strings.Contains(err.Error(), "classified by rule decimal-radix") {
		t.Fatalf("proposal for an identity literal = %v", err)
	}
}
