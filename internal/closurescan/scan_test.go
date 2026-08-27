package closurescan

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/closureledger"
	"overgo/internal/repoanalysis"
)

func TestClosureScanSnapshotConsumer(t *testing.T) {
	snapshot := scanTestSnapshot(t, map[string]string{
		"internal/live.go":      "package p\nconst Limit = 3\n",
		"internal/other.go":     "package p\nconst Other = 4\n",
		"internal/live_test.go": "package p\nconst TestLimit = 5\n",
	})
	candidates, err := ScanSnapshot(snapshot, []string{"internal/live.go"}, CandidateConstants)
	if err != nil {
		t.Fatal(err)
	}
	if len(candidates) != 1 || candidates[0].Name != "Limit" {
		t.Fatalf("candidates = %+v", candidates)
	}
}

func TestScanIncludesFunctionLocalConstants(t *testing.T) {
	snapshot := scanTestSnapshot(t, map[string]string{"internal/policy.go": `package policy
const base = 3
func derive() {
	const base = 5
	const local = base * 4
}
`})
	candidates, err := ScanSnapshot(snapshot, nil, CandidateConstants)
	if err != nil {
		t.Fatal(err)
	}
	byScope := map[string]Candidate{}
	for _, candidate := range candidates {
		byScope[candidate.Scope+"/"+candidate.Name] = candidate
	}
	if got := byScope["package/base"]; got.Value != "3" || got.Expression != "3" {
		t.Fatalf("package base = %+v", got)
	}
	if got := byScope["derive/base"]; got.Value != "5" || got.Line == 0 {
		t.Fatalf("local base = %+v", got)
	}
	if got := byScope["derive/local"]; got.Value != "20" || got.Expression != "base * 4" {
		t.Fatalf("derived local = %+v", got)
	}
}

func TestCandidateIdentityIncludesEvaluatedSourceBinding(t *testing.T) {
	snapshot := scanTestSnapshot(t, map[string]string{
		"internal/policy.go": "package policy\nconst allocation = 4 * 8\n",
	})
	before, err := ScanSnapshot(snapshot, nil, CandidateConstants)
	if err != nil || len(before) != 1 {
		t.Fatalf("before = %+v, %v", before, err)
	}
	rewritten, err := snapshot.Overlay(map[string][]byte{
		"internal/policy.go": []byte("package policy\nconst allocation = 2 * 16\n"),
	})
	if err != nil {
		t.Fatal(err)
	}
	after, err := ScanSnapshot(rewritten, nil, CandidateConstants)
	if err != nil || len(after) != 1 {
		t.Fatalf("after = %+v, %v", after, err)
	}
	left, right := before[0], after[0]
	if left.Package != "internal" || left.Scope != "package" ||
		left.Line == 0 || left.SourceID == "" || left.CallsiteID == "" || left.Expression != "4 * 8" || left.Value != "32" {
		t.Fatalf("source binding = %+v", left)
	}
	if left.DeclarationKey() != right.DeclarationKey() || left.ExactKey() == right.ExactKey() || right.Value != left.Value {
		t.Fatalf("before = %+v, after = %+v", left, right)
	}
}

func TestStructuralBindingSurvivesUnrelatedEdit(t *testing.T) {
	snapshot := scanTestSnapshot(t, map[string]string{
		"internal/policy.go": "package policy\nconst Limit = 8\nfunc use() int { return Limit }\n",
	})
	before, err := ScanSnapshot(snapshot, nil, CandidateConstants)
	if err != nil || len(before) != 1 {
		t.Fatalf("before = %+v, %v", before, err)
	}
	rewritten, err := snapshot.Overlay(map[string][]byte{
		"internal/policy.go": []byte("package policy\n\n// unrelated documentation\nconst Limit = 8\nfunc use() int { return Limit }\n"),
	})
	if err != nil {
		t.Fatal(err)
	}
	after, err := ScanSnapshot(rewritten, nil, CandidateConstants)
	if err != nil || len(after) != 1 {
		t.Fatalf("after = %+v, %v", after, err)
	}
	left, right := before[0], after[0]
	if left.StructuralID == "" || left.StructuralID != right.StructuralID || left.SourceID != right.SourceID ||
		left.CallsiteID != right.CallsiteID || left.OwnerID == right.OwnerID || left.Line == right.Line {
		t.Fatalf("before = %+v, after = %+v", left, right)
	}
}

func TestTypedCandidatesBindEverySiteKind(t *testing.T) {
	snapshot := scanTestSnapshot(t, map[string]string{"internal/policy.go": `package policy
const Window = 8
func decide(values []float64) bool {
	return quantile(values, 0.9) > 3
}
`})
	candidates, err := ScanSnapshot(snapshot, nil, CandidateAll)
	if err != nil {
		t.Fatal(err)
	}
	want := map[closureledger.BindingKind]bool{
		closureledger.BindingConstant: false, closureledger.BindingLiteral: false, closureledger.BindingAssumption: false,
	}
	keys := map[string]bool{}
	for _, candidate := range candidates {
		binding, err := candidate.Binding()
		if err != nil {
			t.Fatalf("bind %+v: %v", candidate, err)
		}
		if binding.Kind != candidate.Kind || binding.CallsiteID == "" || keys[candidate.ExactKey()] {
			t.Fatalf("candidate = %+v, binding = %+v", candidate, binding)
		}
		keys[candidate.ExactKey()] = true
		want[candidate.Kind] = true
	}
	for kind, found := range want {
		if !found {
			t.Fatalf("missing %s candidate in %+v", kind, candidates)
		}
	}
}

func TestLiteralCensusAccountsForEveryProductionLiteral(t *testing.T) {
	snapshot := scanTestSnapshot(t, map[string]string{
		"internal/policy.go": `package policy
var packageDefault = 7
const named = 8
func use(values []int) int {
	local := 3
	_ = [4]int{}
	_ = values[5]
	consume(6)
	if local > 9 { return 10 }
	return 11
}
`,
		"internal/policy_test.go": "package policy\nvar testOnly = 12\n",
		"internal/generated.go":   "// Code generated by fixture. DO NOT EDIT.\npackage policy\nvar generated = 13\n",
	})
	sites, err := CensusLiterals(snapshot, nil)
	if err != nil {
		t.Fatal(err)
	}
	expected := map[string]LiteralContext{
		"7": LiteralAssignment, "3": LiteralAssignment, "4": LiteralExtent, "5": LiteralIndex,
		"6": LiteralCall, "9": LiteralComparison, "10": LiteralReturn, "11": LiteralReturn,
	}
	if len(sites) != len(expected) {
		t.Fatalf("sites = %+v", sites)
	}
	for _, site := range sites {
		context, ok := expected[site.Value]
		if !ok || site.Context != context || site.SourceID == "" || site.Line == 0 {
			t.Fatalf("site = %+v", site)
		}
		delete(expected, site.Value)
	}
	if len(expected) != 0 {
		t.Fatalf("missing = %v", expected)
	}
}

func TestStructuralZerosAreNotRuntimePolicy(t *testing.T) {
	source := `package policy
import "fmt"
import "errors"
func values(input []int) []int {
	if len(input) == 0 { return make([]int, 0, cap(input)) }
	for cursor := uint64(0); cursor < uint64(len(input)); cursor++ { break }
	for offset := 0; offset < len(input); offset++ { break }
	for index := range input { if index > 0 { break } }
	if input[0] > 0 { return input }
	return nil
}
func decode() (int, error) { return 0, fmt.Errorf("invalid") }
func decodeJoined() (int, error) { return 0, errors.Join(errors.New("invalid")) }
`
	snapshot := scanTestSnapshot(t, map[string]string{"internal/policy.go": source})
	candidates, err := ScanSnapshot(snapshot, nil, CandidateLiterals)
	if err != nil {
		t.Fatal(err)
	}
	for _, candidate := range candidates {
		if candidate.Policy {
			t.Fatalf("structural identity classified as runtime policy: %+v", candidates)
		}
	}
	if len(candidates) != strings.Count(source, "0") {
		t.Fatalf("literal authority candidates=%+v", candidates)
	}
}

func TestLiteralClassificationUsesSyntaxAndOwnership(t *testing.T) {
	snapshot := scanTestSnapshot(t, map[string]string{"internal/policy.go": `package policy
func admitted(n int) bool { return n > 3 || n > 4096 }
`})
	sites, err := CensusLiterals(snapshot, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(sites) != 2 {
		t.Fatalf("sites = %+v", sites)
	}
	for _, site := range sites {
		if site.Context != LiteralComparison || site.Scope != "admitted" || site.Package != "internal" {
			t.Fatalf("site = %+v", site)
		}
	}
	if sites[0].Value != "3" || sites[1].Value != "4096" {
		t.Fatalf("values = %q, %q", sites[0].Value, sites[1].Value)
	}
}

func TestLiteralClassUsesContextAndExactAuthority(t *testing.T) {
	snapshot := scanTestSnapshot(t, map[string]string{"internal/policy.go": `package policy
import "math"
func classify(n int, value float64) {
	_ = make([]byte, 8)
	_ = n >> 3
	_ = math.Pow(value, 2)
	if n == 0 || n > 512 {}
}
`})
	sites, err := CensusLiterals(snapshot, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]LiteralClass{
		"8": LiteralCapacity, "3": LiteralFormat, "2": LiteralMathematical,
		"0": LiteralStructural, "512": LiteralUnknown,
	}
	for _, site := range sites {
		if site.Class != want[site.Value] {
			t.Fatalf("literal %s class = %s, want %s", site.Value, site.Class, want[site.Value])
		}
		if site.Value != "512" {
			continue
		}
		candidate := site.Candidate()
		binding, err := candidate.Binding()
		if err != nil {
			t.Fatal(err)
		}
		document, err := closureledger.New(
			candidate.Name, json.RawMessage(candidate.ValueJSON()), closureledger.TierImplementation,
			closureledger.StatusClosed, "Fixed admission policy.", []closureledger.SourceBinding{binding},
			"Replace policy owner.", "Policy owner change.", binding.Owner,
		)
		if err != nil {
			t.Fatal(err)
		}
		classified := ClassifyLiteralAuthorities([]LiteralSite{site}, []closureledger.Document{document})
		if classified[0].Class != LiteralPolicy || site.Class != LiteralUnknown {
			t.Fatalf("authority class = %s, source class = %s", classified[0].Class, site.Class)
		}
	}
}

func TestTestCensusSeparatesFixturesFromPolicyCopies(t *testing.T) {
	snapshot := scanTestSnapshot(t, map[string]string{
		"internal/process.go": `package process
const ProcessFailure = 64
func run() int { return ProcessFailure }
`,
		"internal/process_test.go": `package process
const expectedFailure = 64
func verify() {
	fixture := []int{2, 4}
	_ = fixture
	if run() != 64 { panic("failure") }
}
`,
	})
	sites, err := CensusTestLiterals(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	classes := map[TestLiteralClass]int{}
	for _, site := range sites {
		classes[site.Class]++
		if site.Class == TestPolicyCopy && len(site.ProductionMatches) == 0 {
			t.Fatalf("unbound policy copy = %+v", site)
		}
	}
	if classes[TestPolicyCopy] != 1 || classes[TestFixture] != 2 || classes[TestAssertion] != 1 {
		t.Fatalf("classes = %v, sites = %+v", classes, sites)
	}
}

func TestAssumptionCensusFindsDecisionSurfaces(t *testing.T) {
	snapshot := scanTestSnapshot(t, map[string]string{"internal/decision.go": `package decision
func decide(variance float64, values []float64, tensorRows, limit int) bool {
	normalize(values)
	if variance < quantile(values, 0.9) { return true }
	if euclideanDistance(values, values) > 0 { return true }
	return tensorRows > limit
}
`})
	hints, err := CensusAssumptions(snapshot, nil)
	if err != nil {
		t.Fatal(err)
	}
	kinds := map[AssumptionKind]bool{}
	for _, hint := range hints {
		kinds[hint.Kind] = true
		if hint.SourceID == "" || hint.Scope != "decide" || strings.Contains(hint.Expression, "normalize") {
			t.Fatalf("hint = %+v", hint)
		}
	}
	for _, kind := range []AssumptionKind{AssumptionMoment, AssumptionQuantile, AssumptionGeometry, AssumptionShape} {
		if !kinds[kind] {
			t.Fatalf("missing %s in %+v", kind, hints)
		}
	}
}

func TestImportedOperatorOwnsAssumptionPolicy(t *testing.T) {
	snapshot := scanTestSnapshot(t, map[string]string{"internal/decision.go": `package decision
import "example.org/shared"
func decide(builder *shared.Builder, value float64) bool {
	if shared.ValidEffectiveRank(value) { return true }
	if builder.ValidEffectiveRank(value) { return true }
	return localEffectiveRank(value)
}
`})
	candidates, err := ScanSnapshot(snapshot, nil, CandidateAssumptions)
	if err != nil {
		t.Fatal(err)
	}
	var imported, local bool
	for _, candidate := range candidates {
		if strings.HasPrefix(candidate.Expression, "shared.") {
			imported = !candidate.Policy
		} else {
			local = candidate.Policy
		}
	}
	if !imported || !local {
		t.Fatalf("assumption authority = %+v", candidates)
	}
}

func TestMethodReceiverDoesNotCreateAssumption(t *testing.T) {
	snapshot := scanTestSnapshot(t, map[string]string{"internal/decision.go": `package decision
type value struct { Shape shape }
type shape struct{}
func (shape) Equal(shape) bool { return true }
func check(left, right value) bool { return left.Shape.Equal(right.Shape) }
`})
	candidates, err := ScanSnapshot(snapshot, nil, CandidateAssumptions)
	if err != nil {
		t.Fatal(err)
	}
	if len(candidates) != 0 {
		t.Fatalf("method receiver assumptions = %v", candidates)
	}
}

func TestRepeatedPolicyLiterals(t *testing.T) {
	source := `package policy
func first(n int, values []int) bool {
	_ = 3 * 3
	return n > 4096 && values[3] > 0 && "strict" != ""
}
func second(n int) bool {
	_ = 6 / 3
	return n <= 4096 && "strict" != ""
}
`
	snapshot := scanTestSnapshot(t, map[string]string{"internal/policy.go": source})
	ranked, err := RepeatedPolicyLiterals(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range ranked {
		if row.Value == "3" {
			t.Fatalf("structural math literal ranked: %+v", row)
		}
	}
	if len(ranked) == 0 || ranked[0].Value != "4096" || len(ranked[0].Functions) != 2 {
		t.Fatalf("ranked = %+v", ranked)
	}
}

func scanTestSnapshot(t *testing.T, files map[string]string) repoanalysis.SourceSnapshot {
	t.Helper()
	root := t.TempDir()
	for relative, content := range files {
		path := filepath.Join(root, filepath.FromSlash(relative))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	snapshot, err := repoanalysis.DiscoverGo(root, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}
