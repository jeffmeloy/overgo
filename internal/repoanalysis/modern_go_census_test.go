package repoanalysis

import (
	"context"
	"encoding/json"
	"errors"
	"go/ast"
	"go/token"
	"go/types"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"overgo/internal/gosource"
)

func TestModernGoCensusCoversApplicableCatalog(t *testing.T) {
	root := modernGoTestRepository(t, `package sample

func identity(value any) any { return value }
`)
	census, err := BuildModernGoCensus(root, ModernGoTargetVersion)
	if err != nil {
		t.Fatal(err)
	}
	applicable, err := ModernGoApplicableGuidelines(ModernGoTargetVersion)
	if err != nil {
		t.Fatal(err)
	}
	if len(census.Findings) != len(applicable) {
		t.Fatalf("findings = %d, applicable catalog = %d", len(census.Findings), len(applicable))
	}
	for index, finding := range census.Findings {
		if finding.ID != applicable[index].ID || !finding.Measured || finding.Risk == "" {
			t.Errorf("finding %d = %+v, guideline = %s", index, finding, applicable[index].ID)
		}
		if finding.InspectedFiles == 0 || finding.TypedFiles == 0 {
			t.Errorf("finding %s coverage = %d inspected, %d typed", finding.ID, finding.InspectedFiles, finding.TypedFiles)
		}
	}
}

func TestEveryApplicableModernGoGuidelineMeasured(t *testing.T) {
	census, err := modernGoRepositoryCensus()
	if err != nil {
		t.Fatal(err)
	}
	applicable, err := ModernGoApplicableGuidelines(ModernGoTargetVersion)
	if err != nil {
		t.Fatal(err)
	}
	if len(census.Findings) != len(applicable) {
		t.Fatalf("measured findings=%d applicable=%d", len(census.Findings), len(applicable))
	}
	for index, guideline := range applicable {
		finding := census.Findings[index]
		if finding.ID != guideline.ID || !finding.Measured || finding.InspectedFiles == 0 || finding.TypedFiles == 0 {
			t.Errorf("guideline %s measurement=%+v", guideline.ID, finding)
		}
	}
}

func TestModernGoCensusIsTypeAware(t *testing.T) {
	root := modernGoTestRepository(t, `package sample

func copyMap(dst, src map[string]int) {
	for key, value := range src {
		dst[key] = value
	}
}

func copySlice(dst, src []int) {
	for index, value := range src {
		dst[index] = value
	}
}
`)
	census, err := BuildModernGoCensus(root, ModernGoTargetVersion)
	if err != nil {
		t.Fatal(err)
	}
	finding := modernGoTestFinding(t, census, "maps_copy")
	if len(finding.Candidates) != 1 || finding.Candidates[0].Symbol != "copyMap" || !finding.Candidates[0].TypeChecked {
		t.Fatalf("maps_copy findings = %+v", finding.Candidates)
	}
	if slices.ContainsFunc(finding.Candidates, func(site ModernGoSite) bool { return site.Symbol == "copySlice" }) {
		t.Fatal("slice copy was classified as a map copy")
	}
}

func TestModernGoCensusClassifiesRiskAndOwnership(t *testing.T) {
	root := modernGoTestRepository(t, `package sample

import (
	"context"
	"sync"
)

type message struct {
	Count int `+"`json:\"count,omitempty\"`"+`
}

func worker(ctx context.Context) {
	var group sync.WaitGroup
	group.Add(1)
	go func() { defer group.Done() }()
	_ = context.Background()
}
`)
	census, err := BuildModernGoCensus(root, ModernGoTargetVersion)
	if err != nil {
		t.Fatal(err)
	}
	waitgroup := modernGoTestFinding(t, census, "sync_waitgroup_go")
	if waitgroup.Risk != ModernGoRiskConcurrency || len(waitgroup.Candidates) == 0 {
		t.Fatalf("waitgroup finding = %+v", waitgroup)
	}
	for _, site := range waitgroup.Candidates {
		if site.Path != "internal/sample/sample.go" || site.Package != "example/internal/sample" || site.Symbol != "worker" || site.Line == 0 {
			t.Errorf("waitgroup site lacks ownership: %+v", site)
		}
	}
	jsonFinding := modernGoTestFinding(t, census, "json_omitzero")
	if jsonFinding.Risk != ModernGoRiskWire || len(jsonFinding.Candidates) != 1 || jsonFinding.Candidates[0].Symbol != "package" {
		t.Fatalf("JSON finding = %+v", jsonFinding)
	}
}

func TestModernGoCensusDeterministic(t *testing.T) {
	root := modernGoTestRepository(t, `package sample

import "strings"

func parts(value string) {
	for _, part := range strings.Split(value, ",") {
		_ = part
	}
}
`)
	first, err := BuildModernGoCensus(root, ModernGoTargetVersion)
	if err != nil {
		t.Fatal(err)
	}
	second, err := BuildModernGoCensus(root, ModernGoTargetVersion)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatal("identical source produced different modern-Go censuses")
	}
}

func TestModernGoCensusPackageIsolation(t *testing.T) {
	const imported = `package sample
import workers "sync"
func work() {}
func launch(wait *workers.WaitGroup) {
	wait.Add(1)
	go func() { defer wait.Done(); work() }()
}
`
	const local = `package sample
import workers "sync"
var _ workers.WaitGroup
type localWait struct{}
func (*localWait) Add(int) {}
func (*localWait) Done() {}
func work() {}
func launch(wait *localWait) {
	wait.Add(1)
	go func() { defer wait.Done(); work() }()
}
`
	root := modernGoTestRepository(t, imported)
	other := "internal/other/sample.go"
	if err := os.MkdirAll(filepath.Join(root, "internal/other"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, other), []byte(imported), 0o644); err != nil {
		t.Fatal(err)
	}
	var previous ModernGoCensus
	for _, changed := range []bool{false, true} {
		if changed {
			if err := os.WriteFile(filepath.Join(root, "internal/sample/sample.go"), []byte(local), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		census, err := BuildModernGoCensus(root, ModernGoTargetVersion)
		if err != nil {
			t.Fatal(err)
		}
		var paths []string
		for _, site := range modernGoTestFinding(t, census, "sync_waitgroup_go").Candidates {
			if !site.TypeChecked || site.Symbol != "launch" {
				t.Fatalf("import binding lost: %+v", site)
			}
			paths = append(paths, site.Path)
		}
		expected := []string{other, "internal/sample/sample.go"}
		if changed {
			expected = []string{other}
			if census.SourceIdentity == previous.SourceIdentity {
				t.Fatal("new source retained the previous census identity")
			}
		}
		if !slices.Equal(paths, expected) {
			t.Fatalf("package-local types leaked across groups: got %v, want %v", paths, expected)
		}
		previous = census
	}
	current, err := BuildModernGoCensus(root, ModernGoTargetVersion)
	if err != nil || !reflect.DeepEqual(previous, current) {
		t.Fatalf("same source changed across census invocations: %v", err)
	}
}

func TestModernGoCensusDiscoveryOrderInvariant(t *testing.T) {
	root := filepath.Join("..", "..")
	forward, err := DiscoverGo(root, "cmd", "internal")
	if err != nil {
		t.Fatal(err)
	}
	reverse, err := DiscoverGo(root, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	selection, err := gosource.HostBuildSelection(root, "./cmd/...", "./internal/...")
	if err != nil {
		t.Fatal(err)
	}
	first, err := ModernGoCensusSnapshot(forward, selection, ModernGoTargetVersion)
	if err != nil {
		t.Fatal(err)
	}
	second, err := ModernGoCensusSnapshot(reverse, selection, ModernGoTargetVersion)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("modern-Go census depends on discovery order:\nforward=%+v\nreverse=%+v",
			modernGoTestFinding(t, first, "json_omitzero").Candidates,
			modernGoTestFinding(t, second, "json_omitzero").Candidates)
	}
}

func TestModernGoManualIdiomRewriteConformance(t *testing.T) {
	root := modernGoTestRepository(t, `package sample

import (
	"sort"
	"time"
)

func computed() string { return "fallback" }

func manual(values []string, value, other string) []string {
	if value == "" {
		value = other
	}
	if other == "" {
		other = computed()
	}
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	select {
	case <-ticker.C:
	default:
	}
	sort.Strings(values)
	return append([]string{}, values...)
}

func resetTicker() {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	ticker.Reset(2 * time.Second)
}
`)
	first, err := RewriteModernGoManualIdioms(root)
	if err != nil {
		t.Fatal(err)
	}
	if first.Fallbacks != 1 || first.Tickers != 1 || first.SliceClones != 1 || first.TypedSorts != 1 || len(first.Files) != 1 {
		t.Fatalf("first rewrite=%+v", first)
	}
	content, err := os.ReadFile(filepath.Join(root, "internal", "sample", "sample.go"))
	if err != nil {
		t.Fatal(err)
	}
	text := string(content)
	for _, required := range []string{"cmp.Or(value, other)", "ticker := time.Tick(time.Second)", "case <-ticker:", "slices.Sort(values)", "slices.Clone(values)", "other = computed()", "ticker.Reset(2 * time.Second)"} {
		if !strings.Contains(text, required) {
			t.Errorf("rewrite omits %q\n%s", required, text)
		}
	}
	second, err := RewriteModernGoManualIdioms(root)
	if err != nil {
		t.Fatal(err)
	}
	if second.Fallbacks != 0 || second.Tickers != 0 || second.SliceClones != 0 || second.TypedSorts != 0 || len(second.Files) != 0 {
		t.Fatalf("second rewrite=%+v", second)
	}
}

func TestModernGoMatcherConformance(t *testing.T) {
	root := modernGoTestRepository(t, `package sample

func fallback(value, other string) string {
	if value == "" {
		value = other
	}
	return value
}

func upsert(values []string, index int, value string) []string {
	if index >= 0 {
		values[index] = value
	} else {
		values = append(values, value)
	}
	return values
}

func reverse(values []int) {
	for left, right := 0, len(values)-1; left < right; left, right = left+1, right-1 {
		values[left], values[right] = values[right], values[left]
	}
}

func pair(left, right int) (int, int) {
	left, right = right, left
	return left, right
}

func compact(values []int) []int {
	result := values[:0]
	for _, value := range values {
		if len(result) == 0 || result[len(result)-1] != value {
			result = append(result, value)
		}
	}
	return result
}

func selectPositive(values []int) []int {
	var result []int
	for _, value := range values {
		if value > 0 {
			result = append(result, value)
		}
	}
	return result
}

func captures(values []int) {
	for _, value := range values {
		value := value
		go func() { _ = value }()
	}
	for index := range values {
		go func(index int) { _ = index }(index)
	}
}

func contains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func aligned(values, peers []string) bool {
	for index, value := range values {
		if peers[index] != value {
			return false
		}
	}
	return true
}
`)
	census, err := BuildModernGoCensus(root, ModernGoTargetVersion)
	if err != nil {
		t.Fatal(err)
	}
	for _, expected := range []struct{ rule, symbol string }{
		{"cmp_or", "fallback"}, {"slices_reverse", "reverse"}, {"slices_compact", "compact"},
	} {
		rule, symbol := expected.rule, expected.symbol
		finding := modernGoTestFinding(t, census, rule)
		if len(finding.Candidates) != 1 || finding.Candidates[0].Symbol != symbol {
			t.Errorf("%s findings = %+v", rule, finding.Candidates)
		}
	}
	loopCapture := modernGoTestFinding(t, census, "loopvar_capture")
	if len(loopCapture.Candidates) != 2 {
		t.Errorf("loopvar_capture findings = %+v", loopCapture.Candidates)
	}
	containsFinding := modernGoTestFinding(t, census, "slices_contains")
	if len(containsFinding.Candidates) != 1 || containsFinding.Candidates[0].Symbol != "contains" {
		t.Errorf("slices_contains findings = %+v", containsFinding.Candidates)
	}
}

func TestModernGoNewExpressionConformance(t *testing.T) {
	legacyRoot := modernGoTestRepository(t, `package sample

func pointer(value int) *int { return &value }
`)
	legacy, err := BuildModernGoCensus(legacyRoot, ModernGoTargetVersion)
	switch err {
	case nil:
	default:
		t.Fatal(err)
	}
	legacyFinding := modernGoTestFinding(t, legacy, "new_expression")
	if len(legacyFinding.Candidates) != 1 {
		t.Fatalf("legacy candidates = %+v", legacyFinding.Candidates)
	}

	modernRoot := modernGoTestRepository(t, `package sample

func pointer(value int) *int { return new(value) }
`)
	modern, err := BuildModernGoCensus(modernRoot, ModernGoTargetVersion)
	switch err {
	case nil:
	default:
		t.Fatal(err)
	}
	modernFinding := modernGoTestFinding(t, modern, "new_expression")
	if len(modernFinding.Candidates) != 0 || len(modernFinding.Adopted) != 1 {
		t.Fatalf("modern finding = %+v", modernFinding)
	}
}

func TestNoLegacyUnitStepIntegerRangeOutsideNumericRuntime(t *testing.T) {
	census, err := modernGoRepositoryCensus()
	if err != nil {
		t.Fatal(err)
	}
	finding := modernGoTestFinding(t, census, "range_over_int")
	for _, site := range finding.Candidates {
		if !site.NumericRuntime {
			t.Errorf("legacy unit-step integer loop remains outside numerical runtime: %s:%d %s", site.Path, site.Line, site.Symbol)
		}
	}
}

func TestModernGoNumericRangeResolution(t *testing.T) {
	census, err := modernGoRepositoryCensus()
	if err != nil {
		t.Fatal(err)
	}
	finding := modernGoTestFinding(t, census, "range_over_int")
	for _, site := range finding.Candidates {
		if site.NumericRuntime {
			t.Errorf("legacy unit-step integer loop remains in numerical runtime: %s:%d %s", site.Path, site.Line, site.Symbol)
		}
	}
}

func TestModernGoNumericRangeRewriteConformance(t *testing.T) {
	root := modernGoTestRepositoryPath(t, "internal/hostmath/sample.go", `package hostmath

func transform(values []int, count int) {
	for index := 0; index < len(values); index++ { values[index]++ }
	var slot int
	for slot = 0; slot < count; slot++ { values[slot]++ }
	for attempt := 0; attempt < 3; attempt++ { values[0]++ }
	for index := 0; index < count; index++ { count-- }
	for index := 0; index < count; index += 2 { values[index]++ }
	for index := 0; index < changing(); index++ { values[index]++ }
	var position float64
	for position = 0; position < float64(count); position++ {}
}
func changing() int { return 1 }
`)
	modernGoAssertCandidateCounts(t, root, []modernGoExpectedCount{{"range_over_int", 2}})
	rewritten, files, err := RewriteModernGoNumericIntegerRanges(root)
	if err != nil {
		t.Fatal(err)
	}
	if rewritten != 2 || len(files) != 1 || files[0] != "internal/hostmath/sample.go" {
		t.Fatalf("numeric range rewrite = %d edits in %v", rewritten, files)
	}
	modernGoAssertCandidateCounts(t, root, []modernGoExpectedCount{{"range_over_int", 0}})
	content, err := os.ReadFile(filepath.Join(root, "internal", "hostmath", "sample.go"))
	if err != nil {
		t.Fatal(err)
	}
	text := string(content)
	for _, required := range []string{"for index := range len(values)", "for slot = 0; slot < count; slot++", "for range 3", "index += 2", "index < changing()", "position < float64(count)"} {
		if !strings.Contains(text, required) {
			t.Errorf("numeric range rewrite omits %q:\n%s", required, text)
		}
	}
	rewritten, files, err = RewriteModernGoNumericIntegerRanges(root)
	if err != nil {
		t.Fatal(err)
	}
	if rewritten != 0 || len(files) != 0 {
		t.Fatalf("second numeric range rewrite = %d edits in %v", rewritten, files)
	}
}

func TestNoRedundantLoopVariableCapture(t *testing.T) {
	census, err := modernGoRepositoryCensus()
	if err != nil {
		t.Fatal(err)
	}
	finding := modernGoTestFinding(t, census, "loopvar_capture")
	if len(finding.Candidates) != 0 {
		t.Fatalf("redundant loop captures remain: %+v", finding.Candidates)
	}
}

func TestModernGoSliceSearchConformance(t *testing.T) {
	root := modernGoTestRepository(t, `package sample

import "strings"

func contains(values []string, target string) bool {
	for _, value := range values {
		if value == target { return true }
	}
	return false
}

func index(values []string, target string) int {
	for position, value := range values {
		if value == target { return position }
	}
	return -1
}

func indexFunc(values []string, prefix string) int {
	for position, value := range values {
		if strings.HasPrefix(value, prefix) { return position }
	}
	return -1
}

func aligned(values, peers []string) bool {
	for position, value := range values {
		if peers[position] != value { return false }
	}
	return true
}
`)
	census, err := BuildModernGoCensus(root, ModernGoTargetVersion)
	if err != nil {
		t.Fatal(err)
	}
	for _, expected := range []struct{ rule, symbol string }{
		{"slices_contains", "contains"}, {"slices_index", "index"}, {"slices_index_func", "indexFunc"},
	} {
		finding := modernGoTestFinding(t, census, expected.rule)
		if len(finding.Candidates) != 1 || finding.Candidates[0].Symbol != expected.symbol {
			t.Errorf("%s findings = %+v", expected.rule, finding.Candidates)
		}
	}
}

func TestModernGoCollectionMutationConformance(t *testing.T) {
	root := modernGoTestRepository(t, `package sample

func cloneMap(source map[string]int) map[string]int {
	target := make(map[string]int, len(source))
	for key, value := range source { target[key] = value }
	return target
}
func copyMap(target, source map[string]int) {
	for key, value := range source { target[key] = value }
}
func deleteMap(values map[string]int) {
	for key, value := range values { if value == 0 { delete(values, key) } }
}
func clearMap(values map[string]int) {
	for key := range values { delete(values, key) }
}
func clearSlice(values []int) {
	for index := range values { values[index] = 0 }
}
func cloneSlice(values []int) []int { return append([]int{}, values...) }
func clip(values []int) []int { return values[:len(values):len(values)] }
func compact(values []int) []int {
	result := values[:0]
	for _, value := range values {
		if len(result) == 0 || result[len(result)-1] != value { result = append(result, value) }
	}
	return result
}
func reverse(values []int) {
	for left, right := 0, len(values)-1; left < right; left, right = left+1, right-1 {
		values[left], values[right] = values[right], values[left]
	}
}
func filter(values []int) []int {
	result := values[:0]
	for _, value := range values { if value != 0 { result = append(result, value) } }
	return result
}
func keyedCopy(target, source map[string]int) {
	for key, value := range source { target[valueString(value)] = len(key) }
}
func valueString(value int) string { return "" }
`)
	modernGoAssertCandidateCounts(t, root, []modernGoExpectedCount{
		{"maps_clone", 1}, {"maps_copy", 2}, {"maps_delete_func", 1}, {"clear", 2},
		{"slices_clone", 1}, {"slices_clip", 1}, {"slices_compact", 1}, {"slices_reverse", 1},
	})
}

func TestModernGoMapIteratorConformance(t *testing.T) {
	root := modernGoTestRepository(t, `package sample

import "sort"

func keys(values map[string]int) []string {
	var result []string
	for key := range values { result = append(result, key) }
	return result
}
func values(source map[string]int) []int {
	var result []int
	for _, value := range source { result = append(result, value) }
	return result
}
func sorted(source map[string]int) []string {
	var result []string
	for key := range source { result = append(result, key) }
	sort.Strings(result)
	return result
}
func collect(sequence func(func(int) bool)) []int {
	var result []int
	for value := range sequence { result = append(result, value) }
	return result
}
func sum(source map[string]int) int {
	result := 0
	for _, value := range source { result += value }
	return result
}
func ordinarySort(values []string) { sort.Strings(values) }
`)
	modernGoAssertCandidateCounts(t, root, []modernGoExpectedCount{
		{"maps_keys_values_iter", 3}, {"slices_collect", 1}, {"slices_sorted", 1},
	})
}

func TestModernGoSortingConformance(t *testing.T) {
	root := modernGoTestRepository(t, `package sample

import "sort"

type item struct { name string; rank int }

func natural(values []string) { sort.Strings(values) }
func ordered(values []item) {
	sort.Slice(values, func(left, right int) bool { return values[left].name < values[right].name })
}
func stable(values []item) {
	sort.SliceStable(values, func(left, right int) bool { return values[left].rank < values[right].rank })
}
func search(values []int, target int) int {
	return sort.Search(len(values), func(index int) bool { return values[index] >= target })
}
`)
	modernGoAssertCandidateCounts(t, root, []modernGoExpectedCount{
		{"slices_sort", 1}, {"slices_sort_func", 2},
	})
}

func TestModernGoCutConformance(t *testing.T) {
	modernGoAssertSourceCandidateCounts(t, `package sample

import (
	"bytes"
	"strings"
)

func prefix(value string) string {
	if strings.HasPrefix(value, "pre/") { value = strings.TrimPrefix(value, "pre/") }
	return value
}
func suffix(value string) string {
	if strings.HasSuffix(value, ".json") { value = strings.TrimSuffix(value, ".json") }
	return value
}
func splitString(value string) string {
	if index := strings.Index(value, "/"); index >= 0 { return value[index+1:] }
	return value
}
func splitBytes(value []byte) []byte {
	if index := bytes.IndexByte(value, ':'); index != -1 { return value[:index] }
	return value
}
func standalone(value string) string { return strings.TrimPrefix(value, "pre/") }
`, []modernGoExpectedCount{
		{"strings_cut_prefix_suffix", 2}, {"strings_cut", 1}, {"bytes_cut", 1},
	})
}

func TestModernGoSplitSequenceConformance(t *testing.T) {
	modernGoAssertSourceCandidateCounts(t, `package sample

import (
	"bytes"
	"strings"
)

func split(value string) { for part := range strings.Split(value, "/") { _ = part } }
func fields(value string) { for part := range strings.Fields(value) { _ = part } }
func byteSplit(value []byte) { for part := range bytes.Split(value, []byte{'/'}) { _ = part } }
func byteFields(value []byte) { for part := range bytes.Fields(value) { _ = part } }
func materialize(value string) int { parts := strings.Split(value, "/"); return len(parts) }
`, []modernGoExpectedCount{{"strings_split_seq", 4}})
}

func TestModernGoTextAllocationConformance(t *testing.T) {
	modernGoAssertSourceCandidateCounts(t, `package sample

import "fmt"

type item struct { label string }

func cloneString(value string) string { return string([]byte(value)) }
func cloneBytes(value []byte) []byte { return append([]byte{}, value...) }
func cloneStrings(value []string) []string { return append([]string{}, value...) }
func appendFormat(buffer []byte, value int) []byte {
	return append(buffer, fmt.Sprintf("%d", value)...)
}
func nestedFormat(items []item, value int) []item {
	return append(items, item{label: fmt.Sprintf("%d", value)})
}
`, []modernGoExpectedCount{
		{"strings_clone", 1}, {"bytes_clone", 1}, {"fmt_appendf", 1},
	})
}

func TestModernGoWaitGroupConformance(t *testing.T) {
	modernGoAssertSourceCandidateCounts(t, `package sample

import "sync"

func safe(wait *sync.WaitGroup) {
	wait.Add(1)
	go func() { defer wait.Done(); work() }()
}
func panicking(wait *sync.WaitGroup) {
	wait.Add(1)
	go func() { defer wait.Done(); panic("boom") }()
}
func batch(wait *sync.WaitGroup) {
	wait.Add(2)
	go func() { defer wait.Done(); work() }()
}
func work() {}
`, []modernGoExpectedCount{{"sync_waitgroup_go", 1}})
}

func TestModernGoOnceAndAtomicConformance(t *testing.T) {
	modernGoAssertSourceCandidateCounts(t, `package sample

import (
	"sync"
	"sync/atomic"
)

func onceFunction(once *sync.Once) { once.Do(func() { work() }) }
func onceLocalMutation(once *sync.Once) {
	once.Do(func() {
		local := struct{ value int }{}
		local.value = compute()
	})
}
func onceValue(once *sync.Once) int {
	value := 0
	once.Do(func() { value = compute() })
	return value
}
func legacyAtomic(value *int64) int64 { return atomic.LoadInt64(value) }
func typedAtomic(value *atomic.Int64) int64 { return value.Load() }
func work() {}
func compute() int { return 1 }
`, []modernGoExpectedCount{
		{"sync_once_func", 2}, {"sync_once_value", 1}, {"atomic_types", 1},
	})
}

func TestModernGoContextConformance(t *testing.T) {
	modernGoAssertSourceCandidateCounts(t, `package sample

import (
	"context"
	"sync"
	"time"
)

func legacyCleanup(ctx context.Context) { go func() { <-ctx.Done(); cleanup() }() }
func worker(ctx context.Context) { go func() { work(); <-ctx.Done() }() }
func waitGroup(wait *sync.WaitGroup) { go func() { wait.Done() }() }
func ownedCleanup(ctx context.Context) func() { return context.AfterFunc(ctx, cleanup) }
func legacyTimeout(ctx context.Context) { _, _ = context.WithTimeout(ctx, time.Second) }
func causedTimeout(ctx context.Context) { _, _ = context.WithTimeoutCause(ctx, time.Second, context.DeadlineExceeded) }
func legacyCancel(ctx context.Context) { _, _ = context.WithCancel(ctx) }
func causedCancel(ctx context.Context) { _, _ = context.WithCancelCause(ctx) }
func cleanup() {}
func work() {}
`, []modernGoExpectedCount{
		{"context_after_func", 1}, {"context_timeout_deadline_cause", 1}, {"context_cancel_cause", 1},
	})
}

func TestModernGoTestingContextConformance(t *testing.T) {
	source := `package sample

import (
	"context"
	"testing"
)

func TestOwned(test *testing.T) {
	_ = context.Background()
	_ = test.Context()
	(&executor{}).Execute(context.Background())
	test.Cleanup(func() { cleanupContext(test.Context()) })
	fixtureCleanup(test, func() { cleanupContext(test.Context()) })
}
func helper() { _ = context.Background() }
func helperWithOwner(benchmark *testing.B) { _ = context.TODO() }
type executor struct{}
func (*executor) Execute(context.Context) {}
func cleanupContext(context.Context) {}
func fixtureCleanup(test testing.TB, cleanup func()) { test.Cleanup(cleanup) }
`
	root := modernGoTestRepositoryFile(t, "sample_test.go", source)
	modernGoAssertCandidateCounts(t, root, []modernGoExpectedCount{{"testing_t_context", 3}})
	rewritten, files, err := RewriteModernGoTestingContexts(root)
	if err != nil {
		t.Fatal(err)
	}
	if rewritten != 5 || len(files) != 1 || files[0] != "internal/sample/sample_test.go" {
		t.Fatalf("testing context rewrite = %d edits in %v", rewritten, files)
	}
	modernGoAssertCandidateCounts(t, root, []modernGoExpectedCount{{"testing_t_context", 0}})
	content, err := os.ReadFile(filepath.Join(root, "internal", "sample", "sample_test.go"))
	if err != nil {
		t.Fatal(err)
	}
	text := string(content)
	if strings.Count(text, "test.Context()") != 5 || strings.Count(text, "benchmark.Context()") != 1 ||
		strings.Count(text, "context.WithoutCancel(test.Context())") != 3 ||
		strings.Count(text, "context.Background()") != 1 {
		t.Fatalf("testing context rewrite output:\n%s", text)
	}
}

func TestModernGoBenchmarkLoopConformance(t *testing.T) {
	source := `package sample

import "testing"

func BenchmarkRange(benchmark *testing.B) {
	for range benchmark.N { work() }
}
func BenchmarkClassic(benchmark *testing.B) {
	for index := 0; index < benchmark.N; index++ { work() }
}
func BenchmarkModern(benchmark *testing.B) {
	for benchmark.Loop() { work() }
}
type counter struct { N int }
func unrelated(value counter) {
	for range value.N { work() }
}
func work() {}
`
	root := modernGoTestRepositoryFile(t, "sample_test.go", source)
	modernGoAssertCandidateCounts(t, root, []modernGoExpectedCount{{"testing_b_loop", 2}})
	rewritten, files, err := RewriteModernGoBenchmarkLoops(root)
	if err != nil {
		t.Fatal(err)
	}
	if rewritten != 1 || len(files) != 1 || files[0] != "internal/sample/sample_test.go" {
		t.Fatalf("benchmark loop rewrite = %d edits in %v", rewritten, files)
	}
	modernGoAssertCandidateCounts(t, root, []modernGoExpectedCount{{"testing_b_loop", 1}})
	rewritten, files, err = RewriteModernGoBenchmarkLoops(root)
	if err != nil {
		t.Fatal(err)
	}
	if rewritten != 0 || len(files) != 0 {
		t.Fatalf("second benchmark loop rewrite = %d edits in %v", rewritten, files)
	}
}

func TestModernGoErrorConformance(t *testing.T) {
	source := `package sample

import (
	"errors"
	"fmt"
	"io"
)

type typedError struct{}
func (typedError) Error() string { return "typed" }
type ordinary struct{}
func inspect(err error) bool {
	var typed typedError
	legacyType := errors.As(err, &typed)
	_, modernType := errors.AsType[typedError](err)
	legacyIdentity := err == io.EOF
	modernIdentity := errors.Is(err, io.EOF)
	_ = fmt.Errorf("ordinary values: %s %d and 100%%", "value", 1)
	_ = fmt.Errorf("wrapped values: %w %+v", err, 1)
	_ = fmt.Errorf("joined values: %[2]w and %+w", "unused", err, io.EOF)
	_ = errors.Join(err, io.EOF)
	left, right := &ordinary{}, &ordinary{}
	_ = left == right
	return legacyType || modernType || legacyIdentity || modernIdentity
}
`
	root := modernGoTestRepository(t, source)
	modernGoAssertCandidateCounts(t, root, []modernGoExpectedCount{
		{"errors_as_type", 1}, {"errors_is", 1}, {"errors_join", 1},
	})
	rewritten, files, err := RewriteModernGoErrorIdentityComparisons(root)
	if err != nil {
		t.Fatal(err)
	}
	if rewritten != 1 || len(files) != 1 || files[0] != "internal/sample/sample.go" {
		t.Fatalf("error identity rewrite = %d edits in %v", rewritten, files)
	}
	modernGoAssertCandidateCounts(t, root, []modernGoExpectedCount{{"errors_is", 0}})
	rewritten, files, err = RewriteModernGoErrorIdentityComparisons(root)
	if err != nil {
		t.Fatal(err)
	}
	if rewritten != 0 || len(files) != 0 {
		t.Fatalf("second error identity rewrite = %d edits in %v", rewritten, files)
	}
	t.Run("partial imports require method proof", func(t *testing.T) {
		root := modernGoTestRepository(t, `package sample
import (
	"example/internal/review"
	"io/fs"
)
type census struct { review.Review; Total int }
type inheritedError struct { *fs.PathError }
type wrongSignature struct{}
func (wrongSignature) Error() int { return 0 }
func unresolved(a, b census) bool { return a == b }
func nonError(a, b wrongSignature) bool { return a == b }
func imported(a, b *fs.PathError) bool { return a == b }
func promoted(a, b inheritedError) bool { return a == b }
`)
		directory := filepath.Join(root, "internal", "review")
		if err := os.MkdirAll(directory, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(directory, "review.go"), []byte("package review\ntype Review struct { Count int }\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		// Host selection resolves the local package, while the export importer
		// leaves its embedding unresolved. That is not proof of Error() string.
		census, err := BuildModernGoCensus(root, ModernGoTargetVersion)
		if err != nil {
			t.Fatal(err)
		}
		finding := modernGoTestFinding(t, census, "errors_is")
		var symbols []string
		for _, site := range finding.Candidates {
			symbols = append(symbols, site.Symbol)
		}
		slices.Sort(symbols)
		if !slices.Equal(symbols, []string{"imported", "promoted"}) {
			t.Fatalf("error method proofs = %v; unresolved and wrong signatures must remain unchanged", symbols)
		}
		rewritten, files, err := RewriteModernGoErrorIdentityComparisons(root)
		if err != nil || rewritten != len(symbols) || len(files) != 1 {
			t.Fatalf("proven error rewrites = %d files=%v: %v", rewritten, files, err)
		}
		modernGoAssertCandidateCounts(t, root, []modernGoExpectedCount{{"errors_is", 0}})
	})
}

func TestModernGoJSONTagConformance(t *testing.T) {
	source := `package sample

import "time"

type customZero int
func (value customZero) IsZero() bool { return value == 1 }
type document struct {
	Count int ` + "`json:\"count,omitempty\"`" + `
	Name string ` + "`json:\"name,omitempty\"`" + `
	Ready bool ` + "`json:\"ready,omitempty\"`" + `
	Custom customZero ` + "`json:\"custom,omitempty\"`" + `
	Created time.Time ` + "`json:\"created,omitempty\"`" + `
}
`
	root := modernGoTestRepository(t, source)
	modernGoAssertCandidateCounts(t, root, []modernGoExpectedCount{{"json_omitzero", 5}})
	rewritten, files, err := RewriteModernGoOmitZeroEquivalent(root)
	if err != nil {
		t.Fatal(err)
	}
	if rewritten != 3 || len(files) != 1 || files[0] != "internal/sample/sample.go" {
		t.Fatalf("omitzero rewrite = %d edits in %v", rewritten, files)
	}
	modernGoAssertCandidateCounts(t, root, []modernGoExpectedCount{{"json_omitzero", 2}})
	rewritten, files, err = RewriteModernGoOmitZeroEquivalent(root)
	if err != nil {
		t.Fatal(err)
	}
	if rewritten != 0 || len(files) != 0 {
		t.Fatalf("second omitzero rewrite = %d edits in %v", rewritten, files)
	}
}

func TestModernGoJSONTagRejectsInvalidTypeInformation(t *testing.T) {
	typeExpression := ast.NewIdent("unresolvedImportedType")
	field := &ast.Field{
		Type: typeExpression,
		Tag:  &ast.BasicLit{Kind: token.STRING, Value: "`json:\"value,omitempty\"`"},
	}
	info := &types.Info{Types: map[ast.Expr]types.TypeAndValue{
		typeExpression: {Type: types.Typ[types.Invalid]},
	}}
	if modernGoOmitEmptyCandidate(field, info) {
		t.Fatal("invalid type information classified as a JSON omitzero candidate")
	}
}

func TestModernGoExtremaConformance(t *testing.T) {
	source := `package sample

import "math"

func integerMaximum(left, right int) int {
	if right > left { left = right }
	return left
}
func floatMinimum(left, right float64) float64 {
	if right < left { left = right }
	return left
}
func libraryMaximum(left, right float64) float64 { return math.Max(left, right) }
func libraryMinimum(left, right float64) float64 { return math.Min(left, right) }
func sliceMaximum(values []float64) float64 {
	maximum := values[0]
	for _, value := range values[1:] {
		if value > maximum { maximum = value }
	}
	return maximum
}
func zeroFloor(values []int) int {
	maximum := 0
	for _, value := range values {
		if value > maximum { maximum = value // non-negative floor
		}
	}
	return maximum
}
func unequal(left, right int) int {
	if left != right { left = right }
	return left
}
func conditional(enabled bool, left, right int) int {
	if !enabled { left = right } else if right > left { left = right }
	return left
}
func empty(enabled bool) {
	if enabled { // intentionally empty
	}
}
`
	root := modernGoTestRepository(t, source)
	modernGoAssertCandidateCounts(t, root, []modernGoExpectedCount{{"min_max", 7}, {"slices_max_min", 2}})
	rewritten, files, err := RewriteModernGoExtrema(root)
	if err != nil {
		t.Fatal(err)
	}
	if rewritten != 7 || len(files) != 1 || files[0] != "internal/sample/sample.go" {
		t.Fatalf("extrema rewrite = %d edits in %v", rewritten, files)
	}
	modernGoAssertCandidateCounts(t, root, []modernGoExpectedCount{{"min_max", 0}, {"slices_max_min", 0}})
	content, err := os.ReadFile(filepath.Join(root, "internal", "sample", "sample.go"))
	if err != nil {
		t.Fatal(err)
	}
	text := string(content)
	for _, required := range []string{"left = max(left, right)", "left = min(left, right)", "maximum := slices.Max(values)", "maximum = max(maximum, value) // non-negative floor", "else {\n\t\tleft = max(left, right)"} {
		if !strings.Contains(text, required) {
			t.Errorf("extrema rewrite omits %q:\n%s", required, text)
		}
	}
	if strings.Contains(text, `"math"`) || !strings.Contains(text, `"slices"`) || !strings.Contains(text, "left != right") {
		t.Fatalf("extrema rewrite imports or retained branch are wrong:\n%s", text)
	}
	rewritten, files, err = RewriteModernGoExtrema(root)
	if err != nil {
		t.Fatal(err)
	}
	if rewritten != 0 || len(files) != 0 {
		t.Fatalf("second extrema rewrite = %d edits in %v", rewritten, files)
	}
}

func TestModernGoExtremaRewritesExactMathCallsWithoutTypes(t *testing.T) {
	root := modernGoTestRepository(t, `package sample

import "math"

func maximum(left, right float64) float64 {
	missingTypeInformation()
	return math.Max(left, right)
}
func minimum(left, right float64) float64 {
	if right < left { left = right }
	return left
}
`)
	rewritten, files, err := RewriteModernGoExtrema(root)
	if err != nil {
		t.Fatal(err)
	}
	if rewritten != 2 || len(files) != 1 {
		t.Fatalf("untyped exact extrema rewrite = %d edits in %v", rewritten, files)
	}
	content, err := os.ReadFile(filepath.Join(root, "internal", "sample", "sample.go"))
	if err != nil {
		t.Fatal(err)
	}
	if text := string(content); !strings.Contains(text, "return max(left, right)") ||
		!strings.Contains(text, "left = min(left, right)") || strings.Contains(text, `"math"`) {
		t.Fatalf("untyped exact extrema rewrite output:\n%s", text)
	}
}

func TestExtremaNaNSignedZeroAndEmptyPolicy(t *testing.T) {
	nan := math.NaN()
	negativeZero := math.Copysign(0, -1)
	positiveZero := 0.0
	if !math.IsNaN(min(1.0, nan)) || !math.IsNaN(max(1.0, nan)) ||
		!math.IsNaN(slices.Min([]float64{1, nan})) || !math.IsNaN(slices.Max([]float64{1, nan})) {
		t.Fatal("floating extrema did not propagate NaN")
	}
	if !math.Signbit(min(negativeZero, positiveZero)) || math.Signbit(max(negativeZero, positiveZero)) {
		t.Fatal("floating extrema did not select -0 for min and +0 for max")
	}
	operations := []struct {
		name      string
		operation func()
	}{
		{"min", func() { _ = slices.Min([]int{}) }},
		{"max", func() { _ = slices.Max([]int{}) }},
	}
	for _, check := range operations {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("slices.%s accepted empty input", check.name)
				}
			}()
			check.operation()
		}()
	}
}

func TestModernGoHTTPRoutingResolution(t *testing.T) {
	root := modernGoTestRepository(t, `package sample

import "net/http"

func routes(mux *http.ServeMux, handler http.HandlerFunc) {
	mux.HandleFunc("/legacy", handler)
	mux.HandleFunc("GET /modern/{id}", handler)
}
`)
	modernGoAssertCandidateCounts(t, root, []modernGoExpectedCount{{"http_servemux_patterns", 1}})
	census, err := BuildModernGoCensus(root, ModernGoTargetVersion)
	if err != nil {
		t.Fatal(err)
	}
	if adopted := modernGoTestFinding(t, census, "http_servemux_patterns").Adopted; len(adopted) != 1 {
		t.Fatalf("method-aware ServeMux adoption = %+v", adopted)
	}
}

func TestJSONPresenceGoldenCorpus(t *testing.T) {
	type legacy struct {
		Count int    `json:"count,omitzero"`
		Name  string `json:"name,omitzero"`
		Ready bool   `json:"ready,omitzero"`
	}
	type modern struct {
		Count int    `json:"count,omitzero"`
		Name  string `json:"name,omitzero"`
		Ready bool   `json:"ready,omitzero"`
	}
	fixtures := []struct {
		legacy legacy
		modern modern
	}{
		{},
		{legacy{Count: 3, Name: "ready", Ready: true}, modern{Count: 3, Name: "ready", Ready: true}},
	}
	for index, fixture := range fixtures {
		legacyJSON, err := json.Marshal(fixture.legacy)
		if err != nil {
			t.Fatal(err)
		}
		modernJSON, err := json.Marshal(fixture.modern)
		if err != nil {
			t.Fatal(err)
		}
		if string(legacyJSON) != string(modernJSON) {
			t.Fatalf("fixture %d: legacy=%s modern=%s", index, legacyJSON, modernJSON)
		}
	}
}

func TestZeroAndAbsentRemainDistinguishable(t *testing.T) {
	type modern struct {
		Created time.Time `json:"created,omitzero"`
	}
	legacyType := reflect.StructOf([]reflect.StructField{{
		Name: "Created", Type: reflect.TypeFor[time.Time](), Tag: `json:"created,omitempty"`,
	}})
	legacyJSON, err := json.Marshal(reflect.New(legacyType).Elem().Interface())
	if err != nil {
		t.Fatal(err)
	}
	modernJSON, err := json.Marshal(modern{})
	if err != nil {
		t.Fatal(err)
	}
	if string(legacyJSON) == string(modernJSON) || string(modernJSON) != `{}` {
		t.Fatalf("zero struct presence: legacy=%s modern=%s", legacyJSON, modernJSON)
	}
}

func TestWaitGroupLaunchOrdering(t *testing.T) {
	var wait sync.WaitGroup
	started := make(chan struct{})
	release := make(chan struct{})
	wait.Go(func() {
		close(started)
		<-release
	})
	<-started
	close(release)
	wait.Wait()
}

func TestWaitGroupPanicAndCancellationContract(t *testing.T) {
	ctx, cancel := context.WithCancelCause(t.Context())
	var wait sync.WaitGroup
	observed := make(chan error, 1)
	wait.Go(func() {
		<-ctx.Done()
		observed <- ctx.Err()
	})
	cancel(context.Canceled)
	wait.Wait()
	if err := <-observed; !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation = %v", err)
	}

	marker := &struct{}{}
	recovered := make(chan any, 1)
	wait.Go(func() {
		defer func() { recovered <- recover() }()
		panic(marker)
	})
	wait.Wait()
	if value := <-recovered; value != marker {
		t.Fatalf("recovered panic = %v", value)
	}
}

type modernGoExpectedCount struct {
	rule  string
	count int
}

func modernGoAssertCandidateCounts(t *testing.T, root string, expectedCounts []modernGoExpectedCount) {
	t.Helper()
	census, err := BuildModernGoCensus(root, ModernGoTargetVersion)
	if err != nil {
		t.Fatal(err)
	}
	for _, expected := range expectedCounts {
		if candidates := modernGoTestFinding(t, census, expected.rule).Candidates; len(candidates) != expected.count {
			t.Errorf("%s candidates = %+v, want %d", expected.rule, candidates, expected.count)
		}
	}
}

func modernGoAssertSourceCandidateCounts(t *testing.T, source string, expectedCounts []modernGoExpectedCount) {
	t.Helper()
	modernGoAssertCandidateCounts(t, modernGoTestRepository(t, source), expectedCounts)
}

func modernGoAssertTestSourceCandidateCounts(t *testing.T, source string, expectedCounts []modernGoExpectedCount) {
	t.Helper()
	modernGoAssertCandidateCounts(t, modernGoTestRepositoryFile(t, "sample_test.go", source), expectedCounts)
}

func modernGoTestRepository(t *testing.T, source string) string {
	t.Helper()
	return modernGoTestRepositoryFile(t, "sample.go", source)
}

func modernGoTestRepositoryFile(t *testing.T, name, source string) string {
	return modernGoTestRepositoryPath(t, filepath.Join("internal", "sample", name), source)
}

func modernGoTestRepositoryPath(t *testing.T, name, source string) string {
	t.Helper()
	root := t.TempDir()
	fullName := filepath.Join(root, filepath.FromSlash(name))
	directory := filepath.Dir(fullName)
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module example\n\ngo 1.26\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(fullName, []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}

func modernGoTestFinding(t *testing.T, census ModernGoCensus, id string) ModernGoFinding {
	t.Helper()
	for _, finding := range census.Findings {
		if finding.ID == id {
			return finding
		}
	}
	t.Fatalf("finding %s is absent", id)
	return ModernGoFinding{}
}
