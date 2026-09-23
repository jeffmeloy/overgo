package plan

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// reviewID names evidence by the digest of a label; the review compares
// identities, never their bytes.
func reviewID(t *testing.T, kind artifact.Kind, text string) artifact.ID {
	digest := sha256.Sum256([]byte(text))
	id, err := artifact.ParseID(strings.ToLower(kind.String()) + ":sha256:" + hex.EncodeToString(digest[:]))
	if err != nil {
		t.Fatalf("review identity %q: %v", text, err)
	}
	return id
}

// TestPostRowOptimizationReview binds the review's four derivations and the
// disposition rule: the costliest package and every skipped fixture from the
// suite-cost evidence, the schema that grew the store most by bytes and by
// count, a moved harness surface, and dispatch refusal until each key has a
// row or a reason -- never both, never neither, and a row must exist.
func TestPostRowOptimizationReview(t *testing.T) {
	t.Parallel()
	evidence := reviewID(t, artifact.KindEvidence, "suite cost")
	slow, fast := 40.25, 12.5
	record, err := json.Marshal(map[string]any{
		"result": reviewID(t, artifact.KindEvidence, "gate result").String(),
		"invocations": []map[string]any{
			{"step": "test-short", "wall_ns": 1, "executions": []map[string]any{
				{"package": "overgo/internal/fast", "action": "pass", "started": true, "elapsed_seconds": fast},
			}},
			{"step": "test-rest", "wall_ns": 2, "executions": []map[string]any{
				{"package": "overgo/internal/slow", "action": "pass", "started": true, "elapsed_seconds": slow},
				{"package": "overgo/internal/unmeasured", "action": "pass", "started": true},
			}, "skipped": []string{"overgo/cmd/generate: TestDecodeVideoFileFFmpeg", "overgo/cmd/generate: TestEncodeVideoFileFFmpeg"}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	fromCost, err := SuiteCostCandidates(record, evidence)
	if err != nil {
		t.Fatal(err)
	}
	if len(fromCost) != 2 || fromCost[0].Key != "suite-cost:overgo/internal/slow" || !strings.Contains(fromCost[0].Measure, "40.25s") ||
		!strings.Contains(fromCost[0].Measure, "test-rest") || fromCost[0].Evidence != evidence ||
		fromCost[1].Key != "skipped-fixture:test-rest:overgo/cmd/generate" || !strings.HasPrefix(fromCost[1].Measure, "2 fixtures") {
		t.Fatalf("suite cost candidates = %+v", fromCost)
	}

	big := artifact.Descriptor{ID: reviewID(t, artifact.KindEvidence, "big"), Size: 100, Schema: "test/big/v1"}
	small := artifact.Descriptor{ID: reviewID(t, artifact.KindEvidence, "small"), Size: 5, Schema: "test/small/v1"}
	other := artifact.Descriptor{ID: reviewID(t, artifact.KindEvidence, "other"), Size: 5, Schema: "test/small/v1"}
	declared := artifact.Descriptor{ID: reviewID(t, artifact.KindEvidence, "declared only"), Size: 1000, Schema: "test/declared/v1"}
	growth := StoreGrowthCandidates(overgodb.HeadBoundDelta{
		PreviousSequence: 10, Sequence: 12,
		Artifacts: []artifact.Descriptor{big, small, other, declared},
		Contents:  []artifact.ID{big.ID, small.ID, other.ID},
	})
	if len(growth) != 2 || growth[0].Key != "store-growth:bytes:test/big/v1" || !strings.Contains(growth[0].Measure, "100 bytes over 1 documents, store commits 11..12") ||
		growth[1].Key != "store-growth:count:test/small/v1" || !strings.Contains(growth[1].Measure, "2 documents, 10 bytes") {
		t.Fatalf("store growth candidates = %+v", growth)
	}
	if StoreGrowthCandidates(overgodb.HeadBoundDelta{Artifacts: []artifact.Descriptor{declared}}) != nil {
		t.Fatal("declared-only artifacts counted as growth")
	}

	moves, err := SurfaceMoveCandidates([]byte(`{"production_nodes":181732}`), []byte(`{"production_nodes":181809,"packages":9}`), "8527831950fa")
	if err != nil || len(moves) != 1 || moves[0].Key != "surface-move:production-nodes:8527831950fa" || moves[0].Measure != "+77 production nodes (181732 -> 181809)" {
		t.Fatalf("surface move candidates = (%+v, %v)", moves, err)
	}
	for _, after := range []string{`{"production_nodes":5}`, `{"production_nodes":4}`} {
		if held, err := SurfaceMoveCandidates([]byte(`{"production_nodes":5}`), []byte(after), "8527831950fa"); err != nil || held != nil {
			t.Fatalf("held or reduced surface %s = (%+v, %v)", after, held, err)
		}
	}

	document := Plan{Items: []Item{{ID: "slow-suite-split", Status: StatusOpen, Steps: []Step{{ID: "do", Status: StatusOpen}}}}}
	candidates := slices.Concat(fromCost, growth, moves)
	if pending := UnreviewedCandidates(candidates, document); len(pending) != len(candidates) {
		t.Fatalf("unreviewed before any disposition = %d, want %d", len(pending), len(candidates))
	}
	if _, err := RecordDisposition(document, OptimizationDisposition{Key: fromCost[0].Key, Row: "no-such-row"}); err == nil {
		t.Fatal("disposition naming an absent row was recorded")
	}
	if _, err := RecordDisposition(document, OptimizationDisposition{Key: fromCost[0].Key, Row: "slow-suite-split", Reason: "both"}); err == nil {
		t.Fatal("disposition with both a row and a reason was recorded")
	}
	if _, err := RecordDisposition(document, OptimizationDisposition{Key: fromCost[0].Key}); err == nil {
		t.Fatal("disposition with neither a row nor a reason was recorded")
	}
	document, err = RecordDisposition(document, OptimizationDisposition{Key: fromCost[0].Key, Row: "slow-suite-split"})
	if err != nil {
		t.Fatal(err)
	}
	for _, candidate := range candidates[1:] {
		if document, err = RecordDisposition(document, OptimizationDisposition{Key: candidate.Key, Reason: "measured, below the ratchet"}); err != nil {
			t.Fatal(err)
		}
	}
	if pending := UnreviewedCandidates(candidates, document); len(pending) != 0 {
		t.Fatalf("unreviewed after dispositions = %+v", pending)
	}
	if !slices.IsSortedFunc(document.Reviews, func(left, right OptimizationDisposition) int { return strings.Compare(left.Key, right.Key) }) {
		t.Fatalf("dispositions are not canonical: %+v", document.Reviews)
	}
	if err := validateReviews(append(slices.Clone(document.Reviews), document.Reviews[0])); err == nil {
		t.Fatal("repeated disposition key validated")
	}
}

// TestReviewKeysAreAnswerable holds a lane that skips hundreds of fixtures to
// one candidate per step and package, carrying the count as its measure, in
// a deterministic order after the costliest package.
func TestReviewKeysAreAnswerable(t *testing.T) {
	t.Parallel()
	var deviceSkips []string
	for index := range 302 {
		deviceSkips = append(deviceSkips, "overgo/internal/cuda/executor: TestKernel"+strings.Repeat("x", index%7)+string(rune('A'+index%26)))
	}
	deviceSkips = append(deviceSkips, "overgo/internal/inference: TestDecodeDevice", "overgo/internal/inference: TestPrefillDevice")
	elapsed := 3.5
	record, err := json.Marshal(map[string]any{
		"result": reviewID(t, artifact.KindEvidence, "gate result").String(),
		"invocations": []map[string]any{
			{"step": "test-device", "executions": []map[string]any{
				{"package": "overgo/internal/cuda/executor", "action": "pass", "started": true, "elapsed_seconds": elapsed},
			}, "skipped": deviceSkips},
			{"step": "test", "skipped": []string{"overgo/cmd/generate: TestDecodeVideoFileFFmpeg"}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	candidates, err := SuiteCostCandidates(record, reviewID(t, artifact.KindEvidence, "suite cost"))
	if err != nil {
		t.Fatal(err)
	}
	var keys, measures []string
	for _, candidate := range candidates {
		keys = append(keys, candidate.Key)
		measures = append(measures, candidate.Measure)
	}
	wantKeys := []string{
		"suite-cost:overgo/internal/cuda/executor",
		"skipped-fixture:test-device:overgo/internal/cuda/executor",
		"skipped-fixture:test-device:overgo/internal/inference",
		"skipped-fixture:test:overgo/cmd/generate",
	}
	if !slices.Equal(keys, wantKeys) || !strings.HasPrefix(measures[1], "302 fixtures") ||
		!strings.HasPrefix(measures[2], "2 fixtures") || !strings.HasPrefix(measures[3], "1 fixtures") {
		t.Fatalf("grouped candidates = %v / %v", keys, measures)
	}
}

// TestOptimizationReviewMerge unions dispositions three ways: the side that
// changed a key wins, a key dropped by one side and kept unchanged by the
// other is dropped, and two different edits of one key conflict.
func TestOptimizationReviewMerge(t *testing.T) {
	t.Parallel()
	rowA := OptimizationDisposition{Key: "suite-cost:a", Row: "row-a"}
	rowB := OptimizationDisposition{Key: "suite-cost:a", Row: "row-b"}
	reason := OptimizationDisposition{Key: "store-growth:bytes:x", Reason: "below the ratchet"}
	kept := OptimizationDisposition{Key: "surface-move:production-nodes:8527831950fa", Row: "surface-move-preflight"}
	merged, err := mergeReviews(
		[]OptimizationDisposition{rowA, kept},
		[]OptimizationDisposition{rowA, kept, reason},
		[]OptimizationDisposition{rowB},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(merged, []OptimizationDisposition{rowB, reason}) {
		t.Fatalf("merged reviews = %+v", merged)
	}
	if _, err := mergeReviews([]OptimizationDisposition{rowA}, []OptimizationDisposition{rowB}, []OptimizationDisposition{{Key: "suite-cost:a", Reason: "elsewhere"}}); err == nil {
		t.Fatal("two different edits of one key merged")
	}
}
