package plan

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/gitauthority"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
	"overgo/internal/worklease"
)

// Every landed row leaves measured evidence in the store: what the gate's
// tests cost, what the landing added to the store, what it did to the
// harness surface, which fixtures ran without credit. The optimization
// review turns that evidence into typed candidates and makes the re-plan
// answer each one -- with a row or with a reason -- before the next row is
// dispatched, so optimization happens as you go instead of when it hurts.

const (
	// CandidateSuiteCost names the costliest measured test package of a landing.
	CandidateSuiteCost = "suite-cost"
	// CandidateStoreGrowth names the schema that grew the store most.
	CandidateStoreGrowth = "store-growth"
	// CandidateSurfaceMove names a moved harness production surface.
	CandidateSurfaceMove = "surface-move"
	// CandidateSkippedFixture names a fixture test that ran without credit.
	CandidateSkippedFixture = "skipped-fixture"
)

// OptimizationCandidate is one measured robustness or efficiency lever a
// landed row's gate evidence names. Key is stable across landings -- the
// package, the schema, the fixture -- so one disposition answers it until
// a different subject tops the measure.
type OptimizationCandidate struct {
	Kind     string      `json:"kind"`
	Key      string      `json:"key"`
	Measure  string      `json:"measure"`
	Evidence artifact.ID `json:"evidence,omitzero"`
}

// OptimizationDisposition is the re-plan's answer to one candidate key: the
// row that addresses it, or the reason none is filed. Exactly one is set.
type OptimizationDisposition struct {
	Key    string `json:"key"`
	Row    string `json:"row,omitzero"`
	Reason string `json:"reason,omitzero"`
}

func validateReviews(reviews []OptimizationDisposition) error {
	seen := map[string]bool{}
	for _, review := range reviews {
		if review.Key == "" || !validAutomationDetail(review.Key) || seen[review.Key] {
			return fmt.Errorf("plan: optimization review key %q is invalid or repeated", review.Key)
		}
		seen[review.Key] = true
		if (review.Row == "") == (review.Reason == "") {
			return fmt.Errorf("plan: optimization review %q needs exactly one of a row or a reason", review.Key)
		}
		if review.Row != "" && !worklease.ValidPlanID(review.Row) {
			return fmt.Errorf("plan: optimization review %q names an invalid row %q", review.Key, review.Row)
		}
		if review.Reason != "" && !validAutomationDetail(review.Reason) {
			return fmt.Errorf("plan: optimization review %q has an invalid reason", review.Key)
		}
	}
	return nil
}

// mergeReviews unions dispositions by key under the plan's three-way rule;
// an absent disposition is the zero value, so a drop merges like an edit.
func mergeReviews(base, local, upstream []OptimizationDisposition) ([]OptimizationDisposition, error) {
	key := func(review OptimizationDisposition) string { return review.Key }
	baseByKey, localByKey, upstreamByKey := indexByID(base, key), indexByID(local, key), indexByID(upstream, key)
	var merged []OptimizationDisposition
	for _, id := range unionOrder(key, base, local, upstream) {
		review, err := mergeText("optimization review "+id, baseByKey[id], localByKey[id], upstreamByKey[id])
		if err != nil {
			return nil, err
		}
		if review != (OptimizationDisposition{}) {
			merged = append(merged, review)
		}
	}
	return merged, nil
}

// UnreviewedCandidates returns the candidates whose key the plan has not
// dispositioned.
func UnreviewedCandidates(candidates []OptimizationCandidate, document Plan) []OptimizationCandidate {
	return slices.DeleteFunc(slices.Clone(candidates), func(candidate OptimizationCandidate) bool {
		return slices.ContainsFunc(document.Reviews, func(review OptimizationDisposition) bool { return review.Key == candidate.Key })
	})
}

// RecordDisposition returns the plan with one disposition recorded; a row
// disposition must name an item the plan holds.
func RecordDisposition(document Plan, review OptimizationDisposition) (Plan, error) {
	if review.Row != "" && !slices.ContainsFunc(document.Items, func(item Item) bool { return item.ID == review.Row }) {
		return Plan{}, fmt.Errorf("plan: optimization review %q names no plan item %q", review.Key, review.Row)
	}
	document.Reviews = slices.Clone(document.Reviews)
	document.Reviews = slices.DeleteFunc(document.Reviews, func(existing OptimizationDisposition) bool { return existing.Key == review.Key })
	document.Reviews = append(document.Reviews, review)
	slices.SortFunc(document.Reviews, func(left, right OptimizationDisposition) int { return strings.Compare(left.Key, right.Key) })
	if err := validateReviews(document.Reviews); err != nil {
		return Plan{}, err
	}
	return document, nil
}

// suiteCostRecord is the plan's reading of the gate's suite-cost document:
// the invocations with their package executions and skipped fixtures.
type suiteCostRecord struct {
	Result      artifact.ID `json:"result"`
	Invocations []struct {
		Step       string                          `json:"step"`
		WallNS     uint64                          `json:"wall_ns"`
		Executions []testevidence.PackageExecution `json:"executions"`
		Skipped    []string                        `json:"skipped"`
	} `json:"invocations"`
}

// SuiteCostCandidates names the costliest measured package of the landing
// and, per step and package, the fixtures that ran without credit.
func SuiteCostCandidates(data []byte, evidence artifact.ID) ([]OptimizationCandidate, error) {
	var record suiteCostRecord
	if err := json.Unmarshal(data, &record); err != nil {
		return nil, fmt.Errorf("plan: decode suite cost evidence: %w", err)
	}
	// Invocations and executions are recorded in order, so the first
	// costliest wins deterministically without a tie-break.
	var costliest testevidence.PackageExecution
	costliestStep := ""
	// Skipped fixtures group by step and package: a lane that skips
	// hundreds of tests is a handful of answerable subjects, not hundreds.
	skipped := map[string]int{}
	for _, invocation := range record.Invocations {
		step := cmp.Or(invocation.Step, "the test phase")
		for _, execution := range invocation.Executions {
			if execution.Elapsed == nil || !execution.Started {
				continue
			}
			if costliest.Elapsed == nil || *execution.Elapsed > *costliest.Elapsed {
				costliest, costliestStep = execution, step
			}
		}
		for _, fixture := range invocation.Skipped {
			owner, _, _ := strings.Cut(fixture, ": ")
			skipped[step+":"+owner]++
		}
	}
	var candidates []OptimizationCandidate
	if costliest.Elapsed != nil {
		candidates = append(candidates, OptimizationCandidate{
			Kind: CandidateSuiteCost, Key: CandidateSuiteCost + ":" + costliest.Package,
			Measure:  fmt.Sprintf("%gs in %s", *costliest.Elapsed, costliestStep),
			Evidence: evidence,
		})
	}
	for _, group := range slices.Sorted(maps.Keys(skipped)) {
		candidates = append(candidates, OptimizationCandidate{
			Kind: CandidateSkippedFixture, Key: CandidateSkippedFixture + ":" + group,
			Measure: fmt.Sprintf("%d fixtures skipped without evidence credit", skipped[group]), Evidence: evidence,
		})
	}
	return candidates, nil
}

// StoreGrowthCandidates names the schema that added the most durable bytes
// and the schema that added the most documents across the landing's store
// commits.
func StoreGrowthCandidates(delta overgodb.HeadBoundDelta) []OptimizationCandidate {
	descriptors := map[artifact.ID]artifact.Descriptor{}
	for _, descriptor := range delta.Artifacts {
		descriptors[descriptor.ID] = descriptor
	}
	bytesBySchema := map[string]int64{}
	countBySchema := map[string]int{}
	for _, id := range delta.Contents {
		descriptor := descriptors[id]
		bytesBySchema[descriptor.Schema] += int64(descriptor.Size)
		countBySchema[descriptor.Schema]++
	}
	if len(countBySchema) == 0 {
		return nil
	}
	// Sorted schemas make the first maximum the deterministic winner.
	schemas := slices.Sorted(maps.Keys(countBySchema))
	byBytes := slices.MaxFunc(schemas, func(left, right string) int { return cmp.Compare(bytesBySchema[left], bytesBySchema[right]) })
	byCount := slices.MaxFunc(schemas, func(left, right string) int { return cmp.Compare(countBySchema[left], countBySchema[right]) })
	window := fmt.Sprintf("store commits %d..%d", delta.PreviousSequence+1, delta.Sequence)
	candidates := []OptimizationCandidate{{
		Kind: CandidateStoreGrowth, Key: CandidateStoreGrowth + ":bytes:" + cmp.Or(byBytes, "(schemaless)"),
		Measure: fmt.Sprintf("%d bytes over %d documents, %s", bytesBySchema[byBytes], countBySchema[byBytes], window),
	}}
	if byCount != byBytes {
		candidates = append(candidates, OptimizationCandidate{
			Kind: CandidateStoreGrowth, Key: CandidateStoreGrowth + ":count:" + cmp.Or(byCount, "(schemaless)"),
			Measure: fmt.Sprintf("%d documents, %d bytes, %s", countBySchema[byCount], bytesBySchema[byCount], window),
		})
	}
	return candidates
}

// harnessSurfaceBaselinePath is the committed harness surface baseline.
const harnessSurfaceBaselinePath = "docs/harness_surface_baseline.json"

// SurfaceMoveCandidates compares the harness surface baseline before and
// after a landing and names a grown production surface. The key carries the
// landing, so growth is answered every time it happens: the surface is to be
// reduced or held, and a landing that grows it names the paydown.
func SurfaceMoveCandidates(before, after []byte, commit string) ([]OptimizationCandidate, error) {
	var nodes [2]struct {
		Count int64 `json:"production_nodes"`
	}
	for index, data := range [][]byte{before, after} {
		if err := json.Unmarshal(data, &nodes[index]); err != nil {
			return nil, fmt.Errorf("plan: decode harness baseline around the landing: %w", err)
		}
	}
	previous, current := nodes[0].Count, nodes[1].Count
	// Only growth asks for an answer: the surface is to be reduced or held.
	if current <= previous {
		return nil, nil
	}
	return []OptimizationCandidate{{
		Kind: CandidateSurfaceMove, Key: CandidateSurfaceMove + ":production-nodes:" + commit,
		Measure: fmt.Sprintf("%+d production nodes (%d -> %d)", current-previous, previous, current),
	}}, nil
}

// ReviewLandedCompletion derives the optimization candidates of the
// completion revision names. A revision that is not a prepared completion,
// or whose successful attempt the store does not hold, yields none: the
// completion authority judges that, not the review.
func ReviewLandedCompletion(ctx context.Context, store *overgodb.Store, repository, revision string) ([]OptimizationCandidate, error) {
	landed, found, err := readLandedCompletion(ctx, store, repository, revision)
	if err != nil || !found {
		return nil, err
	}
	repository, commit, attempt := landed.repository, landed.commit, landed.attempt
	var candidates []OptimizationCandidate
	children, err := store.Children(ctx, attempt.Result)
	if err != nil {
		return nil, err
	}
	for _, edge := range children {
		if edge.Parent != attempt.Result || edge.Relation != artifact.RelationDependsOn {
			continue
		}
		content, found, readErr := artifact.ReadContent(ctx, store, edge.Child)
		if readErr != nil {
			return nil, readErr
		}
		if !found || content.Descriptor.Schema != runrecord.SuiteCostSchema {
			continue
		}
		fromCost, costErr := SuiteCostCandidates(content.Data, edge.Child)
		if costErr != nil {
			return nil, costErr
		}
		candidates = append(candidates, fromCost...)
	}
	growth, err := storeGrowthOfCompletion(ctx, store, landed.preparation)
	if err != nil {
		return nil, err
	}
	candidates = append(candidates, growth...)
	before, beforeErr := gitauthority.Query(ctx, repository, "show", "--end-of-options", commit+"^:"+harnessSurfaceBaselinePath)
	after, afterErr := gitauthority.Query(ctx, repository, "show", "--end-of-options", commit+":"+harnessSurfaceBaselinePath)
	if beforeErr == nil && afterErr == nil {
		moves, moveErr := SurfaceMoveCandidates(before, after, commit)
		if moveErr != nil {
			return nil, moveErr
		}
		candidates = append(candidates, moves...)
	}
	return candidates, nil
}

// landedCompletion is a completion commit and the successful gate attempt
// that landed it.
type landedCompletion struct {
	repository, commit string
	preparation        artifact.ID
	attempt            runrecord.AttemptRecord
}

// readLandedCompletion resolves revision to the successful attempt of its
// prepared completion; a revision that is not one, or whose attempt the
// store does not hold, is not found.
func readLandedCompletion(ctx context.Context, store *overgodb.Store, repository, revision string) (landedCompletion, bool, error) {
	if store == nil {
		return landedCompletion{}, false, errors.New("plan: a landed completion is read from the store")
	}
	repository, commit, err := resolveCompletionRevision(ctx, repository, revision)
	if err != nil {
		return landedCompletion{}, false, err
	}
	message, err := gitauthority.Query(ctx, repository, "log", "-1", "--format=%B", "--end-of-options", commit)
	if err != nil {
		return landedCompletion{}, false, err
	}
	trailers, hasCompletion, err := parseCompletionTrailers(string(message))
	if err != nil {
		return landedCompletion{}, false, fmt.Errorf("plan: landed completion %.12s: %w", commit, err)
	}
	if !hasCompletion || !trailers.preparation.Valid() {
		return landedCompletion{}, false, nil
	}
	attempts, err := runrecord.AttemptsForPreparation(ctx, store, trailers.preparation)
	if err != nil {
		return landedCompletion{}, false, err
	}
	index := slices.IndexFunc(attempts, func(attempt runrecord.AttemptRecord) bool {
		return attempt.CodeCommit == commit && attempt.Outcome == runrecord.OutcomeSucceeded
	})
	if index < 0 {
		return landedCompletion{}, false, nil
	}
	return landedCompletion{repository: repository, commit: commit, preparation: trailers.preparation, attempt: attempts[index]}, true, nil
}

// LandedGateResult reads the gate result that landed revision, when revision
// is a prepared completion whose successful attempt the store holds.
func LandedGateResult(ctx context.Context, store *overgodb.Store, repository, revision string) (runrecord.GateResult, bool, error) {
	landed, found, err := readLandedCompletion(ctx, store, repository, revision)
	if err != nil || !found {
		return runrecord.GateResult{}, false, err
	}
	result, err := runrecord.RequireGateResult(ctx, store, landed.attempt.Result)
	return result, err == nil, err
}

// storeGrowthOfCompletion measures what the store gained between the gate's
// preparation and its finalization: exactly the landing's own commits.
func storeGrowthOfCompletion(ctx context.Context, store *overgodb.Store, preparation artifact.ID) ([]OptimizationCandidate, error) {
	start, found, err := store.ArtifactIntroduction(ctx, preparation)
	if err != nil || !found {
		return nil, err
	}
	finalization, found, err := runrecord.GateFinalizationForPreparation(ctx, store, preparation)
	if err != nil || !found {
		return nil, err
	}
	end, found, err := store.ArtifactIntroduction(ctx, finalization.ID)
	if err != nil || !found || end.Sequence <= start.Sequence {
		return nil, err
	}
	delta, resync, err := store.DeltasSince(ctx, start.Commit, start.Sequence, overgodb.ProjectionContractVersion(), int(end.Sequence-start.Sequence))
	if err != nil || resync {
		return nil, err
	}
	return StoreGrowthCandidates(delta), nil
}
