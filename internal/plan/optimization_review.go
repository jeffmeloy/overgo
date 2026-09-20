package plan

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

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

// mergeReviews unions dispositions by key: a side that changed a key wins
// over one that kept it, a key both sides changed differently conflicts,
// and a key one side dropped while the other kept it unchanged is dropped.
func mergeReviews(base, local, upstream []OptimizationDisposition) ([]OptimizationDisposition, error) {
	key := func(review OptimizationDisposition) string { return review.Key }
	baseByKey, localByKey, upstreamByKey := indexByID(base, key), indexByID(local, key), indexByID(upstream, key)
	var merged []OptimizationDisposition
	for _, id := range unionOrder(key, base, local, upstream) {
		baseReview, inBase := baseByKey[id]
		localReview, inLocal := localByKey[id]
		upstreamReview, inUpstream := upstreamByKey[id]
		switch {
		case inLocal && inUpstream:
			switch {
			case localReview == upstreamReview, inBase && upstreamReview == baseReview:
				merged = append(merged, localReview)
			case inBase && localReview == baseReview:
				merged = append(merged, upstreamReview)
			default:
				return nil, fmt.Errorf("plan projection: optimization review %q edited on both sides", id)
			}
		case inLocal:
			if !inBase || localReview != baseReview {
				merged = append(merged, localReview)
			}
		case inUpstream:
			if !inBase || upstreamReview != baseReview {
				merged = append(merged, upstreamReview)
			}
		}
	}
	return merged, nil
}

// UnreviewedCandidates returns the candidates whose key the plan has not
// dispositioned.
func UnreviewedCandidates(candidates []OptimizationCandidate, document Plan) []OptimizationCandidate {
	reviewed := map[string]bool{}
	for _, review := range document.Reviews {
		reviewed[review.Key] = true
	}
	return slices.DeleteFunc(slices.Clone(candidates), func(candidate OptimizationCandidate) bool {
		return reviewed[candidate.Key]
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
// and every fixture the landing skipped without credit.
func SuiteCostCandidates(data []byte, evidence artifact.ID) ([]OptimizationCandidate, error) {
	var record suiteCostRecord
	if err := json.Unmarshal(data, &record); err != nil {
		return nil, fmt.Errorf("plan: decode suite cost evidence: %w", err)
	}
	// Invocations and executions are recorded in order, so the first
	// costliest wins deterministically without a tie-break.
	var costliest testevidence.PackageExecution
	costliestStep := ""
	var candidates []OptimizationCandidate
	skipped := map[string]bool{}
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
			if skipped[fixture] {
				continue
			}
			skipped[fixture] = true
			candidates = append(candidates, OptimizationCandidate{
				Kind: CandidateSkippedFixture, Key: CandidateSkippedFixture + ":" + fixture,
				Measure: "skipped without evidence credit in " + step, Evidence: evidence,
			})
		}
	}
	if costliest.Elapsed != nil {
		candidates = append([]OptimizationCandidate{{
			Kind: CandidateSuiteCost, Key: CandidateSuiteCost + ":" + costliest.Package,
			Measure:  fmt.Sprintf("%s in %s", (time.Duration(*costliest.Elapsed * float64(time.Second))).Round(time.Millisecond), costliestStep),
			Evidence: evidence,
		}}, candidates...)
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
	top := func(better func(left, right string) bool) string {
		selected := ""
		first := true
		for schema := range countBySchema {
			if first || better(schema, selected) {
				selected, first = schema, false
			}
		}
		return selected
	}
	byBytes := top(func(left, right string) bool {
		return bytesBySchema[left] > bytesBySchema[right] || bytesBySchema[left] == bytesBySchema[right] && left < right
	})
	byCount := top(func(left, right string) bool {
		return countBySchema[left] > countBySchema[right] || countBySchema[left] == countBySchema[right] && left < right
	})
	window := fmt.Sprintf("store commits %d..%d", delta.PreviousSequence+1, delta.Sequence)
	label := func(schema string) string { return cmp.Or(schema, "(schemaless)") }
	candidates := []OptimizationCandidate{{
		Kind: CandidateStoreGrowth, Key: CandidateStoreGrowth + ":bytes:" + label(byBytes),
		Measure: fmt.Sprintf("%d bytes over %d documents, %s", bytesBySchema[byBytes], countBySchema[byBytes], window),
	}}
	if byCount != byBytes {
		candidates = append(candidates, OptimizationCandidate{
			Kind: CandidateStoreGrowth, Key: CandidateStoreGrowth + ":count:" + label(byCount),
			Measure: fmt.Sprintf("%d documents, %d bytes, %s", countBySchema[byCount], bytesBySchema[byCount], window),
		})
	}
	return candidates
}

// harnessSurfaceBaselinePath is the committed harness surface baseline.
const harnessSurfaceBaselinePath = "docs/harness_surface_baseline.json"

// SurfaceMoveCandidates compares the harness surface baseline before and
// after a landing and names a moved production surface. The key carries the
// landing, so growth is answered every time it happens: the surface is to be
// reduced or held, and a landing that grows it names the paydown.
func SurfaceMoveCandidates(before, after []byte, commit string) ([]OptimizationCandidate, error) {
	type baseline struct {
		ProductionNodes int64 `json:"production_nodes"`
	}
	var previous, current baseline
	if err := json.Unmarshal(before, &previous); err != nil {
		return nil, fmt.Errorf("plan: decode harness baseline before the landing: %w", err)
	}
	if err := json.Unmarshal(after, &current); err != nil {
		return nil, fmt.Errorf("plan: decode harness baseline after the landing: %w", err)
	}
	if previous.ProductionNodes == current.ProductionNodes {
		return nil, nil
	}
	return []OptimizationCandidate{{
		Kind: CandidateSurfaceMove, Key: CandidateSurfaceMove + ":production-nodes:" + commit,
		Measure: fmt.Sprintf("%+d production nodes (%d -> %d)", current.ProductionNodes-previous.ProductionNodes, previous.ProductionNodes, current.ProductionNodes),
	}}, nil
}

// ReviewLandedCompletion derives the optimization candidates of the
// completion revision names. A revision that is not a prepared completion,
// or whose successful attempt the store does not hold, yields none: the
// completion authority judges that, not the review.
func ReviewLandedCompletion(ctx context.Context, store *overgodb.Store, repository, revision string) ([]OptimizationCandidate, error) {
	if store == nil {
		return nil, errors.New("plan: optimization review requires the store")
	}
	repository, commit, err := resolveCompletionRevision(ctx, repository, revision)
	if err != nil {
		return nil, err
	}
	message, err := gitauthority.Query(ctx, repository, "log", "-1", "--format=%B", "--end-of-options", commit)
	if err != nil {
		return nil, err
	}
	trailers, hasCompletion, err := parseCompletionTrailers(string(message))
	if err != nil {
		return nil, fmt.Errorf("plan: optimization review of %.12s: %w", commit, err)
	}
	if !hasCompletion || !trailers.preparation.Valid() {
		return nil, nil
	}
	attempts, err := runrecord.AttemptsForPreparation(ctx, store, trailers.preparation)
	if err != nil {
		return nil, err
	}
	index := slices.IndexFunc(attempts, func(attempt runrecord.AttemptRecord) bool {
		return attempt.CodeCommit == commit && attempt.Outcome == runrecord.OutcomeSucceeded
	})
	if index < 0 {
		return nil, nil
	}
	attempt := attempts[index]
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
	growth, err := storeGrowthOfCompletion(ctx, store, trailers.preparation)
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

// FormatCandidates renders candidates one per line for the dispatcher.
func FormatCandidates(candidates []OptimizationCandidate) []string {
	lines := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		lines = append(lines, fmt.Sprintf("review pending: %s %s: %s", candidate.Kind, candidate.Key, candidate.Measure))
	}
	return lines
}
