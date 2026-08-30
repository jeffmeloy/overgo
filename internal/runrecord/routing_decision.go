package runrecord

import (
	"cmp"
	"context"
	"errors"
	"math/bits"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/invocation"
	"overgo/internal/textcheck"
)

const (
	routingDecisionMediaType = "application/vnd.overgo.routing-decision+json"
	routingDecisionSchema    = "overgo/routing-decision/v1"
)

// RouteDisposition is the closed outcome of evidence-based capability routing.
type RouteDisposition string

const (
	// RouteSelected means one unique evidence-dominant capability was selected.
	RouteSelected RouteDisposition = "selected"
	// RouteDecisionRequired means the eligible frontier is incomparable.
	RouteDecisionRequired RouteDisposition = "decision-required"
	// RouteRefused means no capability is eligible for the declared intent.
	RouteRefused RouteDisposition = "refused"
)

// RouteRejection explains why a considered capability was outside the eligible frontier.
type RouteRejection string

const (
	// RouteEligible means a candidate may enter the evidence frontier.
	RouteEligible RouteRejection = "eligible"
	// RouteUnauthorized means the active authority excludes a candidate.
	RouteUnauthorized RouteRejection = "unauthorized"
	// RouteBoundaryDisallowed means the declared transport boundary excludes a candidate.
	RouteBoundaryDisallowed RouteRejection = "boundary-disallowed"
	// RouteFailed means typed terminal evidence excludes a candidate.
	RouteFailed RouteRejection = "failed"
)

func (value RouteRejection) valid() bool {
	return slices.Contains([]RouteRejection{RouteEligible, RouteUnauthorized, RouteBoundaryDisallowed, RouteFailed}, value)
}

// RouteMeasurement carries exact comparable observations. Ratios use Attempts
// as their denominator, so selection needs no floating point weights.
type RouteMeasurement struct {
	Attempts      uint64 `json:"attempts"`
	Succeeded     uint64 `json:"succeeded"`
	EvidenceUnits uint64 `json:"evidence_units"`
	CostUnits     uint64 `json:"cost_units"`
	CostObserved  bool   `json:"cost_observed,omitzero"`
	LatencyNS     uint64 `json:"latency_ns"`
}

// RoutingCandidateObservation records every considered exact execution selection.
type RoutingCandidateObservation struct {
	Capability  artifact.ID         `json:"capability"`
	Manual      artifact.ID         `json:"manual"`
	Selection   artifact.ID         `json:"selection"`
	Probe       artifact.ID         `json:"probe"`
	Boundary    invocation.Boundary `json:"boundary"`
	Effect      invocation.Class    `json:"effect"`
	Head        artifact.CommitID   `json:"head"`
	Rejection   RouteRejection      `json:"rejection"`
	Measurement RouteMeasurement    `json:"measurement"`
}

// RoutingDecision is immutable selection evidence. It owns no execution path.
type RoutingDecision struct {
	Version     uint16                        `json:"version"`
	Intent      string                        `json:"intent"`
	Effect      invocation.Class              `json:"effect"`
	Authority   artifact.ID                   `json:"authority"`
	Head        artifact.CommitID             `json:"head"`
	Candidates  []RoutingCandidateObservation `json:"candidates"`
	Disposition RouteDisposition              `json:"disposition"`
	Winner      artifact.ID                   `json:"winner,omitzero"`
	Previous    artifact.ID                   `json:"previous,omitzero"`
	Failure     artifact.ID                   `json:"failure,omitzero"`
	ID          artifact.ID                   `json:"-"`
}

var routingDecisionCodec = artifact.JSONDocumentCodec(
	"routing decision", artifact.KindEvidence, routingDecisionMediaType, routingDecisionSchema,
	canonicalizeRoutingDecision,
	func(value RoutingDecision) artifact.ID { return value.ID },
	func(value *RoutingDecision, id artifact.ID) { value.ID = id },
	cloneRoutingDecision,
)

// NewRoutingDecision validates and identifies one route decision.
func NewRoutingDecision(value RoutingDecision) (RoutingDecision, error) {
	return routingDecisionCodec.NewPrepared(value, func(value *RoutingDecision) {
		value.Version, value.ID = artifact.InitialDocumentVersion, artifact.ID{}
	})
}

// Content returns the canonical route-decision document.
func (value RoutingDecision) Content() (artifact.Content, error) {
	return routingDecisionCodec.Content(value)
}

// ValidateIdentity proves that no observed route fact changed after selection.
func (value RoutingDecision) ValidateIdentity() error {
	return routingDecisionCodec.ValidateIdentity(value)
}

// RequireRoutingDecision loads one exact durable decision and proves that its
// complete stored lineage still exists. Domain owners reverify their typed
// current authority before executing the selected route.
func RequireRoutingDecision(ctx context.Context, reader artifact.Reader, id artifact.ID) (RoutingDecision, error) {
	return routingDecisionCodec.RequireExactLineage(ctx, reader, id, RoutingDecision.Lineage)
}

// PublishRoutingDecision commits a decision only at the exact repository head
// against which its owners were verified.
func PublishRoutingDecision(
	ctx context.Context,
	repository artifact.Repository,
	value RoutingDecision,
	expected artifact.CommitID,
) (RoutingDecision, artifact.CommitID, error) {
	if ctx == nil || repository == nil || value.Head != expected {
		return RoutingDecision{}, artifact.CommitID{}, errors.New("run record: routing decision publication authority differs")
	}
	identified, err := NewRoutingDecision(value)
	if err != nil {
		return RoutingDecision{}, artifact.CommitID{}, err
	}
	content, err := identified.Content()
	if err != nil {
		return RoutingDecision{}, artifact.CommitID{}, err
	}
	batch, err := artifact.NewDocumentBatch(
		"routing/decision/"+identified.ID.String(), []artifact.Content{content}, identified.Lineage(), nil,
	)
	if err != nil {
		return RoutingDecision{}, artifact.CommitID{}, err
	}
	batch.ExpectedHead = &expected
	commit, err := artifact.CommitBatch(ctx, repository, batch)
	return identified, commit, err
}

// Lineage binds the decision to its authority and every considered capability fact.
func (value RoutingDecision) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, routingDecisionParents(value)...)
}

func routingDecisionParents(value RoutingDecision) []artifact.ID {
	parents := []artifact.ID{value.Authority, value.Previous, value.Failure}
	for _, candidate := range value.Candidates {
		// Selection is a derived identity, not a stored document.
		parents = append(parents, candidate.Capability, candidate.Manual, candidate.Probe)
	}
	parents = slices.DeleteFunc(parents, func(id artifact.ID) bool { return !id.Valid() })
	slices.SortFunc(parents, artifact.CompareID)
	return slices.Compact(parents)
}

func canonicalizeRoutingDecision(value *RoutingDecision) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		!textcheck.Bounded(value.Intent, artifact.MaxContentBytes, "\x00\r\n\t ") ||
		strings.Trim(value.Intent, "abcdefghijklmnopqrstuvwxyz0123456789-") != "" ||
		!value.Effect.Valid() || value.Authority.Kind() != artifact.KindEvidence || !value.Head.Valid() ||
		len(value.Candidates) == 0 {
		return errors.New("run record: invalid routing decision")
	}
	if value.Previous.Valid() != value.Failure.Valid() || value.Previous.Valid() &&
		(value.Previous.Kind() != artifact.KindEvidence || value.Failure.Kind() != artifact.KindEvidence || value.Previous == value.Failure) {
		return errors.New("run record: invalid routing fallback lineage")
	}
	value.Candidates = slices.Clone(value.Candidates)
	slices.SortFunc(value.Candidates, func(left, right RoutingCandidateObservation) int {
		return artifact.CompareID(left.Selection, right.Selection)
	})
	eligible := 0
	seen := make(map[artifact.ID]struct{}, len(value.Candidates))
	for _, candidate := range value.Candidates {
		if candidate.Capability.Kind() != artifact.KindProfile || candidate.Manual.Kind() != artifact.KindRecipe ||
			candidate.Selection.Kind() != artifact.KindProfile || candidate.Probe.Kind() != artifact.KindEvidence ||
			!candidate.Boundary.Valid() || !candidate.Effect.Valid() || !candidate.Head.Valid() ||
			!candidate.Rejection.valid() || !validRouteMeasurement(candidate.Measurement, candidate.Rejection == RouteEligible) {
			return errors.New("run record: invalid routing candidate")
		}
		if _, duplicate := seen[candidate.Selection]; duplicate {
			return errors.New("run record: duplicate routing selection")
		}
		seen[candidate.Selection] = struct{}{}
		if candidate.Rejection == RouteEligible {
			eligible++
		}
	}
	frontier := evidenceRouteFrontier(value.Candidates)
	value.Disposition, value.Winner = RouteRefused, artifact.ID{}
	if len(frontier) == 1 {
		value.Disposition, value.Winner = RouteSelected, value.Candidates[frontier[0]].Selection
	} else if len(frontier) > 1 {
		value.Disposition = RouteDecisionRequired
	} else if eligible != 0 {
		return errors.New("run record: eligible route has no Pareto frontier")
	}
	return nil
}

func validRouteMeasurement(value RouteMeasurement, required bool) bool {
	if value.Attempts == 0 {
		return !required && value.Succeeded == 0 && value.EvidenceUnits == 0 &&
			value.CostUnits == 0 && !value.CostObserved && value.LatencyNS == 0
	}
	return value.Succeeded <= value.Attempts && (value.CostObserved || value.CostUnits == 0)
}

// evidenceRouteFrontier returns stable indexes for all non-dominated eligible observations.
func evidenceRouteFrontier(candidates []RoutingCandidateObservation) []int {
	eligible := make([]int, 0, len(candidates))
	for index, candidate := range candidates {
		if candidate.Rejection == RouteEligible {
			eligible = append(eligible, index)
		}
	}
	frontier := make([]int, 0, len(eligible))
	for _, candidate := range eligible {
		dominated := false
		for _, other := range eligible {
			if candidate != other && routeMeasurementDominates(candidates[other].Measurement, candidates[candidate].Measurement) {
				dominated = true
				break
			}
		}
		if !dominated {
			frontier = append(frontier, candidate)
		}
	}
	return frontier
}

func routeMeasurementDominates(left, right RouteMeasurement) bool {
	if left.CostObserved != right.CostObserved {
		return false
	}
	success := compareRouteRatio(left.Succeeded, left.Attempts, right.Succeeded, right.Attempts)
	evidence := compareRouteRatio(left.EvidenceUnits, left.Attempts, right.EvidenceUnits, right.Attempts)
	cost := 0
	if left.CostObserved {
		cost = compareRouteRatio(left.CostUnits, left.Attempts, right.CostUnits, right.Attempts)
	}
	latency := compareRouteRatio(left.LatencyNS, left.Attempts, right.LatencyNS, right.Attempts)
	return success >= 0 && evidence >= 0 && cost <= 0 && latency <= 0 &&
		(success > 0 || evidence > 0 || cost < 0 || latency < 0)
}

func compareRouteRatio(leftNumerator, leftDenominator, rightNumerator, rightDenominator uint64) int {
	leftHigh, leftLow := bits.Mul64(leftNumerator, rightDenominator)
	rightHigh, rightLow := bits.Mul64(rightNumerator, leftDenominator)
	return cmp.Or(cmp.Compare(leftHigh, rightHigh), cmp.Compare(leftLow, rightLow))
}

func cloneRoutingDecision(value RoutingDecision) RoutingDecision {
	value.Candidates = slices.Clone(value.Candidates)
	return value
}
