package evaluation

import (
	"context"
	"errors"
	"slices"
	"strings"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/executionfailure"
	"overgo/internal/invocation"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// EvidenceRouteRequest names the stored authority and the boundary policy for
// one route. Effect, head, selection, eligibility, and measurements are loaded
// or derived by the compiler rather than asserted by its caller.
type EvidenceRouteRequest struct {
	Intent            string                `json:"intent"`
	Authority         artifact.ID           `json:"authority"`
	AllowedBoundaries []invocation.Boundary `json:"allowed_boundaries"`
}

// EvidenceRouteCandidate contains only durable owner identities. A candidate
// without a current production probe is not executable and cannot enter the
// routing decision as a synthetic selection.
type EvidenceRouteCandidate struct {
	Probe  artifact.ID `json:"probe"`
	Manual artifact.ID `json:"manual"`
}

// CompileEvidenceRoute verifies current production probes, exact manuals, and
// an accepted typed authority at one repository head, then publishes the
// provider-neutral routing decision with compare-and-set publication.
func CompileEvidenceRoute(
	ctx context.Context,
	repository artifact.Repository,
	request EvidenceRouteRequest,
	candidates []EvidenceRouteCandidate,
) (runrecord.RoutingDecision, artifact.CommitID, error) {
	if ctx == nil || repository == nil || strings.TrimSpace(request.Intent) == "" ||
		request.Intent != strings.TrimSpace(request.Intent) || request.Authority.Kind() != artifact.KindEvidence ||
		len(request.AllowedBoundaries) == 0 || len(candidates) == 0 {
		return runrecord.RoutingDecision{}, artifact.CommitID{}, errors.New("evaluation: invalid evidence route request")
	}
	head, _ := repository.Head()
	allowed := slices.Clone(request.AllowedBoundaries)
	slices.Sort(allowed)
	allowed = slices.Compact(allowed)
	if slices.ContainsFunc(allowed, func(boundary invocation.Boundary) bool { return !boundary.Valid() }) {
		return runrecord.RoutingDecision{}, artifact.CommitID{}, errors.New("evaluation: invalid route boundary")
	}
	authority, err := requireRouteAuthority(ctx, repository, request.Authority)
	if err != nil {
		return runrecord.RoutingDecision{}, artifact.CommitID{}, err
	}
	observed := make([]runrecord.RoutingCandidateObservation, len(candidates))
	var routeCase artifact.ID
	var effect invocation.Class
	for index, candidate := range candidates {
		probe, manual, boundary, measurement, loadErr := requireRouteCandidate(ctx, repository, candidate)
		if loadErr != nil {
			return runrecord.RoutingDecision{}, artifact.CommitID{}, loadErr
		}
		if index == 0 {
			routeCase, effect = probe.Case, manual.Effect
		} else if probe.Case != routeCase || manual.Effect != effect {
			return runrecord.RoutingDecision{}, artifact.CommitID{}, errors.New("evaluation: route candidates do not share one case and effect")
		}
		rejection := runrecord.RouteEligible
		if !slices.Contains(authority.Evidence, probe.ID) || !slices.Contains(authority.Evidence, manual.ID) {
			rejection = runrecord.RouteUnauthorized
		} else if !slices.Contains(allowed, boundary) {
			rejection = runrecord.RouteBoundaryDisallowed
		}
		observed[index] = runrecord.RoutingCandidateObservation{
			Capability: probe.ModelCapability, Manual: manual.ID, Selection: probe.Selection, Probe: probe.ID,
			Boundary: boundary, Effect: manual.Effect, Head: head, Rejection: rejection, Measurement: measurement,
		}
	}
	if authority.Subject != routeCase {
		return runrecord.RoutingDecision{}, artifact.CommitID{}, errors.New("evaluation: route authority addresses another activation case")
	}
	decision, err := runrecord.NewRoutingDecision(runrecord.RoutingDecision{
		Intent: request.Intent, Effect: effect, Authority: authority.ID, Head: head, Candidates: observed,
	})
	if err != nil {
		return runrecord.RoutingDecision{}, artifact.CommitID{}, err
	}
	return runrecord.PublishRoutingDecision(ctx, repository, decision, head)
}

// FallbackEvidenceRoute consumes the stored selected decision and the current
// typed terminal failure receipt. A retry publishes a new decision before an
// alternate selection is observable; non-retry dispositions return no route.
func FallbackEvidenceRoute(
	ctx context.Context,
	repository artifact.Repository,
	previousID, terminalID artifact.ID,
) (runrecord.RoutingDecision, artifact.CommitID, executionfailure.Disposition, error) {
	if ctx == nil || repository == nil {
		return runrecord.RoutingDecision{}, artifact.CommitID{}, executionfailure.Disposition{}, errors.New("evaluation: fallback authority is absent")
	}
	head, _ := repository.Head()
	previous, err := runrecord.RequireRoutingDecision(ctx, repository, previousID)
	if err != nil || previous.Disposition != runrecord.RouteSelected || !previous.Winner.Valid() {
		return runrecord.RoutingDecision{}, artifact.CommitID{}, executionfailure.Disposition{}, errors.Join(errors.New("evaluation: failed route decision is invalid"), err)
	}
	terminal, err := runrecord.RequireTerminalAttemptReceipt(ctx, repository, terminalID)
	if err != nil || terminal.Operation != previous.ID || terminal.Outcome != runrecord.OutcomeFailed || terminal.Disposition == nil {
		return runrecord.RoutingDecision{}, artifact.CommitID{}, executionfailure.Disposition{}, errors.Join(errors.New("evaluation: fallback terminal receipt differs"), err)
	}
	disposition := *terminal.Disposition
	winnerIndex := slices.IndexFunc(previous.Candidates, func(candidate runrecord.RoutingCandidateObservation) bool {
		return candidate.Selection == previous.Winner
	})
	if winnerIndex < 0 || terminal.Capability != previous.Candidates[winnerIndex].Capability {
		return runrecord.RoutingDecision{}, artifact.CommitID{}, disposition, errors.New("evaluation: fallback failure addresses another capability")
	}
	if disposition.Decision != executionfailure.DecisionRetry {
		return runrecord.RoutingDecision{}, artifact.CommitID{}, disposition, nil
	}
	authority, err := requireRouteAuthority(ctx, repository, previous.Authority)
	if err != nil {
		return runrecord.RoutingDecision{}, artifact.CommitID{}, disposition, err
	}
	candidates := make([]runrecord.RoutingCandidateObservation, len(previous.Candidates))
	for index, prior := range previous.Candidates {
		probe, manual, boundary, measurement, requireErr := requireRouteCandidate(ctx, repository, EvidenceRouteCandidate{
			Probe: prior.Probe, Manual: prior.Manual,
		})
		if requireErr != nil || probe.Selection != prior.Selection || probe.ModelCapability != prior.Capability ||
			manual.Effect != previous.Effect || boundary != prior.Boundary || measurement != prior.Measurement ||
			!slices.Contains(authority.Evidence, probe.ID) || !slices.Contains(authority.Evidence, manual.ID) {
			return runrecord.RoutingDecision{}, artifact.CommitID{}, disposition, errors.Join(errors.New("evaluation: fallback route authority changed"), requireErr)
		}
		prior.Head = head
		if index == winnerIndex {
			prior.Rejection = runrecord.RouteFailed
		}
		candidates[index] = prior
	}
	decision, err := runrecord.NewRoutingDecision(runrecord.RoutingDecision{
		Intent: previous.Intent, Effect: previous.Effect, Authority: previous.Authority, Head: head,
		Candidates: candidates, Previous: previous.ID, Failure: terminal.ID,
	})
	if err != nil {
		return runrecord.RoutingDecision{}, artifact.CommitID{}, disposition, err
	}
	decision, commit, err := runrecord.PublishRoutingDecision(ctx, repository, decision, head)
	return decision, commit, disposition, err
}

func requireRouteAuthority(ctx context.Context, reader artifact.Reader, id artifact.ID) (recipe.Decision, error) {
	authority, err := recipe.RequireDecision(ctx, reader, id)
	if err != nil || authority.Outcome != recipe.DecisionAccepted || authority.Tier != recipe.EvidenceProduction {
		return recipe.Decision{}, errors.Join(errors.New("evaluation: routing authority is not an accepted production decision"), err)
	}
	if _, found, requireErr := reader.Artifact(ctx, authority.Decider.Derivation); requireErr != nil || !found {
		return recipe.Decision{}, errors.Join(errors.New("evaluation: routing derivation authority is absent"), requireErr)
	}
	return authority, nil
}

func requireRouteCandidate(
	ctx context.Context,
	reader artifact.Reader,
	candidate EvidenceRouteCandidate,
) (CapabilityProbeResult, agenttool.Manual, invocation.Boundary, runrecord.RouteMeasurement, error) {
	probe, err := RequireCapabilityProbeResult(ctx, reader, candidate.Probe)
	if err != nil || probe.Outcome != runrecord.OutcomeSucceeded {
		return CapabilityProbeResult{}, agenttool.Manual{}, "", runrecord.RouteMeasurement{}, errors.Join(errors.New("evaluation: route probe is not current successful production evidence"), err)
	}
	manual, err := agenttool.RequireManual(ctx, reader, candidate.Manual)
	if err != nil || manual.Capability != probe.ModelCapability {
		return CapabilityProbeResult{}, agenttool.Manual{}, "", runrecord.RouteMeasurement{}, errors.Join(errors.New("evaluation: route manual addresses another capability"), err)
	}
	boundary, err := probeRouteBoundary(probe.Entry)
	if err != nil {
		return CapabilityProbeResult{}, agenttool.Manual{}, "", runrecord.RouteMeasurement{}, err
	}
	check, err := recipe.RequireDecision(ctx, reader, probe.CheckDecision)
	if err != nil || check.Outcome != recipe.DecisionAccepted {
		return CapabilityProbeResult{}, agenttool.Manual{}, "", runrecord.RouteMeasurement{}, errors.Join(errors.New("evaluation: route probe check is not accepted"), err)
	}
	latency, measured := probe.Resources.Measure(runrecord.ResourceWallNS)
	if !measured {
		return CapabilityProbeResult{}, agenttool.Manual{}, "", runrecord.RouteMeasurement{}, errors.New("evaluation: route probe lacks measured latency")
	}
	measurement := runrecord.RouteMeasurement{
		Attempts: 1, Succeeded: 1, EvidenceUnits: uint64(len(check.Evidence)), LatencyNS: latency,
	}
	if cost, costMeasured := probe.Resources.Measure(runrecord.ResourceCostUnits); costMeasured {
		measurement.CostUnits, measurement.CostObserved = cost, true
	}
	return probe, manual, boundary, measurement, nil
}

func probeRouteBoundary(entry ProductionCapabilityEntry) (invocation.Boundary, error) {
	switch entry {
	case ProductionEntryLocalPlacement:
		return invocation.BoundaryInternal, nil
	case ProductionEntryPeerPlacement:
		return invocation.BoundaryPeer, nil
	case ProductionEntryAgentCoordinator:
		return invocation.BoundaryAgent, nil
	default:
		return "", errors.New("evaluation: unknown production route boundary")
	}
}
