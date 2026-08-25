package modelrecipe

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// PeerReplicaPolicy derives replica count from requested concurrency and measured capacity.
type PeerReplicaPolicy struct {
	MinimumReplicas       uint64 `json:"minimum_replicas"`
	MaximumReplicas       uint64 `json:"maximum_replicas"`
	TargetConcurrency     uint64 `json:"target_concurrency"`
	ConcurrencyPerReplica uint64 `json:"concurrency_per_replica"`
	MaximumMeasuredNS     uint64 `json:"maximum_measured_ns,omitempty"`
	MaximumDeviceBytes    uint64 `json:"maximum_device_bytes,omitempty"`
	AllowLocal            bool   `json:"allow_local,omitempty"`
	AllowPeers            bool   `json:"allow_peers,omitempty"`
}

// PeerPlacementCandidate names one enrolled peer and exact compatibility evidence.
type PeerPlacementCandidate struct {
	Peer          artifact.ID `json:"peer"`
	Compatibility artifact.ID `json:"compatibility"`
}

// PeerPlacementRequest selects the active model recipe and requested service policy.
type PeerPlacementRequest struct {
	Model            artifact.ID              `json:"model"`
	Task             recipe.Task              `json:"task"`
	NowUnixNS        int64                    `json:"now_unix_ns"`
	LocalObservation artifact.ID              `json:"local_observation,omitzero"`
	Policy           PeerReplicaPolicy        `json:"policy"`
	Peers            []PeerPlacementCandidate `json:"peers,omitempty"`
}

// PeerPlacementRefusal preserves why one candidate could not host a replica.
type PeerPlacementRefusal struct {
	Peer   artifact.ID `json:"peer,omitzero"`
	Reason string      `json:"reason"`
	Detail string      `json:"detail"`
}

// PeerReplicaPlacement is one whole-model replica and its exact evidence.
type PeerReplicaPlacement struct {
	Index         int                        `json:"index"`
	Peer          artifact.ID                `json:"peer,omitzero"`
	Environment   artifact.ID                `json:"environment"`
	Endpoint      string                     `json:"endpoint,omitempty"`
	Compatibility artifact.ID                `json:"compatibility,omitzero"`
	Publication   artifact.ID                `json:"publication,omitzero"`
	Heartbeat     artifact.ID                `json:"heartbeat,omitzero"`
	Observation   artifact.ID                `json:"observation"`
	MeasuredNS    uint64                     `json:"measured_ns"`
	Resources     runrecord.ServingResources `json:"resources"`
	Locality      []artifact.Location        `json:"locality"`
}

// PeerPlacementPlan is one deterministic whole-model placement decision.
type PeerPlacementPlan struct {
	Identity   artifact.ID            `json:"identity"`
	Selection  artifact.ID            `json:"selection"`
	Model      artifact.ID            `json:"model"`
	Task       recipe.Task            `json:"task"`
	Recipe     artifact.ID            `json:"recipe"`
	Resources  artifact.ID            `json:"resources"`
	Policy     artifact.ID            `json:"policy"`
	Components []ComponentSession     `json:"components"`
	Replicas   []PeerReplicaPlacement `json:"replicas"`
	Refusals   []PeerPlacementRefusal `json:"refusals,omitempty"`
}

type peerPlacementPlanIdentity struct {
	Selection  artifact.ID            `json:"selection"`
	Model      artifact.ID            `json:"model"`
	Task       recipe.Task            `json:"task"`
	Recipe     artifact.ID            `json:"recipe"`
	Resources  artifact.ID            `json:"resources"`
	Policy     artifact.ID            `json:"policy"`
	Components []ComponentSession     `json:"components"`
	Replicas   []PeerReplicaPlacement `json:"replicas"`
}

// CompilePeerPlacementPlan resolves active authority and selects distinct whole-model replicas.
func CompilePeerPlacementPlan(
	ctx context.Context,
	repository artifact.Repository,
	request PeerPlacementRequest,
) (PeerPlacementPlan, error) {
	if ctx == nil || repository == nil || request.Model.Kind() != artifact.KindModel || !request.Task.Valid() ||
		request.NowUnixNS <= 0 {
		return PeerPlacementPlan{}, errors.New("model recipe: invalid peer placement request")
	}
	desired, policyID, err := compilePeerReplicaPolicy(request.Policy)
	if err != nil {
		return PeerPlacementPlan{}, err
	}
	selection, err := ResolveActiveExecution(ctx, repository, request.Model, request.Task, SessionWarm)
	if err != nil {
		return PeerPlacementPlan{}, fmt.Errorf("model recipe: active placement recipe unavailable: %w", err)
	}
	artifacts := servingPlacementArtifacts(selection.Program.Definition())
	var candidates []PeerReplicaPlacement
	var refusals []PeerPlacementRefusal
	if request.Policy.AllowLocal {
		candidate, reason, candidateErr := localPlacementCandidate(ctx, repository, selection, artifacts, request)
		if candidateErr != nil {
			return PeerPlacementPlan{}, candidateErr
		}
		if reason != nil {
			refusals = append(refusals, *reason)
		} else {
			candidates = append(candidates, candidate)
		}
	}
	if request.Policy.AllowPeers {
		seen := make(map[artifact.ID]struct{}, len(request.Peers))
		for _, peer := range request.Peers {
			if _, duplicate := seen[peer.Peer]; duplicate {
				return PeerPlacementPlan{}, errors.New("model recipe: duplicate peer placement candidate")
			}
			seen[peer.Peer] = struct{}{}
			candidate, reason := remotePlacementCandidate(ctx, repository, selection, artifacts, request, peer)
			if reason != nil {
				refusals = append(refusals, *reason)
			} else {
				candidates = append(candidates, candidate)
			}
		}
	}
	slices.SortFunc(candidates, comparePeerReplicaPlacement)
	slices.SortFunc(refusals, func(left, right PeerPlacementRefusal) int {
		if order := artifact.CompareID(left.Peer, right.Peer); order != 0 {
			return order
		}
		if order := cmp.Compare(left.Reason, right.Reason); order != 0 {
			return order
		}
		return cmp.Compare(left.Detail, right.Detail)
	})
	if len(candidates) < desired {
		detail := "eligible replicas do not satisfy requested service policy"
		if len(refusals) != 0 {
			detail = refusals[0].Reason + ": " + refusals[0].Detail
		}
		return PeerPlacementPlan{}, fmt.Errorf("model recipe: insufficient peer replicas: %s", detail)
	}
	replicas := slices.Clone(candidates[:desired])
	for index := range replicas {
		replicas[index].Index = index
	}
	components := slices.Clone(selection.Resources.Components)
	identity, err := peerPlacementIdentity(peerPlacementPlanIdentity{
		Selection: selection.Identity, Model: request.Model, Task: request.Task, Recipe: selection.Activation.Definition.ID,
		Resources: selection.Resources.Identity, Policy: policyID, Components: components, Replicas: replicas,
	})
	if err != nil {
		return PeerPlacementPlan{}, err
	}
	return PeerPlacementPlan{
		Identity: identity, Selection: selection.Identity, Model: request.Model,
		Task: request.Task, Recipe: selection.Activation.Definition.ID, Resources: selection.Resources.Identity, Policy: policyID,
		Components: components, Replicas: replicas, Refusals: refusals,
	}, nil
}

// ValidateIdentity proves that no execution-bearing placement field changed after compilation.
func (plan PeerPlacementPlan) ValidateIdentity() error {
	id, err := peerPlacementIdentity(peerPlacementPlanIdentity{
		Selection: plan.Selection, Model: plan.Model, Task: plan.Task, Recipe: plan.Recipe, Resources: plan.Resources,
		Policy: plan.Policy, Components: plan.Components, Replicas: plan.Replicas,
	})
	if err != nil || id != plan.Identity {
		return errors.Join(errors.New("model recipe: peer placement plan identity differs"), err)
	}
	return nil
}

func peerPlacementIdentity(value peerPlacementPlanIdentity) (artifact.ID, error) {
	return artifact.JSONID(artifact.KindProfile, value)
}

func compilePeerReplicaPolicy(policy PeerReplicaPolicy) (int, artifact.ID, error) {
	if policy.MinimumReplicas == 0 || policy.MaximumReplicas < policy.MinimumReplicas ||
		policy.TargetConcurrency == 0 || policy.ConcurrencyPerReplica == 0 || !policy.AllowLocal && !policy.AllowPeers {
		return 0, artifact.ID{}, errors.New("model recipe: invalid peer replica policy")
	}
	rounded, ok := checked.RoundUpMultiple(policy.TargetConcurrency, policy.ConcurrencyPerReplica)
	if !ok {
		return 0, artifact.ID{}, errors.New("model recipe: peer replica derivation overflow")
	}
	replicas := rounded / policy.ConcurrencyPerReplica
	replicas = max(replicas, policy.MinimumReplicas)
	if replicas > policy.MaximumReplicas {
		return 0, artifact.ID{}, errors.New("model recipe: requested concurrency exceeds replica policy")
	}
	count, ok := checked.Int(replicas)
	if !ok {
		return 0, artifact.ID{}, errors.New("model recipe: peer replica count exceeds host bounds")
	}
	id, err := artifact.JSONID(artifact.KindProfile, policy)
	return count, id, err
}

func localPlacementCandidate(
	ctx context.Context,
	repository artifact.Repository,
	selection CapabilityEvidenceSelection,
	artifacts []artifact.ID,
	request PeerPlacementRequest,
) (PeerReplicaPlacement, *PeerPlacementRefusal, error) {
	observation, err := runrecord.RequireServingObservation(ctx, repository, request.LocalObservation)
	if err != nil || observation.Model != request.Model || observation.Recipe != selection.Activation.Definition.ID ||
		observation.Task != request.Task || observation.Outcome != runrecord.OutcomeSucceeded {
		return PeerReplicaPlacement{}, &PeerPlacementRefusal{Reason: "observation", Detail: "local serving observation differs"}, nil
	}
	locality, ok, err := placementLocality(ctx, repository, artifacts, false, "")
	if err != nil {
		return PeerReplicaPlacement{}, nil, err
	}
	if !ok {
		return PeerReplicaPlacement{}, &PeerPlacementRefusal{Reason: "artifact-locality", Detail: "local artifacts are unavailable"}, nil
	}
	if reason := placementResourceRefusal(request.Policy, observation); reason != "" {
		return PeerReplicaPlacement{}, &PeerPlacementRefusal{Reason: "resources", Detail: reason}, nil
	}
	return PeerReplicaPlacement{
		Environment: observation.Environment, Observation: observation.ID,
		MeasuredNS: observation.MeasuredNS, Resources: observation.Resources, Locality: locality,
	}, nil, nil
}

func remotePlacementCandidate(
	ctx context.Context,
	repository artifact.Repository,
	selection CapabilityEvidenceSelection,
	artifacts []artifact.ID,
	request PeerPlacementRequest,
	peer PeerPlacementCandidate,
) (PeerReplicaPlacement, *PeerPlacementRefusal) {
	refuse := func(reason, detail string) (PeerReplicaPlacement, *PeerPlacementRefusal) {
		return PeerReplicaPlacement{}, &PeerPlacementRefusal{Peer: peer.Peer, Reason: reason, Detail: detail}
	}
	authority, err := ResolvePeerServingAuthority(ctx, repository, peer.Peer, request.NowUnixNS)
	if err != nil {
		return refuse("lease", err.Error())
	}
	compatibility, err := resolveRemotePeerCompatibility(
		ctx, repository, peer.Compatibility, request.Model, selection.Activation.Definition.ID,
		selection.Resources.Identity, request.Task,
	)
	if err != nil || compatibility.PeerEnvironment != authority.Enrollment.Environment ||
		compatibility.PeerCapability != authority.Capability.ID {
		return refuse("compatibility", "peer compatibility differs from enrolled serving authority")
	}
	locality, ok, err := placementLocality(ctx, repository, artifacts, true, authority.Capability.Endpoint)
	if err != nil || !ok {
		return refuse("artifact-locality", "peer artifacts are unavailable at the declared endpoint")
	}
	observation, err := runrecord.RequireServingObservation(ctx, repository, compatibility.PeerObservation)
	if err != nil {
		return refuse("observation", "peer serving observation is unavailable")
	}
	if reason := placementResourceRefusal(request.Policy, observation); reason != "" {
		return refuse("resources", reason)
	}
	return PeerReplicaPlacement{
		Peer: peer.Peer, Environment: authority.Enrollment.Environment, Endpoint: authority.Capability.Endpoint,
		Compatibility: compatibility.ID, Publication: authority.Publication.ID, Heartbeat: authority.Heartbeat.ID,
		Observation: observation.ID, MeasuredNS: observation.MeasuredNS, Resources: observation.Resources, Locality: locality,
	}, nil
}

func placementLocality(
	ctx context.Context,
	reader artifact.Reader,
	artifacts []artifact.ID,
	remote bool,
	endpoint string,
) ([]artifact.Location, bool, error) {
	var result []artifact.Location
	for _, id := range artifacts {
		locations, err := reader.Locations(ctx, id)
		if err != nil {
			return nil, false, err
		}
		matched := slices.DeleteFunc(slices.Clone(locations), func(location artifact.Location) bool {
			if remote {
				return location.Kind != artifact.LocationRemote || location.Value != endpoint
			}
			return location.Kind != artifact.LocationFile && location.Kind != artifact.LocationDirectory
		})
		if len(matched) == 0 {
			return nil, false, nil
		}
		result = append(result, matched...)
	}
	slices.SortFunc(result, func(left, right artifact.Location) int {
		if order := artifact.CompareID(left.Artifact, right.Artifact); order != 0 {
			return order
		}
		if order := cmp.Compare(left.Kind, right.Kind); order != 0 {
			return order
		}
		return cmp.Compare(left.Value, right.Value)
	})
	return result, true, nil
}

func placementResourceRefusal(policy PeerReplicaPolicy, observation runrecord.ServingObservation) string {
	if policy.MaximumMeasuredNS != 0 && observation.MeasuredNS > policy.MaximumMeasuredNS {
		return "measured latency exceeds policy"
	}
	if policy.MaximumDeviceBytes != 0 && observation.Resources.PeakDeviceBytes > policy.MaximumDeviceBytes {
		return "peak device memory exceeds policy"
	}
	return ""
}

func servingPlacementArtifacts(definition recipe.Definition) []artifact.ID {
	var result []artifact.ID
	for _, dependency := range definition.Dependencies {
		switch dependency.Role {
		case recipe.DependencyModel, recipe.DependencyTokenizer, recipe.DependencyProjector,
			recipe.DependencyAdapter, recipe.DependencyCheckpoint:
			if !slices.Contains(result, dependency.Artifact) {
				result = append(result, dependency.Artifact)
			}
		}
	}
	slices.SortFunc(result, artifact.CompareID)
	return result
}

func comparePeerReplicaPlacement(left, right PeerReplicaPlacement) int {
	if order := cmp.Compare(left.MeasuredNS, right.MeasuredNS); order != 0 {
		return order
	}
	if order := cmp.Compare(left.Resources.PeakDeviceBytes, right.Resources.PeakDeviceBytes); order != 0 {
		return order
	}
	return artifact.CompareID(left.Peer, right.Peer)
}
