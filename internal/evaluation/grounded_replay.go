package evaluation

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/executionfailure"
	"overgo/internal/runrecord"
)

const (
	groundedReplaySummaryMediaType = "application/vnd.overgo.grounded-replay-summary+json"
	groundedReplaySummarySchema    = "overgo/grounded-replay-summary/v1"
)

type groundedReplaySummary struct {
	Version   uint16        `json:"version"`
	Artifacts []artifact.ID `json:"artifacts"`
	ID        artifact.ID   `json:"-"`
}

var groundedReplaySummaryCodec = artifact.JSONDocumentCodec(
	"grounded replay summary", artifact.KindOutput,
	groundedReplaySummaryMediaType, groundedReplaySummarySchema,
	canonicalizeGroundedReplaySummary,
	func(value groundedReplaySummary) artifact.ID { return value.ID },
	func(value *groundedReplaySummary, id artifact.ID) { value.ID = id },
	func(value groundedReplaySummary) groundedReplaySummary {
		value.Artifacts = slices.Clone(value.Artifacts)
		return value
	},
)

// GroundedReplaySource is one exact source document group whose immutable
// identity must close the consumed retrieval receipts used by the replay.
type GroundedReplaySource struct {
	Kind      artifact.Kind                    `json:"kind"`
	Documents []dataset.AgentRetrievalDocument `json:"documents"`
}

// SupervisedGroundedReplayRequest names existing authorities and bounded
// policies. The replay derives projections, measurements, routing outcomes,
// supervision decisions, and promotion claims; callers cannot supply them.
type SupervisedGroundedReplayRequest struct {
	RunRecipe          artifact.ID                                  `json:"run_recipe"`
	EpisodeBounds      dataset.CapabilityEpisodeProjectionBounds    `json:"episode_bounds"`
	EpisodeSources     []runrecord.CapabilityEpisodeAuthoritySource `json:"episode_sources"`
	ArcTrajectory      artifact.ID                                  `json:"arc_trajectory"`
	ArcBounds          dataset.InteractionArcProjectionBounds       `json:"arc_bounds"`
	ArcBranch          string                                       `json:"arc_branch"`
	ArcSelection       dataset.InteractionSelectionBounds           `json:"arc_selection"`
	ArcTokenizer       artifact.ID                                  `json:"arc_tokenizer"`
	ArcCounter         artifact.ID                                  `json:"arc_counter"`
	RetrievalSources   []GroundedReplaySource                       `json:"retrieval_sources"`
	BaselineCase       RetrievalCase                                `json:"baseline_case"`
	BaselineReceipt    artifact.ID                                  `json:"baseline_receipt"`
	TrialCase          RetrievalCase                                `json:"trial_case"`
	TrialReceipt       artifact.ID                                  `json:"trial_receipt"`
	Transform          dataset.DatasetTransformSpec                 `json:"transform"`
	Probes             []CapabilityProbeEvidence                    `json:"probes"`
	Route              EvidenceRouteRequest                         `json:"route"`
	RouteCandidates    []EvidenceRouteCandidate                     `json:"route_candidates"`
	RouteFailure       runrecord.FailureObservation                 `json:"route_failure"`
	RouteMaxAttempts   uint32                                       `json:"route_max_attempts"`
	Trajectories       [3]artifact.ID                               `json:"trajectories"`
	KnowledgeAdmission artifact.ID                                  `json:"knowledge_admission"`
}

// SupervisedGroundedReplayResult is the exact durable closure produced by one
// supervised replay. Run consumes these artifacts as verified inputs rather
// than adding produced-by edges to their closed owner lineages.
type SupervisedGroundedReplayResult struct {
	Run           runrecord.Run                       `json:"run"`
	Summary       artifact.ID                         `json:"summary"`
	Episode       dataset.CapabilityEpisodeProjection `json:"episode"`
	ArcProjection dataset.InteractionArcProjection    `json:"arc_projection"`
	ArcSelection  dataset.InteractionArcSelection     `json:"arc_selection"`
	Probes        []CapabilityProbeResult             `json:"probes"`
	Route         runrecord.RoutingDecision           `json:"route"`
	RouteFailure  runrecord.TerminalAttemptReceipt    `json:"route_failure"`
	Fallback      runrecord.RoutingDecision           `json:"fallback"`
	Trajectories  [2]TrajectoryHealthDecision         `json:"trajectories"`
	Baseline      RetrievalEvaluation                 `json:"baseline"`
	Trial         RetrievalEvaluation                 `json:"trial"`
	Transform     dataset.DatasetTransform            `json:"transform"`
	Proposal      EpisodeKnowledgeProposal            `json:"proposal"`
	Promotion     KnowledgePromotionEvidence          `json:"promotion"`
}

// RunSupervisedGroundedReplay is the single production composition root for
// the supervised grounded-capability proof. It calls domain owners directly
// over one repository, and every publishing owner either commits against its
// observed head or refuses concurrent drift.
func RunSupervisedGroundedReplay(
	ctx context.Context,
	repository artifact.Repository,
	request SupervisedGroundedReplayRequest,
) (SupervisedGroundedReplayResult, error) {
	if err := validateSupervisedGroundedReplayRequest(ctx, repository, request); err != nil {
		return SupervisedGroundedReplayResult{}, err
	}

	transform, err := requireGroundedReplayTransform(ctx, repository, request.Transform)
	if err != nil {
		return SupervisedGroundedReplayResult{}, err
	}
	publishedSources, err := publishGroundedReplaySources(ctx, repository, request.RetrievalSources)
	if err != nil {
		return SupervisedGroundedReplayResult{}, err
	}
	baselineReceipt, err := requireGroundedReplayReceipt(ctx, repository, request.BaselineReceipt, publishedSources)
	if err != nil {
		return SupervisedGroundedReplayResult{}, err
	}
	trialReceipt, err := requireGroundedReplayReceipt(ctx, repository, request.TrialReceipt, publishedSources)
	if err != nil {
		return SupervisedGroundedReplayResult{}, err
	}
	baseline, err := PublishRetrievalEvaluation(ctx, repository, request.BaselineCase, baselineReceipt.ID)
	if err != nil {
		return SupervisedGroundedReplayResult{}, err
	}
	trial, err := PublishRetrievalEvaluation(ctx, repository, request.TrialCase, trialReceipt.ID)
	if err != nil {
		return SupervisedGroundedReplayResult{}, err
	}

	episode, err := runrecord.BuildCapabilityEpisodeProjection(
		ctx, repository, request.EpisodeBounds, request.EpisodeSources,
	)
	if err != nil {
		return SupervisedGroundedReplayResult{}, err
	}
	episodeBatch, err := episode.Batch("evaluation/grounded-replay/episodes/" + episode.ID.String())
	if err != nil {
		return SupervisedGroundedReplayResult{}, errors.Join(errors.New("evaluation: publish grounded episode projection"), err)
	}
	if commitErr := commitGroundedReplayBatch(ctx, repository, episodeBatch); commitErr != nil {
		return SupervisedGroundedReplayResult{}, errors.Join(errors.New("evaluation: publish grounded episode projection"), commitErr)
	}
	episode, err = runrecord.RequireCapabilityEpisodeProjection(ctx, repository, episode.ID)
	if err != nil {
		return SupervisedGroundedReplayResult{}, err
	}

	arcProjection, arcs, err := runrecord.BuildInteractionArcProjection(
		ctx, repository, request.ArcTrajectory, request.ArcBounds,
	)
	if err != nil {
		return SupervisedGroundedReplayResult{}, err
	}
	arcBatch, err := arcProjection.Batch("evaluation/grounded-replay/arcs/"+arcProjection.ID.String(), arcs)
	if err != nil {
		return SupervisedGroundedReplayResult{}, errors.Join(errors.New("evaluation: publish grounded interaction arcs"), err)
	}
	if commitErr := commitGroundedReplayBatch(ctx, repository, arcBatch); commitErr != nil {
		return SupervisedGroundedReplayResult{}, errors.Join(errors.New("evaluation: publish grounded interaction arcs"), commitErr)
	}
	arcProjection, _, err = runrecord.RequireInteractionArcProjection(ctx, repository, arcProjection.ID)
	if err != nil {
		return SupervisedGroundedReplayResult{}, err
	}
	policy, err := dataset.NewInteractionArcMeasurementPolicy(request.ArcTokenizer, request.ArcCounter)
	if err != nil {
		return SupervisedGroundedReplayResult{}, err
	}
	policyBatch, err := policy.Batch("evaluation/grounded-replay/arc-policy/" + policy.ID.String())
	if err != nil {
		return SupervisedGroundedReplayResult{}, errors.Join(errors.New("evaluation: publish grounded arc policy"), err)
	}
	if commitErr := commitGroundedReplayBatch(ctx, repository, policyBatch); commitErr != nil {
		return SupervisedGroundedReplayResult{}, errors.Join(errors.New("evaluation: publish grounded arc policy"), commitErr)
	}
	arcSelection, err := runrecord.SelectInteractionArcs(
		ctx, repository, arcProjection.ID, request.ArcBranch, request.ArcSelection, policy.ID,
	)
	if err != nil {
		return SupervisedGroundedReplayResult{}, err
	}
	arcSelection, err = runrecord.RequireInteractionArcSelection(ctx, repository, arcSelection.ID)
	if err != nil {
		return SupervisedGroundedReplayResult{}, err
	}

	probes := make([]CapabilityProbeResult, len(request.Probes))
	for index, evidence := range request.Probes {
		probe, _, publishErr := PublishProductionCapabilityProbe(ctx, repository, evidence)
		if publishErr != nil {
			return SupervisedGroundedReplayResult{}, publishErr
		}
		probes[index], err = RequireCapabilityProbeResult(ctx, repository, probe.ID)
		if err != nil {
			return SupervisedGroundedReplayResult{}, err
		}
	}
	if err := requireGroundedRouteCandidates(probes, request.RouteCandidates); err != nil {
		return SupervisedGroundedReplayResult{}, err
	}
	route, _, err := CompileEvidenceRoute(ctx, repository, request.Route, request.RouteCandidates)
	if err != nil {
		return SupervisedGroundedReplayResult{}, err
	}
	routeFailure, fallback, err := groundReplayFallback(
		ctx, repository, route, request.RouteFailure, request.RouteMaxAttempts,
	)
	if err != nil {
		return SupervisedGroundedReplayResult{}, err
	}

	var trajectoryDecisions [2]TrajectoryHealthDecision
	for index := range trajectoryDecisions {
		decision, _, superviseErr := SuperviseTrajectory(
			ctx, repository, request.Trajectories[index], request.Trajectories[index+1],
		)
		if superviseErr != nil {
			return SupervisedGroundedReplayResult{}, superviseErr
		}
		trajectoryDecisions[index], err = RequireTrajectoryHealthDecision(ctx, repository, decision.ID)
		if err != nil {
			return SupervisedGroundedReplayResult{}, err
		}
	}
	if trajectoryDecisions[0].Decision != executionfailure.DecisionRetireSession ||
		trajectoryDecisions[1].Decision != executionfailure.DecisionResume {
		return SupervisedGroundedReplayResult{}, errors.New("evaluation: supervised replay did not demonstrate stop and recovery")
	}

	proposal, err := PublishEpisodeKnowledgeProposal(ctx, repository, request.KnowledgeAdmission)
	if err != nil {
		return SupervisedGroundedReplayResult{}, err
	}
	promotion, err := PublishEpisodeKnowledgePromotion(ctx, repository, proposal.ID, baseline.ID, trial.ID)
	if err != nil {
		return SupervisedGroundedReplayResult{}, err
	}
	proposal, err = RequireEpisodeKnowledgeProposal(ctx, repository, proposal.ID)
	if err != nil {
		return SupervisedGroundedReplayResult{}, err
	}
	promotion, err = RequireKnowledgePromotionEvidence(ctx, repository, promotion.ID)
	if err != nil {
		return SupervisedGroundedReplayResult{}, err
	}

	result := SupervisedGroundedReplayResult{
		Episode: episode, ArcProjection: arcProjection, ArcSelection: arcSelection,
		Probes: probes, Route: route, RouteFailure: routeFailure, Fallback: fallback,
		Trajectories: trajectoryDecisions, Baseline: baseline, Trial: trial,
		Transform: transform, Proposal: proposal, Promotion: promotion,
	}
	run, summary, err := publishGroundedReplayRun(ctx, repository, request.RunRecipe, result)
	if err != nil {
		return SupervisedGroundedReplayResult{}, err
	}
	result.Run, result.Summary = run, summary
	return result, nil
}

func validateSupervisedGroundedReplayRequest(
	ctx context.Context,
	repository artifact.Repository,
	request SupervisedGroundedReplayRequest,
) error {
	if ctx == nil || repository == nil || request.RunRecipe.Kind() != artifact.KindRecipe ||
		len(request.EpisodeSources) == 0 || request.ArcTrajectory.Kind() != artifact.KindEvidence ||
		request.ArcTokenizer.Kind() != artifact.KindTokenizer || request.ArcCounter.Kind() != artifact.KindProfile ||
		len(request.RetrievalSources) == 0 || len(request.Probes) < 2 ||
		len(request.RouteCandidates) != len(request.Probes) || request.RouteMaxAttempts < 2 ||
		request.KnowledgeAdmission.Kind() != artifact.KindEvidence {
		return errors.New("evaluation: supervised grounded replay request is incomplete")
	}
	for _, id := range request.Trajectories {
		if id.Kind() != artifact.KindEvidence {
			return errors.New("evaluation: supervised grounded replay trajectory is invalid")
		}
	}
	for _, id := range []artifact.ID{request.RunRecipe, request.ArcTokenizer, request.ArcCounter} {
		if _, found, err := repository.Artifact(ctx, id); err != nil || !found {
			return errors.Join(errors.New("evaluation: supervised grounded replay authority is absent"), err)
		}
	}
	return nil
}

func requireGroundedReplayTransform(
	ctx context.Context,
	repository artifact.Repository,
	spec dataset.DatasetTransformSpec,
) (dataset.DatasetTransform, error) {
	identified, err := dataset.NewDatasetTransform(spec)
	if err != nil {
		return dataset.DatasetTransform{}, err
	}
	if _, found, readErr := repository.Artifact(ctx, identified.ID()); readErr != nil {
		return dataset.DatasetTransform{}, readErr
	} else if !found {
		content, contentErr := identified.Content()
		if contentErr != nil {
			return dataset.DatasetTransform{}, contentErr
		}
		batch, batchErr := artifact.NewDocumentBatch(
			"evaluation/grounded-replay/transform/"+identified.ID().String(),
			[]artifact.Content{content}, identified.Lineage(), nil,
		)
		if batchErr != nil {
			return dataset.DatasetTransform{}, errors.Join(errors.New("evaluation: publish grounded dataset transform"), batchErr)
		}
		if commitErr := commitGroundedReplayBatch(ctx, repository, batch); commitErr != nil {
			return dataset.DatasetTransform{}, errors.Join(errors.New("evaluation: publish grounded dataset transform"), commitErr)
		}
	}
	stored, err := dataset.RequireDatasetTransform(ctx, repository, identified.ID())
	if err != nil || stored.ID() != identified.ID() {
		return dataset.DatasetTransform{}, errors.Join(errors.New("evaluation: grounded dataset transform differs"), err)
	}
	return stored, nil
}

func publishGroundedReplaySources(
	ctx context.Context,
	repository artifact.Repository,
	sources []GroundedReplaySource,
) (map[artifact.ID]struct{}, error) {
	result := make(map[artifact.ID]struct{})
	for _, source := range sources {
		documents, err := dataset.PublishAgentRetrievalSource(ctx, repository, source.Kind, source.Documents)
		if err != nil {
			return nil, err
		}
		for _, document := range documents {
			result[document.Source] = struct{}{}
		}
	}
	if len(result) == 0 {
		return nil, errors.New("evaluation: grounded retrieval sources are absent")
	}
	return result, nil
}

func requireGroundedReplayReceipt(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
	sources map[artifact.ID]struct{},
) (runrecord.RetrievalReceipt, error) {
	receipt, err := runrecord.RequireConsumedRetrievalReceipt(ctx, reader, id)
	if err != nil {
		return runrecord.RetrievalReceipt{}, err
	}
	if len(receipt.Results) == 0 {
		return runrecord.RetrievalReceipt{}, errors.New("evaluation: grounded retrieval receipt has no source closure")
	}
	for _, result := range receipt.Results {
		if _, found := sources[result.Source]; !found {
			return runrecord.RetrievalReceipt{}, errors.New("evaluation: grounded retrieval receipt names an unverified source")
		}
	}
	return receipt, nil
}

func requireGroundedRouteCandidates(probes []CapabilityProbeResult, candidates []EvidenceRouteCandidate) error {
	available := make(map[artifact.ID]struct{}, len(probes))
	for _, probe := range probes {
		available[probe.ID] = struct{}{}
	}
	for _, candidate := range candidates {
		if _, found := available[candidate.Probe]; !found {
			return errors.New("evaluation: grounded route candidate lacks replayed production evidence")
		}
		delete(available, candidate.Probe)
	}
	if len(available) != 0 {
		return errors.New("evaluation: grounded route does not consume every production probe")
	}
	return nil
}

func groundReplayFallback(
	ctx context.Context,
	repository artifact.Repository,
	route runrecord.RoutingDecision,
	failure runrecord.FailureObservation,
	maxAttempts uint32,
) (runrecord.TerminalAttemptReceipt, runrecord.RoutingDecision, error) {
	winnerIndex := slices.IndexFunc(route.Candidates, func(candidate runrecord.RoutingCandidateObservation) bool {
		return candidate.Selection == route.Winner
	})
	if route.Disposition != runrecord.RouteSelected || winnerIndex < 0 {
		return runrecord.TerminalAttemptReceipt{}, runrecord.RoutingDecision{}, errors.New("evaluation: grounded replay route has no selected capability")
	}
	observation, err := runrecord.PublishFailureObservation(ctx, repository, failure)
	if err != nil {
		return runrecord.TerminalAttemptReceipt{}, runrecord.RoutingDecision{}, err
	}
	normalization, err := runrecord.PublishFailureNormalization(
		ctx, repository, runrecord.NormalizeFailureObservation(observation),
	)
	if err != nil {
		return runrecord.TerminalAttemptReceipt{}, runrecord.RoutingDecision{}, err
	}
	disposition := executionfailure.Decide(executionfailure.Situation{
		Cause: normalization.Cause, Attempts: 1, MaxAttempts: maxAttempts,
	})
	if disposition.Decision != executionfailure.DecisionRetry {
		return runrecord.TerminalAttemptReceipt{}, runrecord.RoutingDecision{}, errors.New("evaluation: grounded replay fallback evidence is not retryable")
	}
	terminal, err := runrecord.PublishTerminalAttemptReceipt(ctx, repository, runrecord.TerminalAttemptReceipt{
		Operation: route.ID, Capability: route.Candidates[winnerIndex].Capability, Outcome: runrecord.OutcomeFailed,
		FailureObservation: observation.ID, FailureNormalization: normalization.ID, Disposition: &disposition,
		Gaps: []string{
			runrecord.GapProcessTermination, runrecord.GapResources,
			runrecord.GapToolOutput, runrecord.GapTranscript,
		},
		ObservedUnixNS: observation.ObservedUnixNS,
	})
	if err != nil {
		return runrecord.TerminalAttemptReceipt{}, runrecord.RoutingDecision{}, err
	}
	fallback, _, _, err := FallbackEvidenceRoute(ctx, repository, route.ID, terminal.ID)
	if err != nil || !fallback.Winner.Valid() || fallback.Winner == route.Winner {
		return runrecord.TerminalAttemptReceipt{}, runrecord.RoutingDecision{}, errors.Join(
			errors.New("evaluation: grounded replay fallback did not select an alternate capability"), err,
		)
	}
	return terminal, fallback, nil
}

func publishGroundedReplayRun(
	ctx context.Context,
	repository artifact.Repository,
	recipe artifact.ID,
	result SupervisedGroundedReplayResult,
) (runrecord.Run, artifact.ID, error) {
	inputs := []artifact.ID{
		result.Episode.ID, result.ArcProjection.ID, result.ArcSelection.ID,
		result.Route.ID, result.RouteFailure.ID, result.Fallback.ID,
		result.Trajectories[0].ID, result.Trajectories[1].ID,
		result.Baseline.ID, result.Trial.ID, result.Transform.ID(),
		result.Proposal.ID, result.Promotion.ID,
	}
	for _, probe := range result.Probes {
		inputs = append(inputs, probe.ID)
	}
	slices.SortFunc(inputs, artifact.CompareID)
	inputs = slices.Compact(inputs)
	summary, err := groundedReplaySummaryCodec.New(groundedReplaySummary{
		Version: artifact.InitialDocumentVersion, Artifacts: inputs,
	})
	if err != nil {
		return runrecord.Run{}, artifact.ID{}, err
	}
	run, err := runrecord.NewRun(recipe, runrecord.OutcomeSucceeded, inputs, []artifact.ID{summary.ID}, "")
	if err != nil {
		return runrecord.Run{}, artifact.ID{}, err
	}
	summaryContent, err := groundedReplaySummaryCodec.Content(summary)
	if err != nil {
		return runrecord.Run{}, artifact.ID{}, err
	}
	runContent, err := run.Content()
	if err != nil {
		return runrecord.Run{}, artifact.ID{}, err
	}
	lineage := append(summary.Lineage(), run.Lineage()...)
	batch, err := artifact.NewDocumentBatch(
		"evaluation/grounded-replay/run/"+run.ID.String(),
		[]artifact.Content{summaryContent, runContent}, lineage, nil,
	)
	if err != nil {
		return runrecord.Run{}, artifact.ID{}, errors.Join(errors.New("evaluation: publish grounded replay run"), err)
	}
	if commitErr := commitGroundedReplayBatch(ctx, repository, batch); commitErr != nil {
		return runrecord.Run{}, artifact.ID{}, errors.Join(errors.New("evaluation: publish grounded replay run"), commitErr)
	}
	storedRun, err := requireGroundedReplayRun(ctx, repository, run.ID)
	if err != nil {
		return runrecord.Run{}, artifact.ID{}, err
	}
	if _, err := requireGroundedReplaySummary(ctx, repository, summary.ID, storedRun.ID); err != nil {
		return runrecord.Run{}, artifact.ID{}, err
	}
	return storedRun, summary.ID, nil
}

func canonicalizeGroundedReplaySummary(value *groundedReplaySummary) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || len(value.Artifacts) == 0 {
		return errors.New("evaluation: invalid grounded replay summary")
	}
	value.Artifacts = slices.Clone(value.Artifacts)
	slices.SortFunc(value.Artifacts, artifact.CompareID)
	value.Artifacts = slices.Compact(value.Artifacts)
	if slices.ContainsFunc(value.Artifacts, func(id artifact.ID) bool { return !id.Valid() }) {
		return errors.New("evaluation: invalid grounded replay summary")
	}
	return nil
}

// Lineage binds the summary to every artifact produced by the supervised replay.
func (value groundedReplaySummary) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.Artifacts...)
}

func requireGroundedReplayRun(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (runrecord.Run, error) {
	value, err := runrecord.RequireRun(ctx, reader, id)
	if err != nil {
		return runrecord.Run{}, err
	}
	expected := slices.DeleteFunc(value.Lineage(), func(edge artifact.Lineage) bool { return edge.Child != value.ID })
	if err := requireGroundedReplayLineage(ctx, reader, value.ID, expected); err != nil {
		return runrecord.Run{}, err
	}
	return value, nil
}

func requireGroundedReplaySummary(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
	run artifact.ID,
) (groundedReplaySummary, error) {
	value, err := groundedReplaySummaryCodec.Require(ctx, reader, id)
	if err != nil {
		return groundedReplaySummary{}, err
	}
	expected := append(value.Lineage(), artifact.Lineage{
		Child: value.ID, Parent: run, Relation: artifact.RelationProducedBy,
	})
	if err := requireGroundedReplayLineage(ctx, reader, value.ID, expected); err != nil {
		return groundedReplaySummary{}, err
	}
	return value, nil
}

func requireGroundedReplayLineage(
	ctx context.Context,
	reader artifact.Reader,
	child artifact.ID,
	expected []artifact.Lineage,
) error {
	stored, err := reader.Parents(ctx, child)
	if err != nil {
		return err
	}
	if len(stored) != len(expected) {
		return errors.New("evaluation: grounded replay stored lineage differs")
	}
	for _, edge := range expected {
		if edge.Child != child || !slices.Contains(stored, edge) {
			return errors.New("evaluation: grounded replay stored lineage differs")
		}
		if _, found, requireErr := reader.Artifact(ctx, edge.Parent); requireErr != nil || !found {
			return errors.Join(errors.New("evaluation: grounded replay lineage parent is absent"), requireErr)
		}
	}
	return nil
}

func commitGroundedReplayBatch(ctx context.Context, repository artifact.Repository, batch artifact.Batch) error {
	head, _ := repository.Head()
	batch.ExpectedHead = &head
	_, err := artifact.CommitBatch(ctx, repository, batch)
	if errors.Is(err, artifact.ErrNoChange) {
		return nil
	}
	return err
}
