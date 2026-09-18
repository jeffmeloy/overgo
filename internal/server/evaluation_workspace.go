package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

type EvaluationCapability struct {
	Model  artifact.ID                `json:"model"`
	Recipe artifact.ID                `json:"recipe"`
	Suite  evaluation.SuiteDescriptor `json:"suite"`
}

type EvaluationHistoryEntry = evaluation.HistoryEntry

type EvaluationReport struct {
	ID        artifact.ID     `json:"id"`
	MediaType string          `json:"media_type"`
	Schema    string          `json:"schema"`
	Body      json.RawMessage `json:"body"`
}

type EvaluationFailure struct {
	Name        string          `json:"name,omitzero"`
	Observation json.RawMessage `json:"observation"`
}

type EvaluationMetricDelta struct {
	Name      string              `json:"name"`
	Left      float64             `json:"left"`
	Right     float64             `json:"right"`
	Delta     float64             `json:"delta"`
	Direction runrecord.Direction `json:"direction"`
	Improved  bool                `json:"improved"`
}

type EvaluationComparison struct {
	Left    artifact.ID             `json:"left"`
	Right   artifact.ID             `json:"right"`
	Metrics []EvaluationMetricDelta `json:"metrics"`
}

type EvaluationWorkspaceAPI interface {
	EvaluationCapabilities(context.Context) ([]EvaluationCapability, error)
	ExecuteEvaluation(context.Context, artifact.ID, []artifact.ID, operation.Reporter) (operation.Completion, error)
	ExecuteSupervisedGroundedReplay(context.Context, artifact.ID, evaluation.SupervisedGroundedReplayRequest, operation.Reporter) (operation.Completion, error)
	EvaluationHistory(context.Context, artifact.ID) ([]EvaluationHistoryEntry, error)
	EvaluationReport(context.Context, artifact.ID) (EvaluationReport, error)
	EvaluationFailures(context.Context, artifact.ID) ([]EvaluationFailure, error)
	CompareEvaluations(context.Context, artifact.ID, artifact.ID) (EvaluationComparison, error)
}

type EvaluationWorkspace struct {
	repository   *overgodb.Store
	campaign     *evaluation.Campaign
	model        artifact.ID
	recipe       artifact.ID
	suites       []evaluation.CompiledSuite
	byPlan       map[artifact.ID]int
	historyLimit int
}

func NewEvaluationWorkspace(
	repository *overgodb.Store,
	runtime evaluation.Runtime,
	identity modelrecipe.ProgramIdentity,
	environment runrecord.Environment,
	commit string,
	historyLimit int,
	suitePaths []string,
) (*EvaluationWorkspace, error) {
	if historyLimit <= 0 {
		return nil, errors.New("evaluation workspace: history limit is invalid")
	}
	campaign, err := evaluation.NewCampaign(repository, runtime, identity, environment, commit)
	if err != nil {
		return nil, err
	}
	workspace := &EvaluationWorkspace{
		repository: repository, campaign: campaign, model: identity.Model, recipe: identity.Recipe,
		suites: make([]evaluation.CompiledSuite, 0, len(suitePaths)), byPlan: make(map[artifact.ID]int, len(suitePaths)),
		historyLimit: historyLimit,
	}
	admit := func(compiled evaluation.CompiledSuite) error {
		plan := compiled.Descriptor().Plan
		if _, duplicate := workspace.byPlan[plan]; duplicate {
			return errors.New("evaluation workspace: duplicate compiled plan")
		}
		workspace.byPlan[plan] = len(workspace.suites)
		workspace.suites = append(workspace.suites, compiled)
		return nil
	}
	for _, path := range suitePaths {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		compiled, err := evaluation.CompileSuite(data, campaign.Authorities())
		if err != nil {
			return nil, fmt.Errorf("evaluation workspace: compile %q: %w", path, err)
		}
		if err := admit(compiled); err != nil {
			return nil, err
		}
	}
	if len(suitePaths) == 0 {
		// No suite files named: the store is the configuration. The active
		// benchmark catalog compiles into this model's suites, filtered by
		// the model's declared eval domain, so the workbench evaluates out
		// of the box against benchmarks its scores mean something on.
		derived, _, err := evaluation.DeriveStoreSuites(context.Background(), repository, campaign.Authorities())
		if err != nil {
			return nil, fmt.Errorf("evaluation workspace: derive store suites: %w", err)
		}
		domains, declared, err := modelartifact.EvalDomains(context.Background(), repository, identity.Model)
		if err != nil {
			return nil, err
		}
		for _, compiled := range evaluation.FilterSuitesForDomains(derived, domains, declared) {
			if err := admit(compiled); err != nil {
				return nil, err
			}
		}
	}
	if len(workspace.suites) == 0 {
		return nil, errors.New("evaluation workspace: suites are absent")
	}
	sort.Slice(workspace.suites, func(i, j int) bool {
		return workspace.suites[i].Descriptor().Plan.String() < workspace.suites[j].Descriptor().Plan.String()
	})
	for index := range workspace.suites {
		workspace.byPlan[workspace.suites[index].Descriptor().Plan] = index
	}
	return workspace, nil
}

func (workspace *EvaluationWorkspace) EvaluationCapabilities(context.Context) ([]EvaluationCapability, error) {
	if workspace == nil || workspace.campaign == nil {
		return nil, errors.New("evaluation workspace: unavailable")
	}
	capabilities := make([]EvaluationCapability, len(workspace.suites))
	for index, suite := range workspace.suites {
		capabilities[index] = EvaluationCapability{Model: workspace.model, Recipe: workspace.recipe, Suite: suite.Descriptor()}
	}
	return capabilities, nil
}

func (workspace *EvaluationWorkspace) ExecuteEvaluation(
	ctx context.Context,
	model artifact.ID,
	plans []artifact.ID,
	reporter operation.Reporter,
) (operation.Completion, error) {
	if workspace == nil || model != workspace.model || len(plans) == 0 || reporter == nil {
		return operation.Completion{}, errors.New("evaluation workspace: request is not admitted")
	}
	intent, err := artifact.JSONID(artifact.KindEvidence, struct {
		Version uint16        `json:"version"`
		Model   artifact.ID   `json:"model"`
		Recipe  artifact.ID   `json:"recipe"`
		Plans   []artifact.ID `json:"plans"`
	}{Version: artifact.InitialDocumentVersion, Model: model, Recipe: workspace.recipe, Plans: slices.Clone(plans)})
	if err != nil {
		return operation.Completion{}, err
	}
	return operation.ExecuteReentrant(ctx, workspace.repository, reporter, operation.Request{
		Task: recipe.TaskInference, Recipe: workspace.recipe,
	}, intent, func(ctx context.Context) (operation.Completion, error) {
		return workspace.executeEvaluation(ctx, plans, reporter)
	})
}

func (workspace *EvaluationWorkspace) executeEvaluation(
	ctx context.Context,
	plans []artifact.ID,
	reporter operation.Reporter,
) (operation.Completion, error) {
	total := uint64(len(plans))
	reporter.Progress(0, &total)
	outputs := make([]artifact.ID, 0, 2*len(plans))
	var lastRun artifact.ID
	seen := make(map[artifact.ID]struct{}, len(plans))
	for index, plan := range plans {
		suiteIndex, admitted := workspace.byPlan[plan]
		if _, duplicate := seen[plan]; !admitted || duplicate {
			return operation.Completion{Run: lastRun, Outputs: outputs}, errors.New("evaluation workspace: compiled plan is not admitted")
		}
		seen[plan] = struct{}{}
		result, err := workspace.campaign.Evaluate(ctx, workspace.suites[suiteIndex])
		if err != nil {
			if result.Run.Valid() {
				lastRun = result.Run
			}
			return operation.Completion{Run: lastRun, Outputs: outputs}, err
		}
		lastRun = result.Run
		outputs = append(outputs, result.Evaluation, result.Report)
		for _, metric := range result.Metrics {
			reporter.Metric(operation.Metric{Name: metric.Name, Value: metric.Value, Unit: metric.Unit})
		}
		reporter.Progress(uint64(index+1), &total)
	}
	reporter.Publishing()
	return operation.Completion{Run: lastRun, Outputs: outputs}, nil
}

// ExecuteSupervisedGroundedReplay runs one admitted replay through the existing operation owner.
func (workspace *EvaluationWorkspace) ExecuteSupervisedGroundedReplay(
	ctx context.Context,
	model artifact.ID,
	request evaluation.SupervisedGroundedReplayRequest,
	reporter operation.Reporter,
) (operation.Completion, error) {
	if workspace == nil || workspace.repository == nil || model != workspace.model ||
		request.RunRecipe != workspace.recipe || reporter == nil {
		return operation.Completion{}, errors.New("evaluation workspace: grounded replay request is not admitted")
	}
	intent, err := artifact.JSONID(artifact.KindEvidence, struct {
		Version uint16                                     `json:"version"`
		Model   artifact.ID                                `json:"model"`
		Recipe  artifact.ID                                `json:"recipe"`
		Request evaluation.SupervisedGroundedReplayRequest `json:"request"`
	}{
		Version: artifact.InitialDocumentVersion,
		Model:   model,
		Recipe:  workspace.recipe,
		Request: request,
	})
	if err != nil {
		return operation.Completion{}, err
	}
	return operation.ExecuteReentrant(ctx, workspace.repository, reporter, operation.Request{
		Task: recipe.TaskInference, Recipe: workspace.recipe,
	}, intent, func(ctx context.Context) (operation.Completion, error) {
		total := uint64(1)
		result, err := evaluation.RunSupervisedGroundedReplay(ctx, workspace.repository, request)
		if err != nil {
			return operation.Completion{}, err
		}
		reporter.Progress(total, &total)
		reporter.Publishing()
		return operation.Completion{Run: result.Run.ID, Outputs: groundedReplayOutputs(result)}, nil
	})
}

func groundedReplayOutputs(result evaluation.SupervisedGroundedReplayResult) []artifact.ID {
	outputs := []artifact.ID{
		result.Summary,
		result.Episode.ID,
		result.ArcProjection.ID,
		result.ArcSelection.ID,
		result.Route.ID,
		result.RouteFailure.ID,
		result.Fallback.ID,
		result.Trajectories[0].ID,
		result.Trajectories[1].ID,
		result.Baseline.ID,
		result.Trial.ID,
		result.Transform.ID(),
		result.Proposal.ID,
		result.Promotion.ID,
	}
	for _, probe := range result.Probes {
		outputs = append(outputs, probe.ID)
	}
	seen := make(map[artifact.ID]struct{}, len(outputs))
	unique := outputs[:0]
	for _, id := range outputs {
		if !id.Valid() {
			continue
		}
		if _, duplicate := seen[id]; duplicate {
			continue
		}
		seen[id] = struct{}{}
		unique = append(unique, id)
	}
	return unique
}

func (workspace *EvaluationWorkspace) EvaluationHistory(ctx context.Context, model artifact.ID) ([]EvaluationHistoryEntry, error) {
	if workspace == nil || workspace.repository == nil || model != workspace.model {
		return nil, errors.New("evaluation workspace: model is not selected")
	}
	return workspace.campaign.History(ctx, workspace.suites, workspace.historyLimit)
}

func (workspace *EvaluationWorkspace) EvaluationReport(ctx context.Context, id artifact.ID) (EvaluationReport, error) {
	if workspace == nil || workspace.repository == nil || id.Kind() != artifact.KindEvaluation {
		return EvaluationReport{}, errors.New("evaluation workspace: invalid report")
	}
	content, found, err := artifact.ReadContent(ctx, workspace.repository, id)
	if err != nil {
		return EvaluationReport{}, err
	}
	if !found {
		return EvaluationReport{}, errors.New("evaluation workspace: report is absent")
	}
	return EvaluationReport{
		ID: id, MediaType: content.Descriptor.MediaType, Schema: content.Descriptor.Schema,
		Body: slices.Clone(content.Data),
	}, nil
}

func (workspace *EvaluationWorkspace) EvaluationFailures(ctx context.Context, id artifact.ID) ([]EvaluationFailure, error) {
	report, err := workspace.EvaluationReport(ctx, id)
	if err != nil {
		return nil, err
	}
	failures, err := evaluation.FailedObservations(report.MediaType, report.Schema, report.Body)
	if err != nil {
		return nil, err
	}
	result := make([]EvaluationFailure, len(failures))
	for index, failure := range failures {
		result[index] = EvaluationFailure{Name: failure.Name, Observation: failure.Body}
	}
	return result, nil
}

func (workspace *EvaluationWorkspace) CompareEvaluations(ctx context.Context, left, right artifact.ID) (EvaluationComparison, error) {
	leftRecord, err := workspace.loadEvaluation(ctx, left)
	if err != nil {
		return EvaluationComparison{}, err
	}
	rightRecord, err := workspace.loadEvaluation(ctx, right)
	if err != nil {
		return EvaluationComparison{}, err
	}
	if leftRecord.Dataset != rightRecord.Dataset {
		return EvaluationComparison{}, errors.New("evaluation workspace: comparison datasets differ")
	}
	rightByName := make(map[string]runrecord.Metric, len(rightRecord.Metrics))
	for _, metric := range rightRecord.Metrics {
		rightByName[metric.Name] = metric
	}
	deltas := make([]EvaluationMetricDelta, 0, len(leftRecord.Metrics))
	for _, leftMetric := range leftRecord.Metrics {
		rightMetric, found := rightByName[leftMetric.Name]
		if !found || leftMetric.Unit != rightMetric.Unit || leftMetric.Direction != rightMetric.Direction {
			return EvaluationComparison{}, errors.New("evaluation workspace: comparison metric contracts differ")
		}
		delta := rightMetric.Value - leftMetric.Value
		deltas = append(deltas, EvaluationMetricDelta{
			Name: leftMetric.Name, Left: leftMetric.Value, Right: rightMetric.Value, Delta: delta,
			Direction: leftMetric.Direction,
			Improved: leftMetric.Direction == runrecord.DirectionMaximize && delta > 0 ||
				leftMetric.Direction == runrecord.DirectionMinimize && delta < 0,
		})
	}
	if len(deltas) != len(rightRecord.Metrics) {
		return EvaluationComparison{}, errors.New("evaluation workspace: comparison metric sets differ")
	}
	return EvaluationComparison{Left: left, Right: right, Metrics: deltas}, nil
}

func (workspace *EvaluationWorkspace) loadEvaluation(ctx context.Context, id artifact.ID) (runrecord.Evaluation, error) {
	if id.Kind() != artifact.KindEvaluation {
		return runrecord.Evaluation{}, errors.New("evaluation workspace: invalid evaluation identity")
	}
	record, err := runrecord.RequireEvaluation(ctx, workspace.repository, id)
	if err != nil || record.Recipe != workspace.recipe {
		return runrecord.Evaluation{}, errors.New("evaluation workspace: evaluation is not selected-model evidence")
	}
	return record, nil
}

type evaluationRunRequest struct {
	Model          artifact.ID                                 `json:"model"`
	Plans          []artifact.ID                               `json:"plans,omitempty"`
	GroundedReplay *evaluation.SupervisedGroundedReplayRequest `json:"grounded_replay,omitempty"`
}

func (h *Handler) evaluationCapabilities(response http.ResponseWriter, request *http.Request) {
	workspace := h.config.Evaluation
	if workspace == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "evaluation workspace is unavailable")
		return
	}
	capabilities, err := workspace.EvaluationCapabilities(request.Context())
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	writeJSON(response, http.StatusOK, capabilities)
}

func (h *Handler) evaluationRun(response http.ResponseWriter, request *http.Request) {
	workspace := h.config.Evaluation
	if workspace == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "evaluation workspace is unavailable")
		return
	}
	var body evaluationRunRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	capabilities, err := workspace.EvaluationCapabilities(request.Context())
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	recipeID, err := admitEvaluationRequest(capabilities, body, h.config.MaxStoredResponses)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	id, err := h.operations.Submit(context.WithoutCancel(request.Context()), operation.Request{
		Task: recipe.TaskInference, Recipe: recipeID,
	}, func(ctx context.Context, reporter operation.Reporter) (operation.Completion, error) {
		return h.executeObservedOperation(ctx, reporter, recipe.TaskInference, recipeID, body.Model,
			func(ctx context.Context, reporter operation.Reporter) (operation.Completion, error) {
				if body.GroundedReplay != nil {
					return workspace.ExecuteSupervisedGroundedReplay(ctx, body.Model, *body.GroundedReplay, reporter)
				}
				return workspace.ExecuteEvaluation(ctx, body.Model, body.Plans, reporter)
			})
	})
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	writeJSON(response, http.StatusAccepted, workflowResponse{Operation: id})
}

func admitEvaluationRequest(capabilities []EvaluationCapability, request evaluationRunRequest, limit int) (artifact.ID, error) {
	if !request.Model.Valid() || limit <= 0 || (len(request.Plans) == 0) == (request.GroundedReplay == nil) {
		return artifact.ID{}, errors.New("evaluation workspace: exactly one bounded evaluation request is required")
	}
	admitted := make(map[artifact.ID]artifact.ID, len(capabilities))
	modelRecipes := make(map[artifact.ID]struct{})
	for _, capability := range capabilities {
		if capability.Model == request.Model && capability.Recipe.Kind() == artifact.KindRecipe && capability.Suite.Plan.Kind() == artifact.KindProfile {
			admitted[capability.Suite.Plan] = capability.Recipe
			modelRecipes[capability.Recipe] = struct{}{}
		}
	}
	if request.GroundedReplay != nil {
		if _, ok := modelRecipes[request.GroundedReplay.RunRecipe]; !ok {
			return artifact.ID{}, errors.New("evaluation workspace: grounded replay recipe is not admitted for model")
		}
		if err := admitGroundedReplayRequest(*request.GroundedReplay, limit); err != nil {
			return artifact.ID{}, err
		}
		return request.GroundedReplay.RunRecipe, nil
	}
	if len(request.Plans) > limit {
		return artifact.ID{}, errors.New("evaluation workspace: plan selection exceeds the admitted bound")
	}
	seen := make(map[artifact.ID]struct{}, len(request.Plans))
	var recipeID artifact.ID
	for _, plan := range request.Plans {
		planRecipe, ok := admitted[plan]
		if !ok {
			return artifact.ID{}, errors.New("evaluation workspace: plan is not admitted for model")
		}
		if _, duplicate := seen[plan]; duplicate {
			return artifact.ID{}, errors.New("evaluation workspace: duplicate plan")
		}
		if recipeID.Valid() && recipeID != planRecipe {
			return artifact.ID{}, errors.New("evaluation workspace: plans span recipes")
		}
		recipeID = planRecipe
		seen[plan] = struct{}{}
	}
	return recipeID, nil
}

func admitGroundedReplayRequest(request evaluation.SupervisedGroundedReplayRequest, limit int) error {
	bounded32 := func(values ...uint32) bool {
		for _, value := range values {
			if value == 0 || uint64(value) > uint64(limit) {
				return false
			}
		}
		return true
	}
	bounded64 := func(max uint64, values ...uint64) bool {
		for _, value := range values {
			if value == 0 || value > max {
				return false
			}
		}
		return true
	}
	if request.RunRecipe.Kind() != artifact.KindRecipe {
		return errors.New("evaluation workspace: grounded replay recipe is invalid")
	}
	if _, err := request.EpisodeBounds.Identify(); err != nil ||
		!bounded32(
			request.EpisodeBounds.MaximumEpisodes,
			request.EpisodeBounds.MaximumEventsPerEpisode,
			request.EpisodeBounds.MaximumCallsPerEpisode,
			request.EpisodeBounds.MaximumReferencesPerEpisode,
		) || len(request.EpisodeSources) == 0 || len(request.EpisodeSources) > limit ||
		uint64(len(request.EpisodeSources)) > uint64(request.EpisodeBounds.MaximumEpisodes) {
		return errors.Join(errors.New("evaluation workspace: grounded replay episode projection is unbounded"), err)
	}
	for _, source := range request.EpisodeSources {
		if source.Attempt.Kind() != artifact.KindEvidence || source.Trajectory.Kind() != artifact.KindEvidence {
			return errors.New("evaluation workspace: grounded replay episode source is invalid")
		}
	}
	if request.ArcTrajectory.Kind() != artifact.KindEvidence || request.ArcTokenizer.Kind() != artifact.KindTokenizer ||
		request.ArcCounter.Kind() != artifact.KindProfile || strings.TrimSpace(request.ArcBranch) != request.ArcBranch ||
		len(request.ArcBranch) > maxRequestBytes {
		return errors.New("evaluation workspace: grounded replay interaction authority is invalid")
	}
	if _, err := request.ArcBounds.Identify(); err != nil ||
		!bounded32(
			request.ArcBounds.MaximumBranches,
			request.ArcBounds.MaximumArcs,
			request.ArcBounds.MaximumProjectedSequences,
			request.ArcBounds.MaximumToolPairs,
			request.ArcBounds.MaximumReferences,
		) {
		return errors.Join(errors.New("evaluation workspace: grounded replay interaction projection is unbounded"), err)
	}
	if _, err := request.ArcSelection.Identify(); err != nil ||
		!bounded64(uint64(maxRequestBytes), request.ArcSelection.MaxTokens, request.ArcSelection.MaxBytes) ||
		!bounded64(uint64(limit), request.ArcSelection.MaxDocuments, request.ArcSelection.MaxResults) ||
		!bounded32(request.ArcSelection.MaxDepth) {
		return errors.Join(errors.New("evaluation workspace: grounded replay interaction selection is unbounded"), err)
	}
	if len(request.RetrievalSources) == 0 || len(request.RetrievalSources) > limit {
		return errors.New("evaluation workspace: grounded replay retrieval sources are unbounded")
	}
	totalDocuments := 0
	for _, source := range request.RetrievalSources {
		if source.Kind != artifact.KindFile && source.Kind != artifact.KindDatasetShard ||
			len(source.Documents) == 0 || len(source.Documents) > limit-totalDocuments {
			return errors.New("evaluation workspace: grounded replay retrieval source is invalid")
		}
		totalDocuments += len(source.Documents)
		for _, document := range source.Documents {
			if document.Source.Valid() || document.Text == "" || len(document.Structure) > limit {
				return errors.New("evaluation workspace: grounded replay retrieval document is invalid")
			}
		}
	}
	baseline, baselineErr := evaluation.NewRetrievalCase(request.BaselineCase)
	trial, trialErr := evaluation.NewRetrievalCase(request.TrialCase)
	if baselineErr != nil || trialErr != nil || baseline.ID == trial.ID ||
		len(baseline.Judgments) > limit || len(trial.Judgments) > limit ||
		request.BaselineReceipt.Kind() != artifact.KindEvidence || request.TrialReceipt.Kind() != artifact.KindEvidence ||
		request.BaselineReceipt == request.TrialReceipt {
		return errors.Join(errors.New("evaluation workspace: grounded replay retrieval comparison is invalid"), baselineErr, trialErr)
	}
	if len(request.Transform.Inputs) == 0 || len(request.Transform.Inputs) > limit ||
		len(request.Transform.Outputs) == 0 || len(request.Transform.Outputs) > limit {
		return errors.New("evaluation workspace: grounded replay transform is unbounded")
	}
	if _, err := dataset.NewDatasetTransform(request.Transform); err != nil {
		return errors.Join(errors.New("evaluation workspace: grounded replay transform is invalid"), err)
	}
	if len(request.Probes) < 2 || len(request.Probes) > limit || len(request.RouteCandidates) != len(request.Probes) ||
		request.RouteMaxAttempts < 2 || uint64(request.RouteMaxAttempts) > uint64(limit) {
		return errors.New("evaluation workspace: grounded replay route is unbounded")
	}
	for _, probe := range request.Probes {
		if probe.Case.Kind() != artifact.KindRecipe || probe.Profile.Kind() != artifact.KindProfile ||
			probe.Placement.ID.Kind() != artifact.KindProfile || probe.Environment.Kind() != artifact.KindEvidence ||
			probe.Trace.Kind() != artifact.KindEvidence || probe.StageReceipt.Kind() != artifact.KindEvidence ||
			probe.CheckDecision.Kind() != artifact.KindEvidence || probe.Observation.Kind() != artifact.KindEvidence {
			return errors.New("evaluation workspace: grounded replay production probe is invalid")
		}
	}
	if strings.TrimSpace(request.Route.Intent) != request.Route.Intent || request.Route.Intent == "" ||
		len(request.Route.Intent) > maxRequestBytes || request.Route.Authority.Kind() != artifact.KindEvidence ||
		len(request.Route.AllowedBoundaries) == 0 || len(request.Route.AllowedBoundaries) > limit {
		return errors.New("evaluation workspace: grounded replay route request is invalid")
	}
	for _, boundary := range request.Route.AllowedBoundaries {
		if !boundary.Valid() {
			return errors.New("evaluation workspace: grounded replay route boundary is invalid")
		}
	}
	seenCandidates := make(map[artifact.ID]struct{}, len(request.RouteCandidates))
	for _, candidate := range request.RouteCandidates {
		if candidate.Probe.Kind() != artifact.KindEvidence || candidate.Manual.Kind() != artifact.KindRecipe {
			return errors.New("evaluation workspace: grounded replay route candidate is invalid")
		}
		if _, duplicate := seenCandidates[candidate.Probe]; duplicate {
			return errors.New("evaluation workspace: grounded replay route candidate is duplicated")
		}
		seenCandidates[candidate.Probe] = struct{}{}
	}
	if request.RouteFailure.Version != 0 && request.RouteFailure.Version != artifact.InitialDocumentVersion || request.RouteFailure.ID.Valid() ||
		strings.TrimSpace(request.RouteFailure.Source) == "" || request.RouteFailure.Message == "" ||
		request.RouteFailure.ObservedUnixNS <= 0 || request.RouteFailure.Run.Valid() {
		return errors.New("evaluation workspace: grounded replay failure observation is invalid")
	}
	seenTrajectories := make(map[artifact.ID]struct{}, len(request.Trajectories))
	for _, trajectory := range request.Trajectories {
		if trajectory.Kind() != artifact.KindEvidence {
			return errors.New("evaluation workspace: grounded replay trajectory is invalid")
		}
		if _, duplicate := seenTrajectories[trajectory]; duplicate {
			return errors.New("evaluation workspace: grounded replay trajectory is duplicated")
		}
		seenTrajectories[trajectory] = struct{}{}
	}
	if request.KnowledgeAdmission.Kind() != artifact.KindEvidence {
		return errors.New("evaluation workspace: grounded replay knowledge admission is invalid")
	}
	return nil
}

func (h *Handler) evaluationHistory(response http.ResponseWriter, request *http.Request) {
	workspace, model, ok := h.evaluationQuery(response, request, "model", artifact.KindModel)
	if !ok {
		return
	}
	history, err := workspace.EvaluationHistory(request.Context(), model)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	writeJSON(response, http.StatusOK, history)
}

func (h *Handler) evaluationReport(response http.ResponseWriter, request *http.Request) {
	workspace, id, ok := h.evaluationQuery(response, request, "id", artifact.KindEvaluation)
	if !ok {
		return
	}
	report, err := workspace.EvaluationReport(request.Context(), id)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	writeJSON(response, http.StatusOK, report)
}

func (h *Handler) evaluationFailures(response http.ResponseWriter, request *http.Request) {
	workspace, id, ok := h.evaluationQuery(response, request, "id", artifact.KindEvaluation)
	if !ok {
		return
	}
	failures, err := workspace.EvaluationFailures(request.Context(), id)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	writeJSON(response, http.StatusOK, failures)
}

func (h *Handler) evaluationCompare(response http.ResponseWriter, request *http.Request) {
	workspace := h.config.Evaluation
	if workspace == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "evaluation workspace is unavailable")
		return
	}
	left, err := parseQueryArtifact(request, "left", artifact.KindEvaluation)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	right, err := parseQueryArtifact(request, "right", artifact.KindEvaluation)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	comparison, err := workspace.CompareEvaluations(request.Context(), left, right)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	writeJSON(response, http.StatusOK, comparison)
}

func (h *Handler) evaluationQuery(
	response http.ResponseWriter,
	request *http.Request,
	name string,
	kind artifact.Kind,
) (EvaluationWorkspaceAPI, artifact.ID, bool) {
	workspace := h.config.Evaluation
	if workspace == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "evaluation workspace is unavailable")
		return nil, artifact.ID{}, false
	}
	id, err := parseQueryArtifact(request, name, kind)
	if err != nil {
		writeInvalidRequest(response, err)
		return nil, artifact.ID{}, false
	}
	return workspace, id, true
}

func parseQueryArtifact(request *http.Request, name string, kind artifact.Kind) (artifact.ID, error) {
	id, err := artifact.ParseID(request.URL.Query().Get(name))
	if err != nil || id.Kind() != kind {
		return artifact.ID{}, fmt.Errorf("evaluation workspace: invalid %s identity", name)
	}
	return id, nil
}
