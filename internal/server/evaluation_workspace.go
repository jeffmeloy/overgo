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

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
)

type EvaluationCapability struct {
	Model  artifact.ID                `json:"model"`
	Recipe artifact.ID                `json:"recipe"`
	Suite  evaluation.SuiteDescriptor `json:"suite"`
}

type EvaluationHistoryEntry struct {
	Evaluation artifact.ID        `json:"evaluation,omitempty"`
	Run        artifact.ID        `json:"run"`
	Report     artifact.ID        `json:"report,omitempty"`
	Dataset    artifact.ID        `json:"dataset"`
	Recipe     artifact.ID        `json:"recipe"`
	Outcome    runrecord.Outcome  `json:"outcome"`
	Failure    string             `json:"failure,omitempty"`
	CodeCommit string             `json:"code_commit,omitempty"`
	MeasuredNS uint64             `json:"measured_ns,omitempty"`
	Metrics    []runrecord.Metric `json:"metrics"`
}

type EvaluationReport struct {
	ID        artifact.ID     `json:"id"`
	MediaType string          `json:"media_type"`
	Schema    string          `json:"schema"`
	Body      json.RawMessage `json:"body"`
}

type EvaluationFailure struct {
	Name        string          `json:"name,omitempty"`
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
	EvaluationHistory(context.Context, artifact.ID) ([]EvaluationHistoryEntry, error)
	EvaluationReport(context.Context, artifact.ID) (EvaluationReport, error)
	EvaluationFailures(context.Context, artifact.ID) ([]EvaluationFailure, error)
	CompareEvaluations(context.Context, artifact.ID, artifact.ID) (EvaluationComparison, error)
}

type EvaluationWorkspace struct {
	repository *repodb.Store
	campaign   *evaluation.Campaign
	model      artifact.ID
	recipe     artifact.ID
	suites     []evaluation.CompiledSuite
	byPlan     map[artifact.ID]int
}

func NewEvaluationWorkspace(
	repository *repodb.Store,
	runtime evaluation.Runtime,
	identity modelrecipe.ProgramIdentity,
	environment runrecord.Environment,
	commit string,
	suitePaths []string,
) (*EvaluationWorkspace, error) {
	campaign, err := evaluation.NewCampaign(repository, runtime, identity, environment, commit)
	if err != nil {
		return nil, err
	}
	workspace := &EvaluationWorkspace{
		repository: repository, campaign: campaign, model: identity.Model, recipe: identity.Recipe,
		suites: make([]evaluation.CompiledSuite, 0, len(suitePaths)), byPlan: make(map[artifact.ID]int, len(suitePaths)),
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
		plan := compiled.Descriptor().Plan
		if _, duplicate := workspace.byPlan[plan]; duplicate {
			return nil, errors.New("evaluation workspace: duplicate compiled plan")
		}
		workspace.byPlan[plan] = len(workspace.suites)
		workspace.suites = append(workspace.suites, compiled)
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

func (workspace *EvaluationWorkspace) EvaluationHistory(ctx context.Context, model artifact.ID) ([]EvaluationHistoryEntry, error) {
	if workspace == nil || workspace.repository == nil || model != workspace.model {
		return nil, errors.New("evaluation workspace: model is not selected")
	}
	evaluationResult, err := workspace.repository.Query(ctx, repodb.Query{Kind: artifact.KindEvaluation, MaxResults: repodb.MaxQueryResults})
	if err != nil {
		return nil, err
	}
	evaluationByRun := make(map[artifact.ID]runrecord.Evaluation)
	for _, descriptor := range evaluationResult.Artifacts {
		if descriptor.MediaType != runrecord.EvaluationMediaType || descriptor.Schema != runrecord.EvaluationSchema {
			continue
		}
		content, found, err := workspace.repository.Content(ctx, descriptor.ID)
		if err != nil {
			return nil, err
		}
		if !found {
			continue
		}
		record, err := runrecord.ParseEvaluation(content.Data)
		if err != nil || record.Recipe != workspace.recipe {
			continue
		}
		evaluationByRun[record.Run] = record
	}
	runResult, err := workspace.repository.Query(ctx, repodb.Query{Kind: artifact.KindRun, MaxResults: repodb.MaxQueryResults})
	if err != nil {
		return nil, err
	}
	entries := make([]EvaluationHistoryEntry, 0, len(evaluationByRun))
	for _, descriptor := range runResult.Artifacts {
		if descriptor.MediaType != runrecord.RunMediaType {
			continue
		}
		runContent, found, err := workspace.repository.Content(ctx, descriptor.ID)
		if err != nil || !found {
			if err != nil {
				return nil, err
			}
			continue
		}
		run, err := runrecord.ParseRun(runContent.Data)
		if err != nil || run.Recipe != workspace.recipe || len(run.Inputs) == 0 {
			continue
		}
		suiteIndex, evaluationPlan := workspace.byPlan[run.Inputs[0]]
		if !evaluationPlan {
			continue
		}
		record := evaluationByRun[run.ID]
		entry := EvaluationHistoryEntry{
			Evaluation: record.ID, Run: run.ID, Dataset: workspace.suites[suiteIndex].Descriptor().Dataset, Recipe: run.Recipe,
			Outcome: run.Outcome, Failure: run.Failure, CodeCommit: run.CodeCommit,
			MeasuredNS: run.MeasuredNS, Metrics: slices.Clone(record.Metrics),
		}
		if len(run.Outputs) > 0 {
			entry.Report = run.Outputs[0]
		}
		entries = append(entries, entry)
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Run.String() < entries[j].Run.String() })
	return entries, nil
}

func (workspace *EvaluationWorkspace) EvaluationReport(ctx context.Context, id artifact.ID) (EvaluationReport, error) {
	if workspace == nil || workspace.repository == nil || id.Kind() != artifact.KindEvaluation {
		return EvaluationReport{}, errors.New("evaluation workspace: invalid report")
	}
	content, found, err := workspace.repository.Content(ctx, id)
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
	content, found, err := workspace.repository.Content(ctx, id)
	if err != nil {
		return runrecord.Evaluation{}, err
	}
	if !found {
		return runrecord.Evaluation{}, errors.New("evaluation workspace: evaluation is absent")
	}
	record, err := runrecord.ParseEvaluation(content.Data)
	if err != nil || record.Recipe != workspace.recipe {
		return runrecord.Evaluation{}, errors.New("evaluation workspace: evaluation is not selected-model evidence")
	}
	return record, nil
}

type evaluationRunRequest struct {
	Model artifact.ID   `json:"model"`
	Plans []artifact.ID `json:"plans"`
}

func (h *Handler) evaluationCapabilities(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
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
	if !requireMethod(response, request, http.MethodPost) {
		return
	}
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
	if err := admitEvaluationRequest(capabilities, body); err != nil {
		writeInvalidRequest(response, err)
		return
	}
	recipeID := capabilities[0].Recipe
	id, err := h.operations.Submit(context.WithoutCancel(request.Context()), operation.Request{
		Task: recipe.TaskInference, Recipe: recipeID,
	}, func(ctx context.Context, reporter operation.Reporter) (operation.Completion, error) {
		return workspace.ExecuteEvaluation(ctx, body.Model, body.Plans, reporter)
	})
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	writeJSON(response, http.StatusAccepted, workflowResponse{Operation: id})
}

func admitEvaluationRequest(capabilities []EvaluationCapability, request evaluationRunRequest) error {
	if !request.Model.Valid() || len(request.Plans) == 0 {
		return errors.New("evaluation workspace: model and plans are required")
	}
	admitted := make(map[artifact.ID]struct{}, len(capabilities))
	for _, capability := range capabilities {
		if capability.Model == request.Model && capability.Recipe.Kind() == artifact.KindRecipe && capability.Suite.Plan.Kind() == artifact.KindProfile {
			admitted[capability.Suite.Plan] = struct{}{}
		}
	}
	seen := make(map[artifact.ID]struct{}, len(request.Plans))
	for _, plan := range request.Plans {
		if _, ok := admitted[plan]; !ok {
			return errors.New("evaluation workspace: plan is not admitted for model")
		}
		if _, duplicate := seen[plan]; duplicate {
			return errors.New("evaluation workspace: duplicate plan")
		}
		seen[plan] = struct{}{}
	}
	return nil
}

func (h *Handler) evaluationHistory(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
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
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
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
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
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
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
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
