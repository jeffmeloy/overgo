package server

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	"overgo/internal/artifact"
	"overgo/internal/bridgegraph"
	"overgo/internal/composition"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/strictjson"
)

type compositionCompletion struct {
	ContractTested   bool `json:"contract_tested"`
	RuntimeWired     bool `json:"runtime_wired"`
	CUDAVerified     bool `json:"cuda_verified"`
	EvidencePromoted bool `json:"evidence_promoted"`
	ProductionActive bool `json:"production_active"`
}

type compositionGraphView struct {
	Source    artifact.ID          `json:"source"`
	Capture   artifact.ID          `json:"capture_contract"`
	Bridge    artifact.ID          `json:"bridge"`
	Weights   artifact.ID          `json:"weights"`
	Operator  bridgegraph.Operator `json:"operator"`
	Injection artifact.ID          `json:"injection_contract"`
	Target    artifact.ID          `json:"target"`
}

type compositionTrainingView struct {
	Policy      artifact.ID `json:"policy"`
	MetricNames []string    `json:"metric_names"`
}

type compositionEvaluationView struct {
	Evidence                artifact.ID `json:"evidence"`
	Policy                  artifact.ID `json:"policy"`
	HeldOutSplit            artifact.ID `json:"held_out_split"`
	RegressionSet           artifact.ID `json:"regression_set"`
	Evaluator               artifact.ID `json:"evaluator"`
	TrialCount              int         `json:"trial_count"`
	WorstHeldOutGain        float64     `json:"worst_held_out_gain"`
	WorstSourceDependence   float64     `json:"worst_source_dependence"`
	WorstRegression         float64     `json:"worst_regression"`
	SeedSpread              float64     `json:"seed_spread"`
	WorstLatencyIncrease    float64     `json:"worst_latency_increase"`
	WorstDeviceByteIncrease uint64      `json:"worst_device_byte_increase"`
}

type compositionRuntimeView struct {
	Plan          artifact.ID                      `json:"plan"`
	CacheIdentity artifact.ID                      `json:"cache_identity"`
	Sessions      modelrecipe.ComponentSessionPlan `json:"sessions"`
}

type compositionGenerationView struct {
	Promotion artifact.ID `json:"promotion"`
	Evidence  artifact.ID `json:"evidence"`
	Output    artifact.ID `json:"output"`
}

type compositionWorkflowView struct {
	Recipe     artifact.ID                `json:"recipe"`
	Source     artifact.ID                `json:"source"`
	Target     artifact.ID                `json:"target"`
	Task       recipe.Task                `json:"task"`
	Compatible bool                       `json:"compatible"`
	Refusal    string                     `json:"refusal,omitempty"`
	Active     bool                       `json:"active"`
	Graph      *compositionGraphView      `json:"graph,omitempty"`
	Training   *compositionTrainingView   `json:"training,omitempty"`
	Evaluation *compositionEvaluationView `json:"evaluation,omitempty"`
	Runtime    *compositionRuntimeView    `json:"runtime,omitempty"`
	Generation *compositionGenerationView `json:"generation,omitempty"`
	Completion compositionCompletion      `json:"completion"`
}

type compositionInventoryResponse struct {
	Compositions []compositionWorkflowView `json:"compositions"`
}

type compositionActivationRequest struct {
	Recipe   artifact.ID  `json:"recipe"`
	Previous *artifact.ID `json:"previous,omitempty"`
}

type compositionActivationResponse struct {
	Recipe artifact.ID `json:"recipe"`
	Alias  string      `json:"alias"`
	Commit string      `json:"commit"`
}

type compositionGenerationRequest struct {
	Source artifact.ID `json:"source"`
	Target artifact.ID `json:"target"`
	Task   recipe.Task `json:"task"`
}

func (h *Handler) compositionInventory(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	if h.repository == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "composition repository is unavailable")
		return
	}
	views := make([]compositionWorkflowView, 0)
	_, err := overgodb.VisitDecodedDocuments(request.Context(), h.repository, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindRecipe, MediaType: composition.CompositionRecipeMediaType, Schema: composition.CompositionRecipeSchema,
		}}, Order: overgodb.DocumentNewestFirst,
	}, composition.ParseCompositionRecipe, func(_ overgodb.DocumentView, value composition.CompositionRecipe) error {
		views = append(views, h.compositionWorkflowView(request.Context(), value))
		return nil
	})
	if err != nil {
		writeError(response, http.StatusInternalServerError, "composition_inventory_error", err.Error())
		return
	}
	writeJSON(response, http.StatusOK, compositionInventoryResponse{
		Compositions: views,
	})
}

func (h *Handler) compositionWorkflowView(ctx context.Context, value composition.CompositionRecipe) compositionWorkflowView {
	view := compositionWorkflowView{Recipe: value.ID}
	view.Source, view.Target, view.Task = value.SourceModel, value.TargetModel, value.Task
	bridge, err := composition.LoadBridgeDefinition(ctx, h.repository, value.BridgeDefinition)
	if err != nil {
		view.Refusal = err.Error()
		return view
	}
	promotion, err := composition.LoadRepresentationBridgePromotion(ctx, h.repository, value.Promotion)
	if err != nil {
		view.Refusal = err.Error()
		return view
	}
	view.Compatible = true
	view.Graph = &compositionGraphView{
		Source: value.SourceModel, Capture: value.SourceContract,
		Bridge: value.BridgeDefinition, Weights: value.BridgeWeights,
		Operator: bridge.Graph.Operator, Injection: value.TargetContract, Target: value.TargetModel,
	}
	view.Training = &compositionTrainingView{
		Policy: value.TrainingPolicy, MetricNames: []string{"bridge_loss", "bridge_update_l2"},
	}
	view.Evaluation = &compositionEvaluationView{
		Evidence: value.Promotion, Policy: value.PromotionPolicy,
		HeldOutSplit: promotion.HeldOutSplit, RegressionSet: promotion.RegressionSet,
		Evaluator: promotion.Evaluator, TrialCount: len(promotion.Trials),
		WorstHeldOutGain:      promotion.WorstHeldOutGain,
		WorstSourceDependence: promotion.WorstSourceDependence,
		WorstRegression:       promotion.WorstRegression, SeedSpread: promotion.SeedSpread,
		WorstLatencyIncrease:    promotion.WorstLatencyIncrease,
		WorstDeviceByteIncrease: promotion.WorstDeviceByteIncrease,
	}
	view.Completion.ContractTested = true
	view.Completion.EvidencePromoted = true
	active, found, activeErr := composition.ActiveComposition(ctx, h.repository, value.SourceModel, value.TargetModel, value.Task)
	if activeErr != nil {
		view.Compatible = false
		view.Refusal = activeErr.Error()
		return view
	}
	view.Active = found && active.ID == value.ID
	view.Completion.ProductionActive = view.Active
	if !view.Active {
		return view
	}
	plan, err := composition.CompileCompositionExecutionPlan(ctx, h.repository, value.SourceModel, value.TargetModel, value.Task)
	if err != nil {
		view.Compatible = false
		view.Refusal = err.Error()
		return view
	}
	view.Runtime = &compositionRuntimeView{Plan: plan.ID, CacheIdentity: plan.CacheIdentity, Sessions: plan.Sessions}
	view.Completion.RuntimeWired = true
	generation, err := inference.ResolveCompositeGenerationSurface(ctx, h.repository, value.SourceModel, value.TargetModel, value.Task)
	if err == nil {
		view.Generation = &compositionGenerationView{
			Promotion: generation.Promotion, Evidence: generation.Evidence, Output: generation.Output,
		}
		view.Completion.CUDAVerified = true
	}
	return view
}

func (h *Handler) compositeGeneration(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodPost) {
		return
	}
	if h.repository == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "composition repository is unavailable")
		return
	}
	var input compositionGenerationRequest
	if err := strictjson.Decode(request.Body, &input); err != nil {
		writeInvalidRequest(response, err)
		return
	}
	selection, err := inference.ResolveCompositeGenerationSurface(
		request.Context(), h.repository, input.Source, input.Target, input.Task,
	)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	writeJSON(response, http.StatusOK, selection)
}

func (h *Handler) activateComposition(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodPost) {
		return
	}
	if h.repository == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "composition repository is unavailable")
		return
	}
	var input compositionActivationRequest
	if err := strictjson.Decode(request.Body, &input); err != nil || !input.Recipe.Valid() {
		writeInvalidRequest(response, errors.Join(err, errors.New("valid composition recipe is required")))
		return
	}
	value, err := composition.LoadCompositionRecipe(request.Context(), h.repository, input.Recipe)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	previous := input.Previous
	if previous == nil {
		if current, found, resolveErr := composition.ActiveComposition(
			request.Context(), h.repository, value.SourceModel, value.TargetModel, value.Task,
		); resolveErr != nil {
			writeInvalidRequest(response, resolveErr)
			return
		} else if found {
			previous = &current.ID
		}
	}
	alias, err := composition.ActiveCompositionAlias(value.SourceModel, value.TargetModel, value.Task)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	_, sequence := h.repository.Head()
	batch, err := value.ActivationBatch(
		request.Context(), h.repository,
		fmt.Sprintf("composition.activate.%d.%s", sequence+1, value.ID.String()), previous,
	)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	commit, err := artifact.CommitBatch(request.Context(), h.repository, batch)
	if err != nil {
		if errors.Is(err, artifact.ErrCommitPrecondition) {
			writeError(response, http.StatusConflict, "composition_activation_conflict", err.Error())
			return
		}
		writeError(response, http.StatusInternalServerError, "composition_activation_error", err.Error())
		return
	}
	writeJSON(response, http.StatusOK, compositionActivationResponse{
		Recipe: value.ID, Alias: alias, Commit: commit.String(),
	})
}
