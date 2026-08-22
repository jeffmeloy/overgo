package server

import (
	"context"
	"errors"
	"net/http"

	"overgo/internal/artifact"
	"overgo/internal/bridgegraph"
	"overgo/internal/composition"
	"overgo/internal/repodb"
	"overgo/internal/representation"
	"overgo/internal/strictjson"
)

// CompositionEvidence exposes the exact training, evaluation, and promotion
// authorities that govern one production candidate.
type CompositionEvidence struct {
	TrainingPolicy  artifact.ID `json:"training_policy"`
	PromotionPolicy artifact.ID `json:"promotion_policy"`
	Promotion       artifact.ID `json:"promotion"`
	HeldOutSplit    artifact.ID `json:"held_out_split"`
	RegressionSet   artifact.ID `json:"regression_set"`
	Evaluator       artifact.ID `json:"evaluator"`
}

// CompositionCandidate is one RepoDB-owned composition lifecycle view.
type CompositionCandidate struct {
	ID         artifact.ID                                `json:"id"`
	Compatible bool                                       `json:"compatible"`
	Refusal    string                                     `json:"refusal,omitempty"`
	Active     bool                                       `json:"active"`
	Recipe     *composition.CompositionRecipe             `json:"recipe,omitempty"`
	Source     *representation.Contract                   `json:"source,omitempty"`
	Target     *representation.Contract                   `json:"target,omitempty"`
	Graph      *bridgegraph.Definition                    `json:"graph,omitempty"`
	Evidence   *CompositionEvidence                       `json:"evidence,omitempty"`
	Promotion  *composition.RepresentationBridgePromotion `json:"promotion,omitempty"`
	Runtime    *composition.CompositionExecutionPlan      `json:"runtime,omitempty"`
}

// CompositionInventory is a bounded snapshot of every composition recipe.
type CompositionInventory struct {
	Head       string                 `json:"head"`
	Sequence   uint64                 `json:"sequence"`
	Truncated  bool                   `json:"truncated,omitempty"`
	Candidates []CompositionCandidate `json:"candidates"`
}

// CompositionActivation records the exact alias transition committed by an
// activation request.
type CompositionActivation struct {
	Recipe artifact.ID `json:"recipe"`
	Commit string      `json:"commit"`
}

// CompositionWorkflowAPI lets an alternate workspace provide the same HTTP
// and GUI lifecycle without coupling the surface to one storage backend.
type CompositionWorkflowAPI interface {
	// CompositionInventory returns a bounded, immutable lifecycle snapshot.
	CompositionInventory(context.Context) (CompositionInventory, error)
	// ActivateComposition commits one exact compare-and-set alias transition.
	ActivateComposition(context.Context, artifact.ID, *artifact.ID) (CompositionActivation, error)
}

type repositoryCompositionWorkspace struct {
	store *repodb.Store
	limit int
}

// CompositionInventory resolves all exact composition authorities in RepoDB.
func (workspace repositoryCompositionWorkspace) CompositionInventory(ctx context.Context) (CompositionInventory, error) {
	if workspace.store == nil || workspace.limit <= 0 {
		return CompositionInventory{}, errors.New("composition workspace: repository is absent")
	}
	result, err := workspace.store.Query(ctx, repodb.Query{
		Schema: composition.CompositionRecipeSchema, MaxResults: workspace.limit,
	})
	if err != nil {
		return CompositionInventory{}, err
	}
	inventory := CompositionInventory{
		Head: result.Head.String(), Sequence: result.Sequence, Truncated: result.Truncated,
		Candidates: make([]CompositionCandidate, 0, len(result.Artifacts)),
	}
	for _, descriptor := range result.Artifacts {
		candidate := CompositionCandidate{ID: descriptor.ID}
		value, loadErr := composition.LoadCompositionRecipe(ctx, workspace.store, descriptor.ID)
		if loadErr != nil {
			candidate.Refusal = loadErr.Error()
			inventory.Candidates = append(inventory.Candidates, candidate)
			continue
		}
		source, loadErr := representation.LoadContract(ctx, workspace.store, value.SourceContract)
		if loadErr != nil {
			candidate.Refusal = loadErr.Error()
			inventory.Candidates = append(inventory.Candidates, candidate)
			continue
		}
		target, loadErr := representation.LoadContract(ctx, workspace.store, value.TargetContract)
		if loadErr != nil {
			candidate.Refusal = loadErr.Error()
			inventory.Candidates = append(inventory.Candidates, candidate)
			continue
		}
		bridge, loadErr := composition.LoadBridgeDefinition(ctx, workspace.store, value.BridgeDefinition)
		if loadErr != nil {
			candidate.Refusal = loadErr.Error()
			inventory.Candidates = append(inventory.Candidates, candidate)
			continue
		}
		promotion, loadErr := composition.LoadRepresentationBridgePromotion(ctx, workspace.store, value.Promotion)
		if loadErr != nil {
			candidate.Refusal = loadErr.Error()
			inventory.Candidates = append(inventory.Candidates, candidate)
			continue
		}
		candidate.Compatible = true
		candidate.Recipe, candidate.Source, candidate.Target = &value, &source, &target
		candidate.Graph, candidate.Promotion = &bridge.Graph, &promotion
		candidate.Evidence = &CompositionEvidence{
			TrainingPolicy: value.TrainingPolicy, PromotionPolicy: value.PromotionPolicy,
			Promotion: value.Promotion, HeldOutSplit: promotion.HeldOutSplit,
			RegressionSet: promotion.RegressionSet, Evaluator: promotion.Evaluator,
		}
		active, found, activeErr := composition.ActiveComposition(
			ctx, workspace.store, value.SourceModel, value.TargetModel, value.Task,
		)
		if activeErr != nil {
			return CompositionInventory{}, activeErr
		}
		candidate.Active = found && active.ID == value.ID
		if candidate.Active {
			plan, compileErr := composition.CompileCompositionExecutionPlan(
				ctx, workspace.store, value.SourceModel, value.TargetModel, value.Task,
			)
			if compileErr != nil {
				return CompositionInventory{}, compileErr
			}
			candidate.Runtime = &plan
		}
		inventory.Candidates = append(inventory.Candidates, candidate)
	}
	return inventory, nil
}

// ActivateComposition commits one recipe-scoped active alias transition.
func (workspace repositoryCompositionWorkspace) ActivateComposition(
	ctx context.Context,
	id artifact.ID,
	previous *artifact.ID,
) (CompositionActivation, error) {
	if workspace.store == nil || id.Kind() != artifact.KindRecipe {
		return CompositionActivation{}, errors.New("composition workspace: activation authority is invalid")
	}
	value, err := composition.LoadCompositionRecipe(ctx, workspace.store, id)
	if err != nil {
		return CompositionActivation{}, err
	}
	snapshot, err := workspace.store.Snapshot(ctx)
	if err != nil {
		return CompositionActivation{}, err
	}
	batch, err := value.ActivationBatch(
		ctx, workspace.store, "server/composition/activation/"+snapshot.Head.String(), previous,
	)
	if err != nil {
		return CompositionActivation{}, err
	}
	commit, err := workspace.store.Commit(ctx, batch)
	return CompositionActivation{Recipe: id, Commit: commit.String()}, err
}

type compositionActivationRequest struct {
	Recipe   artifact.ID  `json:"recipe"`
	Previous *artifact.ID `json:"previous,omitempty"`
}

func (h *Handler) compositionInventory(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	if workspace, ok := h.generator.(CompositionWorkflowAPI); ok {
		h.writeCompositionInventory(response, request, workspace)
		return
	}
	store, release, ok := h.openBrowseStore(response)
	if !ok {
		return
	}
	defer release()
	h.writeCompositionInventory(response, request, repositoryCompositionWorkspace{
		store: store, limit: h.config.MaxStoredResponses,
	})
}

func (h *Handler) writeCompositionInventory(
	response http.ResponseWriter,
	request *http.Request,
	workspace CompositionWorkflowAPI,
) {
	inventory, err := workspace.CompositionInventory(request.Context())
	if err != nil {
		writeError(response, http.StatusInternalServerError, "composition_error", err.Error())
		return
	}
	writeJSON(response, http.StatusOK, inventory)
}

func (h *Handler) compositionActivate(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodPost) {
		return
	}
	var body compositionActivationRequest
	if err := strictjson.Decode(request.Body, &body); err != nil {
		writeInvalidRequest(response, err)
		return
	}
	workspace, ok := h.generator.(CompositionWorkflowAPI)
	if !ok {
		if h.repository == nil {
			writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "composition activation is not configured")
			return
		}
		workspace = repositoryCompositionWorkspace{store: h.repository}
	}
	activation, err := workspace.ActivateComposition(request.Context(), body.Recipe, body.Previous)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	writeJSON(response, http.StatusOK, activation)
}
