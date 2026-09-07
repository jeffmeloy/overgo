package server

import (
	"context"
	"net/http"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// artifactLineageResponse is what the store records around one artifact:
// the runs that produced it (each with its recipe and inputs), the runs
// that consumed it (each with its outputs), and the next steps, every
// active generation capability whose declared slot takes the artifact's
// kind.
type artifactLineageResponse struct {
	ID        artifact.ID          `json:"id"`
	MediaType string               `json:"media_type,omitzero"`
	Producers []artifactLineageRun `json:"producers"`
	Consumers []artifactLineageRun `json:"consumers"`
	Next      []artifactNextStep   `json:"next"`
}

// artifactLineageRun is one run beside an artifact with what it read and wrote.
type artifactLineageRun struct {
	Run     artifact.ID            `json:"run"`
	Recipe  artifact.ID            `json:"recipe"`
	Outcome runrecord.Outcome      `json:"outcome"`
	Inputs  []artifactLineageInput `json:"inputs"`
	Outputs []artifact.ID          `json:"outputs"`
}

// artifactLineageInput is a run input with the facts a page labels it by
// (a JSON document under an input schema is the request; a file is media).
type artifactLineageInput struct {
	ID        artifact.ID `json:"id"`
	MediaType string      `json:"media_type,omitzero"`
	Schema    string      `json:"schema,omitzero"`
}

// artifactNextStep is a capability whose declared slot takes the artifact:
// the mode, the model and the slot a page opens with the artifact in it.
type artifactNextStep struct {
	Task    recipe.Task `json:"task"`
	Recipe  artifact.ID `json:"recipe"`
	Name    string      `json:"name"`
	Control string      `json:"control"`
	Label   string      `json:"label"`
}

// artifactLineage answers /artifacts/lineage?id=: the artifact's producers,
// consumers and next steps from the store's records and the declared
// capabilities alone.
func (h *Handler) artifactLineage(response http.ResponseWriter, request *http.Request) {
	id, err := artifact.ParseID(request.URL.Query().Get("id"))
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	store, ok := h.requireBrowseStore(response, request)
	if !ok {
		return
	}
	ctx := request.Context()
	descriptor, found, err := store.Artifact(ctx, id)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "overgodb_error", err.Error())
		return
	}
	if !found {
		writeError(response, http.StatusNotFound, "not_found", "artifact not found")
		return
	}
	lineage := artifactLineageResponse{ID: id, MediaType: descriptor.MediaType, Producers: []artifactLineageRun{}, Consumers: []artifactLineageRun{}, Next: []artifactNextStep{}}
	parents, err := store.Parents(ctx, id)
	if err == nil {
		lineage.Producers, err = lineageRuns(ctx, store, parents, artifact.RelationProducedBy, func(edge artifact.Lineage) artifact.ID { return edge.Parent })
	}
	var children []artifact.Lineage
	if err == nil {
		children, err = store.Children(ctx, id)
	}
	if err == nil {
		lineage.Consumers, err = lineageRuns(ctx, store, children, artifact.RelationDependsOn, func(edge artifact.Lineage) artifact.ID { return edge.Child })
	}
	if err != nil {
		writeError(response, http.StatusInternalServerError, "overgodb_error", err.Error())
		return
	}
	if workspace, declared := h.generator.(WorkflowWorkspaceAPI); declared {
		capabilities, err := workspace.WorkflowCapabilities(ctx, WorkflowGeneration)
		if err != nil {
			writeGenerationError(response, err)
			return
		}
		lineage.Next = nextSteps(capabilities, descriptor.MediaType)
	}
	writeJSON(response, http.StatusOK, lineage)
}

// lineageRuns reads the run records at the run end of every edge of the
// relation, with each run's inputs described.
func lineageRuns(ctx context.Context, store *overgodb.Store, edges []artifact.Lineage, relation artifact.Relation, end func(artifact.Lineage) artifact.ID) ([]artifactLineageRun, error) {
	runs := make([]artifactLineageRun, 0)
	for _, edge := range edges {
		id := end(edge)
		if edge.Relation != relation || id.Kind() != artifact.KindRun {
			continue
		}
		run, err := runrecord.RequireRun(ctx, store, id)
		if err != nil {
			return nil, err
		}
		inputs := make([]artifactLineageInput, 0, len(run.Inputs))
		for _, input := range run.Inputs {
			descriptor, _, err := store.Artifact(ctx, input)
			if err != nil {
				return nil, err
			}
			inputs = append(inputs, artifactLineageInput{ID: input, MediaType: descriptor.MediaType, Schema: descriptor.Schema})
		}
		runs = append(runs, artifactLineageRun{Run: run.ID, Recipe: run.Recipe, Outcome: run.Outcome, Inputs: inputs, Outputs: run.Outputs})
	}
	return runs, nil
}

// nextSteps lists, per accepting slot, every capability a page can run
// whose declared slot media takes the media type; a refused capability
// is never a next step.
func nextSteps(capabilities []WorkflowCapability, mediaType string) []artifactNextStep {
	steps := make([]artifactNextStep, 0)
	for _, capability := range capabilities {
		if capability.Refusal != "" {
			continue
		}
		for _, control := range capability.Controls {
			if control.Type == WorkflowControlArtifact && control.Media != "" && strings.HasPrefix(mediaType, control.Media+"/") {
				steps = append(steps, artifactNextStep{Task: capability.Task, Recipe: capability.Recipe, Name: capability.Name, Control: control.Name, Label: control.Label})
			}
		}
	}
	return steps
}
