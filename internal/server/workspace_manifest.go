package server

import (
	"context"
	_ "embed"
	"errors"
	"net/http"
	"slices"

	"overgo/internal/recipe"
	"overgo/internal/strictjson"
	"overgo/internal/textcheck"
)

//go:embed workspace_manifest.json
var workspaceManifestJSON []byte

type workspaceSection struct {
	ID    string `json:"id"`
	Label string `json:"label"`
}

type workspaceTabDeclaration struct {
	ID         string `json:"id"`
	Label      string `json:"label"`
	Section    string `json:"section"`
	Capability string `json:"capability"`
}

type workspaceManifestDeclaration struct {
	Version  uint16                    `json:"version"`
	Sections []workspaceSection        `json:"sections"`
	Tabs     []workspaceTabDeclaration `json:"tabs"`
}

type workspaceTab struct {
	ID      string `json:"id"`
	Label   string `json:"label"`
	Section string `json:"section"`
	Enabled bool   `json:"enabled"`
	Refusal string `json:"refusal,omitempty"`
}

type workspaceManifestResponse struct {
	Version  uint16             `json:"version"`
	Sections []workspaceSection `json:"sections"`
	Tabs     []workspaceTab     `json:"tabs"`
}

func (h *Handler) workspaceManifest(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	declaration, err := parseWorkspaceManifest()
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	result := workspaceManifestResponse{
		Version: declaration.Version, Sections: slices.Clone(declaration.Sections),
		Tabs: make([]workspaceTab, len(declaration.Tabs)),
	}
	for index, tab := range declaration.Tabs {
		enabled, refusal := h.workspaceCapability(request.Context(), tab.Capability)
		result.Tabs[index] = workspaceTab{
			ID: tab.ID, Label: tab.Label, Section: tab.Section, Enabled: enabled, Refusal: refusal,
		}
	}
	writeJSON(response, http.StatusOK, result)
}

func parseWorkspaceManifest() (workspaceManifestDeclaration, error) {
	var declaration workspaceManifestDeclaration
	if err := strictjson.DecodeBytes(workspaceManifestJSON, &declaration); err != nil {
		return workspaceManifestDeclaration{}, err
	}
	if declaration.Version != 1 || len(declaration.Sections) == 0 || len(declaration.Tabs) == 0 {
		return workspaceManifestDeclaration{}, errors.New("server: invalid workspace manifest envelope")
	}
	sections := make(map[string]struct{}, len(declaration.Sections))
	for _, section := range declaration.Sections {
		if !textcheck.LowerIdentifier(section.ID, len(section.ID)) || section.Label == "" {
			return workspaceManifestDeclaration{}, errors.New("server: invalid workspace section")
		}
		if _, duplicate := sections[section.ID]; duplicate {
			return workspaceManifestDeclaration{}, errors.New("server: duplicate workspace section")
		}
		sections[section.ID] = struct{}{}
	}
	tabs := make(map[string]struct{}, len(declaration.Tabs))
	for _, tab := range declaration.Tabs {
		_, sectionFound := sections[tab.Section]
		if !textcheck.LowerIdentifier(tab.ID, len(tab.ID)) || tab.Label == "" || tab.Capability == "" || !sectionFound {
			return workspaceManifestDeclaration{}, errors.New("server: invalid workspace tab")
		}
		if _, duplicate := tabs[tab.ID]; duplicate {
			return workspaceManifestDeclaration{}, errors.New("server: duplicate workspace tab")
		}
		tabs[tab.ID] = struct{}{}
	}
	return declaration, nil
}

func (h *Handler) workspaceCapability(ctx context.Context, capability string) (bool, string) {
	analysis := h.analysisCapabilities()
	supported := false
	switch capability {
	case "analysis.logits":
		supported = analysis.Logits
	case "analysis.vocabulary":
		supported = analysis.Vocabulary
	case "analysis.hidden-states":
		supported = analysis.HiddenStates
	case "analysis.attention":
		supported = analysis.Attention
	case "analysis.tensors":
		supported = analysis.Tensors
	case "model.properties":
		_, supported = h.generator.(ModelPropertiesAPI)
	case "agent":
		supported = h.agentCoordinator != nil
	case "repository", "operations":
		supported = h.repository != nil
	case "evaluation":
		_, supported = h.generator.(EvaluationWorkspaceAPI)
	case "automation":
		_, supported = h.generator.(AutomationWorkspaceAPI)
	case "workflow.generation":
		supported = h.hasWorkspaceCapability(ctx, WorkflowGeneration, "")
	case "workflow.training":
		supported = h.hasWorkspaceCapability(ctx, WorkflowTraining, "")
	case "workflow.model-builder":
		supported = h.hasWorkspaceCapability(ctx, WorkflowModelBuild, "")
	case "workflow.export":
		supported = h.hasWorkspaceCapability(ctx, WorkflowExport, "")
	case "workflow.image":
		supported = h.hasWorkspaceCapability(ctx, WorkflowGeneration, recipe.TaskImageGen)
	case "workflow.speech":
		supported = h.hasWorkspaceCapability(ctx, WorkflowGeneration, recipe.TaskSpeech)
	}
	if supported {
		return true, ""
	}
	return false, "Server capability " + capability + " is unavailable"
}

func (h *Handler) hasWorkspaceCapability(ctx context.Context, kind WorkflowKind, task recipe.Task) bool {
	workspace, ok := h.generator.(WorkflowWorkspaceAPI)
	if !ok {
		return false
	}
	capabilities, err := workspace.WorkflowCapabilities(ctx, kind)
	if err != nil {
		return false
	}
	if task == "" {
		return len(capabilities) != 0
	}
	for _, capability := range capabilities {
		if capability.Task == task {
			return true
		}
	}
	return false
}
