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
	// Module names the client file under webui/mod/ that registers the tab;
	// empty means the file is named after the tab. The shell's loader reads
	// it, so a tab is a manifest entry plus a module file and nothing else.
	Module string `json:"module,omitzero"`
}

// module returns the client module file name (without directory or
// extension) that registers the tab.
func (tab workspaceTabDeclaration) module() string {
	if tab.Module != "" {
		return tab.Module
	}
	return tab.ID
}

// validWorkspaceModule: a module file name is lower-case letters, digits and
// underscores, the shape of every file under webui/mod/.
func validWorkspaceModule(name string) bool {
	if name == "" {
		return false
	}
	for index := range len(name) {
		character := name[index]
		if !(character >= 'a' && character <= 'z' || character >= '0' && character <= '9' || character == '_') {
			return false
		}
	}
	return true
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
	Module  string `json:"module"`
	Enabled bool   `json:"enabled"`
	Refusal string `json:"refusal,omitzero"`
	// Action names what would enable a refused workspace.
	Action string `json:"action,omitzero"`
}

// workspaceEnablingAction says what enables the capability a refused tab needs.
func workspaceEnablingAction(capability string) string {
	switch capability {
	case "analysis.logits", "analysis.vocabulary", "analysis.hidden-states", "analysis.attention", "analysis.tensors", "model.properties":
		return "Serve a local model; analysis reads its logits, states and tensors."
	case "agent":
		return "Define an agent in the store; the coordinator opens with it."
	case "repository", "operations", "explorer":
		return "Run the server over a store (-repo)."
	case "embeddings":
		return "Serve a model activated for embedding."
	case "rerank":
		return "Serve a model activated for rerank."
	case "evaluation":
		return "Launch from a clean checkout or pass -evaluation-commit."
	case "workflow.training":
		return "Launch with -training."
	case "workflow.model-builder":
		return "Launch with -model-builder."
	case "workflow.export", "workflow.generation", "workflow.image", "workflow.video", "workflow.speech", "workflow.vqa", "workflow.transcription":
		return "Register a media model activated for the task; the workspace opens over the store."
	}
	return "The documented launch does not open this workspace."
}

type workspaceManifestResponse struct {
	Version  uint16                      `json:"version"`
	Sections []workspaceSection          `json:"sections"`
	Tabs     []workspaceTab              `json:"tabs"`
	Model    *workspaceModelCapabilities `json:"model,omitempty"`
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
			ID: tab.ID, Label: tab.Label, Section: tab.Section, Module: tab.module(), Enabled: enabled, Refusal: refusal,
		}
		if !enabled {
			result.Tabs[index].Action = workspaceEnablingAction(tab.Capability)
		}
	}
	if document, ok := h.workspaceModelCapabilities(request.Context()); ok {
		result.Model = &document
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
		if !validWorkspaceModule(tab.module()) {
			return workspaceManifestDeclaration{}, errors.New("server: invalid workspace tab module")
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
	case "conversation":
		supported = analysis.Logits || h.hasWorkspaceCapability(ctx, WorkflowGeneration, "")
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
	case "embeddings":
		supported = slices.Contains(h.modelCapabilities(), "embedding")
	case "rerank":
		supported = slices.Contains(h.modelCapabilities(), "rerank")
	case "explorer":
		supported = true
	case "evaluation":
		// The launch supplies the evaluation workspace through Config, the
		// way cmd/server does; the generator never carries it.
		supported = h.config.Evaluation != nil
	case "automation":
		_, supported = h.generator.(AutomationWorkspaceAPI)
	case "peer":
		_, supported = h.generator.(PeerWorkspaceAPI)
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
	case "workflow.video":
		supported = h.hasWorkspaceCapability(ctx, WorkflowGeneration, recipe.TaskVideoGen)
	case "workflow.speech":
		supported = h.hasWorkspaceCapability(ctx, WorkflowGeneration, recipe.TaskSpeech)
	case "workflow.vqa":
		supported = h.hasWorkspaceCapability(ctx, WorkflowGeneration, recipe.TaskVQA)
	case "workflow.transcription":
		supported = h.hasWorkspaceCapability(ctx, WorkflowGeneration, recipe.TaskTranscription)
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
