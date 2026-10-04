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
	// ViewOf names the tab this one is a view of: it has no navigation entry
	// of its own and its owner shows it (Inspect's analysis views).
	ViewOf string `json:"view_of,omitzero"`
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
	ViewOf  string `json:"view_of,omitzero"`
	Enabled bool   `json:"enabled"`
	Refusal string `json:"refusal,omitzero"`
	// Action names what would enable a refused workspace.
	Action string `json:"action,omitzero"`
}

// workspaceRefusal says in plain words why a refused tab's workspace is
// closed on this server and what opens it, naming no flag or command.
func workspaceRefusal(capability string) (reason, action string) {
	switch capability {
	case "conversation":
		return "No model that can chat is being served.", "Choose a model to serve from the model picker."
	case "analysis", "analysis.logits", "analysis.vocabulary", "analysis.hidden-states", "analysis.attention", "analysis.tensors", "model.properties":
		return "This server is not serving a local model to look inside.", "Serve a local model to inspect its logits, states and tensors."
	case "agent":
		return "No agent is defined yet.", "Define an agent in the store and it opens here."
	case "repository", "operations", "explorer":
		return "This server runs without a store, so it keeps no conversations, runs or operations.", "Restart the server over a store to keep them."
	case "scorecard":
		return "This server has no scorecard to read: it runs without a store or without the launcher's scorecard reader.", "Restart the server over a store to see what each model measures."
	case "embeddings":
		return "The served model does not make embeddings.", "Serve a model activated for embedding."
	case "rerank":
		return "The served model does not rerank.", "Serve a model activated for reranking."
	case "evaluation":
		return "Evaluations need to know exactly which code ran them, and this server cannot tell.", "Start the server from a clean checkout to run evaluations."
	case "workflow.training":
		return "Training is not turned on for this server.", "Restart the server with training turned on."
	case "workflow.model-builder":
		return "The model builder is not turned on for this server.", "Restart the server with the model builder turned on."
	case "workflow.export":
		return "No model is registered for export yet.", "Register and activate a model in the Library to export it."
	case "workflow.generation", "workflow.image", "workflow.video", "workflow.speech", "workflow.vqa", "workflow.transcription":
		return "No model is registered and activated for this kind of generation.", "Register and activate one in the Library."
	}
	return "This server was started without this workspace.", "Restart the server with it turned on."
}

type workspaceManifestResponse struct {
	Version  uint16                      `json:"version"`
	Sections []workspaceSection          `json:"sections"`
	Tabs     []workspaceTab              `json:"tabs"`
	Model    *workspaceModelCapabilities `json:"model,omitempty"`
}

func (h *Handler) workspaceManifest(response http.ResponseWriter, request *http.Request) {
	declaration, err := parseWorkspaceManifest()
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	result := workspaceManifestResponse{
		Version: declaration.Version, Sections: slices.Clone(declaration.Sections),
		Tabs: make([]workspaceTab, len(declaration.Tabs)),
	}
	// The tabs and composer modes share this request's listing per kind.
	listing := workspaceListing{}
	for index, tab := range declaration.Tabs {
		enabled := h.workspaceCapability(request.Context(), listing, tab.Capability)
		result.Tabs[index] = workspaceTab{
			ID: tab.ID, Label: tab.Label, Section: tab.Section, Module: tab.module(), ViewOf: tab.ViewOf, Enabled: enabled,
		}
		if !enabled {
			result.Tabs[index].Refusal, result.Tabs[index].Action = workspaceRefusal(tab.Capability)
		}
	}
	if document, ok := h.workspaceModelCapabilities(request.Context(), listing); ok {
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
	// A view belongs to a declared tab in its section that is not itself a view.
	for _, tab := range declaration.Tabs {
		if tab.ViewOf == "" {
			continue
		}
		owner := slices.IndexFunc(declaration.Tabs, func(candidate workspaceTabDeclaration) bool { return candidate.ID == tab.ViewOf })
		if owner < 0 || declaration.Tabs[owner].ViewOf != "" || declaration.Tabs[owner].Section != tab.Section {
			return workspaceManifestDeclaration{}, errors.New("server: workspace view names no tab that shows it")
		}
	}
	return declaration, nil
}

func (h *Handler) workspaceCapability(ctx context.Context, listing workspaceListing, capability string) bool {
	analysis := h.analysisCapabilities()
	supported := false
	switch capability {
	case "conversation":
		supported = analysis.Logits || h.hasWorkspaceCapability(ctx, listing, workflowGeneration, "")
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
		_, supported = h.generator.(modelPropertiesAPI)
	case "analysis":
		// Inspect opens when it has a view to show.
		_, properties := h.generator.(modelPropertiesAPI)
		supported = properties || analysis.Logits || analysis.Vocabulary || analysis.HiddenStates || analysis.Attention || analysis.Tensors
	case "agent":
		supported = h.agentCoordinator != nil
	case "repository", "operations":
		supported = h.repository != nil
	case "scorecard":
		supported = h.repository != nil && h.config.Scorecard != nil
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
	case "workflow.generation":
		supported = h.hasWorkspaceCapability(ctx, listing, workflowGeneration, "")
	case "workflow.training":
		supported = h.hasWorkspaceCapability(ctx, listing, WorkflowTraining, "")
	case "workflow.model-builder":
		supported = h.hasWorkspaceCapability(ctx, listing, workflowModelBuild, "")
	case "workflow.export":
		supported = h.hasWorkspaceCapability(ctx, listing, workflowExport, "")
	case "workflow.image":
		supported = h.hasWorkspaceCapability(ctx, listing, workflowGeneration, recipe.TaskImageGen)
	case "workflow.video":
		supported = h.hasWorkspaceCapability(ctx, listing, workflowGeneration, recipe.TaskVideoGen)
	case "workflow.speech":
		supported = h.hasWorkspaceCapability(ctx, listing, workflowGeneration, recipe.TaskSpeech)
	case "workflow.vqa":
		supported = h.hasWorkspaceCapability(ctx, listing, workflowGeneration, recipe.TaskVQA)
	case "workflow.transcription":
		supported = h.hasWorkspaceCapability(ctx, listing, workflowGeneration, recipe.TaskTranscription)
	}
	return supported
}

// workspaceListing: one request's workflow capabilities by kind, each
// acquired once; a failed listing lists none.
type workspaceListing map[WorkflowKind][]WorkflowCapability

func (h *Handler) hasWorkspaceCapability(ctx context.Context, listing workspaceListing, kind WorkflowKind, task recipe.Task) bool {
	if h.config.Workflows == nil {
		return false
	}
	capabilities, acquired := listing[kind]
	if !acquired {
		listed, err := h.config.Workflows.WorkflowCapabilities(ctx, kind)
		if err == nil {
			capabilities = listed
		}
		listing[kind] = capabilities
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
