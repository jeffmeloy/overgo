package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"testing"

	"overgo/internal/operation"
	"overgo/internal/recipe"
)

func TestWorkspaceManifest(t *testing.T) {
	t.Parallel()
	handler := newTestHandlerWithRepository(t, &fakeGenerator{})
	response := serveTestRequest(handler, http.MethodGet, "/workspace/manifest", "")
	if response.Code != http.StatusOK {
		t.Fatalf("workspace manifest status=%d body=%s", response.Code, response.Body.String())
	}
	var manifest workspaceManifestResponse
	if err := json.Unmarshal(response.Body.Bytes(), &manifest); err != nil {
		t.Fatal(err)
	}
	// The conversation is the first tab, so the front page opens on it.
	if manifest.Version != 1 || len(manifest.Sections) == 0 || len(manifest.Tabs) == 0 || manifest.Tabs[0].ID != "chat" {
		t.Fatalf("workspace manifest = %+v", manifest)
	}
	if routesByPath["/catalog/models"].Authentication != routePublic {
		t.Error("the header's evidence line reads /catalog/models, which must stay public")
	}
	sections := make(map[string]bool, len(manifest.Sections))
	for _, section := range manifest.Sections {
		sections[section.ID] = true
	}
	seen := make(map[string]bool, len(manifest.Tabs))
	for _, tab := range manifest.Tabs {
		if seen[tab.ID] || !sections[tab.Section] || tab.Label == "" {
			t.Fatalf("invalid workspace tab = %+v", tab)
		}
		seen[tab.ID] = true
	}
	for _, required := range []string{"chat", "datasets", "training-jobs", "compositions", "attention"} {
		if !seen[required] {
			t.Fatalf("workspace manifest lacks %q", required)
		}
	}
	// Five places; the analysis tabs are Inspect's views, with no navigation entry of their own.
	views := 0
	for _, tab := range manifest.Tabs {
		if tab.ViewOf == "" {
			continue
		}
		views++
		if tab.ViewOf != "inspect" || tab.Section != "inspect" || !seen[tab.ViewOf] {
			t.Errorf("view %+v does not belong to Inspect", tab)
		}
	}
	if len(manifest.Sections) != 5 || views != 6 {
		t.Fatalf("workspace manifest has %d places and %d views, want 5 and 6", len(manifest.Sections), views)
	}
}

// countingWorkflows lists a generation capability for each task it holds
// and counts the listings it is asked for, by kind.
type countingWorkflows struct {
	tasks  []recipe.Task
	listed map[WorkflowKind]int
}

func (workflows *countingWorkflows) WorkflowCapabilities(_ context.Context, kind WorkflowKind) ([]WorkflowCapability, error) {
	workflows.listed[kind]++
	if kind != workflowGeneration {
		return nil, nil
	}
	capabilities := make([]WorkflowCapability, len(workflows.tasks))
	for index, task := range workflows.tasks {
		capabilities[index] = WorkflowCapability{Task: task}
	}
	return capabilities, nil
}

func (*countingWorkflows) ExecuteWorkflow(context.Context, WorkflowKind, WorkflowCapability, json.RawMessage, operation.Reporter) (operation.Completion, error) {
	return operation.Completion{}, errors.New("the manifest executes nothing")
}

// TestManifestAcquiresEachCapabilityOnce: one manifest request lists each
// workflow kind at most once, its tabs and composer modes reading that one
// listing, and the next request lists afresh and shows what changed.
func TestManifestAcquiresEachCapabilityOnce(t *testing.T) {
	t.Parallel()
	workflows := &countingWorkflows{tasks: []recipe.Task{recipe.TaskImageGen, recipe.TaskSpeech}}
	handler, err := New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, Analysis: testAnalysisPolicy, Workflows: workflows}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	manifest := func() (map[string]bool, map[string]bool) {
		t.Helper()
		workflows.listed = map[WorkflowKind]int{}
		response := serveTestRequest(handler, http.MethodGet, "/workspace/manifest", "")
		var manifest workspaceManifestResponse
		if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &manifest) != nil || manifest.Model == nil {
			t.Fatalf("workspace manifest status=%d body=%s", response.Code, response.Body.String())
		}
		for kind, count := range workflows.listed {
			if count != 1 {
				t.Fatalf("one manifest listed %s %d times", kind, count)
			}
		}
		tabs, modes := map[string]bool{}, map[string]bool{}
		for _, tab := range manifest.Tabs {
			tabs[tab.ID] = tab.Enabled
		}
		for _, mode := range manifest.Model.Modes {
			modes[mode.ID] = mode.Enabled
		}
		return tabs, modes
	}
	tabs, modes := manifest()
	if workflows.listed[workflowGeneration] != 1 || !tabs["generation"] || !modes["image-gen"] || !modes["speech"] || modes["video-gen"] {
		t.Fatalf("listed %v; tabs %v, modes %v", workflows.listed, tabs, modes)
	}
	workflows.tasks = []recipe.Task{recipe.TaskVideoGen}
	if tabs, modes = manifest(); !tabs["generation"] || modes["image-gen"] || modes["speech"] || !modes["video-gen"] {
		t.Fatalf("the next manifest kept the earlier listing: tabs %v, modes %v", tabs, modes)
	}
	workflows.tasks = nil
	if tabs, modes = manifest(); tabs["generation"] || modes["video-gen"] {
		t.Fatalf("a manifest over no capability: tabs %v, modes %v", tabs, modes)
	}
}

func TestWorkspaceManifestCapabilityRefusal(t *testing.T) {
	t.Parallel()
	response := serveTestRequest(newTestHandler(t, &fakeGenerator{}), http.MethodGet, "/workspace/manifest", "")
	var manifest workspaceManifestResponse
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &manifest) != nil {
		t.Fatalf("workspace manifest status=%d body=%s", response.Code, response.Body.String())
	}
	byID := make(map[string]workspaceTab, len(manifest.Tabs))
	for _, tab := range manifest.Tabs {
		byID[tab.ID] = tab
	}
	if !byID["chat"].Enabled || byID["attention"].Enabled || byID["attention"].Refusal == "" ||
		byID["datasets"].Enabled || byID["datasets"].Refusal == "" {
		t.Fatalf("workspace capability projection = %+v", byID)
	}
}
