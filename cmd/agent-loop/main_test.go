package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/agentloop"
	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

const cliSessionSteps = 1

// The delegated fixture's figures.
//   - delegatedSteps bounds the invocation's session above the steps the test admits
//   - notesMode = the permission the tool server writes the workspace file with
const (
	delegatedSteps = 4
	notesMode      = 0o600
)

// TestDelegatedStepAdmitsByContractAndRestores holds a delegated session to its task and to its checkpoints.
//   - a mutation of a scope the task allows is admitted, and a mutation of a scope it does not allow is refused
//   - a mutation is previewed in one run and approved in the next, because each run restores the session from its interaction chain
//   - the approved mutation's workspace file is checkpointed, and -restore puts it back behind its own preview and approval
func TestDelegatedStepAdmitsByContractAndRestores(t *testing.T) {
	// Serial: the delegated workspace is the working directory, which t.Chdir sets for the whole process.
	repository, workspace := t.TempDir(), t.TempDir()
	t.Chdir(workspace)
	const notes, session = "notes.txt", "delegated-1"
	if err := os.WriteFile(filepath.Join(workspace, notes), []byte("before"), notesMode); err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		var call struct {
			Path string `json:"path"`
			Text string `json:"text"`
		}
		if err := json.NewDecoder(request.Body).Decode(&call); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		target := filepath.Join(workspace, call.Path)
		if request.URL.Path == "/write" {
			if err := os.WriteFile(target, []byte(call.Text), notesMode); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
		}
		data, err := os.ReadFile(target)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]string{"text": string(data)})
	}))
	defer server.Close()

	path := agenttool.Field{Name: "path", Kind: agenttool.FieldString, Required: true}
	inWorkspace := agenttool.EffectCeiling{Targets: []agenttool.EffectTargetBinding{{Scope: agenttool.EffectScopeWorkspace, Argument: path.Name}}}
	var manuals []agenttool.Manual
	var granted []artifact.ID
	for _, draft := range []agenttool.Manual{
		{Name: "notes.read", Description: "Read one workspace file.", Effect: agenttool.EffectInspection,
			Arguments: []agenttool.Field{path}, Ceiling: inWorkspace,
			Transport: agenttool.Transport{Kind: agenttool.TransportHTTP, URL: server.URL + "/read"}},
		{Name: "notes.write", Description: "Write one workspace file.", Effect: agenttool.EffectMutation,
			Arguments: []agenttool.Field{path, {Name: "text", Kind: agenttool.FieldString, Required: true}}, Ceiling: inWorkspace,
			Transport: agenttool.Transport{Kind: agenttool.TransportHTTP, URL: server.URL + "/write"}},
		{Name: "host.write", Description: "Change host state.", Effect: agenttool.EffectMutation,
			Ceiling:   agenttool.EffectCeiling{Targets: []agenttool.EffectTargetBinding{{Scope: agenttool.EffectScopeHost, Value: "registry"}}},
			Transport: agenttool.Transport{Kind: agenttool.TransportHTTP, URL: server.URL + "/host"}},
	} {
		manual, err := agenttool.NewManual(draft)
		if err != nil {
			t.Fatal(err)
		}
		manuals, granted = append(manuals, manual), append(granted, manual.ID)
	}
	snapshot, err := agenttool.NewCatalogSnapshot(manuals)
	if err != nil {
		t.Fatal(err)
	}
	worker, err := recipe.NewAgentDefinition(recipe.AgentDefinition{
		Name: "worker", Prompt: testutil.ArtifactID(t, artifact.KindFile, "delegated prompt"),
		ModelRecipe: testutil.ArtifactID(t, artifact.KindRecipe, "delegated model recipe"),
		ToolManuals: granted, Policies: []artifact.ID{testutil.ArtifactID(t, artifact.KindProfile, "delegated policy")},
	})
	if err != nil {
		t.Fatal(err)
	}
	task, err := recipe.NewAgentTaskContract(recipe.AgentTaskContract{
		Agent: worker.ID, Objective: "Edit the notes file.", Scope: []string{notes},
		AllowedEffects: []string{"external:mutation", "workspace:mutation"},
		Acceptance: []recipe.AcceptanceCriterion{
			{Name: "notes", Scope: notes, Verifier: testutil.ArtifactID(t, artifact.KindRecipe, "verify notes")},
		},
		Verification:    []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "delegated gate")},
		Budget:          testutil.ArtifactID(t, artifact.KindEvidence, "delegated budget"),
		PauseConditions: []string{"drift"},
	})
	if err != nil {
		t.Fatal(err)
	}
	delegation, err := recipe.NewDelegatedAgentInvocation(recipe.DelegatedAgentInvocation{
		Task: task.ID, Worker: worker.ID,
		Grant:     recipe.AgentCapabilityGrant{Manuals: granted, AllowedEffects: []string{"inspection", "mutation"}},
		Scheduler: recipe.AgentSchedulerPolicy{MaxSteps: delegatedSteps, MaxParallel: 1},
		Catalog:   snapshot.ID, Model: testutil.ArtifactID(t, artifact.KindModel, "delegated model"),
	})
	if err != nil {
		t.Fatal(err)
	}
	var contents []artifact.Content
	for _, encode := range []func() (artifact.Content, error){snapshot.ArtifactContent, worker.ArtifactContent, task.Content, delegation.Content} {
		content, err := encode()
		if err != nil {
			t.Fatal(err)
		}
		contents = append(contents, content)
	}
	batch, err := artifact.NewDocumentBatch("delegated-loop-fixture", contents, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(repository)
	if err != nil {
		t.Fatal(err)
	}
	_, publishErr := agenttool.PublishManualCatalog(t.Context(), store, manuals)
	_, commitErr := artifact.CommitBatch(t.Context(), store, batch)
	if err := errors.Join(publishErr, commitErr, store.Close()); err != nil {
		t.Fatal(err)
	}

	command := func(args ...string) (string, error) {
		var out bytes.Buffer
		err := run(append([]string{"-repo", repository, "-delegated", delegation.ID.String()}, args...), &out)
		return out.String(), err
	}
	step := func(tool, arguments string, args ...string) (string, error) {
		return command(append([]string{"-session", session, "-tool", tool, "-arguments", arguments}, args...)...)
	}
	read, write := `{"path":"notes.txt"}`, `{"path":"notes.txt","text":"after"}`
	if out, err := step("notes.read", read); err != nil || !strings.Contains(out, "before") {
		t.Fatalf("inspection = %q, %v, want the file read", out, err)
	}
	if _, err := step("host.write", "{}"); err == nil || !strings.Contains(err.Error(), "does not grant host:mutation") {
		t.Fatalf("a host mutation = %v, want the task contract's refusal", err)
	}
	out, err := step("notes.write", write, "-preview")
	var previewed struct {
		Decision agentloop.DecisionPreview `json:"decision"`
	}
	if err != nil || json.Unmarshal([]byte(out), &previewed) != nil || !previewed.Decision.Operation.Valid() {
		t.Fatalf("mutation preview = %q, %v, want the decision's operation", out, err)
	}
	if out, err := step("notes.write", write, "-approve", previewed.Decision.Operation.String()); err != nil || !strings.Contains(out, "after") {
		t.Fatalf("approved mutation in a later run = %q, %v, want it admitted over the restored inspection", out, err)
	}
	if data, err := os.ReadFile(filepath.Join(workspace, notes)); err != nil || string(data) != "after" {
		t.Fatalf("mutated file = %q, %v, want after", data, err)
	}

	store, err = overgodb.Open(repository)
	if err != nil {
		t.Fatal(err)
	}
	interaction, found, err := runrecord.ResolveInteraction(t.Context(), store, session+"-step-2")
	if err := errors.Join(err, store.Close()); err != nil || !found || len(interaction.Tools) != 1 {
		t.Fatalf("mutation step = %+v, found=%t, %v, want one checkpoint on its interaction", interaction, found, err)
	}
	checkpoint := interaction.Tools[0].String()
	if _, err := command("-restore", checkpoint); err == nil || !strings.Contains(err.Error(), "restore approval is absent") {
		t.Fatalf("restore with no grant = %v, want it refused", err)
	}
	out, err = command("-restore", checkpoint, "-preview")
	var request struct {
		Operation artifact.ID `json:"operation"`
	}
	if err != nil || json.Unmarshal([]byte(out), &request) != nil || !request.Operation.Valid() {
		t.Fatalf("restore preview = %q, %v, want the restore's operation", out, err)
	}
	if _, err := command("-restore", checkpoint, "-approve", previewed.Decision.Operation.String()); err == nil {
		t.Fatal("a restore approved under the mutation's operation was admitted")
	}
	if out, err := command("-restore", checkpoint, "-approve", request.Operation.String()); err != nil || strings.TrimSpace(out) == "" {
		t.Fatalf("approved restore = %q, %v, want its receipt", out, err)
	}
	if data, err := os.ReadFile(filepath.Join(workspace, notes)); err != nil || string(data) != "before" {
		t.Fatalf("restored file = %q, %v, want before", data, err)
	}
}

func TestAgentLoopProposesGatedStep(t *testing.T) {
	repository := t.TempDir()
	store, err := overgodb.Open(repository)
	if err != nil {
		t.Fatal(err)
	}
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "cli-loop-recipe")
	modelID := testutil.ArtifactID(t, artifact.KindModel, "cli-loop-model")
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "cli-loop/identity", Artifacts: []artifact.Descriptor{{ID: recipeID}},
	}); err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"seen":true}`))
	}))
	defer server.Close()
	inspect, err := agenttool.NewManual(agenttool.Manual{
		Name: "probe.read", Description: "Read state for the CLI fixture.",
		Effect:    agenttool.EffectInspection,
		Transport: agenttool.Transport{Kind: agenttool.TransportHTTP, URL: server.URL},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agenttool.PublishManualCatalog(t.Context(), store, []agenttool.Manual{inspect}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := run([]string{
		"-repo", repository, "-session", "cli-1", "-tool", "probe.read",
		"-recipe", recipeID.String(), "-model", modelID.String(),
		"-max-steps", strconv.Itoa(cliSessionSteps),
	}, &out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), `"seen":true`) {
		t.Fatalf("loop output = %q", out.String())
	}
}
