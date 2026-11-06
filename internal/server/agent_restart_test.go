package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/testcheck"
)

// TestAMutationAfterARestartRuns: a restarted server knows nothing of an agent's session but what the store holds, so the
// approval names the agent's definition as the proposal does, and a mutation inspected before the restart runs after it.
func TestAMutationAfterARestartRuns(t *testing.T) {
	t.Parallel()
	fixture := newAgentWorkspaceFixture(t, nil, nil, nil)
	defer fixture.store.Close()
	manuals := testcheck.Must(agenttool.StandardManuals())(t)
	write := testcheck.Must(agenttool.NewManual(agenttool.Manual{
		Name: "store.commit", Description: "A mutation manual for the approval gate.",
		Effect:    agenttool.EffectMutation,
		Ceiling:   agenttool.EffectCeiling{Targets: []agenttool.EffectTargetBinding{{Scope: agenttool.EffectScopeRepository, Value: "overgodb"}}},
		Transport: agenttool.Transport{Kind: agenttool.TransportArgv, Program: "git", Args: []string{"status"}},
	}))(t)
	if _, err := agenttool.PublishArgvPolicy(t.Context(), fixture.store, []string{"git"}); err != nil {
		t.Fatal(err)
	}
	if _, err := agenttool.PublishManualCatalog(t.Context(), fixture.store, append(manuals, write)); err != nil {
		t.Fatal(err)
	}
	definition := agentDefinitionInput{
		Name: "research-agent", Prompt: fixture.prompt, ModelRecipe: fixture.generator.description.Identity.Recipe,
		ToolManuals: []artifact.ID{fixture.manual, write.ID}, Policies: []artifact.ID{fixture.policy},
	}
	published := serveTestRequest(fixture.handler, http.MethodPost, "/agents/definitions", marshalTestJSON(t, definition))
	var created struct {
		ID artifact.ID `json:"id"`
	}
	if err := json.Unmarshal(published.Body.Bytes(), &created); published.Code != http.StatusCreated || err != nil {
		t.Fatalf("agent publish status=%d body=%s", published.Code, published.Body.String())
	}
	activateAgentFromAPI(t, fixture.handler, created.ID)
	inspect := serveTestRequest(fixture.handler, http.MethodPost, "/agents/step", `{"agent":"research-agent","session":"alpha","tool":"store.head"}`)
	if inspect.Code != http.StatusOK {
		t.Fatalf("inspection status=%d body=%s", inspect.Code, inspect.Body.String())
	}

	restarted := testcheck.Must(New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, Repository: fixture.store}, fixture.generator))(t)
	defer restarted.Close()
	body := `"agent":"research-agent","session":"alpha","tool":"store.commit","arguments":{}`
	preview := serveTestRequest(restarted, http.MethodPost, "/agents/approval", `{`+body+`}`)
	var previewed struct {
		Operation string `json:"operation"`
	}
	if err := json.Unmarshal(preview.Body.Bytes(), &previewed); preview.Code != http.StatusOK || err != nil || previewed.Operation == "" {
		t.Fatalf("preview status=%d body=%s", preview.Code, preview.Body.String())
	}
	approved := serveTestRequest(restarted, http.MethodPost, "/agents/step", `{`+body+`,"approval":"`+previewed.Operation+`"}`)
	if approved.Code != http.StatusOK || !strings.Contains(approved.Body.String(), `"steps":2`) {
		t.Fatalf("approved mutation status=%d body=%s", approved.Code, approved.Body.String())
	}
}
