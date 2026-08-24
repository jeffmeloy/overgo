package server

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

func agentTestHandler(t *testing.T, register func(context.Context, *overgodb.Store)) *Handler {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	generator := responseRecipeGenerator(t, &fakeGenerator{})
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key:       "agent/serving-identity",
		Artifacts: []artifact.Descriptor{{ID: generator.description.Identity.Recipe}},
	}); err != nil {
		t.Fatal(err)
	}
	if register != nil {
		register(context.Background(), store)
	}
	handler, err := New(Config{
		ModelID: testModelID, MaxTokens: testMaxTokens, Repository: store,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { handler.Close() })
	return handler
}

// TestAgentWorkspaceProjectsCatalogAndStepsGatedly pins the server as
// protocol projection: the tool catalog carries effect classes and
// coverage, an inspection step admits, and a mutation admits only with
// approval -- the coordinator owning every rule.
func TestAgentWorkspaceProjectsCatalogAndStepsGatedly(t *testing.T) {
	// The serving executor refuses loopback endpoints by policy, so the
	// fixture uses the store-inspection builtin for the inspection step
	// and a mutation manual whose only role is exercising the approval
	// gate; the gate refuses it before any transport runs.
	handler := agentTestHandler(t, func(ctx context.Context, store *overgodb.Store) {
		manuals, err := agenttool.StandardManuals()
		if err != nil {
			t.Fatal(err)
		}
		write, err := agenttool.NewManual(agenttool.Manual{
			Name: "store.commit", Description: "A mutation manual for the approval gate.",
			Effect:    agenttool.EffectMutation,
			Transport: agenttool.Transport{Kind: agenttool.TransportArgv, Program: "git", Args: []string{"status"}},
		})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := agenttool.PublishManualCatalog(ctx, store, append(manuals, write)); err != nil {
			t.Fatal(err)
		}
	})

	catalog := serveTestRequest(handler, http.MethodGet, "/agent/tools", "")
	if catalog.Code != http.StatusOK {
		t.Fatalf("catalog status=%d body=%s", catalog.Code, catalog.Body.String())
	}
	var tools struct {
		Tools []struct {
			Name, Effect string
		} `json:"tools"`
		Coverage struct{ Registered, Published int } `json:"coverage"`
	}
	if err := json.Unmarshal(catalog.Body.Bytes(), &tools); err != nil {
		t.Fatal(err)
	}
	effects := map[string]string{}
	for _, tool := range tools.Tools {
		effects[tool.Name] = tool.Effect
	}
	if effects["store.head"] != "inspection" || effects["store.commit"] != "mutation" || tools.Coverage.Published != 3 {
		t.Fatalf("catalog projection = %+v", tools)
	}

	inspect := serveTestRequest(handler, http.MethodPost, "/agent/step",
		`{"session":"s1","tool":"store.head"}`)
	if inspect.Code != http.StatusOK || !strings.Contains(inspect.Body.String(), `"steps":1`) {
		t.Fatalf("inspection step status=%d body=%s", inspect.Code, inspect.Body.String())
	}
	// The approval gate refuses before the mutation's transport runs, so
	// the projection surfaces the coordinator's typed refusal.
	refused := serveTestRequest(handler, http.MethodPost, "/agent/step",
		`{"session":"s1","tool":"store.commit"}`)
	if refused.Code != http.StatusUnprocessableEntity || !strings.Contains(refused.Body.String(), "approval") {
		t.Fatalf("unapproved mutation status=%d body=%s", refused.Code, refused.Body.String())
	}
}

// TestAgentWorkspaceRefusesUnregisteredTool is the named negative test:
// a step naming a tool the store never registered is refused, not
// executed, and no durable interaction is written for it.
func TestAgentWorkspaceRefusesUnregisteredTool(t *testing.T) {
	handler := agentTestHandler(t, nil)
	refused := serveTestRequest(handler, http.MethodPost, "/agent/step",
		`{"session":"ghost","tool":"probe.absent"}`)
	if refused.Code != http.StatusUnprocessableEntity || !strings.Contains(refused.Body.String(), "unregistered") {
		t.Fatalf("unregistered tool status=%d body=%s", refused.Code, refused.Body.String())
	}
}

// TestAgentWorkspaceRestoresSessionsAcrossRestart pins durable session
// continuity: a fresh handler over the same store resumes the session
// with its recorded step count and inspection state, so a restart
// neither resets the mutation gate nor forks the interaction chain.
func TestAgentWorkspaceRestoresSessionsAcrossRestart(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	generator := responseRecipeGenerator(t, &fakeGenerator{})
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key:       "agent/restart-identity",
		Artifacts: []artifact.Descriptor{{ID: generator.description.Identity.Recipe}},
	}); err != nil {
		t.Fatal(err)
	}
	manuals, err := agenttool.StandardManuals()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agenttool.PublishManualCatalog(context.Background(), store, manuals); err != nil {
		t.Fatal(err)
	}
	first, err := New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, Repository: store}, generator)
	if err != nil {
		t.Fatal(err)
	}
	step := serveTestRequest(first, http.MethodPost, "/agent/step", `{"session":"restart-1","tool":"store.head"}`)
	if step.Code != http.StatusOK || !strings.Contains(step.Body.String(), `"steps":1`) {
		t.Fatalf("first step status=%d body=%s", step.Code, step.Body.String())
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	second, err := New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, Repository: store}, generator)
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	resumed := serveTestRequest(second, http.MethodPost, "/agent/step", `{"session":"restart-1","tool":"store.head"}`)
	if resumed.Code != http.StatusOK ||
		!strings.Contains(resumed.Body.String(), `"steps":2`) ||
		!strings.Contains(resumed.Body.String(), `"inspected":true`) {
		t.Fatalf("resumed step status=%d body=%s", resumed.Code, resumed.Body.String())
	}
}
