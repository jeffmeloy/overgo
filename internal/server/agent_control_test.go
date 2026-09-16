package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

const (
	agentWorkspaceChunkRunes = uint64(128)
	agentWorkspaceCandidates = uint64(4)
)

type agentWorkspaceEmbedder struct{ model artifact.ID }

func (embedder agentWorkspaceEmbedder) ModelIdentity() artifact.ID { return embedder.model }
func (agentWorkspaceEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	return []float32{float32(utf8.RuneCountInString(text)), float32(strings.Count(text, "evidence") + 1)}, nil
}

type agentWorkspaceReranker struct{ policy artifact.ID }

func (reranker agentWorkspaceReranker) PolicyIdentity() artifact.ID { return reranker.policy }
func (agentWorkspaceReranker) Score(_ context.Context, query, text string) (float64, error) {
	if strings.Contains(text, query) {
		return float64(utf8.RuneCountInString(text)), nil
	}
	return 0, nil
}

type agentWorkspaceFixture struct {
	handler   *Handler
	store     *overgodb.Store
	generator *recipeInspectorGenerator
	prompt    artifact.ID
	policy    artifact.ID
	manual    artifact.ID
}

func TestAgentWorkspaceVertical(t *testing.T) {
	fixture := newAgentWorkspaceFixture(t, nil, nil, nil)
	defer fixture.store.Close()
	definition := publishAgentFromAPI(t, fixture, nil, nil)
	activateDefinitionFromAPI(t, fixture.handler, definition, "/agents/activate")

	inventory := serveTestRequest(fixture.handler, http.MethodGet, "/agents", "")
	if inventory.Code != http.StatusOK || !strings.Contains(inventory.Body.String(), "research-agent") ||
		!strings.Contains(inventory.Body.String(), fixture.manual.String()) {
		t.Fatalf("agent inventory status=%d body=%s", inventory.Code, inventory.Body.String())
	}
	paused := serveTestRequest(fixture.handler, http.MethodPost, "/agents/state", `{"name":"research-agent","state":"paused"}`)
	resumed := serveTestRequest(fixture.handler, http.MethodPost, "/agents/state", `{"name":"research-agent","state":"active"}`)
	if paused.Code != http.StatusOK || resumed.Code != http.StatusOK {
		t.Fatalf("agent lifecycle pause=(%d %s) resume=(%d %s)", paused.Code, paused.Body.String(), resumed.Code, resumed.Body.String())
	}
	step := serveTestRequest(fixture.handler, http.MethodPost, "/agents/step",
		`{"agent":"research-agent","session":"vertical","tool":"store.head"}`)
	if step.Code != http.StatusOK || !strings.Contains(step.Body.String(), `"steps":1`) {
		t.Fatalf("agent step status=%d body=%s", step.Code, step.Body.String())
	}
	chat := serveTestRequest(fixture.handler, http.MethodPost, "/agents/chat",
		`{"agent":"research-agent","messages":[{"role":"user","content":"hello"}]}`)
	if chat.Code != http.StatusOK || !strings.Contains(chat.Body.String(), `"choices"`) {
		t.Fatalf("agent chat status=%d body=%s", chat.Code, chat.Body.String())
	}
}

func TestAgentWorkspaceSSE(t *testing.T) {
	fixture := newAgentWorkspaceFixture(t, nil, nil, nil)
	defer fixture.store.Close()
	activateDefinitionFromAPI(t, fixture.handler, publishAgentFromAPI(t, fixture, nil, nil), "/agents/activate")
	ctx, cancel := context.WithCancel(t.Context())
	recorder := &countingRecorder{ResponseRecorder: httptest.NewRecorder(), flushes: make(chan struct{}, agentWorkspaceCandidates)}
	done := make(chan struct{})
	go func() {
		fixture.handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/agents/stream", nil).WithContext(ctx))
		close(done)
	}()
	for range 2 {
		<-recorder.flushes
	}
	cancel()
	<-done
	if recorder.Header().Get("Content-Type") != "text/event-stream" ||
		!strings.Contains(recorder.Body.String(), "event: agent.inventory") ||
		!strings.Contains(recorder.Body.String(), "event: operation.snapshot") {
		t.Fatalf("agent SSE headers=%v body=%s", recorder.Header(), recorder.Body.String())
	}
}

func TestAgentWorkspaceRetrievalEvidence(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	datasetID := testutil.ArtifactID(t, artifact.KindDataset, "agent-retrieval-dataset")
	model := testutil.ArtifactID(t, artifact.KindModel, "response-model")
	policy := testutil.ArtifactID(t, artifact.KindProfile, "agent-rerank-policy")
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key:       "agent/workspace/retrieval-dependencies",
		Artifacts: []artifact.Descriptor{{ID: datasetID}, {ID: model}, {ID: policy}},
	}); err != nil {
		t.Fatal(err)
	}
	embedder := agentWorkspaceEmbedder{model: model}
	reranker := agentWorkspaceReranker{policy: policy}
	documents, err := dataset.PublishAgentRetrievalSource(
		t.Context(), store, artifact.KindFile,
		[]dataset.AgentRetrievalDocument{{Text: "cited evidence for retrieval"}},
	)
	if err != nil {
		t.Fatal(err)
	}
	source := documents[0].Source
	projection, err := (dataset.AgentRetrievalBuilder{Repository: store}).Build(t.Context(), dataset.AgentRetrievalBuild{
		Dataset: datasetID, Documents: documents,
		Policy:   dataset.AgentRetrievalPolicy{MaximumChunkRunes: agentWorkspaceChunkRunes, CandidateLimit: agentWorkspaceCandidates},
		Embedder: embedder, RerankPolicy: policy,
	})
	if err != nil {
		t.Fatal(err)
	}
	fixture := newAgentWorkspaceFixture(t, store, embedder, reranker)
	activateDefinitionFromAPI(t, fixture.handler, publishAgentFromAPI(t, fixture, []artifact.ID{datasetID}, nil), "/agents/activate")
	request := marshalAutomationJSON(t, map[string]any{
		"agent": "research-agent", "projection": projection.ID, "query": "evidence", "limit": 1, "rerank_policy": policy,
	})
	search := serveTestRequest(fixture.handler, http.MethodPost, "/agents/retrieval", request)
	if search.Code != http.StatusOK || !strings.Contains(search.Body.String(), source.String()) ||
		!strings.Contains(search.Body.String(), projection.Dataset.String()) ||
		search.Header().Get(agentRetrievalReceiptHeader) == "" || search.Header().Get(agentRetrievalTraceHeader) == "" {
		t.Fatalf("agent retrieval status=%d body=%s", search.Code, search.Body.String())
	}
	receiptID, err := artifact.ParseID(search.Header().Get(agentRetrievalReceiptHeader))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runrecord.RequireConsumedRetrievalReceipt(t.Context(), store, receiptID); err != nil {
		t.Fatalf("agent retrieval receipt is not replayable: %v", err)
	}
	exactRequest := agentRetrievalRequest{
		Agent: "research-agent", Projection: projection.ID, Query: "evidence", Limit: 1, RerankPolicy: policy,
	}
	originalReranker := fixture.handler.config.AgentReranker
	fixture.handler.config.AgentReranker = agentWorkspaceReranker{
		policy: testutil.ArtifactID(t, artifact.KindProfile, "different-agent-rerank-policy"),
	}
	if _, err := fixture.handler.agentRetrieval(t.Context(), exactRequest); err == nil {
		t.Fatal("agent retrieval admitted a provider outside the requested policy authority")
	}
	fixture.handler.config.AgentReranker = originalReranker
	originalEmbedder := fixture.handler.config.AgentEmbedder
	fixture.handler.config.AgentEmbedder = agentWorkspaceEmbedder{
		model: testutil.ArtifactID(t, artifact.KindModel, "different-agent-retrieval-model"),
	}
	if _, err := fixture.handler.agentRetrieval(t.Context(), exactRequest); err == nil {
		t.Fatal("agent retrieval admitted a provider outside the active model authority")
	}
	fixture.handler.config.AgentEmbedder = originalEmbedder
	step := serveTestRequest(fixture.handler, http.MethodPost, "/agents/step",
		`{"agent":"research-agent","session":"observed","tool":"store.head"}`)
	if step.Code != http.StatusOK {
		t.Fatalf("agent evidence step status=%d body=%s", step.Code, step.Body.String())
	}
	evidence := serveTestRequest(fixture.handler, http.MethodGet, "/agents/evidence?session=research-agent%3Aobserved", "")
	if evidence.Code != http.StatusOK || !strings.Contains(evidence.Body.String(), `"transcript"`) ||
		!strings.Contains(evidence.Body.String(), fixture.manual.String()) || strings.Contains(evidence.Body.String(), `"arguments"`) ||
		strings.Contains(evidence.Body.String(), `"content"`) {
		t.Fatalf("agent evidence status=%d body=%s", evidence.Code, evidence.Body.String())
	}
}

type agentAutomationGenerator struct {
	*recipeInspectorGenerator
	*AutomationWorkspace
}

func TestAgentWorkspaceAutomationAttachment(t *testing.T) {
	automation := newAutomationServerFixture(t)
	defer automation.store.Close()
	automationID := publishAutomationFromAPI(t, automation)
	activate := serveTestRequest(automation.handler, http.MethodPost, "/automations/activate", marshalAutomationJSON(t, map[string]any{
		"definition": automationID,
	}))
	if activate.Code != http.StatusOK {
		t.Fatalf("automation activation status=%d body=%s", activate.Code, activate.Body.String())
	}
	inspector := responseRecipeGenerator(t, &fakeGenerator{})
	generator := &agentAutomationGenerator{
		recipeInspectorGenerator: inspector,
		AutomationWorkspace:      automation.handler.generator.(*automationWorkspaceGenerator).AutomationWorkspace,
	}
	fixture := newAgentWorkspaceFixtureWithGenerator(t, automation.store, generator, nil, nil)
	activateDefinitionFromAPI(t, fixture.handler, publishAgentFromAPI(t, fixture, nil, []artifact.ID{automationID}), "/agents/activate")
	run := serveTestRequest(fixture.handler, http.MethodPost, "/agents/automation", marshalAutomationJSON(t, map[string]any{
		"agent": "research-agent", "automation": automationID, "key": "attached", "inputs": map[string]any{"tokens": "hello"},
	}))
	if run.Code != http.StatusAccepted {
		t.Fatalf("agent automation status=%d body=%s", run.Code, run.Body.String())
	}
	var execution workflowruntime.AutomationExecution
	if err := json.Unmarshal(run.Body.Bytes(), &execution); err != nil || !execution.Operation.Valid() {
		t.Fatalf("agent automation execution=(%+v, %v)", execution, err)
	}
	status, err := fixture.handler.operations.Wait(t.Context(), execution.Operation)
	if err != nil || status.State != operation.StateCompleted {
		t.Fatalf("agent automation operation=(%+v, %v)", status, err)
	}
}

func TestAgentWorkspaceNoHiddenReasoning(t *testing.T) {
	fixture := newAgentWorkspaceFixture(t, nil, nil, nil)
	defer fixture.store.Close()
	activateDefinitionFromAPI(t, fixture.handler, publishAgentFromAPI(t, fixture, nil, nil), "/agents/activate")
	javascript := serveTestRequest(fixture.handler, http.MethodGet, "/mod/agent.js", "").Body.String()
	for _, expected := range []string{
		"schemaForm", "/agents/chat", "overgo.toolStep(", "/agents/retrieval", "/agents/automation", "/agents/evidence",
		"/agents/stream", "PopStateEvent", "overgo.artifactLink",
	} {
		if !strings.Contains(javascript, expected) {
			t.Errorf("agent GUI lacks %q", expected)
		}
	}
	if strings.Contains(javascript, "reasoning_content") || strings.Contains(javascript, "ReasoningContent") {
		t.Fatal("agent GUI projects a private model field")
	}
	source, err := os.ReadFile("agent_control.go")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(source), "InteractionToolCall `json") || strings.Contains(string(source), `json:"content"`) {
		t.Fatal("agent observable copies interaction payloads instead of artifact references")
	}
}

func newAgentWorkspaceFixture(
	t *testing.T,
	store *overgodb.Store,
	embedder dataset.AgentEmbeddingProvider,
	reranker dataset.AgentRerankProvider,
) agentWorkspaceFixture {
	t.Helper()
	return newAgentWorkspaceFixtureWithGenerator(t, store, responseRecipeGenerator(t, &fakeGenerator{}), embedder, reranker)
}

func newAgentWorkspaceFixtureWithGenerator(
	t *testing.T,
	store *overgodb.Store,
	generator Generator,
	embedder dataset.AgentEmbeddingProvider,
	reranker dataset.AgentRerankProvider,
) agentWorkspaceFixture {
	t.Helper()
	if store == nil {
		var err error
		store, err = overgodb.Open(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
	}
	inspector, ok := generator.(interface {
		RecipeRuntimeDescription(recipe.Task) (modelrecipe.RuntimeDescription, error)
	})
	if !ok {
		t.Fatal("agent workspace generator lacks recipe authority")
	}
	description, err := inspector.RecipeRuntimeDescription(recipe.TaskInference)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key:       "agent/workspace/serving-recipe/" + description.Identity.Recipe.String(),
		Artifacts: []artifact.Descriptor{{ID: description.Identity.Recipe}, {ID: description.Identity.Model}},
	}); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		t.Fatal(err)
	}
	manuals, err := agenttool.StandardManuals()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agenttool.PublishManualCatalog(t.Context(), store, manuals); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		t.Fatal(err)
	}
	prompt := commitAutomationServerBlob(t, store, artifact.KindFile, "agent-workspace-prompt")
	policy := commitAutomationServerBlob(t, store, artifact.KindProfile, "agent-workspace-policy")
	handler, err := New(Config{
		ModelID: testModelID, MaxTokens: testMaxTokens, DefaultTemperature: testNeutralTemperature,
		DefaultTopP: testFullTopP, Analysis: testAnalysisPolicy, Repository: store,
		AgentEmbedder: embedder, AgentReranker: reranker,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handler.Close() })
	fixture := agentWorkspaceFixture{
		handler: handler, store: store, prompt: prompt, policy: policy, manual: manuals[0].ID,
	}
	if concrete, concreteOK := generator.(*recipeInspectorGenerator); concreteOK {
		fixture.generator = concrete
	} else if concrete, concreteOK := generator.(*agentAutomationGenerator); concreteOK {
		fixture.generator = concrete.recipeInspectorGenerator
	}
	return fixture
}

func publishAgentFromAPI(
	t *testing.T,
	fixture agentWorkspaceFixture,
	datasets, automations []artifact.ID,
) artifact.ID {
	t.Helper()
	request := AgentDefinitionInput{
		Name: "research-agent", Prompt: fixture.prompt, ModelRecipe: fixture.generator.description.Identity.Recipe,
		ToolManuals: []artifact.ID{fixture.manual}, Datasets: datasets, Automations: automations,
		Policies: []artifact.ID{fixture.policy},
	}
	if provider := fixture.handler.config.AgentReranker; provider != nil && !slices.Contains(request.Policies, provider.PolicyIdentity()) {
		request.Policies = append(request.Policies, provider.PolicyIdentity())
	}
	response := serveTestRequest(fixture.handler, http.MethodPost, "/agents/definitions", marshalAutomationJSON(t, request))
	if response.Code != http.StatusCreated {
		t.Fatalf("agent publish status=%d body=%s", response.Code, response.Body.String())
	}
	var result struct {
		ID artifact.ID `json:"id"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil || !result.ID.Valid() {
		t.Fatalf("agent definition=(%s, %v)", result.ID, err)
	}
	return result.ID
}
