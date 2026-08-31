package main

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/server"
)

type workflowStub struct {
	executed bool
	run      artifact.ID
}

type agentRetrievalRuntimeStub struct {
	embeddings    map[string][]float32
	rank          inference.RankResult
	rankSupported bool
	rankCalled    bool
}

func (stub *agentRetrievalRuntimeStub) Embed(_ context.Context, text string) ([]float32, int, error) {
	vector, found := stub.embeddings[text]
	if !found {
		return nil, 0, errors.New("missing embedding")
	}
	return vector, len(vector), nil
}

func (stub *agentRetrievalRuntimeStub) RankPair(context.Context, string, string) (inference.RankResult, error) {
	stub.rankCalled = true
	return stub.rank, nil
}

func (stub *agentRetrievalRuntimeStub) SupportsRank() bool { return stub.rankSupported }

func (stub *workflowStub) WorkflowCapabilities(context.Context, server.WorkflowKind) ([]server.WorkflowCapability, error) {
	return nil, nil
}

func (stub *workflowStub) ExecuteWorkflow(_ context.Context, _ server.WorkflowKind, _ recipe.Task, _ artifact.ID, _ json.RawMessage, _ operation.Reporter) (operation.Completion, error) {
	stub.executed = true
	return operation.Completion{Run: stub.run}, nil
}

func TestServerRuntimeExecutesDPOWorkflow(t *testing.T) {
	run, err := artifact.IdentifyBytes(artifact.KindRun, []byte("run"))
	if err != nil {
		t.Fatal(err)
	}
	stub := &workflowStub{run: run}
	runtime := &serverRuntime{WorkflowWorkspaceAPI: stub}
	var workspace server.WorkflowWorkspaceAPI = runtime
	if _, err := workspace.ExecuteWorkflow(t.Context(), server.WorkflowTraining, recipe.TaskTraining, artifact.ID{}, nil, nil); err != nil || !stub.executed {
		t.Fatalf("executed=%v err=%v", stub.executed, err)
	}
}

func TestAgentRetrievalProviderUsesRuntimeRankCapability(t *testing.T) {
	model, err := artifact.IdentifyBytes(artifact.KindModel, []byte("model"))
	if err != nil {
		t.Fatal(err)
	}
	policy, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("runtime-policy"))
	if err != nil {
		t.Fatal(err)
	}
	runtime := &agentRetrievalRuntimeStub{
		rankSupported: true,
		rank:          inference.RankResult{Scores: []float32{0.75}},
	}
	provider, err := newAgentRetrievalProvider(runtime, model, policy)
	if err != nil {
		t.Fatal(err)
	}
	if provider.ModelIdentity() != model || provider.PolicyIdentity() != policy {
		t.Fatalf("provider identity = (%s, %s)", provider.ModelIdentity(), provider.PolicyIdentity())
	}
	score, err := provider.Score(t.Context(), "query", "document")
	if err != nil || score != 0.75 || !runtime.rankCalled {
		t.Fatalf("runtime rank score = (%f, %v, called=%v)", score, err, runtime.rankCalled)
	}
}

func TestAgentRetrievalProviderUsesExistingNormalizedEmbeddingFallback(t *testing.T) {
	model, err := artifact.IdentifyBytes(artifact.KindModel, []byte("model"))
	if err != nil {
		t.Fatal(err)
	}
	policy, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("runtime-policy"))
	if err != nil {
		t.Fatal(err)
	}
	runtime := &agentRetrievalRuntimeStub{embeddings: map[string][]float32{
		"query": {1, 0}, "document": {0.6, 0.8},
	}}
	provider, err := newAgentRetrievalProvider(runtime, model, policy)
	if err != nil {
		t.Fatal(err)
	}
	score, err := provider.Score(t.Context(), "query", "document")
	if err != nil || math.Abs(score-0.6) > 1e-6 || runtime.rankCalled {
		t.Fatalf("embedding fallback score = (%f, %v, rank-called=%v)", score, err, runtime.rankCalled)
	}
}
