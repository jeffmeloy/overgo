// Package main assembles the Overgo serving process from typed runtime owners.
package main

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	llamaserver "overgo/internal/server"
)

// agentRetrievalRuntime is the existing model runtime surface needed by the
// dataset retrieval contracts. The adapter owns no second model process.
type agentRetrievalRuntime interface {
	llamaserver.Embedder
	llamaserver.Ranker
	llamaserver.RankCapability
}

// agentRetrievalProvider binds retrieval to the same exact model and runtime
// policy that serve generation. AgentRetrievalProjection verifies the pair, so
// neither an embedding model nor a ranking policy can be substituted alone.
type agentRetrievalProvider struct {
	runtime  agentRetrievalRuntime
	model    artifact.ID
	policy   artifact.ID
	rankPair bool
}

func newAgentRetrievalProvider(
	runtime agentRetrievalRuntime,
	model artifact.ID,
	policy artifact.ID,
) (*agentRetrievalProvider, error) {
	if runtime == nil || model.Kind() != artifact.KindModel || policy.Kind() != artifact.KindProfile {
		return nil, errors.New("server: invalid agent retrieval runtime authority")
	}
	return &agentRetrievalProvider{
		runtime: runtime, model: model, policy: policy, rankPair: runtime.SupportsRank(),
	}, nil
}

// ModelIdentity returns the exact serving model used for retrieval embeddings.
func (provider *agentRetrievalProvider) ModelIdentity() artifact.ID {
	return provider.model
}

// PolicyIdentity returns the exact serving policy used for retrieval ranking.
func (provider *agentRetrievalProvider) PolicyIdentity() artifact.ID {
	return provider.policy
}

// Embed computes one retrieval vector through the active serving runtime.
func (provider *agentRetrievalProvider) Embed(ctx context.Context, text string) ([]float32, error) {
	if ctx == nil || provider == nil || provider.runtime == nil {
		return nil, errors.New("server: agent retrieval embedding runtime is unavailable")
	}
	vector, _, err := provider.runtime.Embed(ctx, text)
	if err != nil {
		return nil, err
	}
	if len(vector) == 0 {
		return nil, errors.New("server: agent retrieval embedding is empty")
	}
	for _, value := range vector {
		if !checked.Finite32(value) {
			return nil, errors.New("server: agent retrieval embedding is non-finite")
		}
	}
	return vector, nil
}

// Score ranks a query-document pair through the active serving runtime.
func (provider *agentRetrievalProvider) Score(ctx context.Context, query, document string) (float64, error) {
	if ctx == nil || provider == nil || provider.runtime == nil {
		return 0, errors.New("server: agent retrieval ranking runtime is unavailable")
	}
	if provider.rankPair {
		result, err := provider.runtime.RankPair(ctx, query, document)
		if err != nil {
			return 0, err
		}
		if len(result.Scores) == 0 || !checked.Finite32(result.Scores[0]) {
			return 0, errors.New("server: agent retrieval rank result is invalid")
		}
		return float64(result.Scores[0]), nil
	}
	queryVector, err := provider.Embed(ctx, query)
	if err != nil {
		return 0, err
	}
	documentVector, err := provider.Embed(ctx, document)
	if err != nil {
		return 0, err
	}
	if len(queryVector) != len(documentVector) {
		return 0, errors.New("server: agent retrieval embedding dimensions differ")
	}
	var score float64
	queryNonzero, documentNonzero := false, false
	for index, queryValue := range queryVector {
		documentValue := documentVector[index]
		queryNonzero = queryNonzero || queryValue != 0
		documentNonzero = documentNonzero || documentValue != 0
		score += float64(queryValue) * float64(documentValue)
	}
	if !queryNonzero || !documentNonzero || !checked.Finite64(score) {
		return 0, errors.New("server: agent retrieval unit embedding is invalid")
	}
	return score, nil
}
