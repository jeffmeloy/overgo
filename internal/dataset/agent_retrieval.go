// Package dataset owns immutable dataset documents and derived projections.
package dataset

import (
	"cmp"
	"context"
	"errors"
	"math"
	"slices"

	"overgo/internal/artifact"
)

const (
	agentRetrievalChunkMediaType     = "application/vnd.overgo.agent-retrieval-chunk+json"
	agentRetrievalChunkSchema        = "overgo/agent-retrieval-chunk/v1"
	agentRetrievalEmbeddingMediaType = "application/vnd.overgo.agent-retrieval-embedding+json"
	agentRetrievalEmbeddingSchema    = "overgo/agent-retrieval-embedding/v1"
	agentRetrievalIndexMediaType     = "application/vnd.overgo.agent-retrieval-index+json"
	agentRetrievalIndexSchema        = "overgo/agent-retrieval-index/v1"
	agentRetrievalAliasRoot          = "dataset/agent-retrieval/"
)

// AgentRetrievalPolicy owns chunk extent, overlap, and bounded candidate selection.
type AgentRetrievalPolicy struct {
	MaximumChunkRunes uint64 `json:"maximum_chunk_runes"`
	OverlapRunes      uint64 `json:"overlap_runes,omitempty"`
	CandidateLimit    uint64 `json:"candidate_limit"`
}

// Identify validates and identifies retrieval policy authority.
func (policy AgentRetrievalPolicy) Identify() (artifact.ID, error) {
	if policy.MaximumChunkRunes == 0 || policy.OverlapRunes >= policy.MaximumChunkRunes || policy.CandidateLimit == 0 {
		return artifact.ID{}, errors.New("dataset: invalid agent retrieval policy")
	}
	return artifact.JSONID(artifact.KindProfile, policy)
}

// AgentRetrievalDocument supplies source-bound text extracted by an existing dataset reader.
type AgentRetrievalDocument struct {
	Source artifact.ID `json:"source"`
	Text   string      `json:"text"`
}

// AgentEmbeddingProvider binds vector generation to an exact model identity.
type AgentEmbeddingProvider interface {
	ModelIdentity() artifact.ID
	Embed(context.Context, string) ([]float32, error)
}

// AgentRerankProvider binds optional second-stage ranking to an exact policy identity.
type AgentRerankProvider interface {
	PolicyIdentity() artifact.ID
	Score(context.Context, string, string) (float64, error)
}

// AgentRetrievalChunk is one immutable, source-cited chunk.
type AgentRetrievalChunk struct {
	Version   uint16      `json:"version"`
	Dataset   artifact.ID `json:"dataset"`
	Source    artifact.ID `json:"source"`
	Policy    artifact.ID `json:"policy"`
	Ordinal   uint64      `json:"ordinal"`
	StartRune uint64      `json:"start_rune"`
	EndRune   uint64      `json:"end_rune"`
	Text      string      `json:"text"`
	ID        artifact.ID `json:"-"`
}

// AgentRetrievalEmbedding binds an exact vector to its chunk and model.
type AgentRetrievalEmbedding struct {
	Version uint16      `json:"version"`
	Chunk   artifact.ID `json:"chunk"`
	Model   artifact.ID `json:"model"`
	Vector  []float32   `json:"vector"`
	ID      artifact.ID `json:"-"`
}

// AgentRetrievalEntry binds one chunk to its embedding.
type AgentRetrievalEntry struct {
	Chunk     artifact.ID `json:"chunk"`
	Embedding artifact.ID `json:"embedding"`
}

// AgentRetrievalProjection is a rebuildable index over immutable artifacts.
type AgentRetrievalProjection struct {
	Version        uint16                `json:"version"`
	Dataset        artifact.ID           `json:"dataset"`
	Policy         artifact.ID           `json:"policy"`
	CandidateLimit uint64                `json:"candidate_limit"`
	EmbeddingModel artifact.ID           `json:"embedding_model"`
	RerankPolicy   artifact.ID           `json:"rerank_policy"`
	Entries        []AgentRetrievalEntry `json:"entries"`
	ID             artifact.ID           `json:"-"`
}

// AgentRetrievalBuild requests one deterministic projection.
type AgentRetrievalBuild struct {
	Dataset      artifact.ID
	Documents    []AgentRetrievalDocument
	Policy       AgentRetrievalPolicy
	Embedder     AgentEmbeddingProvider
	RerankPolicy artifact.ID
}

// AgentRetrievalBuilder publishes dataset-owned retrieval artifacts.
type AgentRetrievalBuilder struct {
	Repository artifact.Repository
}

// AgentRetrievalQuery requests bounded cited results.
type AgentRetrievalQuery struct {
	Projection artifact.ID
	Text       string
	Limit      uint64
	Embedder   AgentEmbeddingProvider
	Reranker   AgentRerankProvider
}

// AgentRetrievalCitation preserves exact source and span identity.
type AgentRetrievalCitation struct {
	Dataset   artifact.ID `json:"dataset"`
	Source    artifact.ID `json:"source"`
	Chunk     artifact.ID `json:"chunk"`
	StartRune uint64      `json:"start_rune"`
	EndRune   uint64      `json:"end_rune"`
	Text      string      `json:"text"`
}

// AgentRetrievalResult carries deterministic semantic and rerank evidence.
type AgentRetrievalResult struct {
	Citation   AgentRetrievalCitation `json:"citation"`
	Embedding  artifact.ID            `json:"embedding"`
	Similarity float64                `json:"similarity"`
	Rerank     float64                `json:"rerank"`
}

var agentRetrievalChunkCodec = artifact.JSONDocumentCodec(
	"agent retrieval chunk", artifact.KindDatasetShard, agentRetrievalChunkMediaType, agentRetrievalChunkSchema,
	canonicalizeAgentRetrievalChunk,
	func(value AgentRetrievalChunk) artifact.ID { return value.ID },
	func(value *AgentRetrievalChunk, id artifact.ID) { value.ID = id }, nil,
)

var agentRetrievalEmbeddingCodec = artifact.JSONDocumentCodec(
	"agent retrieval embedding", artifact.KindTensorSet, agentRetrievalEmbeddingMediaType, agentRetrievalEmbeddingSchema,
	canonicalizeAgentRetrievalEmbedding,
	func(value AgentRetrievalEmbedding) artifact.ID { return value.ID },
	func(value *AgentRetrievalEmbedding, id artifact.ID) { value.ID = id },
	func(value AgentRetrievalEmbedding) AgentRetrievalEmbedding {
		value.Vector = slices.Clone(value.Vector)
		return value
	},
)

var agentRetrievalProjectionCodec = artifact.JSONDocumentCodec(
	"agent retrieval projection", artifact.KindProfile, agentRetrievalIndexMediaType, agentRetrievalIndexSchema,
	canonicalizeAgentRetrievalProjection,
	func(value AgentRetrievalProjection) artifact.ID { return value.ID },
	func(value *AgentRetrievalProjection, id artifact.ID) { value.ID = id },
	func(value AgentRetrievalProjection) AgentRetrievalProjection {
		value.Entries = slices.Clone(value.Entries)
		return value
	},
)

// Build chunks, embeds, and publishes one deterministic projection.
func (builder AgentRetrievalBuilder) Build(ctx context.Context, request AgentRetrievalBuild) (AgentRetrievalProjection, error) {
	policyID, err := request.Policy.Identify()
	if ctx == nil || builder.Repository == nil || request.Dataset.Kind() != artifact.KindDataset ||
		request.Embedder == nil || request.Embedder.ModelIdentity().Kind() != artifact.KindModel ||
		request.RerankPolicy.Kind() != artifact.KindProfile || len(request.Documents) == 0 || err != nil {
		return AgentRetrievalProjection{}, errors.Join(errors.New("dataset: invalid agent retrieval build"), err)
	}
	documents := slices.Clone(request.Documents)
	slices.SortFunc(documents, func(left, right AgentRetrievalDocument) int { return artifact.CompareID(left.Source, right.Source) })
	contents := make([]artifact.Content, 0)
	lineage := make([]artifact.Lineage, 0)
	entries := make([]AgentRetrievalEntry, 0)
	parents := make([]artifact.ID, 0, len(documents)+4)
	for _, id := range []artifact.ID{request.Dataset, policyID, request.Embedder.ModelIdentity(), request.RerankPolicy} {
		if !slices.Contains(parents, id) {
			parents = append(parents, id)
		}
	}
	for index, document := range documents {
		if document.Source.Kind() != artifact.KindFile && document.Source.Kind() != artifact.KindDatasetShard ||
			document.Text == "" || index > 0 && documents[index-1].Source == document.Source {
			return AgentRetrievalProjection{}, errors.New("dataset: invalid agent retrieval source")
		}
		if !slices.Contains(parents, document.Source) {
			parents = append(parents, document.Source)
		}
		chunks, chunkErr := agentRetrievalChunks(request.Dataset, policyID, document, request.Policy)
		if chunkErr != nil {
			return AgentRetrievalProjection{}, chunkErr
		}
		for _, chunk := range chunks {
			vector, embedErr := request.Embedder.Embed(ctx, chunk.Text)
			if embedErr != nil {
				return AgentRetrievalProjection{}, embedErr
			}
			embedding, identifyErr := agentRetrievalEmbeddingCodec.New(AgentRetrievalEmbedding{
				Version: artifact.InitialDocumentVersion, Chunk: chunk.ID,
				Model: request.Embedder.ModelIdentity(), Vector: vector,
			})
			if identifyErr != nil {
				return AgentRetrievalProjection{}, identifyErr
			}
			chunkContent, contentErr := agentRetrievalChunkCodec.Content(chunk)
			if contentErr != nil {
				return AgentRetrievalProjection{}, contentErr
			}
			embeddingContent, contentErr := agentRetrievalEmbeddingCodec.Content(embedding)
			if contentErr != nil {
				return AgentRetrievalProjection{}, contentErr
			}
			contents = append(contents, chunkContent, embeddingContent)
			lineage = append(lineage, artifact.DependencyLineage(chunk.ID, request.Dataset, document.Source, policyID)...)
			lineage = append(lineage, artifact.DependencyLineage(embedding.ID, chunk.ID, request.Embedder.ModelIdentity())...)
			entries = append(entries, AgentRetrievalEntry{Chunk: chunk.ID, Embedding: embedding.ID})
		}
	}
	projection, err := agentRetrievalProjectionCodec.New(AgentRetrievalProjection{
		Version: artifact.InitialDocumentVersion, Dataset: request.Dataset, Policy: policyID,
		CandidateLimit: request.Policy.CandidateLimit, EmbeddingModel: request.Embedder.ModelIdentity(),
		RerankPolicy: request.RerankPolicy, Entries: entries,
	})
	if err != nil {
		return AgentRetrievalProjection{}, err
	}
	projectionContent, err := agentRetrievalProjectionCodec.Content(projection)
	if err != nil {
		return AgentRetrievalProjection{}, err
	}
	contents = append(contents, projectionContent)
	entryParents := make([]artifact.ID, 0, len(entries)*2+len(parents))
	entryParents = append(entryParents, parents...)
	for _, entry := range entries {
		entryParents = append(entryParents, entry.Chunk, entry.Embedding)
	}
	lineage = append(lineage, artifact.DependencyLineage(projection.ID, entryParents...)...)
	aliasName := agentRetrievalAlias(request.Dataset, policyID, request.Embedder.ModelIdentity(), request.RerankPolicy)
	current, found, err := artifact.ResolveAlias(ctx, builder.Repository, aliasName)
	if err != nil {
		return AgentRetrievalProjection{}, err
	}
	if found && current == projection.ID {
		return projection, nil
	}
	alias := artifact.AliasBinding{Name: aliasName, Target: projection.ID}
	if found {
		alias.Previous = artifact.IDPointer(current)
	}
	descriptors := make([]artifact.Descriptor, 0, len(parents))
	for _, id := range parents {
		descriptors = append(descriptors, artifact.Descriptor{ID: id})
	}
	if _, err := artifact.CommitBatch(ctx, builder.Repository, artifact.Batch{
		Key: "dataset/agent-retrieval/" + projection.ID.String(), Artifacts: descriptors,
		Contents: contents, Lineage: lineage, Aliases: []artifact.AliasBinding{alias},
	}); err != nil {
		return AgentRetrievalProjection{}, err
	}
	return projection, nil
}

// Search verifies exact provider identities and returns deterministic cited results.
func (builder AgentRetrievalBuilder) Search(ctx context.Context, query AgentRetrievalQuery) ([]AgentRetrievalResult, error) {
	if ctx == nil || builder.Repository == nil || query.Text == "" || query.Limit == 0 || query.Embedder == nil || query.Reranker == nil {
		return nil, errors.New("dataset: invalid agent retrieval query")
	}
	projection, err := agentRetrievalProjectionCodec.Require(ctx, builder.Repository, query.Projection)
	if err != nil || projection.EmbeddingModel != query.Embedder.ModelIdentity() || projection.RerankPolicy != query.Reranker.PolicyIdentity() {
		return nil, errors.Join(errors.New("dataset: agent retrieval provider identity differs"), err)
	}
	queryVector, err := query.Embedder.Embed(ctx, query.Text)
	if err != nil {
		return nil, err
	}
	results := make([]AgentRetrievalResult, 0, len(projection.Entries))
	for _, entry := range projection.Entries {
		chunk, chunkErr := agentRetrievalChunkCodec.Require(ctx, builder.Repository, entry.Chunk)
		embedding, embeddingErr := agentRetrievalEmbeddingCodec.Require(ctx, builder.Repository, entry.Embedding)
		if chunkErr != nil || embeddingErr != nil || embedding.Chunk != chunk.ID || embedding.Model != projection.EmbeddingModel {
			return nil, errors.Join(errors.New("dataset: agent retrieval entry differs"), chunkErr, embeddingErr)
		}
		similarity, scoreErr := cosineSimilarity(queryVector, embedding.Vector)
		if scoreErr != nil {
			return nil, scoreErr
		}
		results = append(results, AgentRetrievalResult{
			Citation: AgentRetrievalCitation{Dataset: chunk.Dataset, Source: chunk.Source, Chunk: chunk.ID,
				StartRune: chunk.StartRune, EndRune: chunk.EndRune, Text: chunk.Text},
			Embedding: embedding.ID, Similarity: similarity,
		})
	}
	slices.SortFunc(results, func(left, right AgentRetrievalResult) int {
		if order := cmp.Compare(right.Similarity, left.Similarity); order != 0 {
			return order
		}
		return artifact.CompareID(left.Citation.Chunk, right.Citation.Chunk)
	})
	candidates := min(uint64(len(results)), projection.CandidateLimit)
	results = results[:candidates]
	for index := range results {
		rerank, scoreErr := query.Reranker.Score(ctx, query.Text, results[index].Citation.Text)
		if scoreErr != nil || !finiteRetrievalNumber(rerank) {
			return nil, errors.Join(errors.New("dataset: invalid agent rerank score"), scoreErr)
		}
		results[index].Rerank = rerank
	}
	slices.SortFunc(results, func(left, right AgentRetrievalResult) int {
		if order := cmp.Compare(right.Rerank, left.Rerank); order != 0 {
			return order
		}
		if order := cmp.Compare(right.Similarity, left.Similarity); order != 0 {
			return order
		}
		return artifact.CompareID(left.Citation.Chunk, right.Citation.Chunk)
	})
	limit := min(uint64(len(results)), query.Limit)
	limit = min(limit, uint64(len(projection.Entries)))
	return slices.Clone(results[:limit]), nil
}

func agentRetrievalChunks(datasetID, policyID artifact.ID, document AgentRetrievalDocument, policy AgentRetrievalPolicy) ([]AgentRetrievalChunk, error) {
	runes := []rune(document.Text)
	maximum, overlap := int(policy.MaximumChunkRunes), int(policy.OverlapRunes)
	if uint64(maximum) != policy.MaximumChunkRunes || uint64(overlap) != policy.OverlapRunes {
		return nil, errors.New("dataset: agent retrieval chunk policy exceeds host bounds")
	}
	result := make([]AgentRetrievalChunk, 0)
	for start, ordinal := 0, uint64(0); start < len(runes); ordinal++ {
		end := min(start+maximum, len(runes))
		chunk, err := agentRetrievalChunkCodec.New(AgentRetrievalChunk{
			Version: artifact.InitialDocumentVersion, Dataset: datasetID, Source: document.Source, Policy: policyID,
			Ordinal: ordinal, StartRune: uint64(start), EndRune: uint64(end), Text: string(runes[start:end]),
		})
		if err != nil {
			return nil, err
		}
		result = append(result, chunk)
		if end == len(runes) {
			break
		}
		start = end - overlap
	}
	return result, nil
}

func cosineSimilarity(left, right []float32) (float64, error) {
	if len(left) == 0 || len(left) != len(right) {
		return 0, errors.New("dataset: retrieval embedding dimensions differ")
	}
	var dot, leftNorm, rightNorm float64
	for index := range left {
		leftValue, rightValue := float64(left[index]), float64(right[index])
		if !finiteRetrievalNumber(leftValue) || !finiteRetrievalNumber(rightValue) {
			return 0, errors.New("dataset: retrieval embedding is non-finite")
		}
		dot += leftValue * rightValue
		leftNorm += leftValue * leftValue
		rightNorm += rightValue * rightValue
	}
	if leftNorm == 0 || rightNorm == 0 {
		return 0, errors.New("dataset: retrieval embedding norm is zero")
	}
	return dot / math.Sqrt(leftNorm*rightNorm), nil
}

func canonicalizeAgentRetrievalChunk(value *AgentRetrievalChunk) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Dataset.Kind() != artifact.KindDataset ||
		(value.Source.Kind() != artifact.KindFile && value.Source.Kind() != artifact.KindDatasetShard) ||
		value.Policy.Kind() != artifact.KindProfile || value.Text == "" || value.EndRune <= value.StartRune ||
		uint64(len([]rune(value.Text))) != value.EndRune-value.StartRune {
		return errors.New("dataset: invalid agent retrieval chunk")
	}
	return nil
}

func canonicalizeAgentRetrievalEmbedding(value *AgentRetrievalEmbedding) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Chunk.Kind() != artifact.KindDatasetShard ||
		value.Model.Kind() != artifact.KindModel || len(value.Vector) == 0 {
		return errors.New("dataset: invalid agent retrieval embedding")
	}
	for _, number := range value.Vector {
		if !finiteRetrievalNumber(float64(number)) {
			return errors.New("dataset: non-finite agent retrieval embedding")
		}
	}
	value.Vector = slices.Clone(value.Vector)
	return nil
}

func canonicalizeAgentRetrievalProjection(value *AgentRetrievalProjection) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Dataset.Kind() != artifact.KindDataset ||
		value.Policy.Kind() != artifact.KindProfile || value.CandidateLimit == 0 || value.EmbeddingModel.Kind() != artifact.KindModel ||
		value.RerankPolicy.Kind() != artifact.KindProfile || len(value.Entries) == 0 {
		return errors.New("dataset: invalid agent retrieval projection")
	}
	seen := make(map[artifact.ID]struct{}, len(value.Entries))
	for _, entry := range value.Entries {
		if entry.Chunk.Kind() != artifact.KindDatasetShard || entry.Embedding.Kind() != artifact.KindTensorSet {
			return errors.New("dataset: invalid agent retrieval entry")
		}
		if _, duplicate := seen[entry.Chunk]; duplicate {
			return errors.New("dataset: duplicate agent retrieval chunk")
		}
		seen[entry.Chunk] = struct{}{}
	}
	value.Entries = slices.Clone(value.Entries)
	return nil
}

func agentRetrievalAlias(datasetID, policy, model, rerank artifact.ID) string {
	return agentRetrievalAliasRoot + datasetID.String() + "/" + policy.String() + "/" + model.String() + "/" + rerank.String()
}

func finiteRetrievalNumber(value float64) bool {
	return !math.IsNaN(value) && value <= math.MaxFloat64 && value >= -math.MaxFloat64
}
