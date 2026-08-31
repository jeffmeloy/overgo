// Package dataset owns immutable dataset documents and derived projections.
package dataset

import (
	"cmp"
	"context"
	"errors"
	"maps"
	"math"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"

	"overgo/internal/artifact"
	"overgo/internal/checked"
)

const (
	agentRetrievalChunkMediaType     = "application/vnd.overgo.agent-retrieval-chunk+json"
	agentRetrievalChunkSchema        = "overgo/agent-retrieval-chunk/v1"
	agentRetrievalEmbeddingMediaType = "application/vnd.overgo.agent-retrieval-embedding+json"
	agentRetrievalEmbeddingSchema    = "overgo/agent-retrieval-embedding/v1"
	agentRetrievalIndexMediaType     = "application/vnd.overgo.agent-retrieval-index+json"
	agentRetrievalIndexSchema        = "overgo/agent-retrieval-index/v1"
	agentRetrievalSourceMediaType    = "application/vnd.overgo.agent-retrieval-source+json"
	agentRetrievalSourceSchema       = "overgo/agent-retrieval-source/v1"
	agentRetrievalAliasRoot          = "dataset/agent-retrieval/"
)

// AgentRetrievalPolicy owns chunk extent, overlap, and bounded candidate selection.
type AgentRetrievalPolicy struct {
	MaximumChunkRunes uint64 `json:"maximum_chunk_runes"`
	OverlapRunes      uint64 `json:"overlap_runes,omitzero"`
	CandidateLimit    uint64 `json:"candidate_limit"`
}

// AgentRetrievalFacet is the closed set of structural content classes accepted
// by grounded retrieval. The empty value remains compatible with documents
// produced before structural metadata was supplied.
type AgentRetrievalFacet string

const (
	// AgentRetrievalFacetText identifies ordinary prose content.
	AgentRetrievalFacetText AgentRetrievalFacet = "text"
	// AgentRetrievalFacetCode identifies source-code content.
	AgentRetrievalFacetCode AgentRetrievalFacet = "code"
	// AgentRetrievalFacetTable identifies tabular content.
	AgentRetrievalFacetTable AgentRetrievalFacet = "table"
	// AgentRetrievalFacetImage identifies image-derived content.
	AgentRetrievalFacetImage AgentRetrievalFacet = "image"
	// AgentRetrievalFacetRecord identifies structured record content.
	AgentRetrievalFacetRecord AgentRetrievalFacet = "record"
)

// Identify validates and identifies retrieval policy authority.
func (policy AgentRetrievalPolicy) Identify() (artifact.ID, error) {
	if policy.MaximumChunkRunes == 0 || policy.OverlapRunes >= policy.MaximumChunkRunes || policy.CandidateLimit == 0 {
		return artifact.ID{}, errors.New("dataset: invalid agent retrieval policy")
	}
	return artifact.JSONID(artifact.KindProfile, policy)
}

// AgentRetrievalDocument supplies source-bound text extracted by an existing dataset reader.
type AgentRetrievalDocument struct {
	Source    artifact.ID         `json:"source"`
	Text      string              `json:"text"`
	Facet     AgentRetrievalFacet `json:"facet,omitzero"`
	Structure []string            `json:"structure,omitzero"`
	Parent    artifact.ID         `json:"parent,omitzero"`
	Previous  artifact.ID         `json:"previous,omitzero"`
	Next      artifact.ID         `json:"next,omitzero"`
	StartRune uint64              `json:"start_rune,omitzero"`
	StartByte uint64              `json:"start_byte,omitzero"`
}

// agentRetrievalSource is the immutable extraction authority behind cited
// retrieval documents. Its content owns text, structural paths, spans, and
// adjacency; callers cannot attach those claims to an unrelated artifact ID.
type agentRetrievalSource struct {
	Version uint16                     `json:"version"`
	Spans   []agentRetrievalSourceSpan `json:"spans"`
	ID      artifact.ID                `json:"-"`
}

type agentRetrievalSourceSpan struct {
	Text      string              `json:"text"`
	Facet     AgentRetrievalFacet `json:"facet,omitzero"`
	Structure []string            `json:"structure,omitzero"`
	Parent    artifact.ID         `json:"parent,omitzero"`
	Previous  artifact.ID         `json:"previous,omitzero"`
	Next      artifact.ID         `json:"next,omitzero"`
	StartRune uint64              `json:"start_rune,omitzero"`
	StartByte uint64              `json:"start_byte,omitzero"`
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
	Version   uint16              `json:"version"`
	Dataset   artifact.ID         `json:"dataset"`
	Source    artifact.ID         `json:"source"`
	Policy    artifact.ID         `json:"policy"`
	Ordinal   uint64              `json:"ordinal"`
	StartRune uint64              `json:"start_rune"`
	EndRune   uint64              `json:"end_rune"`
	StartByte uint64              `json:"start_byte,omitzero"`
	EndByte   uint64              `json:"end_byte,omitzero"`
	Text      string              `json:"text"`
	Facet     AgentRetrievalFacet `json:"facet,omitzero"`
	Structure []string            `json:"structure,omitzero"`
	Parent    artifact.ID         `json:"parent,omitzero"`
	Previous  artifact.ID         `json:"previous,omitzero"`
	Next      artifact.ID         `json:"next,omitzero"`
	ID        artifact.ID         `json:"-"`
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
	Chunk              artifact.ID         `json:"chunk"`
	Embedding          artifact.ID         `json:"embedding"`
	FilterKeysComplete bool                `json:"filter_keys_complete,omitzero"`
	Source             artifact.ID         `json:"source,omitzero"`
	Facet              AgentRetrievalFacet `json:"facet,omitzero"`
	Parent             artifact.ID         `json:"parent,omitzero"`
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
	Budget       AgentRetrievalBuildBudget
}

// AgentRetrievalBuildBudget bounds source facts, generated chunks, provider
// calls, encoded intermediate bytes, and facts admitted by one build.
type AgentRetrievalBuildBudget struct {
	MaxDocuments         uint64 `json:"max_documents"`
	MaxSourceBytes       uint64 `json:"max_source_bytes"`
	MaxChunks            uint64 `json:"max_chunks"`
	MaxEmbeddingCalls    uint64 `json:"max_embedding_calls"`
	MaxIntermediateBytes uint64 `json:"max_intermediate_bytes"`
	MaxPublishedFacts    uint64 `json:"max_published_facts"`
}

// AgentRetrievalBuildWork reports exact work performed by a build.
type AgentRetrievalBuildWork struct {
	DocumentsInspected uint64 `json:"documents_inspected"`
	SourceBytesLoaded  uint64 `json:"source_bytes_loaded"`
	ChunksBuilt        uint64 `json:"chunks_built"`
	EmbeddingCalls     uint64 `json:"embedding_calls"`
	IntermediateBytes  uint64 `json:"intermediate_bytes"`
	PublishedFacts     uint64 `json:"published_facts"`
}

// AgentRetrievalBuildResult pairs the projection with measured build work.
type AgentRetrievalBuildResult struct {
	Projection AgentRetrievalProjection  `json:"projection"`
	Budget     AgentRetrievalBuildBudget `json:"budget"`
	Work       AgentRetrievalBuildWork   `json:"work"`
}

// AgentRetrievalBuilder publishes dataset-owned retrieval artifacts.
type AgentRetrievalBuilder struct {
	Repository artifact.Repository
}

// AgentRetrievalQuery requests bounded cited results.
type AgentRetrievalQuery struct {
	Projection      artifact.ID
	Text            string
	Limit           uint64
	AllowedDatasets []artifact.ID
	AllowedSources  []artifact.ID
	AllowedFacets   []AgentRetrievalFacet
	AllowedParents  []artifact.ID
	Embedder        AgentEmbeddingProvider
	Reranker        AgentRerankProvider
	Budget          AgentRetrievalSearchBudget
}

// AgentRetrievalSearchBudget bounds compiled-index inspection, decodes,
// loaded bytes, provider calls, and returned facts.
type AgentRetrievalSearchBudget struct {
	MaxEntriesInspected uint64 `json:"max_entries_inspected"`
	MaxChunksLoaded     uint64 `json:"max_chunks_loaded"`
	MaxEmbeddingsLoaded uint64 `json:"max_embeddings_loaded"`
	MaxBytesLoaded      uint64 `json:"max_bytes_loaded"`
	MaxEmbeddingCalls   uint64 `json:"max_embedding_calls"`
	MaxRerankCalls      uint64 `json:"max_rerank_calls"`
	MaxResults          uint64 `json:"max_results"`
}

// AgentRetrievalCitation preserves exact source and span identity.
type AgentRetrievalCitation struct {
	Dataset   artifact.ID         `json:"dataset"`
	Source    artifact.ID         `json:"source"`
	Chunk     artifact.ID         `json:"chunk"`
	StartRune uint64              `json:"start_rune"`
	EndRune   uint64              `json:"end_rune"`
	StartByte uint64              `json:"start_byte,omitzero"`
	EndByte   uint64              `json:"end_byte,omitzero"`
	Text      string              `json:"text"`
	Facet     AgentRetrievalFacet `json:"facet,omitzero"`
	Structure []string            `json:"structure,omitzero"`
	Parent    artifact.ID         `json:"parent,omitzero"`
	Previous  artifact.ID         `json:"previous,omitzero"`
	Next      artifact.ID         `json:"next,omitzero"`
}

// AgentRetrievalResult carries deterministic semantic and rerank evidence.
type AgentRetrievalResult struct {
	Citation         AgentRetrievalCitation `json:"citation"`
	Embedding        artifact.ID            `json:"embedding"`
	Similarity       float64                `json:"similarity"`
	Lexical          uint64                 `json:"lexical"`
	Rerank           float64                `json:"rerank"`
	DenseCandidate   bool                   `json:"dense_candidate,omitzero"`
	LexicalCandidate bool                   `json:"lexical_candidate,omitzero"`
}

// AgentRetrievalWork records exact owner work so callers can distinguish a
// selective query from a broad query that happens to return few results.
type AgentRetrievalWork struct {
	QueryEmbeddings      uint64 `json:"query_embeddings"`
	EntriesInspected     uint64 `json:"entries_inspected"`
	ChunksLoaded         uint64 `json:"chunks_loaded"`
	ChunkBytesLoaded     uint64 `json:"chunk_bytes_loaded"`
	FilteredEntries      uint64 `json:"filtered_entries"`
	EmbeddingsLoaded     uint64 `json:"embeddings_loaded"`
	EmbeddingBytesLoaded uint64 `json:"embedding_bytes_loaded"`
	DenseScored          uint64 `json:"dense_scored"`
	LexicalScored        uint64 `json:"lexical_scored"`
	DenseCandidates      uint64 `json:"dense_candidates"`
	LexicalCandidates    uint64 `json:"lexical_candidates"`
	UnionCandidates      uint64 `json:"union_candidates"`
	RerankCalls          uint64 `json:"rerank_calls"`
	Returned             uint64 `json:"returned"`
}

// AgentRetrievalSearchResult is the auditable form of a grounded retrieval.
// Search remains the compatibility surface for callers that only need hits.
type AgentRetrievalSearchResult struct {
	Results               []AgentRetrievalResult     `json:"results"`
	Budget                AgentRetrievalSearchBudget `json:"budget"`
	Work                  AgentRetrievalWork         `json:"work"`
	DenseCandidateStage   artifact.ID                `json:"dense_candidate_stage"`
	LexicalCandidateStage artifact.ID                `json:"lexical_candidate_stage"`
	UnionCandidateStage   artifact.ID                `json:"union_candidate_stage"`
}

type agentRetrievalCandidateStage string

const (
	agentRetrievalDenseStage   agentRetrievalCandidateStage = "dense"
	agentRetrievalLexicalStage agentRetrievalCandidateStage = "lexical"
	agentRetrievalUnionStage   agentRetrievalCandidateStage = "union"
)

var agentRetrievalChunkCodec = artifact.JSONDocumentCodec(
	"agent retrieval chunk", artifact.KindDatasetShard, agentRetrievalChunkMediaType, agentRetrievalChunkSchema,
	canonicalizeAgentRetrievalChunk,
	func(value AgentRetrievalChunk) artifact.ID { return value.ID },
	func(value *AgentRetrievalChunk, id artifact.ID) { value.ID = id },
	func(value AgentRetrievalChunk) AgentRetrievalChunk {
		value.Structure = slices.Clone(value.Structure)
		return value
	},
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

var agentRetrievalFileSourceCodec = artifact.JSONDocumentCodec(
	"agent retrieval file source", artifact.KindFile, agentRetrievalSourceMediaType, agentRetrievalSourceSchema,
	canonicalizeAgentRetrievalSource,
	func(value agentRetrievalSource) artifact.ID { return value.ID },
	func(value *agentRetrievalSource, id artifact.ID) { value.ID = id },
	cloneAgentRetrievalSource,
)

var agentRetrievalShardSourceCodec = artifact.JSONDocumentCodec(
	"agent retrieval shard source", artifact.KindDatasetShard, agentRetrievalSourceMediaType, agentRetrievalSourceSchema,
	canonicalizeAgentRetrievalSource,
	func(value agentRetrievalSource) artifact.ID { return value.ID },
	func(value *agentRetrievalSource, id artifact.ID) { value.ID = id },
	cloneAgentRetrievalSource,
)

// PublishAgentRetrievalSource stores exact extracted text and structure, then
// returns build documents bound to that content identity. Input documents must
// not carry a caller-selected Source.
func PublishAgentRetrievalSource(
	ctx context.Context,
	repository artifact.Repository,
	kind artifact.Kind,
	documents []AgentRetrievalDocument,
) ([]AgentRetrievalDocument, error) {
	if ctx == nil || repository == nil || len(documents) == 0 {
		return nil, errors.New("dataset: agent retrieval source authority is absent")
	}
	spans := make([]agentRetrievalSourceSpan, len(documents))
	for index, document := range documents {
		if document.Source.Valid() || validateAgentRetrievalSourceFields(document) != nil {
			return nil, errors.New("dataset: invalid agent retrieval source publication")
		}
		spans[index] = agentRetrievalSpan(document)
	}
	codec, err := agentRetrievalSourceCodec(kind)
	if err != nil {
		return nil, err
	}
	source, err := codec.New(agentRetrievalSource{Version: artifact.InitialDocumentVersion, Spans: spans})
	if err != nil {
		return nil, err
	}
	content, err := codec.Content(source)
	if err != nil {
		return nil, err
	}
	batch, err := artifact.NewDocumentBatch(
		"dataset/agent-retrieval-source/"+source.ID.String(), []artifact.Content{content},
		agentRetrievalSourceLineage(source), nil,
	)
	if err != nil {
		return nil, err
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return nil, err
	}
	result := make([]AgentRetrievalDocument, len(source.Spans))
	for index, span := range source.Spans {
		result[index] = agentRetrievalDocument(source.ID, span)
	}
	return result, nil
}

// RequireAgentRetrievalChunk loads one immutable typed chunk for receipt and
// citation verification without exposing the dataset owner's codec.
func RequireAgentRetrievalChunk(ctx context.Context, reader artifact.Reader, id artifact.ID) (AgentRetrievalChunk, error) {
	chunk, err := agentRetrievalChunkCodec.RequireExactLineage(ctx, reader, id, agentRetrievalChunkLineage)
	if err != nil {
		return AgentRetrievalChunk{}, err
	}
	if err := requireAgentRetrievalSourceChunk(ctx, reader, chunk); err != nil {
		return AgentRetrievalChunk{}, err
	}
	return chunk, nil
}

// RequireAgentRetrievalEmbedding loads one immutable typed embedding for
// receipt verification without exposing projection mutation.
func RequireAgentRetrievalEmbedding(ctx context.Context, reader artifact.Reader, id artifact.ID) (AgentRetrievalEmbedding, error) {
	return agentRetrievalEmbeddingCodec.RequireExactLineage(ctx, reader, id, agentRetrievalEmbeddingLineage)
}

// RequireAgentRetrievalProjection loads one immutable typed projection so a
// receipt owner can verify corpus, policy, provider, and candidate-limit links.
func RequireAgentRetrievalProjection(ctx context.Context, reader artifact.Reader, id artifact.ID) (AgentRetrievalProjection, error) {
	return agentRetrievalProjectionCodec.RequireExactLineage(ctx, reader, id, agentRetrievalProjectionLineage)
}

// Build chunks, embeds, and publishes one deterministic projection.
func (builder AgentRetrievalBuilder) Build(ctx context.Context, request AgentRetrievalBuild) (AgentRetrievalProjection, error) {
	result, err := builder.BuildWithWork(ctx, request)
	return result.Projection, err
}

// BuildWithWork applies one finite budget and reports the exact construction
// work. A zero budget derives conservative finite bounds from the request for
// compatibility; partially specified budgets are refused.
func (builder AgentRetrievalBuilder) BuildWithWork(ctx context.Context, request AgentRetrievalBuild) (AgentRetrievalBuildResult, error) {
	policyID, err := request.Policy.Identify()
	if ctx == nil || builder.Repository == nil || request.Dataset.Kind() != artifact.KindDataset ||
		request.Embedder == nil || request.Embedder.ModelIdentity().Kind() != artifact.KindModel ||
		request.RerankPolicy.Kind() != artifact.KindProfile || len(request.Documents) == 0 || err != nil {
		return AgentRetrievalBuildResult{}, errors.Join(errors.New("dataset: invalid agent retrieval build"), err)
	}
	budget, err := resolveAgentRetrievalBuildBudget(request)
	if err != nil {
		return AgentRetrievalBuildResult{}, err
	}
	work := AgentRetrievalBuildWork{}
	if uint64(len(request.Documents)) > budget.MaxDocuments {
		return AgentRetrievalBuildResult{}, errors.New("dataset: agent retrieval build budget exceeded")
	}
	documents := slices.Clone(request.Documents)
	for index := range documents {
		work.DocumentsInspected++
		documents[index].Structure = slices.Clone(documents[index].Structure)
		if err := validateAgentRetrievalDocument(documents[index]); err != nil {
			return AgentRetrievalBuildResult{}, err
		}
		sourceBytes, err := agentRetrievalStoredSize(ctx, builder.Repository, documents[index].Source)
		if err != nil {
			return AgentRetrievalBuildResult{}, err
		}
		work.SourceBytesLoaded, err = agentRetrievalAdd(
			work.SourceBytesLoaded, sourceBytes, "dataset: agent retrieval source byte count overflows",
		)
		if err != nil || work.SourceBytesLoaded > budget.MaxSourceBytes {
			return AgentRetrievalBuildResult{}, errors.Join(errors.New("dataset: agent retrieval build budget exceeded"), err)
		}
		if err := requireAgentRetrievalSourceDocument(ctx, builder.Repository, documents[index]); err != nil {
			return AgentRetrievalBuildResult{}, err
		}
	}
	slices.SortFunc(documents, compareAgentRetrievalDocument)
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
		if index > 0 && compareAgentRetrievalDocument(documents[index-1], document) == 0 {
			return AgentRetrievalBuildResult{}, errors.New("dataset: duplicate agent retrieval document")
		}
		chunks, chunkErr := agentRetrievalChunks(request.Dataset, policyID, document, request.Policy)
		if chunkErr != nil {
			return AgentRetrievalBuildResult{}, chunkErr
		}
		for _, chunk := range chunks {
			if work.ChunksBuilt == budget.MaxChunks || work.EmbeddingCalls == budget.MaxEmbeddingCalls {
				return AgentRetrievalBuildResult{}, errors.New("dataset: agent retrieval build budget exceeded")
			}
			work.ChunksBuilt++
			vector, embedErr := request.Embedder.Embed(ctx, chunk.Text)
			work.EmbeddingCalls++
			if embedErr != nil {
				return AgentRetrievalBuildResult{}, embedErr
			}
			embedding, identifyErr := agentRetrievalEmbeddingCodec.New(AgentRetrievalEmbedding{
				Version: artifact.InitialDocumentVersion, Chunk: chunk.ID,
				Model: request.Embedder.ModelIdentity(), Vector: vector,
			})
			if identifyErr != nil {
				return AgentRetrievalBuildResult{}, identifyErr
			}
			chunkContent, contentErr := agentRetrievalChunkCodec.Content(chunk)
			if contentErr != nil {
				return AgentRetrievalBuildResult{}, contentErr
			}
			publishedFacts := uint64(len([...]artifact.ID{chunk.ID, embedding.ID}))
			if work.PublishedFacts > budget.MaxPublishedFacts ||
				budget.MaxPublishedFacts-work.PublishedFacts < publishedFacts {
				return AgentRetrievalBuildResult{}, errors.New("dataset: agent retrieval build budget exceeded")
			}
			embeddingContent, contentErr := agentRetrievalEmbeddingCodec.Content(embedding)
			if contentErr != nil {
				return AgentRetrievalBuildResult{}, contentErr
			}
			encodedBytes, addErr := agentRetrievalAdd(
				chunkContent.Descriptor.Size, embeddingContent.Descriptor.Size,
				"dataset: agent retrieval build byte count overflows",
			)
			if addErr != nil {
				return AgentRetrievalBuildResult{}, addErr
			}
			work.IntermediateBytes, addErr = agentRetrievalAdd(
				work.IntermediateBytes, encodedBytes, "dataset: agent retrieval build byte count overflows",
			)
			if addErr != nil || work.IntermediateBytes > budget.MaxIntermediateBytes {
				return AgentRetrievalBuildResult{}, errors.Join(errors.New("dataset: agent retrieval build budget exceeded"), addErr)
			}
			work.PublishedFacts += publishedFacts
			contents = append(contents, chunkContent, embeddingContent)
			lineage = append(lineage, agentRetrievalChunkLineage(chunk)...)
			lineage = append(lineage, agentRetrievalEmbeddingLineage(embedding)...)
			entries = append(entries, AgentRetrievalEntry{
				Chunk: chunk.ID, Embedding: embedding.ID, FilterKeysComplete: true,
				Source: chunk.Source, Facet: chunk.Facet, Parent: chunk.Parent,
			})
		}
	}
	projection, err := agentRetrievalProjectionCodec.New(AgentRetrievalProjection{
		Version: artifact.InitialDocumentVersion, Dataset: request.Dataset, Policy: policyID,
		CandidateLimit: request.Policy.CandidateLimit, EmbeddingModel: request.Embedder.ModelIdentity(),
		RerankPolicy: request.RerankPolicy, Entries: entries,
	})
	if err != nil {
		return AgentRetrievalBuildResult{}, err
	}
	projectionContent, err := agentRetrievalProjectionCodec.Content(projection)
	if err != nil {
		return AgentRetrievalBuildResult{}, err
	}
	if work.PublishedFacts == budget.MaxPublishedFacts {
		return AgentRetrievalBuildResult{}, errors.New("dataset: agent retrieval build budget exceeded")
	}
	work.PublishedFacts++
	work.IntermediateBytes, err = agentRetrievalAdd(
		work.IntermediateBytes, projectionContent.Descriptor.Size, "dataset: agent retrieval build byte count overflows",
	)
	if err != nil || work.IntermediateBytes > budget.MaxIntermediateBytes {
		return AgentRetrievalBuildResult{}, errors.Join(errors.New("dataset: agent retrieval build budget exceeded"), err)
	}
	contents = append(contents, projectionContent)
	lineage = append(lineage, agentRetrievalProjectionLineage(projection)...)
	aliasName := agentRetrievalAlias(request.Dataset, policyID, request.Embedder.ModelIdentity(), request.RerankPolicy)
	current, found, err := artifact.ResolveAlias(ctx, builder.Repository, aliasName)
	if err != nil {
		return AgentRetrievalBuildResult{}, err
	}
	if found && current == projection.ID {
		if _, err := RequireAgentRetrievalProjection(ctx, builder.Repository, projection.ID); err != nil {
			return AgentRetrievalBuildResult{}, err
		}
		for _, entry := range projection.Entries {
			if _, err := RequireAgentRetrievalChunk(ctx, builder.Repository, entry.Chunk); err != nil {
				return AgentRetrievalBuildResult{}, err
			}
			if _, err := RequireAgentRetrievalEmbedding(ctx, builder.Repository, entry.Embedding); err != nil {
				return AgentRetrievalBuildResult{}, err
			}
		}
		return AgentRetrievalBuildResult{Projection: projection, Budget: budget, Work: work}, nil
	}
	alias := artifact.AliasBinding{Name: aliasName, Target: projection.ID}
	if found {
		alias.Previous = artifact.IDPointer(current)
	}
	descriptors := make([]artifact.Descriptor, 0, len(parents))
	for _, id := range parents {
		descriptor, present, descriptorErr := builder.Repository.Artifact(ctx, id)
		if descriptorErr != nil {
			return AgentRetrievalBuildResult{}, descriptorErr
		}
		if !present {
			descriptor = artifact.Descriptor{ID: id}
		}
		descriptors = append(descriptors, descriptor)
	}
	if _, err := artifact.CommitBatch(ctx, builder.Repository, artifact.Batch{
		Key: "dataset/agent-retrieval/" + projection.ID.String(), Artifacts: descriptors,
		Contents: contents, Lineage: lineage, Aliases: []artifact.AliasBinding{alias},
	}); err != nil {
		return AgentRetrievalBuildResult{}, err
	}
	return AgentRetrievalBuildResult{Projection: projection, Budget: budget, Work: work}, nil
}

func resolveAgentRetrievalBuildBudget(request AgentRetrievalBuild) (AgentRetrievalBuildBudget, error) {
	budget := request.Budget
	if budget == (AgentRetrievalBuildBudget{}) {
		budget.MaxDocuments = uint64(len(request.Documents))
		var ok bool
		budget.MaxSourceBytes, ok = checked.Mul64(budget.MaxDocuments, artifact.MaxContentBytes)
		if !ok {
			return AgentRetrievalBuildBudget{}, errors.New("dataset: agent retrieval derived build budget overflows")
		}
		for _, document := range request.Documents {
			var err error
			budget.MaxChunks, err = agentRetrievalAdd(
				budget.MaxChunks, uint64(utf8.RuneCountInString(document.Text)),
				"dataset: agent retrieval derived build budget overflows",
			)
			if err != nil {
				return AgentRetrievalBuildBudget{}, err
			}
		}
		budget.MaxEmbeddingCalls = budget.MaxChunks
		factsPerChunk := uint64(len([...]artifact.Kind{artifact.KindDatasetShard, artifact.KindTensorSet}))
		facts, ok := checked.Mul64(budget.MaxChunks, factsPerChunk)
		if !ok {
			return AgentRetrievalBuildBudget{}, errors.New("dataset: agent retrieval derived build budget overflows")
		}
		projectionFacts := uint64(len([...]artifact.Kind{artifact.KindProfile}))
		budget.MaxPublishedFacts, ok = checked.Add64(facts, projectionFacts)
		if !ok {
			return AgentRetrievalBuildBudget{}, errors.New("dataset: agent retrieval derived build budget overflows")
		}
		budget.MaxIntermediateBytes, ok = checked.Mul64(budget.MaxPublishedFacts, artifact.MaxContentBytes)
		if !ok {
			return AgentRetrievalBuildBudget{}, errors.New("dataset: agent retrieval derived build budget overflows")
		}
	}
	if budget.MaxDocuments == 0 || budget.MaxSourceBytes == 0 || budget.MaxChunks == 0 || budget.MaxEmbeddingCalls == 0 ||
		budget.MaxIntermediateBytes == 0 || budget.MaxPublishedFacts == 0 {
		return AgentRetrievalBuildBudget{}, errors.New("dataset: incomplete agent retrieval build budget")
	}
	return budget, nil
}

func resolveAgentRetrievalSearchBudget(
	query AgentRetrievalQuery,
	projection AgentRetrievalProjection,
) (AgentRetrievalSearchBudget, error) {
	budget := query.Budget
	if budget == (AgentRetrievalSearchBudget{}) {
		entries := uint64(len(projection.Entries))
		budget = AgentRetrievalSearchBudget{
			MaxEntriesInspected: entries, MaxChunksLoaded: entries, MaxEmbeddingsLoaded: entries,
			MaxEmbeddingCalls: uint64(len([...]AgentEmbeddingProvider{query.Embedder})),
			MaxRerankCalls:    projection.CandidateLimit, MaxResults: query.Limit,
		}
		factsPerEntry := uint64(len([...]artifact.Kind{artifact.KindDatasetShard, artifact.KindTensorSet}))
		loadedFacts, ok := checked.Mul64(entries, factsPerEntry)
		if !ok {
			return AgentRetrievalSearchBudget{}, errors.New("dataset: agent retrieval derived search budget overflows")
		}
		budget.MaxBytesLoaded, ok = checked.Mul64(loadedFacts, artifact.MaxContentBytes)
		if !ok {
			return AgentRetrievalSearchBudget{}, errors.New("dataset: agent retrieval derived search budget overflows")
		}
	}
	if budget.MaxEntriesInspected == 0 || budget.MaxChunksLoaded == 0 || budget.MaxEmbeddingsLoaded == 0 ||
		budget.MaxBytesLoaded == 0 || budget.MaxEmbeddingCalls == 0 || budget.MaxRerankCalls == 0 || budget.MaxResults == 0 {
		return AgentRetrievalSearchBudget{}, errors.New("dataset: incomplete agent retrieval search budget")
	}
	return budget, nil
}

func agentRetrievalSearchFitsBytes(work AgentRetrievalWork, next, maximum uint64) bool {
	total, ok := checked.Add64(work.ChunkBytesLoaded, work.EmbeddingBytesLoaded)
	if !ok {
		return false
	}
	total, ok = checked.Add64(total, next)
	return ok && total <= maximum
}

// Search verifies exact provider identities and returns deterministic cited
// results. It remains the compatibility wrapper over SearchWithWork.
func (builder AgentRetrievalBuilder) Search(ctx context.Context, query AgentRetrievalQuery) ([]AgentRetrievalResult, error) {
	search, err := builder.SearchWithWork(ctx, query)
	if err != nil {
		return nil, err
	}
	return cloneAgentRetrievalResults(search.Results), nil
}

// SearchWithWork performs structural filter pushdown, ranks dense and lexical
// candidates independently, then derives one total bounded rank-round union
// before invoking the existing reranker. The channel union has no weight knob.
func (builder AgentRetrievalBuilder) SearchWithWork(ctx context.Context, query AgentRetrievalQuery) (AgentRetrievalSearchResult, error) {
	if ctx == nil || builder.Repository == nil || query.Text == "" || query.Limit == 0 || query.Embedder == nil || query.Reranker == nil {
		return AgentRetrievalSearchResult{}, errors.New("dataset: invalid agent retrieval query")
	}
	if err := validateAgentRetrievalQuery(query); err != nil {
		return AgentRetrievalSearchResult{}, err
	}
	projection, err := RequireAgentRetrievalProjection(ctx, builder.Repository, query.Projection)
	if err != nil || projection.EmbeddingModel != query.Embedder.ModelIdentity() || projection.RerankPolicy != query.Reranker.PolicyIdentity() {
		return AgentRetrievalSearchResult{}, errors.Join(errors.New("dataset: agent retrieval provider identity differs"), err)
	}
	if len(query.AllowedDatasets) != 0 && !slices.Contains(query.AllowedDatasets, projection.Dataset) {
		return AgentRetrievalSearchResult{}, errors.New("dataset: retrieval projection is outside active agent authority")
	}
	budget, err := resolveAgentRetrievalSearchBudget(query, projection)
	if err != nil {
		return AgentRetrievalSearchResult{}, err
	}
	if query.Limit > budget.MaxResults {
		return AgentRetrievalSearchResult{}, errors.New("dataset: agent retrieval search budget exceeded")
	}
	search := AgentRetrievalSearchResult{Budget: budget}
	if search.Work.QueryEmbeddings == budget.MaxEmbeddingCalls {
		return AgentRetrievalSearchResult{}, errors.New("dataset: agent retrieval search budget exceeded")
	}
	queryVector, err := query.Embedder.Embed(ctx, query.Text)
	search.Work.QueryEmbeddings++
	if err != nil {
		return AgentRetrievalSearchResult{}, err
	}
	queryID, err := artifact.JSONID(artifact.KindEvidence, query.Text)
	if err != nil {
		return AgentRetrievalSearchResult{}, err
	}
	queryTerms := agentRetrievalTerms(query.Text)
	scored := make([]AgentRetrievalResult, 0, len(projection.Entries))
	for _, entry := range projection.Entries {
		if search.Work.EntriesInspected == budget.MaxEntriesInspected {
			return AgentRetrievalSearchResult{}, errors.New("dataset: agent retrieval search budget exceeded")
		}
		search.Work.EntriesInspected++
		if !agentRetrievalEntryMayMatch(query, entry) {
			search.Work.FilteredEntries++
			continue
		}
		chunkBytes, sizeErr := agentRetrievalStoredSize(ctx, builder.Repository, entry.Chunk)
		if sizeErr != nil {
			return AgentRetrievalSearchResult{}, sizeErr
		}
		if search.Work.ChunksLoaded == budget.MaxChunksLoaded ||
			!agentRetrievalSearchFitsBytes(search.Work, chunkBytes, budget.MaxBytesLoaded) {
			return AgentRetrievalSearchResult{}, errors.New("dataset: agent retrieval search budget exceeded")
		}
		chunk, chunkErr := RequireAgentRetrievalChunk(ctx, builder.Repository, entry.Chunk)
		search.Work.ChunksLoaded++
		search.Work.ChunkBytesLoaded, sizeErr = agentRetrievalAdd(
			search.Work.ChunkBytesLoaded, chunkBytes, "dataset: agent retrieval work byte count overflows",
		)
		if sizeErr != nil {
			return AgentRetrievalSearchResult{}, sizeErr
		}
		if chunkErr != nil || chunk.Dataset != projection.Dataset || chunk.Policy != projection.Policy ||
			!agentRetrievalEntryAgrees(entry, chunk) {
			return AgentRetrievalSearchResult{}, errors.Join(errors.New("dataset: agent retrieval entry differs"), chunkErr)
		}
		if !agentRetrievalChunkMatches(query, chunk) {
			search.Work.FilteredEntries++
			continue
		}
		embeddingBytes, sizeErr := agentRetrievalStoredSize(ctx, builder.Repository, entry.Embedding)
		if sizeErr != nil {
			return AgentRetrievalSearchResult{}, sizeErr
		}
		if search.Work.EmbeddingsLoaded == budget.MaxEmbeddingsLoaded ||
			!agentRetrievalSearchFitsBytes(search.Work, embeddingBytes, budget.MaxBytesLoaded) {
			return AgentRetrievalSearchResult{}, errors.New("dataset: agent retrieval search budget exceeded")
		}
		embedding, embeddingErr := RequireAgentRetrievalEmbedding(ctx, builder.Repository, entry.Embedding)
		search.Work.EmbeddingsLoaded++
		search.Work.EmbeddingBytesLoaded, sizeErr = agentRetrievalAdd(
			search.Work.EmbeddingBytesLoaded, embeddingBytes, "dataset: agent retrieval work byte count overflows",
		)
		if sizeErr != nil {
			return AgentRetrievalSearchResult{}, sizeErr
		}
		if embeddingErr != nil || embedding.Chunk != chunk.ID || embedding.Model != projection.EmbeddingModel {
			return AgentRetrievalSearchResult{}, errors.Join(errors.New("dataset: agent retrieval entry differs"), embeddingErr)
		}
		similarity, scoreErr := cosineSimilarity(queryVector, embedding.Vector)
		if scoreErr != nil || !finiteRetrievalNumber(similarity) {
			return AgentRetrievalSearchResult{}, errors.Join(errors.New("dataset: invalid agent similarity score"), scoreErr)
		}
		search.Work.DenseScored++
		search.Work.LexicalScored++
		scored = append(scored, AgentRetrievalResult{
			Citation: AgentRetrievalCitation{
				Dataset: chunk.Dataset, Source: chunk.Source, Chunk: chunk.ID,
				StartRune: chunk.StartRune, EndRune: chunk.EndRune, StartByte: chunk.StartByte, EndByte: chunk.EndByte,
				Text: chunk.Text, Facet: chunk.Facet, Structure: slices.Clone(chunk.Structure),
				Parent: chunk.Parent, Previous: chunk.Previous, Next: chunk.Next,
			},
			Embedding: embedding.ID, Similarity: similarity, Lexical: agentRetrievalLexicalScore(queryTerms, chunk.Text),
		})
	}

	dense := cloneAgentRetrievalResults(scored)
	slices.SortFunc(dense, func(left, right AgentRetrievalResult) int {
		if order := cmp.Compare(right.Similarity, left.Similarity); order != 0 {
			return order
		}
		return artifact.CompareID(left.Citation.Chunk, right.Citation.Chunk)
	})
	dense = dense[:min(uint64(len(dense)), projection.CandidateLimit)]
	search.Work.DenseCandidates = uint64(len(dense))
	search.DenseCandidateStage, err = agentRetrievalCandidateStageID(
		agentRetrievalDenseStage, projection.ID, queryID, dense,
	)
	if err != nil {
		return AgentRetrievalSearchResult{}, err
	}

	lexical := make([]AgentRetrievalResult, 0, len(scored))
	for _, result := range scored {
		if result.Lexical != 0 {
			lexical = append(lexical, result)
		}
	}
	slices.SortFunc(lexical, func(left, right AgentRetrievalResult) int {
		if order := cmp.Compare(right.Lexical, left.Lexical); order != 0 {
			return order
		}
		return artifact.CompareID(left.Citation.Chunk, right.Citation.Chunk)
	})
	lexical = lexical[:min(uint64(len(lexical)), projection.CandidateLimit)]
	search.Work.LexicalCandidates = uint64(len(lexical))
	search.LexicalCandidateStage, err = agentRetrievalCandidateStageID(
		agentRetrievalLexicalStage, projection.ID, queryID, lexical,
	)
	if err != nil {
		return AgentRetrievalSearchResult{}, err
	}

	candidates := agentRetrievalRankRoundUnion(dense, lexical, projection.CandidateLimit)
	search.Work.UnionCandidates = uint64(len(candidates))
	search.UnionCandidateStage, err = agentRetrievalCandidateStageID(
		agentRetrievalUnionStage, projection.ID, queryID, candidates,
	)
	if err != nil {
		return AgentRetrievalSearchResult{}, err
	}
	for index := range candidates {
		if search.Work.RerankCalls == budget.MaxRerankCalls {
			return AgentRetrievalSearchResult{}, errors.New("dataset: agent retrieval search budget exceeded")
		}
		rerank, scoreErr := query.Reranker.Score(ctx, query.Text, candidates[index].Citation.Text)
		search.Work.RerankCalls++
		if scoreErr != nil || !finiteRetrievalNumber(rerank) {
			return AgentRetrievalSearchResult{}, errors.Join(errors.New("dataset: invalid agent rerank score"), scoreErr)
		}
		candidates[index].Rerank = rerank
	}
	slices.SortFunc(candidates, func(left, right AgentRetrievalResult) int {
		if order := cmp.Compare(right.Rerank, left.Rerank); order != 0 {
			return order
		}
		if order := cmp.Compare(right.Similarity, left.Similarity); order != 0 {
			return order
		}
		if order := cmp.Compare(right.Lexical, left.Lexical); order != 0 {
			return order
		}
		return artifact.CompareID(left.Citation.Chunk, right.Citation.Chunk)
	})
	limit := min(uint64(len(candidates)), query.Limit)
	search.Results = cloneAgentRetrievalResults(candidates[:limit])
	search.Work.Returned = uint64(len(search.Results))
	return search, nil
}

func agentRetrievalChunks(datasetID, policyID artifact.ID, document AgentRetrievalDocument, policy AgentRetrievalPolicy) ([]AgentRetrievalChunk, error) {
	runes := []rune(document.Text)
	byteOffsets := make([]uint64, len(runes)+1)
	runeIndex := 0
	for byteIndex := range document.Text {
		byteOffsets[runeIndex] = uint64(byteIndex)
		runeIndex++
	}
	byteOffsets[len(runes)] = uint64(len(document.Text))
	hostIntLimit := uint64(^uint(0) >> 1)
	if policy.MaximumChunkRunes > hostIntLimit || policy.OverlapRunes > hostIntLimit {
		return nil, errors.New("dataset: agent retrieval chunk policy exceeds host bounds")
	}
	maximum, overlap := int(policy.MaximumChunkRunes), int(policy.OverlapRunes)
	result := make([]AgentRetrievalChunk, 0)
	for start, ordinal := 0, uint64(0); start < len(runes); ordinal++ {
		end := len(runes)
		if maximum < len(runes)-start {
			end = start + maximum
		}
		startRune, runeErr := agentRetrievalAdd(document.StartRune, uint64(start), "dataset: agent retrieval source span overflows")
		endRune, endRuneErr := agentRetrievalAdd(document.StartRune, uint64(end), "dataset: agent retrieval source span overflows")
		startByte, byteErr := agentRetrievalAdd(document.StartByte, byteOffsets[start], "dataset: agent retrieval source span overflows")
		endByte, endByteErr := agentRetrievalAdd(document.StartByte, byteOffsets[end], "dataset: agent retrieval source span overflows")
		if runeErr != nil || endRuneErr != nil || byteErr != nil || endByteErr != nil {
			return nil, errors.Join(runeErr, endRuneErr, byteErr, endByteErr)
		}
		chunk, err := agentRetrievalChunkCodec.New(AgentRetrievalChunk{
			Version: artifact.InitialDocumentVersion, Dataset: datasetID, Source: document.Source, Policy: policyID,
			Ordinal: ordinal, StartRune: startRune, EndRune: endRune, StartByte: startByte, EndByte: endByte,
			Text: string(runes[start:end]), Facet: document.Facet, Structure: slices.Clone(document.Structure),
			Parent: document.Parent, Previous: document.Previous, Next: document.Next,
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

func agentRetrievalSourceCodec(kind artifact.Kind) (artifact.DocumentCodec[agentRetrievalSource], error) {
	switch kind {
	case artifact.KindFile:
		return agentRetrievalFileSourceCodec, nil
	case artifact.KindDatasetShard:
		return agentRetrievalShardSourceCodec, nil
	default:
		return artifact.DocumentCodec[agentRetrievalSource]{}, errors.New("dataset: invalid agent retrieval source kind")
	}
}

func agentRetrievalSpan(document AgentRetrievalDocument) agentRetrievalSourceSpan {
	return agentRetrievalSourceSpan{
		Text: document.Text, Facet: document.Facet, Structure: slices.Clone(document.Structure),
		Parent: document.Parent, Previous: document.Previous, Next: document.Next,
		StartRune: document.StartRune, StartByte: document.StartByte,
	}
}

func agentRetrievalDocument(source artifact.ID, span agentRetrievalSourceSpan) AgentRetrievalDocument {
	return AgentRetrievalDocument{
		Source: source, Text: span.Text, Facet: span.Facet, Structure: slices.Clone(span.Structure),
		Parent: span.Parent, Previous: span.Previous, Next: span.Next,
		StartRune: span.StartRune, StartByte: span.StartByte,
	}
}

func requireAgentRetrievalSource(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (agentRetrievalSource, error) {
	codec, err := agentRetrievalSourceCodec(id.Kind())
	if err != nil {
		return agentRetrievalSource{}, err
	}
	return codec.RequireExactLineage(ctx, reader, id, agentRetrievalSourceLineage)
}

func requireAgentRetrievalSourceDocument(
	ctx context.Context,
	reader artifact.Reader,
	document AgentRetrievalDocument,
) error {
	source, err := requireAgentRetrievalSource(ctx, reader, document.Source)
	if err != nil {
		return errors.Join(errors.New("dataset: agent retrieval source content is absent"), err)
	}
	want := agentRetrievalSpan(document)
	if !slices.ContainsFunc(source.Spans, func(span agentRetrievalSourceSpan) bool {
		return compareAgentRetrievalSourceSpan(span, want) == 0
	}) {
		return errors.New("dataset: agent retrieval document differs from stored source content or structure")
	}
	return nil
}

func requireAgentRetrievalSourceChunk(
	ctx context.Context,
	reader artifact.Reader,
	chunk AgentRetrievalChunk,
) error {
	source, err := requireAgentRetrievalSource(ctx, reader, chunk.Source)
	if err != nil {
		return errors.Join(errors.New("dataset: agent retrieval chunk source is absent"), err)
	}
	for _, span := range source.Spans {
		if span.Facet != chunk.Facet || !slices.Equal(span.Structure, chunk.Structure) ||
			span.Parent != chunk.Parent || span.Previous != chunk.Previous || span.Next != chunk.Next ||
			chunk.StartRune < span.StartRune || chunk.StartByte < span.StartByte {
			continue
		}
		runeStart, byteStart := chunk.StartRune-span.StartRune, chunk.StartByte-span.StartByte
		runeEnd, byteEnd := chunk.EndRune-span.StartRune, chunk.EndByte-span.StartByte
		runes := []rune(span.Text)
		if runeEnd > uint64(len(runes)) || byteEnd > uint64(len(span.Text)) || runeStart >= runeEnd || byteStart >= byteEnd {
			continue
		}
		if string(runes[int(runeStart):int(runeEnd)]) == chunk.Text &&
			span.Text[int(byteStart):int(byteEnd)] == chunk.Text {
			return nil
		}
	}
	return errors.New("dataset: agent retrieval chunk differs from stored source content or structure")
}

func agentRetrievalSourceLineage(value agentRetrievalSource) []artifact.Lineage {
	parents := make([]artifact.ID, 0, len(value.Spans)*3)
	for _, span := range value.Spans {
		parents = appendUniqueRetrievalIDs(parents, span.Parent, span.Previous, span.Next)
	}
	return artifact.DependencyLineage(value.ID, parents...)
}

func agentRetrievalChunkLineage(value AgentRetrievalChunk) []artifact.Lineage {
	parents := appendUniqueRetrievalIDs(nil, value.Dataset, value.Source, value.Policy, value.Parent, value.Previous, value.Next)
	return artifact.DependencyLineage(value.ID, parents...)
}

func agentRetrievalEmbeddingLineage(value AgentRetrievalEmbedding) []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.Chunk, value.Model)
}

func agentRetrievalProjectionLineage(value AgentRetrievalProjection) []artifact.Lineage {
	parents := appendUniqueRetrievalIDs(nil, value.Dataset, value.Policy, value.EmbeddingModel, value.RerankPolicy)
	for _, entry := range value.Entries {
		parents = appendUniqueRetrievalIDs(parents, entry.Chunk, entry.Embedding)
	}
	return artifact.DependencyLineage(value.ID, parents...)
}

func appendUniqueRetrievalIDs(destination []artifact.ID, values ...artifact.ID) []artifact.ID {
	for _, id := range values {
		if id.Valid() && !slices.Contains(destination, id) {
			destination = append(destination, id)
		}
	}
	return destination
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

func canonicalizeAgentRetrievalSource(value *agentRetrievalSource) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || len(value.Spans) == 0 {
		return errors.New("dataset: invalid agent retrieval source document")
	}
	value.Spans = slices.Clone(value.Spans)
	for index := range value.Spans {
		value.Spans[index].Structure = slices.Clone(value.Spans[index].Structure)
		document := agentRetrievalDocument(artifact.ID{}, value.Spans[index])
		if err := validateAgentRetrievalSourceFields(document); err != nil {
			return err
		}
	}
	slices.SortFunc(value.Spans, compareAgentRetrievalSourceSpan)
	for index := range value.Spans {
		if index != 0 && compareAgentRetrievalSourceSpan(value.Spans[index-1], value.Spans[index]) == 0 {
			return errors.New("dataset: duplicate agent retrieval source span")
		}
	}
	return nil
}

func canonicalizeAgentRetrievalChunk(value *AgentRetrievalChunk) error {
	validByteSpan := value != nil && (value.StartByte == 0 && value.EndByte == 0 ||
		value.EndByte > value.StartByte && uint64(len(value.Text)) == value.EndByte-value.StartByte)
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Dataset.Kind() != artifact.KindDataset ||
		(value.Source.Kind() != artifact.KindFile && value.Source.Kind() != artifact.KindDatasetShard) ||
		value.Policy.Kind() != artifact.KindProfile || value.Text == "" || value.EndRune <= value.StartRune ||
		!utf8.ValidString(value.Text) || uint64(len([]rune(value.Text))) != value.EndRune-value.StartRune || !validByteSpan ||
		!validAgentRetrievalFacet(value.Facet) ||
		!validAgentRetrievalStructure(value.Structure) || !validOptionalRetrievalRelation(value.Parent) ||
		!validOptionalRetrievalRelation(value.Previous) || !validOptionalRetrievalRelation(value.Next) {
		return errors.New("dataset: invalid agent retrieval chunk")
	}
	value.Structure = slices.Clone(value.Structure)
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
		if entry.Chunk.Kind() != artifact.KindDatasetShard || entry.Embedding.Kind() != artifact.KindTensorSet ||
			!validOptionalRetrievalSource(entry.Source) || !validAgentRetrievalFacet(entry.Facet) ||
			!validOptionalRetrievalRelation(entry.Parent) || entry.FilterKeysComplete && !entry.Source.Valid() {
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

func validateAgentRetrievalDocument(document AgentRetrievalDocument) error {
	if !validOptionalRetrievalSource(document.Source) || !document.Source.Valid() {
		return errors.New("dataset: invalid agent retrieval source")
	}
	return validateAgentRetrievalSourceFields(document)
}

func validateAgentRetrievalSourceFields(document AgentRetrievalDocument) error {
	if document.Text == "" || !utf8.ValidString(document.Text) ||
		!validAgentRetrievalFacet(document.Facet) || !validAgentRetrievalStructure(document.Structure) ||
		!validOptionalRetrievalRelation(document.Parent) || !validOptionalRetrievalRelation(document.Previous) ||
		!validOptionalRetrievalRelation(document.Next) {
		return errors.New("dataset: invalid agent retrieval source")
	}
	_, runeErr := agentRetrievalAdd(document.StartRune, uint64(len([]rune(document.Text))), "dataset: agent retrieval source span overflows")
	_, byteErr := agentRetrievalAdd(document.StartByte, uint64(len(document.Text)), "dataset: agent retrieval source span overflows")
	if runeErr != nil || byteErr != nil {
		return errors.Join(errors.New("dataset: invalid agent retrieval source span"), runeErr, byteErr)
	}
	return nil
}

func validateAgentRetrievalQuery(query AgentRetrievalQuery) error {
	for _, datasetID := range query.AllowedDatasets {
		if datasetID.Kind() != artifact.KindDataset {
			return errors.New("dataset: invalid allowed retrieval dataset")
		}
	}
	for _, sourceID := range query.AllowedSources {
		if !sourceID.Valid() || !validOptionalRetrievalSource(sourceID) {
			return errors.New("dataset: invalid allowed retrieval source")
		}
	}
	for _, facet := range query.AllowedFacets {
		if facet == "" || !validAgentRetrievalFacet(facet) {
			return errors.New("dataset: invalid allowed retrieval facet")
		}
	}
	for _, parentID := range query.AllowedParents {
		if parentID.Kind() != artifact.KindDatasetShard {
			return errors.New("dataset: invalid allowed retrieval parent")
		}
	}
	return nil
}

func compareAgentRetrievalDocument(left, right AgentRetrievalDocument) int {
	if order := artifact.CompareID(left.Source, right.Source); order != 0 {
		return order
	}
	if order := cmp.Compare(left.StartRune, right.StartRune); order != 0 {
		return order
	}
	if order := cmp.Compare(left.StartByte, right.StartByte); order != 0 {
		return order
	}
	if order := cmp.Compare(left.Facet, right.Facet); order != 0 {
		return order
	}
	if order := compareAgentRetrievalStructure(left.Structure, right.Structure); order != 0 {
		return order
	}
	if order := artifact.CompareID(left.Parent, right.Parent); order != 0 {
		return order
	}
	if order := artifact.CompareID(left.Previous, right.Previous); order != 0 {
		return order
	}
	if order := artifact.CompareID(left.Next, right.Next); order != 0 {
		return order
	}
	return strings.Compare(left.Text, right.Text)
}

func compareAgentRetrievalSourceSpan(left, right agentRetrievalSourceSpan) int {
	return compareAgentRetrievalDocument(
		agentRetrievalDocument(artifact.ID{}, left),
		agentRetrievalDocument(artifact.ID{}, right),
	)
}

func compareAgentRetrievalStructure(left, right []string) int {
	for index := 0; index < min(len(left), len(right)); index++ {
		if order := strings.Compare(left[index], right[index]); order != 0 {
			return order
		}
	}
	return cmp.Compare(len(left), len(right))
}

func validAgentRetrievalFacet(facet AgentRetrievalFacet) bool {
	switch facet {
	case "", AgentRetrievalFacetText, AgentRetrievalFacetCode, AgentRetrievalFacetTable,
		AgentRetrievalFacetImage, AgentRetrievalFacetRecord:
		return true
	default:
		return false
	}
}

func validAgentRetrievalStructure(structure []string) bool {
	for _, component := range structure {
		if component == "" || strings.TrimSpace(component) != component || strings.ContainsAny(component, "\x00\r\n") {
			return false
		}
	}
	return true
}

func validOptionalRetrievalSource(id artifact.ID) bool {
	return id == (artifact.ID{}) || id.Kind() == artifact.KindFile || id.Kind() == artifact.KindDatasetShard
}

func validOptionalRetrievalRelation(id artifact.ID) bool {
	return id == (artifact.ID{}) || id.Kind() == artifact.KindDatasetShard
}

func agentRetrievalStoredSize(ctx context.Context, reader artifact.Reader, id artifact.ID) (uint64, error) {
	descriptor, found, err := reader.Artifact(ctx, id)
	if err != nil || !found || descriptor.ID != id {
		return 0, errors.Join(errors.New("dataset: agent retrieval stored descriptor differs"), err)
	}
	return descriptor.Size, nil
}

func agentRetrievalAdd(base, delta uint64, overflow string) (uint64, error) {
	value, ok := checked.Add64(base, delta)
	if !ok {
		return 0, errors.New(overflow)
	}
	return value, nil
}

func agentRetrievalEntryMayMatch(query AgentRetrievalQuery, entry AgentRetrievalEntry) bool {
	if entry.FilterKeysComplete {
		return (len(query.AllowedSources) == 0 || slices.Contains(query.AllowedSources, entry.Source)) &&
			(len(query.AllowedFacets) == 0 || slices.Contains(query.AllowedFacets, entry.Facet)) &&
			(len(query.AllowedParents) == 0 || slices.Contains(query.AllowedParents, entry.Parent))
	}
	if len(query.AllowedSources) != 0 && entry.Source.Valid() && !slices.Contains(query.AllowedSources, entry.Source) {
		return false
	}
	if len(query.AllowedFacets) != 0 && entry.Facet != "" && !slices.Contains(query.AllowedFacets, entry.Facet) {
		return false
	}
	return len(query.AllowedParents) == 0 || !entry.Parent.Valid() || slices.Contains(query.AllowedParents, entry.Parent)
}

func agentRetrievalChunkMatches(query AgentRetrievalQuery, chunk AgentRetrievalChunk) bool {
	return (len(query.AllowedSources) == 0 || slices.Contains(query.AllowedSources, chunk.Source)) &&
		(len(query.AllowedFacets) == 0 || slices.Contains(query.AllowedFacets, chunk.Facet)) &&
		(len(query.AllowedParents) == 0 || slices.Contains(query.AllowedParents, chunk.Parent))
}

func agentRetrievalEntryAgrees(entry AgentRetrievalEntry, chunk AgentRetrievalChunk) bool {
	if entry.FilterKeysComplete {
		return entry.Source == chunk.Source && entry.Facet == chunk.Facet && entry.Parent == chunk.Parent
	}
	return (!entry.Source.Valid() || entry.Source == chunk.Source) &&
		(entry.Facet == "" || entry.Facet == chunk.Facet) && (!entry.Parent.Valid() || entry.Parent == chunk.Parent)
}

func agentRetrievalRankRoundUnion(dense, lexical []AgentRetrievalResult, limit uint64) []AgentRetrievalResult {
	type membership struct {
		dense   bool
		lexical bool
	}
	byChunk := make(map[artifact.ID]AgentRetrievalResult, len(dense)+len(lexical))
	memberships := make(map[artifact.ID]membership, len(dense)+len(lexical))
	for _, result := range dense {
		byChunk[result.Citation.Chunk] = result
		member := memberships[result.Citation.Chunk]
		member.dense = true
		memberships[result.Citation.Chunk] = member
	}
	for _, result := range lexical {
		byChunk[result.Citation.Chunk] = result
		member := memberships[result.Citation.Chunk]
		member.lexical = true
		memberships[result.Citation.Chunk] = member
	}
	capacity := len(byChunk)
	if limit < uint64(capacity) {
		capacity = int(limit)
	}
	selected := make(map[artifact.ID]struct{}, capacity)
	result := make([]AgentRetrievalResult, 0, capacity)
	for rank := 0; uint64(len(result)) < limit && rank < max(len(dense), len(lexical)); rank++ {
		var round []artifact.ID
		if rank < len(dense) {
			round = append(round, dense[rank].Citation.Chunk)
		}
		if rank < len(lexical) && !slices.Contains(round, lexical[rank].Citation.Chunk) {
			round = append(round, lexical[rank].Citation.Chunk)
		}
		slices.SortFunc(round, artifact.CompareID)
		for _, chunkID := range round {
			if uint64(len(result)) == limit {
				break
			}
			if _, duplicate := selected[chunkID]; duplicate {
				continue
			}
			candidate := byChunk[chunkID]
			candidate.DenseCandidate = memberships[chunkID].dense
			candidate.LexicalCandidate = memberships[chunkID].lexical
			result = append(result, candidate)
			selected[chunkID] = struct{}{}
		}
	}
	return result
}

func agentRetrievalCandidateStageID(
	stage agentRetrievalCandidateStage,
	projection, query artifact.ID,
	values []AgentRetrievalResult,
) (artifact.ID, error) {
	if projection.Kind() != artifact.KindProfile || query.Kind() != artifact.KindEvidence {
		return artifact.ID{}, errors.New("dataset: invalid agent retrieval candidate stage authority")
	}
	type denseCandidate struct {
		Chunk      artifact.ID `json:"chunk"`
		Embedding  artifact.ID `json:"embedding"`
		Similarity float64     `json:"similarity"`
	}
	type lexicalCandidate struct {
		Chunk     artifact.ID `json:"chunk"`
		Embedding artifact.ID `json:"embedding"`
		Lexical   uint64      `json:"lexical"`
	}
	type unionCandidate struct {
		Chunk            artifact.ID `json:"chunk"`
		Embedding        artifact.ID `json:"embedding"`
		Similarity       float64     `json:"similarity"`
		Lexical          uint64      `json:"lexical"`
		DenseCandidate   bool        `json:"dense_candidate,omitzero"`
		LexicalCandidate bool        `json:"lexical_candidate,omitzero"`
	}
	switch stage {
	case agentRetrievalDenseStage:
		candidates := make([]denseCandidate, len(values))
		for index, value := range values {
			if value.Citation.Chunk.Kind() != artifact.KindDatasetShard || value.Embedding.Kind() != artifact.KindTensorSet ||
				!finiteRetrievalNumber(value.Similarity) {
				return artifact.ID{}, errors.New("dataset: invalid dense retrieval candidate stage")
			}
			candidates[index] = denseCandidate{value.Citation.Chunk, value.Embedding, value.Similarity}
		}
		return artifact.JSONID(artifact.KindEvidence, struct {
			Version    uint16                       `json:"version"`
			Stage      agentRetrievalCandidateStage `json:"stage"`
			Projection artifact.ID                  `json:"projection"`
			Query      artifact.ID                  `json:"query"`
			Candidates []denseCandidate             `json:"candidates"`
		}{artifact.InitialDocumentVersion, stage, projection, query, candidates})
	case agentRetrievalLexicalStage:
		candidates := make([]lexicalCandidate, len(values))
		for index, value := range values {
			if value.Citation.Chunk.Kind() != artifact.KindDatasetShard || value.Embedding.Kind() != artifact.KindTensorSet ||
				value.Lexical == 0 {
				return artifact.ID{}, errors.New("dataset: invalid lexical retrieval candidate stage")
			}
			candidates[index] = lexicalCandidate{value.Citation.Chunk, value.Embedding, value.Lexical}
		}
		return artifact.JSONID(artifact.KindEvidence, struct {
			Version    uint16                       `json:"version"`
			Stage      agentRetrievalCandidateStage `json:"stage"`
			Projection artifact.ID                  `json:"projection"`
			Query      artifact.ID                  `json:"query"`
			Candidates []lexicalCandidate           `json:"candidates"`
		}{artifact.InitialDocumentVersion, stage, projection, query, candidates})
	case agentRetrievalUnionStage:
		candidates := make([]unionCandidate, len(values))
		for index, value := range values {
			if value.Citation.Chunk.Kind() != artifact.KindDatasetShard || value.Embedding.Kind() != artifact.KindTensorSet ||
				!finiteRetrievalNumber(value.Similarity) || !value.DenseCandidate && !value.LexicalCandidate {
				return artifact.ID{}, errors.New("dataset: invalid union retrieval candidate stage")
			}
			candidates[index] = unionCandidate{
				value.Citation.Chunk, value.Embedding, value.Similarity, value.Lexical,
				value.DenseCandidate, value.LexicalCandidate,
			}
		}
		return artifact.JSONID(artifact.KindEvidence, struct {
			Version    uint16                       `json:"version"`
			Stage      agentRetrievalCandidateStage `json:"stage"`
			Projection artifact.ID                  `json:"projection"`
			Query      artifact.ID                  `json:"query"`
			Candidates []unionCandidate             `json:"candidates"`
		}{artifact.InitialDocumentVersion, stage, projection, query, candidates})
	default:
		return artifact.ID{}, errors.New("dataset: invalid agent retrieval candidate stage")
	}
}

func agentRetrievalTerms(text string) []string {
	seen := make(map[string]struct{})
	for _, term := range strings.FieldsFunc(strings.ToLower(text), func(value rune) bool {
		return !unicode.IsLetter(value) && !unicode.IsNumber(value)
	}) {
		seen[term] = struct{}{}
	}
	return slices.Sorted(maps.Keys(seen))
}

func agentRetrievalLexicalScore(queryTerms []string, text string) uint64 {
	textTerms := agentRetrievalTerms(text)
	var score uint64
	for _, queryTerm := range queryTerms {
		if _, found := slices.BinarySearch(textTerms, queryTerm); found {
			score++
		}
	}
	return score
}

func cloneAgentRetrievalResults(values []AgentRetrievalResult) []AgentRetrievalResult {
	result := slices.Clone(values)
	for index := range result {
		result[index].Citation.Structure = slices.Clone(result[index].Citation.Structure)
	}
	return result
}

func cloneAgentRetrievalSource(value agentRetrievalSource) agentRetrievalSource {
	value.Spans = slices.Clone(value.Spans)
	for index := range value.Spans {
		value.Spans[index].Structure = slices.Clone(value.Spans[index].Structure)
	}
	return value
}
