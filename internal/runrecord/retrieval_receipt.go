package runrecord

import (
	"context"
	"errors"
	"math"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/dataset"
)

const (
	retrievalReceiptMediaType = "application/vnd.overgo.retrieval-receipt+json"
	retrievalReceiptSchema    = "overgo/retrieval-receipt/v1"
)

// RetrievalFilterBinding is the exact structural filter set used before
// candidate generation. Empty sets mean unrestricted within the projection.
type RetrievalFilterBinding struct {
	Datasets []artifact.ID                 `json:"datasets,omitempty"`
	Sources  []artifact.ID                 `json:"sources,omitempty"`
	Facets   []dataset.AgentRetrievalFacet `json:"facets,omitempty"`
	Parents  []artifact.ID                 `json:"parents,omitempty"`
}

// RetrievalCandidateStages binds each independently ranked candidate stage
// without copying candidate payloads into the run record.
type RetrievalCandidateStages struct {
	Dense   artifact.ID `json:"dense"`
	Lexical artifact.ID `json:"lexical"`
	Union   artifact.ID `json:"union"`
}

// RetrievalReceiptResult is the ordered, payload-free citation selected for
// consumption. Chunk identity remains the text authority.
type RetrievalReceiptResult struct {
	Rank             uint64                      `json:"rank"`
	Dataset          artifact.ID                 `json:"dataset"`
	Source           artifact.ID                 `json:"source"`
	Chunk            artifact.ID                 `json:"chunk"`
	Embedding        artifact.ID                 `json:"embedding"`
	StartRune        uint64                      `json:"start_rune"`
	EndRune          uint64                      `json:"end_rune"`
	StartByte        uint64                      `json:"start_byte,omitzero"`
	EndByte          uint64                      `json:"end_byte,omitzero"`
	TextBytes        uint64                      `json:"text_bytes"`
	Facet            dataset.AgentRetrievalFacet `json:"facet,omitempty"`
	Structure        []string                    `json:"structure,omitempty"`
	Parent           artifact.ID                 `json:"parent,omitzero"`
	Previous         artifact.ID                 `json:"previous,omitzero"`
	Next             artifact.ID                 `json:"next,omitzero"`
	Similarity       float64                     `json:"similarity"`
	Lexical          uint64                      `json:"lexical"`
	Rerank           float64                     `json:"rerank"`
	DenseCandidate   bool                        `json:"dense_candidate,omitzero"`
	LexicalCandidate bool                        `json:"lexical_candidate,omitzero"`
}

// retrievalReceiptRequest binds a search to its exact repository view and
// provider authorities. It contains no policy scalar absent from the search.
type retrievalReceiptRequest struct {
	Query          string
	Head           artifact.CommitID
	Dataset        artifact.ID
	Projection     artifact.ID
	Policy         artifact.ID
	EmbeddingModel artifact.ID
	RerankPolicy   artifact.ID
	Filters        RetrievalFilterBinding
	Limit          uint64
	CandidateLimit uint64
}

// RetrievalReceipt is immutable evidence for context that an interaction
// actually cites. It has no alias or mutable lifecycle.
type RetrievalReceipt struct {
	Version         uint16                     `json:"version"`
	Query           artifact.ID                `json:"query"`
	Head            artifact.CommitID          `json:"head"`
	Dataset         artifact.ID                `json:"dataset"`
	Projection      artifact.ID                `json:"projection"`
	Policy          artifact.ID                `json:"policy"`
	EmbeddingModel  artifact.ID                `json:"embedding_model"`
	RerankPolicy    artifact.ID                `json:"rerank_policy"`
	Filters         RetrievalFilterBinding     `json:"filters"`
	Limit           uint64                     `json:"limit"`
	CandidateLimit  uint64                     `json:"candidate_limit"`
	CandidateStages RetrievalCandidateStages   `json:"candidate_stages"`
	Work            dataset.AgentRetrievalWork `json:"work"`
	Results         []RetrievalReceiptResult   `json:"results,omitempty"`
	ContextBytes    uint64                     `json:"context_bytes,omitzero"`
	ID              artifact.ID                `json:"-"`
}

// RetrievalConsumer binds a consumed search to the exact active execution
// authorities that caused context to be selected.
type RetrievalConsumer struct {
	Recipe       artifact.ID
	Model        artifact.ID
	Operation    artifact.ID
	TaskContract artifact.ID
	Strategy     artifact.ID
}

// ConsumedRetrieval returns the exact search, receipt, and interaction trace
// produced by one head-bound search-and-publication transaction.
type ConsumedRetrieval struct {
	Search  dataset.AgentRetrievalSearchResult
	Receipt RetrievalReceipt
	Trace   InteractionTrace
}

var retrievalReceiptCodec = artifact.JSONDocumentCodec(
	"retrieval receipt", artifact.KindEvidence, retrievalReceiptMediaType, retrievalReceiptSchema,
	canonicalizeRetrievalReceipt,
	func(value RetrievalReceipt) artifact.ID { return value.ID },
	func(value *RetrievalReceipt, id artifact.ID) { value.ID = id },
	cloneRetrievalReceipt,
)

func newRetrievalReceipt(request retrievalReceiptRequest, search dataset.AgentRetrievalSearchResult) (RetrievalReceipt, error) {
	if request.Query == "" {
		return RetrievalReceipt{}, errors.New("run record: retrieval receipt query is empty")
	}
	query, err := artifact.JSONID(artifact.KindEvidence, request.Query)
	if err != nil {
		return RetrievalReceipt{}, err
	}
	results := make([]RetrievalReceiptResult, len(search.Results))
	var contextBytes uint64
	for index, result := range search.Results {
		textBytes := uint64(len(result.Citation.Text))
		if contextBytes > math.MaxUint64-textBytes {
			return RetrievalReceipt{}, errors.New("run record: retrieval context byte count overflows")
		}
		contextBytes += textBytes
		results[index] = RetrievalReceiptResult{
			Rank: uint64(index + 1), Dataset: result.Citation.Dataset, Source: result.Citation.Source,
			Chunk: result.Citation.Chunk, Embedding: result.Embedding,
			StartRune: result.Citation.StartRune, EndRune: result.Citation.EndRune,
			StartByte: result.Citation.StartByte, EndByte: result.Citation.EndByte, TextBytes: textBytes,
			Facet: result.Citation.Facet, Structure: slices.Clone(result.Citation.Structure),
			Parent: result.Citation.Parent, Previous: result.Citation.Previous, Next: result.Citation.Next,
			Similarity: result.Similarity, Lexical: result.Lexical, Rerank: result.Rerank,
			DenseCandidate: result.DenseCandidate, LexicalCandidate: result.LexicalCandidate,
		}
	}
	return retrievalReceiptCodec.New(RetrievalReceipt{
		Version: artifact.InitialDocumentVersion, Query: query, Head: request.Head,
		Dataset: request.Dataset, Projection: request.Projection, Policy: request.Policy,
		EmbeddingModel: request.EmbeddingModel, RerankPolicy: request.RerankPolicy,
		Filters: request.Filters, Limit: request.Limit, CandidateLimit: request.CandidateLimit,
		CandidateStages: RetrievalCandidateStages{
			Dense: search.DenseCandidateStage, Lexical: search.LexicalCandidateStage, Union: search.UnionCandidateStage,
		},
		Work: search.Work, Results: results, ContextBytes: contextBytes,
	})
}

// SearchAndPublishConsumedRetrieval is the production entry for grounded
// context. It executes the dataset-owned search itself, derives the receipt
// from that exact output, and atomically publishes the receipt with the trace
// that consumed it. Callers cannot inject scores, work, or candidate stages.
func SearchAndPublishConsumedRetrieval(
	ctx context.Context,
	repository artifact.Repository,
	query dataset.AgentRetrievalQuery,
	consumer RetrievalConsumer,
) (ConsumedRetrieval, error) {
	if ctx == nil || repository == nil {
		return ConsumedRetrieval{}, errors.New("run record: retrieval publication authority is absent")
	}
	messages := []InteractionMessage{
		{Role: "user", Content: query.Text},
		{Role: "assistant", Content: "grounded retrieval context selected"},
	}
	transcript, err := NewInteractionTranscript(messages)
	if err != nil {
		return ConsumedRetrieval{}, err
	}
	transcriptContent, err := interactionTranscriptCodec.Content(transcript)
	if err != nil {
		return ConsumedRetrieval{}, err
	}
	transcriptBatch, err := artifact.NewDocumentBatch(
		"retrieval/request/"+transcript.ID.String(), []artifact.Content{transcriptContent}, nil, nil,
	)
	if err != nil {
		return ConsumedRetrieval{}, err
	}
	if _, err = artifact.CommitBatch(ctx, repository, transcriptBatch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return ConsumedRetrieval{}, err
	}
	head, _ := repository.Head()
	projection, err := dataset.RequireAgentRetrievalProjection(ctx, repository, query.Projection)
	if err != nil {
		return ConsumedRetrieval{}, err
	}
	search, err := (dataset.AgentRetrievalBuilder{Repository: repository}).SearchWithWork(ctx, query)
	if err != nil {
		return ConsumedRetrieval{}, err
	}
	receipt, err := newRetrievalReceipt(retrievalReceiptRequest{
		Query: query.Text, Head: head, Dataset: projection.Dataset, Projection: projection.ID, Policy: projection.Policy,
		EmbeddingModel: projection.EmbeddingModel, RerankPolicy: projection.RerankPolicy,
		Filters: RetrievalFilterBinding{
			Datasets: slices.Clone(query.AllowedDatasets), Sources: slices.Clone(query.AllowedSources),
			Facets: slices.Clone(query.AllowedFacets), Parents: slices.Clone(query.AllowedParents),
		},
		Limit: query.Limit, CandidateLimit: projection.CandidateLimit,
	}, search)
	if err != nil {
		return ConsumedRetrieval{}, err
	}
	events := make([]InteractionTraceEvent, len(messages))
	for index, message := range messages {
		events[index] = InteractionTraceEvent{
			Sequence: traceSequenceStart + uint32(index), Kind: interactionEventKind(message), Message: message,
		}
	}
	trace, err := NewAgentTrajectory(InteractionTrace{
		Version: artifact.InitialDocumentVersion, Recipe: consumer.Recipe, Model: consumer.Model,
		Operation: consumer.Operation, Request: transcript.ID, Events: events, TaskContract: consumer.TaskContract,
		Strategy: consumer.Strategy, Context: []artifact.ID{receipt.ID}, Terminal: OutcomeSucceeded,
	})
	if err != nil {
		return ConsumedRetrieval{}, err
	}
	storedReceipt, storedTrace, err := publishConsumedRetrievalReceipt(ctx, repository, receipt, trace)
	if err != nil {
		return ConsumedRetrieval{}, err
	}
	return ConsumedRetrieval{Search: search, Receipt: storedReceipt, Trace: storedTrace}, nil
}

func publishConsumedRetrievalReceipt(
	ctx context.Context,
	repository artifact.Repository,
	receipt RetrievalReceipt,
	trace InteractionTrace,
) (RetrievalReceipt, InteractionTrace, error) {
	if ctx == nil || repository == nil || receipt.ValidateIdentity() != nil || trace.ValidateIdentity() != nil ||
		!slices.Contains(trace.Context, receipt.ID) {
		return RetrievalReceipt{}, InteractionTrace{}, errors.New("run record: invalid consumed retrieval publication")
	}
	if head, _ := repository.Head(); head != receipt.Head {
		return RetrievalReceipt{}, InteractionTrace{}, errors.New("run record: retrieval source head moved before admission")
	}
	if err := verifyRetrievalReceiptSources(ctx, repository, receipt); err != nil {
		return RetrievalReceipt{}, InteractionTrace{}, err
	}
	receiptContent, err := retrievalReceiptCodec.Content(receipt)
	if err != nil {
		return RetrievalReceipt{}, InteractionTrace{}, err
	}
	traceContent, err := trace.Content()
	if err != nil {
		return RetrievalReceipt{}, InteractionTrace{}, err
	}
	lineage := append(retrievalReceiptLineage(receipt), trace.Lineage()...)
	batch, err := artifact.NewDocumentBatch(
		"retrieval/consumed/"+receipt.ID.String(), []artifact.Content{receiptContent, traceContent}, lineage, nil,
	)
	if err != nil {
		return RetrievalReceipt{}, InteractionTrace{}, err
	}
	expected := receipt.Head
	batch.ExpectedHead = &expected
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return RetrievalReceipt{}, InteractionTrace{}, err
	}
	return receipt, trace, nil
}

// RequireConsumedRetrievalReceipt replays exact receipt identity, source
// cross-links, and a durable interaction child that cites the receipt.
func RequireConsumedRetrievalReceipt(ctx context.Context, reader artifact.Reader, id artifact.ID) (RetrievalReceipt, error) {
	receipt, err := retrievalReceiptCodec.RequireExactLineage(ctx, reader, id, retrievalReceiptLineage)
	if err != nil {
		return RetrievalReceipt{}, err
	}
	if err := verifyRetrievalReceiptSources(ctx, reader, receipt); err != nil {
		return RetrievalReceipt{}, err
	}
	children, err := reader.Children(ctx, receipt.ID)
	if err != nil {
		return RetrievalReceipt{}, err
	}
	for _, edge := range children {
		if edge.Parent != receipt.ID || edge.Relation != artifact.RelationDependsOn {
			continue
		}
		trace, traceErr := RequireInteractionTrace(ctx, reader, edge.Child)
		if traceErr == nil && slices.Contains(trace.Context, receipt.ID) {
			return receipt, nil
		}
	}
	return RetrievalReceipt{}, errors.New("run record: retrieval receipt has no durable consumer")
}

// ValidateIdentity proves no receipt field changed after admission.
func (value RetrievalReceipt) ValidateIdentity() error {
	return retrievalReceiptCodec.ValidateIdentity(value)
}

func verifyRetrievalReceiptSources(ctx context.Context, reader artifact.Reader, receipt RetrievalReceipt) error {
	projection, err := dataset.RequireAgentRetrievalProjection(ctx, reader, receipt.Projection)
	if err != nil || projection.Dataset != receipt.Dataset || projection.Policy != receipt.Policy ||
		projection.EmbeddingModel != receipt.EmbeddingModel || projection.RerankPolicy != receipt.RerankPolicy ||
		projection.CandidateLimit != receipt.CandidateLimit || receipt.Work.EntriesInspected != uint64(len(projection.Entries)) {
		return errors.Join(errors.New("run record: retrieval projection authority differs"), err)
	}
	entries := make(map[artifact.ID]artifact.ID, len(projection.Entries))
	for _, entry := range projection.Entries {
		entries[entry.Chunk] = entry.Embedding
	}
	for _, result := range receipt.Results {
		if entries[result.Chunk] != result.Embedding {
			return errors.New("run record: retrieval result is outside its projection")
		}
		chunk, chunkErr := dataset.RequireAgentRetrievalChunk(ctx, reader, result.Chunk)
		embedding, embeddingErr := dataset.RequireAgentRetrievalEmbedding(ctx, reader, result.Embedding)
		if chunkErr != nil || embeddingErr != nil || chunk.Dataset != receipt.Dataset || chunk.Policy != receipt.Policy ||
			embedding.Chunk != chunk.ID || embedding.Model != receipt.EmbeddingModel ||
			chunk.Source != result.Source || chunk.StartRune != result.StartRune || chunk.EndRune != result.EndRune ||
			chunk.StartByte != result.StartByte || chunk.EndByte != result.EndByte || uint64(len(chunk.Text)) != result.TextBytes ||
			chunk.Facet != result.Facet || !slices.Equal(chunk.Structure, result.Structure) || chunk.Parent != result.Parent ||
			chunk.Previous != result.Previous || chunk.Next != result.Next {
			return errors.Join(errors.New("run record: retrieval result citation differs"), chunkErr, embeddingErr)
		}
	}
	return nil
}

func retrievalReceiptLineage(value RetrievalReceipt) []artifact.Lineage {
	parents := []artifact.ID{value.Dataset, value.Projection, value.Policy, value.EmbeddingModel, value.RerankPolicy}
	parents = append(parents, value.Filters.Datasets...)
	parents = append(parents, value.Filters.Sources...)
	parents = append(parents, value.Filters.Parents...)
	for _, result := range value.Results {
		parents = append(parents, result.Source, result.Chunk, result.Embedding, result.Parent, result.Previous, result.Next)
	}
	parents = slices.DeleteFunc(parents, func(id artifact.ID) bool { return !id.Valid() })
	slices.SortFunc(parents, artifact.CompareID)
	parents = slices.Compact(parents)
	return artifact.DependencyLineage(value.ID, parents...)
}

func canonicalizeRetrievalReceipt(value *RetrievalReceipt) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Query.Kind() != artifact.KindEvidence ||
		!value.Head.Valid() || value.Dataset.Kind() != artifact.KindDataset || value.Projection.Kind() != artifact.KindProfile ||
		value.Policy.Kind() != artifact.KindProfile || value.EmbeddingModel.Kind() != artifact.KindModel ||
		value.RerankPolicy.Kind() != artifact.KindProfile || value.Limit == 0 || value.CandidateLimit == 0 ||
		value.CandidateStages.Dense.Kind() != artifact.KindEvidence ||
		value.CandidateStages.Lexical.Kind() != artifact.KindEvidence || value.CandidateStages.Union.Kind() != artifact.KindEvidence ||
		uint64(len(value.Results)) > value.Limit || value.Work.Returned != uint64(len(value.Results)) ||
		!validRetrievalReceiptWork(value.Work, value.CandidateLimit) {
		return errors.New("run record: invalid retrieval receipt")
	}
	if err := canonicalizeRetrievalFilters(&value.Filters); err != nil {
		return err
	}
	if len(value.Filters.Datasets) != 0 && !slices.Contains(value.Filters.Datasets, value.Dataset) {
		return errors.New("run record: retrieval dataset is outside receipt filters")
	}
	value.Results = slices.Clone(value.Results)
	var contextBytes uint64
	seen := make(map[artifact.ID]struct{}, len(value.Results))
	for index := range value.Results {
		result := &value.Results[index]
		if result.Rank != uint64(index+1) || result.Dataset != value.Dataset ||
			(result.Source.Kind() != artifact.KindFile && result.Source.Kind() != artifact.KindDatasetShard) ||
			result.Chunk.Kind() != artifact.KindDatasetShard || result.Embedding.Kind() != artifact.KindTensorSet ||
			result.EndRune <= result.StartRune || result.TextBytes == 0 || !finiteReceiptScore(result.Similarity) ||
			!finiteReceiptScore(result.Rerank) || (!result.DenseCandidate && !result.LexicalCandidate) ||
			!validReceiptFacet(result.Facet) || !validReceiptStructure(result.Structure) ||
			!validReceiptRelation(result.Parent) || !validReceiptRelation(result.Previous) || !validReceiptRelation(result.Next) {
			return errors.New("run record: invalid retrieval receipt result")
		}
		if len(value.Filters.Sources) != 0 && !slices.Contains(value.Filters.Sources, result.Source) ||
			len(value.Filters.Facets) != 0 && !slices.Contains(value.Filters.Facets, result.Facet) ||
			len(value.Filters.Parents) != 0 && !slices.Contains(value.Filters.Parents, result.Parent) {
			return errors.New("run record: retrieval result is outside receipt filters")
		}
		if _, duplicate := seen[result.Chunk]; duplicate {
			return errors.New("run record: duplicate retrieval receipt result")
		}
		seen[result.Chunk] = struct{}{}
		result.Structure = slices.Clone(result.Structure)
		if contextBytes > math.MaxUint64-result.TextBytes {
			return errors.New("run record: retrieval receipt byte count overflows")
		}
		contextBytes += result.TextBytes
	}
	if contextBytes != value.ContextBytes {
		return errors.New("run record: retrieval receipt context bytes differ")
	}
	return nil
}

func validRetrievalReceiptWork(work dataset.AgentRetrievalWork, candidateLimit uint64) bool {
	if work.QueryEmbeddings != 1 || work.FilteredEntries > work.EntriesInspected ||
		work.EmbeddingsLoaded > work.ChunksLoaded || work.DenseScored != work.EmbeddingsLoaded ||
		work.LexicalScored != work.EmbeddingsLoaded || work.DenseCandidates > work.DenseScored ||
		work.LexicalCandidates > work.LexicalScored || work.DenseCandidates > candidateLimit ||
		work.LexicalCandidates > candidateLimit || work.UnionCandidates > candidateLimit ||
		work.RerankCalls != work.UnionCandidates || work.Returned > work.UnionCandidates {
		return false
	}
	if (work.ChunksLoaded == 0) != (work.ChunkBytesLoaded == 0) ||
		(work.EmbeddingsLoaded == 0) != (work.EmbeddingBytesLoaded == 0) {
		return false
	}
	return true
}

func canonicalizeRetrievalFilters(value *RetrievalFilterBinding) error {
	value.Datasets = sortedReceiptIDs(value.Datasets)
	value.Sources = sortedReceiptIDs(value.Sources)
	value.Parents = sortedReceiptIDs(value.Parents)
	value.Facets = slices.Clone(value.Facets)
	slices.Sort(value.Facets)
	value.Facets = slices.Compact(value.Facets)
	for _, id := range value.Datasets {
		if id.Kind() != artifact.KindDataset {
			return errors.New("run record: invalid retrieval dataset filter")
		}
	}
	for _, id := range value.Sources {
		if id.Kind() != artifact.KindFile && id.Kind() != artifact.KindDatasetShard {
			return errors.New("run record: invalid retrieval source filter")
		}
	}
	for _, id := range value.Parents {
		if id.Kind() != artifact.KindDatasetShard {
			return errors.New("run record: invalid retrieval parent filter")
		}
	}
	for _, facet := range value.Facets {
		if facet == "" || !validReceiptFacet(facet) {
			return errors.New("run record: invalid retrieval facet filter")
		}
	}
	return nil
}

func sortedReceiptIDs(values []artifact.ID) []artifact.ID {
	values = slices.Clone(values)
	slices.SortFunc(values, artifact.CompareID)
	return slices.Compact(values)
}

func finiteReceiptScore(value float64) bool {
	return checked.Finite64(value)
}

func validReceiptFacet(value dataset.AgentRetrievalFacet) bool {
	switch value {
	case "", dataset.AgentRetrievalFacetText, dataset.AgentRetrievalFacetCode, dataset.AgentRetrievalFacetTable,
		dataset.AgentRetrievalFacetImage, dataset.AgentRetrievalFacetRecord:
		return true
	default:
		return false
	}
}

func validReceiptRelation(value artifact.ID) bool {
	return !value.Valid() || value.Kind() == artifact.KindDatasetShard
}

func validReceiptStructure(value []string) bool {
	for _, component := range value {
		if component == "" || component != strings.TrimSpace(component) || strings.ContainsAny(component, "\x00\r\n") {
			return false
		}
	}
	return true
}

func cloneRetrievalReceipt(value RetrievalReceipt) RetrievalReceipt {
	value.Filters.Datasets = slices.Clone(value.Filters.Datasets)
	value.Filters.Sources = slices.Clone(value.Filters.Sources)
	value.Filters.Facets = slices.Clone(value.Filters.Facets)
	value.Filters.Parents = slices.Clone(value.Filters.Parents)
	value.Results = slices.Clone(value.Results)
	for index := range value.Results {
		value.Results[index].Structure = slices.Clone(value.Results[index].Structure)
	}
	return value
}
