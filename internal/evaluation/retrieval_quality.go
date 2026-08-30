package evaluation

import (
	"cmp"
	"context"
	"errors"
	"math"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/runrecord"
)

const (
	retrievalCaseMediaType       = "application/vnd.overgo.retrieval-case+json"
	retrievalCaseSchema          = "overgo/retrieval-case/v1"
	retrievalEvaluationMediaType = "application/vnd.overgo.retrieval-evaluation+json"
	retrievalEvaluationSchema    = "overgo/retrieval-evaluation/v1"
)

// RetrievalJudgment identifies one exact relevant source span and its ordinal relevance.
type RetrievalJudgment struct {
	Chunk     artifact.ID `json:"chunk"`
	Source    artifact.ID `json:"source"`
	StartRune uint64      `json:"start_rune"`
	EndRune   uint64      `json:"end_rune"`
	Relevance uint8       `json:"relevance"`
}

// RetrievalCase is one group-safe held-out retrieval question.
type RetrievalCase struct {
	Version       uint16              `json:"version"`
	Query         artifact.ID         `json:"query"`
	Split         artifact.ID         `json:"split"`
	Group         string              `json:"group"`
	Judgments     []RetrievalJudgment `json:"judgments,omitempty"`
	ExpectAbstain bool                `json:"expect_abstain,omitzero"`
	ID            artifact.ID         `json:"-"`
}

// RetrievalRankedResult is the evaluator's provider-neutral citation view.
type RetrievalRankedResult struct {
	Chunk     artifact.ID `json:"chunk"`
	Source    artifact.ID `json:"source"`
	StartRune uint64      `json:"start_rune"`
	EndRune   uint64      `json:"end_rune"`
}

// RetrievalWork reports the exact work bound by the consumed retrieval receipt.
type RetrievalWork struct {
	InspectedFacts uint64 `json:"inspected_facts"`
	LoadedFacts    uint64 `json:"loaded_facts"`
	LoadedBytes    uint64 `json:"loaded_bytes"`
	ContextBytes   uint64 `json:"context_bytes"`
	ProviderCost   uint64 `json:"provider_cost"`
}

// RetrievalMetrics separates ranking, citation, abstention, and coverage.
type RetrievalMetrics struct {
	Recall            float64 `json:"recall"`
	ReciprocalRank    float64 `json:"reciprocal_rank"`
	NormalizedDCG     float64 `json:"normalized_dcg"`
	CitationPrecision float64 `json:"citation_precision"`
	CitationRecall    float64 `json:"citation_recall"`
	AbstentionCorrect bool    `json:"abstention_correct"`
	RelevantExpected  uint64  `json:"relevant_expected"`
	RelevantReturned  uint64  `json:"relevant_returned"`
	Returned          uint64  `json:"returned"`
}

// RetrievalEvaluation is immutable evidence over one case and consumed receipt.
type RetrievalEvaluation struct {
	Version uint16           `json:"version"`
	Case    artifact.ID      `json:"case"`
	Receipt artifact.ID      `json:"receipt"`
	Results artifact.ID      `json:"results"`
	Split   artifact.ID      `json:"split"`
	Metrics RetrievalMetrics `json:"metrics"`
	Work    RetrievalWork    `json:"work"`
	ID      artifact.ID      `json:"-"`
}

var retrievalCaseCodec = artifact.JSONDocumentCodec(
	"retrieval case", artifact.KindRecipe, retrievalCaseMediaType, retrievalCaseSchema,
	canonicalizeRetrievalCase,
	func(value RetrievalCase) artifact.ID { return value.ID },
	func(value *RetrievalCase, id artifact.ID) { value.ID = id },
	func(value RetrievalCase) RetrievalCase {
		value.Judgments = slices.Clone(value.Judgments)
		return value
	},
)

var retrievalEvaluationCodec = artifact.JSONDocumentCodec(
	"retrieval evaluation", artifact.KindEvaluation, retrievalEvaluationMediaType, retrievalEvaluationSchema,
	canonicalizeRetrievalEvaluation,
	func(value RetrievalEvaluation) artifact.ID { return value.ID },
	func(value *RetrievalEvaluation, id artifact.ID) { value.ID = id },
	func(value RetrievalEvaluation) RetrievalEvaluation { return value },
)

// NewRetrievalCase validates and identifies one ranking or abstention case.
func NewRetrievalCase(value RetrievalCase) (RetrievalCase, error) {
	return retrievalCaseCodec.NewPrepared(value, prepareRetrievalCase)
}

func prepareRetrievalCase(value *RetrievalCase) {
	value.Version, value.ID = artifact.InitialDocumentVersion, artifact.ID{}
}

func canonicalizeRetrievalCase(value *RetrievalCase) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Query.Kind() != artifact.KindEvidence || value.Split.Kind() != artifact.KindDatasetShard ||
		strings.TrimSpace(value.Group) == "" || value.Group != strings.TrimSpace(value.Group) ||
		(value.ExpectAbstain == (len(value.Judgments) != 0)) {
		return errors.New("evaluation: invalid retrieval case")
	}
	value.Judgments = slices.Clone(value.Judgments)
	slices.SortFunc(value.Judgments, func(left, right RetrievalJudgment) int { return artifact.CompareID(left.Chunk, right.Chunk) })
	for index, judgment := range value.Judgments {
		if judgment.Chunk.Kind() != artifact.KindDatasetShard ||
			(judgment.Source.Kind() != artifact.KindFile && judgment.Source.Kind() != artifact.KindDatasetShard) ||
			judgment.EndRune <= judgment.StartRune || judgment.Relevance == 0 ||
			index > 0 && value.Judgments[index-1].Chunk == judgment.Chunk {
			return errors.New("evaluation: invalid retrieval judgment")
		}
	}
	return nil
}

// EvaluateRetrievalCase loads the consumed receipt and derives citation-aware
// ranking and work metrics without accepting caller-authored results.
func EvaluateRetrievalCase(
	ctx context.Context,
	reader artifact.Reader,
	caseValue RetrievalCase,
	receipt artifact.ID,
) (RetrievalEvaluation, error) {
	validated, err := NewRetrievalCase(caseValue)
	consumed, receiptErr := runrecord.RequireConsumedRetrievalReceipt(ctx, reader, receipt)
	if err != nil || receiptErr != nil || caseValue.ID.Valid() && caseValue.ID != validated.ID || consumed.Query != validated.Query {
		return RetrievalEvaluation{}, errors.Join(errors.New("evaluation: invalid retrieval evidence"), err, receiptErr)
	}
	results := make([]RetrievalRankedResult, len(consumed.Results))
	for index, result := range consumed.Results {
		results[index] = RetrievalRankedResult{
			Chunk: result.Chunk, Source: result.Source, StartRune: result.StartRune, EndRune: result.EndRune,
		}
	}
	seen := make(map[artifact.ID]bool, len(results))
	for _, result := range results {
		if result.Chunk.Kind() != artifact.KindDatasetShard ||
			(result.Source.Kind() != artifact.KindFile && result.Source.Kind() != artifact.KindDatasetShard) ||
			result.EndRune <= result.StartRune || seen[result.Chunk] {
			return RetrievalEvaluation{}, errors.New("evaluation: invalid ranked retrieval result")
		}
		seen[result.Chunk] = true
	}
	resultsID, err := artifact.JSONID(artifact.KindEvidence, results)
	if err != nil {
		return RetrievalEvaluation{}, err
	}
	metrics := retrievalMetrics(validated, results)
	work := RetrievalWork{
		InspectedFacts: consumed.Work.EntriesInspected,
		LoadedFacts:    consumed.Work.ChunksLoaded + consumed.Work.EmbeddingsLoaded,
		LoadedBytes:    consumed.Work.ChunkBytesLoaded + consumed.Work.EmbeddingBytesLoaded,
		ContextBytes:   consumed.ContextBytes,
		ProviderCost:   consumed.Work.QueryEmbeddings + consumed.Work.RerankCalls,
	}
	return retrievalEvaluationCodec.New(RetrievalEvaluation{
		Version: artifact.InitialDocumentVersion, Case: validated.ID, Receipt: receipt, Split: validated.Split,
		Results: resultsID, Metrics: metrics, Work: work,
	})
}

// Content returns the canonical retrieval-case document.
func (value RetrievalCase) Content() (artifact.Content, error) {
	return retrievalCaseCodec.Content(value)
}

// Lineage binds a held-out case to its sealed split and exact judged spans.
func (value RetrievalCase) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.Split}
	for _, judgment := range value.Judgments {
		parents = append(parents, judgment.Chunk, judgment.Source)
	}
	slices.SortFunc(parents, artifact.CompareID)
	parents = slices.Compact(parents)
	return artifact.DependencyLineage(value.ID, parents...)
}

// Content returns the canonical evaluation evidence document.
func (value RetrievalEvaluation) Content() (artifact.Content, error) {
	return retrievalEvaluationCodec.Content(value)
}

// RequireRetrievalEvaluation loads exact citation-aware evaluation evidence.
func RequireRetrievalEvaluation(ctx context.Context, reader artifact.Reader, id artifact.ID) (RetrievalEvaluation, error) {
	value, err := retrievalEvaluationCodec.RequireExactLineage(ctx, reader, id, RetrievalEvaluation.Lineage)
	if err != nil {
		return RetrievalEvaluation{}, err
	}
	caseValue, err := retrievalCaseCodec.RequireExactLineage(ctx, reader, value.Case, RetrievalCase.Lineage)
	if err != nil {
		return RetrievalEvaluation{}, err
	}
	replayed, err := EvaluateRetrievalCase(ctx, reader, caseValue, value.Receipt)
	if err != nil || replayed.ID != value.ID {
		return RetrievalEvaluation{}, errors.Join(errors.New("evaluation: retrieval evaluation differs from consumed evidence"), err)
	}
	return value, nil
}

// ValidateIdentity proves aggregate evidence and ordered-result identity did not change.
func (value RetrievalEvaluation) ValidateIdentity() error {
	return retrievalEvaluationCodec.ValidateIdentity(value)
}

// Lineage binds the evaluation to the case and consumed receipt. The Results
// field is a value digest, not a synthetic catalog parent.
func (value RetrievalEvaluation) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.Case, value.Receipt, value.Split)
}

// PublishRetrievalEvaluation stores the sealed case and evidence atomically,
// without an alias or mutable evaluation lifecycle.
func PublishRetrievalEvaluation(
	ctx context.Context,
	repository artifact.Repository,
	caseValue RetrievalCase,
	receipt artifact.ID,
) (RetrievalEvaluation, error) {
	if ctx == nil || repository == nil {
		return RetrievalEvaluation{}, errors.New("evaluation: retrieval evaluation repository is absent")
	}
	head, _ := repository.Head()
	validated, err := NewRetrievalCase(caseValue)
	if err != nil || caseValue.ID.Valid() && caseValue.ID != validated.ID {
		return RetrievalEvaluation{}, errors.Join(errors.New("evaluation: invalid retrieval case publication"), err)
	}
	value, err := EvaluateRetrievalCase(ctx, repository, validated, receipt)
	if err != nil {
		return RetrievalEvaluation{}, err
	}
	if _, found, readErr := retrievalEvaluationCodec.Read(ctx, repository, value.ID); readErr != nil {
		return RetrievalEvaluation{}, readErr
	} else if found {
		return RequireRetrievalEvaluation(ctx, repository, value.ID)
	}
	caseContent, err := validated.Content()
	if err != nil {
		return RetrievalEvaluation{}, err
	}
	evaluationContent, err := value.Content()
	if err != nil {
		return RetrievalEvaluation{}, err
	}
	lineage := append(validated.Lineage(), value.Lineage()...)
	batch, err := artifact.NewDocumentBatch(
		"evaluation/retrieval/"+value.ID.String(), []artifact.Content{caseContent, evaluationContent}, lineage, nil,
	)
	if err != nil {
		return RetrievalEvaluation{}, err
	}
	batch.ExpectedHead = &head
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return RetrievalEvaluation{}, err
	}
	return value, nil
}

func canonicalizeRetrievalEvaluation(value *RetrievalEvaluation) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Case.Kind() != artifact.KindRecipe ||
		value.Receipt.Kind() != artifact.KindEvidence || value.Results.Kind() != artifact.KindEvidence ||
		value.Split.Kind() != artifact.KindDatasetShard || !finiteRetrievalMetric(value.Metrics.Recall) ||
		!finiteRetrievalMetric(value.Metrics.ReciprocalRank) || !finiteRetrievalMetric(value.Metrics.NormalizedDCG) ||
		!finiteRetrievalMetric(value.Metrics.CitationPrecision) || !finiteRetrievalMetric(value.Metrics.CitationRecall) ||
		value.Metrics.RelevantReturned > value.Metrics.RelevantExpected || value.Metrics.RelevantReturned > value.Metrics.Returned {
		return errors.New("evaluation: invalid retrieval evaluation")
	}
	return nil
}

func finiteRetrievalMetric(value float64) bool {
	return checked.Finite64(value) && value >= 0 && value <= 1
}

func retrievalMetrics(caseValue RetrievalCase, results []RetrievalRankedResult) RetrievalMetrics {
	byChunk := make(map[artifact.ID]RetrievalJudgment, len(caseValue.Judgments))
	for _, judgment := range caseValue.Judgments {
		byChunk[judgment.Chunk] = judgment
	}
	var relevantReturned uint64
	var reciprocalRank, dcg float64
	for index, result := range results {
		judgment, relevant := byChunk[result.Chunk]
		if !relevant || judgment.Source != result.Source || judgment.StartRune != result.StartRune || judgment.EndRune != result.EndRune {
			continue
		}
		relevantReturned++
		reciprocalRank = cmp.Or(reciprocalRank, 1/float64(index+1))
		dcg += (math.Pow(2, float64(judgment.Relevance)) - 1) / math.Log2(float64(index)+2)
	}
	ideal := slices.Clone(caseValue.Judgments)
	slices.SortFunc(ideal, func(left, right RetrievalJudgment) int {
		return cmp.Compare(right.Relevance, left.Relevance)
	})
	var idealDCG float64
	for index, judgment := range ideal[:min(len(results), len(ideal))] {
		idealDCG += (math.Pow(2, float64(judgment.Relevance)) - 1) / math.Log2(float64(index)+2)
	}
	metrics := RetrievalMetrics{
		ReciprocalRank: reciprocalRank, RelevantExpected: uint64(len(caseValue.Judgments)),
		RelevantReturned: relevantReturned, Returned: uint64(len(results)),
		AbstentionCorrect: caseValue.ExpectAbstain == (len(results) == 0),
	}
	if len(caseValue.Judgments) != 0 {
		metrics.Recall = float64(relevantReturned) / float64(len(caseValue.Judgments))
		metrics.CitationRecall = metrics.Recall
		if idealDCG != 0 {
			metrics.NormalizedDCG = dcg / idealDCG
		}
	}
	if len(results) != 0 {
		metrics.CitationPrecision = float64(relevantReturned) / float64(len(results))
	}
	return metrics
}
