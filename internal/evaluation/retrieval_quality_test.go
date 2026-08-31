package evaluation

import (
	"context"
	"math"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

type evaluationEmbedder struct{ model artifact.ID }

func (provider evaluationEmbedder) ModelIdentity() artifact.ID { return provider.model }
func (provider evaluationEmbedder) Embed(context.Context, string) ([]float32, error) {
	return []float32{1, 0}, nil
}

type evaluationReranker struct {
	policy artifact.ID
	scores map[string]float64
}

func (provider evaluationReranker) PolicyIdentity() artifact.ID { return provider.policy }
func (provider evaluationReranker) Score(_ context.Context, _ string, text string) (float64, error) {
	return provider.scores[text], nil
}

type consumedRetrievalFixture struct {
	receipt runrecord.RetrievalReceipt
	search  dataset.AgentRetrievalSearchResult
}

func TestRetrievalEvaluationMetricsAndCoverage(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	fixture := publishConsumedRetrieval(t, store, "metrics", []string{"target alpha", "target beta"}, 2,
		map[string]float64{"target alpha": 1, "target beta": 2}, false)
	if len(fixture.search.Results) != 2 {
		t.Fatalf("search results = %+v", fixture.search)
	}
	relevantResult := fixture.search.Results[1]
	missing := RetrievalJudgment{
		Chunk:  testutil.ArtifactID(t, artifact.KindDatasetShard, "metrics-missing-chunk"),
		Source: testutil.ArtifactID(t, artifact.KindFile, "metrics-missing-source"), StartRune: 0, EndRune: 4, Relevance: 1,
	}
	split := testutil.ArtifactID(t, artifact.KindDatasetShard, "metrics-held-out")
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key:       "retrieval/evaluation/authorities",
		Artifacts: []artifact.Descriptor{{ID: split}, {ID: missing.Chunk}, {ID: missing.Source}},
	}); err != nil {
		t.Fatal(err)
	}
	caseValue, err := NewRetrievalCase(RetrievalCase{
		Query: fixture.receipt.Query, Split: split, Group: "question-family-a",
		Judgments: []RetrievalJudgment{missing, {
			Chunk: relevantResult.Citation.Chunk, Source: relevantResult.Citation.Source,
			StartRune: relevantResult.Citation.StartRune, EndRune: relevantResult.Citation.EndRune, Relevance: 3,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	evidence, err := PublishRetrievalEvaluation(t.Context(), store, caseValue, fixture.receipt.ID)
	if err != nil {
		t.Fatal(err)
	}
	if evidence.Metrics.RelevantExpected != 2 || evidence.Metrics.RelevantReturned != 1 ||
		evidence.Metrics.Returned != 2 || math.Abs(evidence.Metrics.Recall-0.5) > 1e-12 ||
		math.Abs(evidence.Metrics.ReciprocalRank-0.5) > 1e-12 ||
		math.Abs(evidence.Metrics.CitationPrecision-0.5) > 1e-12 || !evidence.Metrics.AbstentionCorrect ||
		evidence.Metrics.NormalizedDCG <= 0 || evidence.Metrics.NormalizedDCG >= 1 ||
		evidence.Work.LoadedBytes == 0 || evidence.Results.Kind() != artifact.KindEvidence {
		t.Fatalf("metrics = %+v, work = %+v", evidence.Metrics, evidence.Work)
	}
	replayed, err := RequireRetrievalEvaluation(t.Context(), store, evidence.ID)
	if err != nil || replayed.ID != evidence.ID {
		t.Fatalf("replayed evaluation = %+v, %v", replayed, err)
	}

	abstentionFixture := publishConsumedRetrieval(t, store, "abstention", []string{"background"}, 1,
		map[string]float64{"background": 1}, true)
	abstention, err := NewRetrievalCase(RetrievalCase{
		Query: abstentionFixture.receipt.Query, Split: split, Group: "question-family-b", ExpectAbstain: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	abstained, err := PublishRetrievalEvaluation(t.Context(), store, abstention, abstentionFixture.receipt.ID)
	if err != nil || !abstained.Metrics.AbstentionCorrect || abstained.Metrics.RelevantExpected != 0 ||
		abstained.Metrics.NormalizedDCG != 0 {
		t.Fatalf("abstention evaluation = %+v, %v", abstained, err)
	}
}

func publishConsumedRetrieval(
	t *testing.T,
	store *overgodb.Store,
	tag string,
	texts []string,
	limit uint64,
	scores map[string]float64,
	filterAll bool,
) consumedRetrievalFixture {
	t.Helper()
	datasetID := testutil.ArtifactID(t, artifact.KindDataset, tag+"-dataset")
	modelID := testutil.ArtifactID(t, artifact.KindModel, tag+"-model")
	rerankID := testutil.ArtifactID(t, artifact.KindProfile, tag+"-rerank")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, tag+"-recipe")
	taskID := testutil.ArtifactID(t, artifact.KindRecipe, tag+"-task")
	operationID := testutil.ArtifactID(t, artifact.KindEvidence, tag+"-operation")
	strategyID := testutil.ArtifactID(t, artifact.KindProfile, tag+"-strategy")
	descriptors := []artifact.Descriptor{
		{ID: datasetID}, {ID: modelID}, {ID: rerankID}, {ID: recipeID}, {ID: taskID},
		{ID: operationID}, {ID: strategyID},
	}
	documents := make([]dataset.AgentRetrievalDocument, len(texts))
	var start uint64
	for index, sourceText := range texts {
		documents[index] = dataset.AgentRetrievalDocument{
			Text: sourceText, Facet: dataset.AgentRetrievalFacetText,
			Structure: []string{"fixture", tag}, StartRune: start, StartByte: start,
		}
		start += uint64(len([]rune(sourceText))) + 1
	}
	absentSource := testutil.ArtifactID(t, artifact.KindFile, tag+"-absent-source")
	if filterAll {
		descriptors = append(descriptors, artifact.Descriptor{ID: absentSource})
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "retrieval/fixture/" + tag, Artifacts: descriptors}); err != nil {
		t.Fatal(err)
	}
	documents, err := dataset.PublishAgentRetrievalSource(t.Context(), store, artifact.KindFile, documents)
	if err != nil {
		t.Fatal(err)
	}
	embedder := evaluationEmbedder{model: modelID}
	reranker := evaluationReranker{policy: rerankID, scores: scores}
	policy := dataset.AgentRetrievalPolicy{MaximumChunkRunes: 128, CandidateLimit: uint64(max(1, len(texts)))}
	builder := dataset.AgentRetrievalBuilder{Repository: store}
	projection, err := builder.Build(t.Context(), dataset.AgentRetrievalBuild{
		Dataset: datasetID, Documents: documents, Policy: policy, Embedder: embedder, RerankPolicy: rerankID,
	})
	if err != nil {
		t.Fatal(err)
	}
	query := dataset.AgentRetrievalQuery{
		Projection: projection.ID, Text: "target", Limit: limit, AllowedDatasets: []artifact.ID{datasetID},
		Embedder: embedder, Reranker: reranker,
	}
	if filterAll {
		query.AllowedSources = []artifact.ID{absentSource}
	}
	consumed, err := runrecord.SearchAndPublishConsumedRetrieval(t.Context(), store, query, runrecord.RetrievalConsumer{
		Recipe: recipeID, Model: modelID, Operation: operationID, TaskContract: taskID, Strategy: strategyID,
	})
	if err != nil {
		t.Fatal(err)
	}
	return consumedRetrievalFixture{receipt: consumed.Receipt, search: consumed.Search}
}
