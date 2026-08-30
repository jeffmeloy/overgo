package runrecord

import (
	"context"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

type receiptEmbeddingProvider struct{ model artifact.ID }

func (provider receiptEmbeddingProvider) ModelIdentity() artifact.ID { return provider.model }

func (provider receiptEmbeddingProvider) Embed(_ context.Context, text string) ([]float32, error) {
	if strings.Contains(text, "target") {
		return []float32{1, 0}, nil
	}
	return []float32{0, 1}, nil
}

type receiptRerankProvider struct{ policy artifact.ID }

func (provider receiptRerankProvider) PolicyIdentity() artifact.ID { return provider.policy }

func (provider receiptRerankProvider) Score(_ context.Context, query, text string) (float64, error) {
	if strings.Contains(text, query) {
		return 2, nil
	}
	return 1, nil
}

func TestRetrievalReceiptReplay(t *testing.T) {
	directory := t.TempDir()
	store, err := overgodb.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	datasetID := testutil.ArtifactID(t, artifact.KindDataset, "receipt-dataset")
	modelID := testutil.ArtifactID(t, artifact.KindModel, "receipt-model")
	rerankID := testutil.ArtifactID(t, artifact.KindProfile, "receipt-rerank")
	parent := testutil.ArtifactID(t, artifact.KindDatasetShard, "receipt-parent")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "receipt-recipe")
	taskID := testutil.ArtifactID(t, artifact.KindRecipe, "receipt-task")
	operationID := testutil.ArtifactID(t, artifact.KindEvidence, "receipt-operation")
	strategyID := testutil.ArtifactID(t, artifact.KindProfile, "receipt-strategy")
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "retrieval/receipt/authorities",
		Artifacts: []artifact.Descriptor{
			{ID: datasetID}, {ID: modelID}, {ID: rerankID}, {ID: parent},
			{ID: recipeID}, {ID: taskID}, {ID: operationID}, {ID: strategyID},
		},
	}); err != nil {
		t.Fatal(err)
	}
	policy := dataset.AgentRetrievalPolicy{MaximumChunkRunes: 64, CandidateLimit: 2}
	embedder := receiptEmbeddingProvider{model: modelID}
	reranker := receiptRerankProvider{policy: rerankID}
	builder := dataset.AgentRetrievalBuilder{Repository: store}
	documents, err := dataset.PublishAgentRetrievalSource(
		t.Context(), store, artifact.KindFile, []dataset.AgentRetrievalDocument{
			{Text: "target evidence", Facet: dataset.AgentRetrievalFacetText,
				Structure: []string{"chapter", "section"}, Parent: parent},
			{Text: "background", Facet: dataset.AgentRetrievalFacetText,
				Structure: []string{"chapter", "appendix"}, Parent: parent, StartRune: 15, StartByte: 15},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	source := documents[0].Source
	projection, err := builder.Build(t.Context(), dataset.AgentRetrievalBuild{
		Dataset: datasetID, Policy: policy, Embedder: embedder, RerankPolicy: rerankID,
		Documents: documents,
	})
	if err != nil {
		t.Fatal(err)
	}
	query := dataset.AgentRetrievalQuery{
		Projection: projection.ID, Text: "target", Limit: 1,
		AllowedDatasets: []artifact.ID{datasetID}, AllowedSources: []artifact.ID{source},
		AllowedFacets:  []dataset.AgentRetrievalFacet{dataset.AgentRetrievalFacetText},
		AllowedParents: []artifact.ID{parent}, Embedder: embedder, Reranker: reranker,
	}
	consumed, err := SearchAndPublishConsumedRetrieval(t.Context(), store, query, RetrievalConsumer{
		Recipe: recipeID, Model: modelID, Operation: operationID, TaskContract: taskID, Strategy: strategyID,
	})
	if err != nil {
		t.Fatal(err)
	}
	search, stored, receipt := consumed.Search, consumed.Receipt, consumed.Receipt
	if len(search.Results) != 1 {
		t.Fatalf("search = %+v", search)
	}
	if stored.ID != receipt.ID || stored.CandidateStages.Dense != search.DenseCandidateStage ||
		stored.CandidateStages.Lexical != search.LexicalCandidateStage ||
		stored.CandidateStages.Union != search.UnionCandidateStage ||
		stored.Work.ChunkBytesLoaded == 0 || stored.Work.EmbeddingBytesLoaded == 0 || stored.ContextBytes == 0 {
		t.Fatalf("receipt omitted replay evidence: %+v", stored)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := overgodb.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	cold, err := RequireConsumedRetrievalReceipt(t.Context(), reopened, receipt.ID)
	if err != nil || cold.ID != receipt.ID || !slices.Equal(cold.Results[0].Structure, []string{"chapter", "section"}) {
		t.Fatalf("cold retrieval receipt = %+v, %v", cold, err)
	}
	forged := cold
	forged.Results = slices.Clone(cold.Results)
	forged.Results[0].EndRune++
	if err := forged.ValidateIdentity(); err == nil {
		t.Fatal("changed citation retained receipt identity")
	}
}
