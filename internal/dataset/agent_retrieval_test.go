package dataset

import (
	"context"
	"strings"
	"testing"
	"unicode/utf8"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

const (
	agentRetrievalChunkRunes = uint64(64)
	agentRetrievalCandidates = uint64(3)
)

type agentRetrievalEmbedder struct{ model artifact.ID }

func (embedder agentRetrievalEmbedder) ModelIdentity() artifact.ID { return embedder.model }
func (agentRetrievalEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	return []float32{float32(utf8.RuneCountInString(text)), float32(strings.Count(text, "target") + 1)}, nil
}

type agentRetrievalReranker struct {
	policy artifact.ID
	calls  *int
}

func (reranker agentRetrievalReranker) PolicyIdentity() artifact.ID { return reranker.policy }
func (reranker agentRetrievalReranker) Score(_ context.Context, query, text string) (float64, error) {
	if reranker.calls != nil {
		*reranker.calls++
	}
	if strings.Contains(text, query) {
		return float64(len([]rune(text))), nil
	}
	return 0, nil
}

type agentRetrievalFixture struct {
	store    *overgodb.Store
	builder  AgentRetrievalBuilder
	build    AgentRetrievalBuild
	sources  []artifact.ID
	reranker agentRetrievalReranker
}

func TestAgentRetrievalProjection(t *testing.T) {
	fixture := newAgentRetrievalFixture(t)
	defer fixture.store.Close()
	projection, err := fixture.builder.Build(context.Background(), fixture.build)
	if err != nil || len(projection.Entries) != len(fixture.sources) || projection.Dataset != fixture.build.Dataset ||
		projection.EmbeddingModel != fixture.build.Embedder.ModelIdentity() || projection.RerankPolicy != fixture.build.RerankPolicy {
		t.Fatalf("retrieval projection=(%+v, %v)", projection, err)
	}
	parents, err := fixture.store.Parents(context.Background(), projection.ID)
	if err != nil || len(parents) < len(projection.Entries)*2 {
		t.Fatalf("retrieval lineage=(%d, %v)", len(parents), err)
	}
}

func TestAgentRetrievalCitation(t *testing.T) {
	fixture := newAgentRetrievalFixture(t)
	defer fixture.store.Close()
	projection, err := fixture.builder.Build(context.Background(), fixture.build)
	if err != nil {
		t.Fatal(err)
	}
	results, err := fixture.builder.Search(context.Background(), AgentRetrievalQuery{
		Projection: projection.ID, Text: "target", Limit: 1,
		Embedder: fixture.build.Embedder, Reranker: fixture.reranker,
	})
	if err != nil || len(results) != 1 || results[0].Citation.Dataset != fixture.build.Dataset ||
		results[0].Citation.Source != fixture.sources[1] || results[0].Citation.Text != "target evidence" ||
		results[0].Citation.EndRune-results[0].Citation.StartRune != uint64(len([]rune(results[0].Citation.Text))) {
		t.Fatalf("retrieval citation=(%+v, %v)", results, err)
	}
}

func TestAgentRetrievalRebuild(t *testing.T) {
	fixture := newAgentRetrievalFixture(t)
	defer fixture.store.Close()
	first, err := fixture.builder.Build(context.Background(), fixture.build)
	if err != nil {
		t.Fatal(err)
	}
	head, _ := fixture.store.Head()
	second, err := fixture.builder.Build(context.Background(), fixture.build)
	secondHead, _ := fixture.store.Head()
	if err != nil || second.ID != first.ID || secondHead != head {
		t.Fatalf("retrieval rebuild=(%s, %s, %s, %s, %v)", first.ID, second.ID, head, secondHead, err)
	}
}

func TestAgentRetrievalIdentityMismatch(t *testing.T) {
	fixture := newAgentRetrievalFixture(t)
	defer fixture.store.Close()
	projection, err := fixture.builder.Build(context.Background(), fixture.build)
	if err != nil {
		t.Fatal(err)
	}
	query := AgentRetrievalQuery{
		Projection: projection.ID, Text: "target", Limit: 1,
		Embedder: agentRetrievalEmbedder{model: testutil.ArtifactID(t, artifact.KindModel, "wrong-embedding-model")},
		Reranker: fixture.reranker,
	}
	if _, err := fixture.builder.Search(context.Background(), query); err == nil || !strings.Contains(err.Error(), "identity differs") {
		t.Fatalf("mismatched embedding identity accepted: %v", err)
	}
	query.Embedder = fixture.build.Embedder
	query.Reranker = agentRetrievalReranker{policy: testutil.ArtifactID(t, artifact.KindProfile, "wrong-rerank-policy")}
	if _, err := fixture.builder.Search(context.Background(), query); err == nil || !strings.Contains(err.Error(), "identity differs") {
		t.Fatalf("mismatched rerank identity accepted: %v", err)
	}
}

func newAgentRetrievalFixture(t *testing.T) agentRetrievalFixture {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	datasetID := testutil.ArtifactID(t, artifact.KindDataset, "retrieval-dataset")
	modelID := testutil.ArtifactID(t, artifact.KindModel, "retrieval-embedding-model")
	rerankID := testutil.ArtifactID(t, artifact.KindProfile, "retrieval-rerank-policy")
	sources := []artifact.ID{
		testutil.ArtifactID(t, artifact.KindFile, "retrieval-source-a"),
		testutil.ArtifactID(t, artifact.KindFile, "retrieval-source-b"),
		testutil.ArtifactID(t, artifact.KindFile, "retrieval-source-c"),
	}
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: "retrieval/fixture", Artifacts: []artifact.Descriptor{
			{ID: datasetID}, {ID: modelID}, {ID: rerankID},
			{ID: sources[0]}, {ID: sources[1]}, {ID: sources[2]},
		},
	}); err != nil {
		store.Close()
		t.Fatal(err)
	}
	build := AgentRetrievalBuild{
		Dataset: datasetID,
		Documents: []AgentRetrievalDocument{
			{Source: sources[0], Text: "background context"},
			{Source: sources[1], Text: "target evidence"},
			{Source: sources[2], Text: "secondary context"},
		},
		Policy: AgentRetrievalPolicy{
			MaximumChunkRunes: agentRetrievalChunkRunes, CandidateLimit: agentRetrievalCandidates,
		},
		Embedder: agentRetrievalEmbedder{model: modelID}, RerankPolicy: rerankID,
	}
	return agentRetrievalFixture{
		store: store, builder: AgentRetrievalBuilder{Repository: store}, build: build, sources: sources,
		reranker: agentRetrievalReranker{policy: rerankID},
	}
}
