package dataset

import (
	"context"
	"errors"
	"reflect"
	"slices"
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
	runes := float32(utf8.RuneCountInString(text))
	targets := float32(strings.Count(text, "target") + 1)
	return []float32{runes, targets}, nil
}

type agentRetrievalScriptedEmbedder struct {
	model   artifact.ID
	vectors map[string][]float32
}

func (embedder agentRetrievalScriptedEmbedder) ModelIdentity() artifact.ID { return embedder.model }
func (embedder agentRetrievalScriptedEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	vector, found := embedder.vectors[text]
	if !found {
		return nil, errors.New("missing scripted retrieval embedding")
	}
	return append([]float32(nil), vector...), nil
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

func (fixture *agentRetrievalFixture) Build(t *testing.T) (AgentRetrievalProjection, error) {
	t.Helper()
	documents := slices.Clone(fixture.build.Documents)
	sources := make([]artifact.ID, len(documents))
	for index := range documents {
		documents[index].Structure = slices.Clone(documents[index].Structure)
		documents[index].Source = artifact.ID{}
		published, err := PublishAgentRetrievalSource(
			t.Context(), fixture.store, artifact.KindFile, []AgentRetrievalDocument{documents[index]},
		)
		if err != nil {
			return AgentRetrievalProjection{}, err
		}
		documents[index], sources[index] = published[0], published[0].Source
	}
	fixture.build.Documents, fixture.sources = documents, sources
	return fixture.builder.Build(t.Context(), fixture.build)
}

func TestAgentRetrievalProjection(t *testing.T) {
	fixture := newAgentRetrievalFixture(t)
	defer fixture.store.Close()
	projection, err := fixture.Build(t)
	if err != nil || len(projection.Entries) != len(fixture.sources) || projection.Dataset != fixture.build.Dataset ||
		projection.EmbeddingModel != fixture.build.Embedder.ModelIdentity() || projection.RerankPolicy != fixture.build.RerankPolicy {
		t.Fatalf("retrieval projection=(%+v, %v)", projection, err)
	}
	parents, err := fixture.store.Parents(t.Context(), projection.ID)
	if err != nil || len(parents) < len(projection.Entries)*2 {
		t.Fatalf("retrieval lineage=(%d, %v)", len(parents), err)
	}
	loadedProjection, err := RequireAgentRetrievalProjection(t.Context(), fixture.store, projection.ID)
	loadedChunk, chunkErr := RequireAgentRetrievalChunk(t.Context(), fixture.store, projection.Entries[0].Chunk)
	loadedEmbedding, embeddingErr := RequireAgentRetrievalEmbedding(t.Context(), fixture.store, projection.Entries[0].Embedding)
	if err != nil || chunkErr != nil || embeddingErr != nil || loadedProjection.ID != projection.ID ||
		loadedChunk.ID != projection.Entries[0].Chunk || loadedEmbedding.ID != projection.Entries[0].Embedding ||
		loadedEmbedding.Chunk != loadedChunk.ID {
		t.Fatalf("typed retrieval loads=(%s, %s, %s, %v, %v, %v)",
			loadedProjection.ID, loadedChunk.ID, loadedEmbedding.ID, err, chunkErr, embeddingErr)
	}
}

func TestAgentRetrievalCitation(t *testing.T) {
	fixture := newAgentRetrievalFixture(t)
	defer fixture.store.Close()
	projection, err := fixture.Build(t)
	if err != nil {
		t.Fatal(err)
	}
	results, err := fixture.builder.Search(t.Context(), AgentRetrievalQuery{
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
	first, err := fixture.Build(t)
	if err != nil {
		t.Fatal(err)
	}
	head, _ := fixture.store.Head()
	second, err := fixture.Build(t)
	secondHead, _ := fixture.store.Head()
	if err != nil || second.ID != first.ID || secondHead != head {
		t.Fatalf("retrieval rebuild=(%s, %s, %s, %s, %v)", first.ID, second.ID, head, secondHead, err)
	}
}

func TestAgentRetrievalIdentityMismatch(t *testing.T) {
	fixture := newAgentRetrievalFixture(t)
	defer fixture.store.Close()
	projection, err := fixture.Build(t)
	if err != nil {
		t.Fatal(err)
	}
	query := AgentRetrievalQuery{
		Projection: projection.ID, Text: "target", Limit: 1,
		Embedder: agentRetrievalEmbedder{model: testutil.ArtifactID(t, artifact.KindModel, "wrong-embedding-model")},
		Reranker: fixture.reranker,
	}
	if _, err := fixture.builder.Search(t.Context(), query); err == nil || !strings.Contains(err.Error(), "identity differs") {
		t.Fatalf("mismatched embedding identity accepted: %v", err)
	}
	query.Embedder = fixture.build.Embedder
	query.Reranker = agentRetrievalReranker{policy: testutil.ArtifactID(t, artifact.KindProfile, "wrong-rerank-policy")}
	if _, err := fixture.builder.Search(t.Context(), query); err == nil || !strings.Contains(err.Error(), "identity differs") {
		t.Fatalf("mismatched rerank identity accepted: %v", err)
	}
}

func TestAgentRetrievalBuildRejectsFabricatedSourceContentOrStructure(t *testing.T) {
	fixture := newAgentRetrievalFixture(t)
	defer fixture.store.Close()
	stored, err := PublishAgentRetrievalSource(t.Context(), fixture.store, artifact.KindFile, []AgentRetrievalDocument{{
		Text: "grounded source text", Facet: AgentRetrievalFacetTable,
		Structure: []string{"chapter", "table"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name   string
		mutate func(*AgentRetrievalDocument)
	}{
		{name: "content", mutate: func(value *AgentRetrievalDocument) { value.Text = "fabricated text" }},
		{name: "structure", mutate: func(value *AgentRetrievalDocument) { value.Structure = []string{"other", "path"} }},
	} {
		t.Run(test.name, func(t *testing.T) {
			document := stored[0]
			document.Structure = slices.Clone(document.Structure)
			test.mutate(&document)
			request := fixture.build
			request.Documents = []AgentRetrievalDocument{document}
			if _, err := fixture.builder.Build(t.Context(), request); err == nil ||
				!strings.Contains(err.Error(), "differs from stored source content or structure") {
				t.Fatalf("fabricated %s was admitted: %v", test.name, err)
			}
		})
	}
}

func TestAgentRetrievalColdReadsRejectForgedLineage(t *testing.T) {
	for _, target := range []string{"source", "chunk", "embedding", "projection"} {
		t.Run(target, func(t *testing.T) {
			fixture := newAgentRetrievalFixture(t)
			defer fixture.store.Close()
			projection, err := fixture.Build(t)
			if err != nil {
				t.Fatal(err)
			}
			entry := projection.Entries[0]
			chunk, err := RequireAgentRetrievalChunk(t.Context(), fixture.store, entry.Chunk)
			if err != nil {
				t.Fatal(err)
			}
			id := map[string]artifact.ID{
				"source": chunk.Source, "chunk": entry.Chunk,
				"embedding": entry.Embedding, "projection": projection.ID,
			}[target]
			extra := testutil.ArtifactID(t, artifact.KindEvidence, "forged-retrieval-lineage-"+target)
			if _, err := fixture.store.Commit(t.Context(), artifact.Batch{
				Key:       "retrieval/forged-lineage/" + target,
				Artifacts: []artifact.Descriptor{{ID: extra}},
				Lineage:   artifact.DependencyLineage(id, extra),
			}); err != nil {
				t.Fatal(err)
			}
			var requireErr error
			switch target {
			case "source", "chunk":
				_, requireErr = RequireAgentRetrievalChunk(t.Context(), fixture.store, entry.Chunk)
			case "embedding":
				_, requireErr = RequireAgentRetrievalEmbedding(t.Context(), fixture.store, entry.Embedding)
			case "projection":
				_, requireErr = RequireAgentRetrievalProjection(t.Context(), fixture.store, projection.ID)
			}
			if requireErr == nil {
				t.Fatalf("%s with added lineage was admitted", target)
			}
			if _, err := fixture.builder.SearchWithWork(t.Context(), AgentRetrievalQuery{
				Projection: projection.ID, Text: "target", Limit: 1,
				Embedder: fixture.build.Embedder, Reranker: fixture.reranker,
			}); err == nil {
				t.Fatalf("search admitted forged %s lineage", target)
			}
		})
	}
}

func TestGroundedHybridRetrievalPreservesStructureAndCitations(t *testing.T) {
	fixture := newAgentRetrievalFixture(t)
	defer fixture.store.Close()
	parent := testutil.ArtifactID(t, artifact.KindDatasetShard, "retrieval-parent")
	previous := testutil.ArtifactID(t, artifact.KindDatasetShard, "retrieval-previous")
	next := testutil.ArtifactID(t, artifact.KindDatasetShard, "retrieval-next")
	if _, err := fixture.store.Commit(t.Context(), artifact.Batch{
		Key: "retrieval/structure", Artifacts: []artifact.Descriptor{{ID: parent}, {ID: previous}, {ID: next}},
	}); err != nil {
		t.Fatal(err)
	}
	text := "café target evidence"
	structure := []string{"chapter-1", "table-2", "row-3"}
	fixture.build.Documents = []AgentRetrievalDocument{{
		Source: fixture.sources[1], Text: text, Facet: AgentRetrievalFacetTable, Structure: structure,
		Parent: parent, Previous: previous, Next: next, StartRune: 17, StartByte: 31,
	}}
	projection, err := fixture.Build(t)
	if err != nil {
		t.Fatal(err)
	}
	search, err := fixture.builder.SearchWithWork(t.Context(), AgentRetrievalQuery{
		Projection: projection.ID, Text: "target", Limit: 1,
		Embedder: fixture.build.Embedder, Reranker: fixture.reranker,
	})
	if err != nil || len(search.Results) != 1 {
		t.Fatalf("grounded retrieval=(%+v, %v)", search, err)
	}
	result := search.Results[0]
	chunkBytes := agentRetrievalStoredBytes(t, fixture.store, projection.Entries[0].Chunk)
	embeddingBytes := agentRetrievalStoredBytes(t, fixture.store, projection.Entries[0].Embedding)
	wantWork := AgentRetrievalWork{
		QueryEmbeddings: 1, EntriesInspected: 1, ChunksLoaded: 1, ChunkBytesLoaded: chunkBytes,
		EmbeddingsLoaded: 1, EmbeddingBytesLoaded: embeddingBytes, DenseScored: 1, LexicalScored: 1,
		DenseCandidates: 1, LexicalCandidates: 1, UnionCandidates: 1, RerankCalls: 1, Returned: 1,
	}
	if result.Citation.Dataset != fixture.build.Dataset || result.Citation.Source != fixture.sources[0] ||
		result.Citation.Text != text || result.Citation.StartRune != 17 ||
		result.Citation.EndRune != 17+uint64(utf8.RuneCountInString(text)) || result.Citation.StartByte != 31 ||
		result.Citation.EndByte != 31+uint64(len(text)) || result.Citation.Facet != AgentRetrievalFacetTable ||
		!reflect.DeepEqual(result.Citation.Structure, structure) || result.Citation.Parent != parent ||
		result.Citation.Previous != previous || result.Citation.Next != next || result.Lexical != 1 ||
		!result.DenseCandidate || !result.LexicalCandidate || !reflect.DeepEqual(search.Work, wantWork) ||
		search.DenseCandidateStage.Kind() != artifact.KindEvidence || search.LexicalCandidateStage.Kind() != artifact.KindEvidence ||
		search.UnionCandidateStage.Kind() != artifact.KindEvidence {
		t.Fatalf("grounded structure/result=(%+v, work=%+v)", result, search.Work)
	}
}

func TestGroundedHybridRetrievalFiltersBeforeCandidateLimit(t *testing.T) {
	fixture := newAgentRetrievalFixture(t)
	defer fixture.store.Close()
	embedder := agentRetrievalScriptedEmbedder{
		model: fixture.build.Embedder.ModelIdentity(),
		vectors: map[string][]float32{
			"query":            {1, 0},
			"dense winner":     {1, 0},
			"allowed evidence": {0, 1},
		},
	}
	fixture.build.Embedder = embedder
	fixture.build.Policy.CandidateLimit = 1
	fixture.build.Documents = []AgentRetrievalDocument{
		{Source: fixture.sources[0], Text: "dense winner", Facet: AgentRetrievalFacetCode},
		{Source: fixture.sources[1], Text: "allowed evidence", Facet: AgentRetrievalFacetTable},
	}
	projection, err := fixture.Build(t)
	if err != nil {
		t.Fatal(err)
	}
	search, err := fixture.builder.SearchWithWork(t.Context(), AgentRetrievalQuery{
		Projection: projection.ID, Text: "query", Limit: 1, AllowedSources: []artifact.ID{fixture.sources[1]},
		AllowedFacets: []AgentRetrievalFacet{AgentRetrievalFacetTable}, Embedder: embedder, Reranker: fixture.reranker,
	})
	if err != nil || len(search.Results) != 1 || search.Results[0].Citation.Source != fixture.sources[1] ||
		search.Work.EntriesInspected != 2 || search.Work.FilteredEntries != 1 || search.Work.ChunksLoaded != 1 ||
		search.Work.EmbeddingsLoaded != 1 || search.Work.DenseScored != 1 || search.Work.DenseCandidates != 1 ||
		search.Work.UnionCandidates != 1 || search.Work.RerankCalls != 1 || search.Work.Returned != 1 {
		t.Fatalf("filter pushdown=(%+v, %v)", search, err)
	}
}

func TestGroundedHybridRetrievalDeterministic(t *testing.T) {
	fixture := newAgentRetrievalFixture(t)
	defer fixture.store.Close()
	embedder := agentRetrievalScriptedEmbedder{
		model: fixture.build.Embedder.ModelIdentity(),
		vectors: map[string][]float32{
			"needle":          {1, 0},
			"dense-only":      {1, 0},
			"dense-second":    {0.8, 0.6},
			"needle evidence": {0, 1},
		},
	}
	fixture.build.Embedder = embedder
	fixture.build.Policy.CandidateLimit = 2
	fixture.build.Documents = []AgentRetrievalDocument{
		{Source: fixture.sources[0], Text: "dense-only", Facet: AgentRetrievalFacetText},
		{Source: fixture.sources[1], Text: "dense-second", Facet: AgentRetrievalFacetText},
		{Source: fixture.sources[2], Text: "needle evidence", Facet: AgentRetrievalFacetText},
	}
	projection, err := fixture.Build(t)
	if err != nil {
		t.Fatal(err)
	}
	query := AgentRetrievalQuery{
		Projection: projection.ID, Text: "needle", Limit: 2, Embedder: embedder, Reranker: fixture.reranker,
	}
	first, err := fixture.builder.SearchWithWork(t.Context(), query)
	if err != nil {
		t.Fatal(err)
	}
	second, err := fixture.builder.SearchWithWork(t.Context(), query)
	if err != nil {
		t.Fatal(err)
	}
	permutedDocuments := slices.Clone(fixture.build.Documents)
	slices.Reverse(permutedDocuments)
	fixture.build.Documents = permutedDocuments
	permutedProjection, err := fixture.Build(t)
	if err != nil {
		t.Fatal(err)
	}
	permuted, err := fixture.builder.SearchWithWork(t.Context(), AgentRetrievalQuery{
		Projection: permutedProjection.ID, Text: "needle", Limit: 2, Embedder: embedder, Reranker: fixture.reranker,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) || !reflect.DeepEqual(first, permuted) || permutedProjection.ID != projection.ID ||
		len(first.Results) != 2 ||
		first.Results[0].Citation.Text != "needle evidence" || first.Results[0].DenseCandidate ||
		!first.Results[0].LexicalCandidate || first.Results[1].Citation.Text != "dense-only" ||
		!first.Results[1].DenseCandidate || first.Results[1].LexicalCandidate || first.Work.DenseCandidates != 2 ||
		first.Work.LexicalCandidates != 1 || first.Work.UnionCandidates != 2 || first.Work.RerankCalls != 2 ||
		first.Work.Returned != 2 {
		t.Fatalf("deterministic hybrid retrieval=(first=%+v, second=%+v, permuted=%+v)", first, second, permuted)
	}
}

func TestGroundedHybridRetrievalCandidateLimitBoundsUnion(t *testing.T) {
	fixture := newAgentRetrievalFixture(t)
	defer fixture.store.Close()
	embedder := agentRetrievalScriptedEmbedder{
		model: fixture.build.Embedder.ModelIdentity(),
		vectors: map[string][]float32{
			"needle":          {1, 0},
			"dense-only":      {1, 0},
			"needle evidence": {0, 1},
		},
	}
	fixture.build.Embedder = embedder
	fixture.build.Policy.CandidateLimit = 1
	fixture.build.Documents = []AgentRetrievalDocument{
		{Source: fixture.sources[0], Text: "dense-only"},
		{Source: fixture.sources[1], Text: "needle evidence"},
	}
	projection, err := fixture.Build(t)
	if err != nil {
		t.Fatal(err)
	}
	search, err := fixture.builder.SearchWithWork(t.Context(), AgentRetrievalQuery{
		Projection: projection.ID, Text: "needle", Limit: 2, Embedder: embedder, Reranker: fixture.reranker,
	})
	if err != nil || len(search.Results) != 1 || search.Work.DenseCandidates != 1 ||
		search.Work.LexicalCandidates != 1 || search.Work.UnionCandidates != 1 ||
		search.Work.RerankCalls != 1 || search.Work.Returned != 1 {
		t.Fatalf("bounded candidate union=(%+v, %v)", search, err)
	}
}

func TestGroundedHybridRetrievalRejectsCrossProjectionChunk(t *testing.T) {
	fixture := newAgentRetrievalFixture(t)
	defer fixture.store.Close()
	projection, err := fixture.Build(t)
	if err != nil {
		t.Fatal(err)
	}
	foreignDataset := testutil.ArtifactID(t, artifact.KindDataset, "foreign-retrieval-dataset")
	foreignPolicy := testutil.ArtifactID(t, artifact.KindProfile, "foreign-retrieval-policy")
	if _, err := fixture.store.Commit(t.Context(), artifact.Batch{
		Key:       "retrieval/foreign-authority",
		Artifacts: []artifact.Descriptor{{ID: foreignDataset}, {ID: foreignPolicy}},
	}); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name    string
		mutate  func(*AgentRetrievalProjection)
		allowed artifact.ID
	}{
		{name: "dataset", mutate: func(value *AgentRetrievalProjection) { value.Dataset = foreignDataset }, allowed: foreignDataset},
		{name: "policy", mutate: func(value *AgentRetrievalProjection) { value.Policy = foreignPolicy }, allowed: projection.Dataset},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			misbound := projection
			misbound.Entries = slices.Clone(projection.Entries)
			test.mutate(&misbound)
			misbound = publishAgentRetrievalProjection(t, fixture.store, misbound)
			_, err := fixture.builder.SearchWithWork(t.Context(), AgentRetrievalQuery{
				Projection: misbound.ID, Text: "target", Limit: 1, AllowedDatasets: []artifact.ID{test.allowed},
				Embedder: fixture.build.Embedder, Reranker: fixture.reranker,
			})
			if err == nil || !strings.Contains(err.Error(), "entry differs") {
				t.Fatalf("misbound projection accepted: %v", err)
			}
		})
	}
}

func TestGroundedHybridRetrievalCompleteAndLegacyFilterKeys(t *testing.T) {
	fixture := newAgentRetrievalFixture(t)
	defer fixture.store.Close()
	parent := testutil.ArtifactID(t, artifact.KindDatasetShard, "filter-parent")
	if _, err := fixture.store.Commit(t.Context(), artifact.Batch{
		Key: "retrieval/filter-parent", Artifacts: []artifact.Descriptor{{ID: parent}},
	}); err != nil {
		t.Fatal(err)
	}
	fixture.build.Documents = []AgentRetrievalDocument{{
		Source: fixture.sources[0], Text: "target evidence", Facet: AgentRetrievalFacetTable, Parent: parent,
	}}
	projection, err := fixture.Build(t)
	if err != nil {
		t.Fatal(err)
	}
	completeMismatch := projection
	completeMismatch.Entries = slices.Clone(projection.Entries)
	completeMismatch.Entries[0].Facet = ""
	completeMismatch.Entries[0].Parent = artifact.ID{}
	completeMismatch = publishAgentRetrievalProjection(t, fixture.store, completeMismatch)
	if _, err := fixture.builder.SearchWithWork(t.Context(), AgentRetrievalQuery{
		Projection: completeMismatch.ID, Text: "target", Limit: 1,
		Embedder: fixture.build.Embedder, Reranker: fixture.reranker,
	}); err == nil || !strings.Contains(err.Error(), "entry differs") {
		t.Fatalf("incomplete exact filter keys accepted: %v", err)
	}

	legacy := projection
	legacy.Entries = slices.Clone(projection.Entries)
	legacy.Entries[0].FilterKeysComplete = false
	legacy.Entries[0].Source = artifact.ID{}
	legacy.Entries[0].Facet = ""
	legacy.Entries[0].Parent = artifact.ID{}
	legacy = publishAgentRetrievalProjection(t, fixture.store, legacy)
	search, err := fixture.builder.SearchWithWork(t.Context(), AgentRetrievalQuery{
		Projection: legacy.ID, Text: "target", Limit: 1,
		AllowedSources: []artifact.ID{fixture.sources[0]}, AllowedFacets: []AgentRetrievalFacet{AgentRetrievalFacetTable},
		AllowedParents: []artifact.ID{parent}, Embedder: fixture.build.Embedder, Reranker: fixture.reranker,
	})
	if err != nil || len(search.Results) != 1 || search.Results[0].Citation.Parent != parent ||
		search.Work.ChunksLoaded != 1 || search.Work.EmbeddingsLoaded != 1 {
		t.Fatalf("legacy filter fallback=(%+v, %v)", search, err)
	}
}

func TestGroundedHybridRetrievalRejectsPolicyAboveHostBounds(t *testing.T) {
	fixture := newAgentRetrievalFixture(t)
	defer fixture.store.Close()
	fixture.build.Policy.MaximumChunkRunes = uint64(^uint(0)>>1) + 1
	if _, err := fixture.Build(t); err == nil ||
		!strings.Contains(err.Error(), "exceeds host bounds") {
		t.Fatalf("above-host retrieval policy accepted: %v", err)
	}
}

func TestGroundedHybridRetrievalCandidateStageDigestsEmpty(t *testing.T) {
	fixture := newAgentRetrievalFixture(t)
	defer fixture.store.Close()
	projection, err := fixture.Build(t)
	if err != nil {
		t.Fatal(err)
	}
	query := AgentRetrievalQuery{
		Projection: projection.ID, Text: "target", Limit: 1,
		AllowedSources: []artifact.ID{testutil.ArtifactID(t, artifact.KindFile, "absent-retrieval-source")},
		Embedder:       fixture.build.Embedder, Reranker: fixture.reranker,
	}
	first, err := fixture.builder.SearchWithWork(t.Context(), query)
	if err != nil {
		t.Fatal(err)
	}
	second, err := fixture.builder.SearchWithWork(t.Context(), query)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) || len(first.Results) != 0 || first.Work.EntriesInspected != 3 ||
		first.Work.FilteredEntries != 3 || first.Work.ChunksLoaded != 0 || first.Work.ChunkBytesLoaded != 0 ||
		first.Work.EmbeddingsLoaded != 0 || first.Work.EmbeddingBytesLoaded != 0 || first.Work.UnionCandidates != 0 ||
		first.DenseCandidateStage.Kind() != artifact.KindEvidence || first.LexicalCandidateStage.Kind() != artifact.KindEvidence ||
		first.UnionCandidateStage.Kind() != artifact.KindEvidence || first.DenseCandidateStage == first.LexicalCandidateStage ||
		first.DenseCandidateStage == first.UnionCandidateStage || first.LexicalCandidateStage == first.UnionCandidateStage {
		t.Fatalf("empty candidate stages=(first=%+v, second=%+v)", first, second)
	}
}

func publishAgentRetrievalProjection(
	t *testing.T,
	store *overgodb.Store,
	value AgentRetrievalProjection,
) AgentRetrievalProjection {
	t.Helper()
	value.ID = artifact.ID{}
	identified, err := agentRetrievalProjectionCodec.New(value)
	if err != nil {
		t.Fatal(err)
	}
	content, err := agentRetrievalProjectionCodec.Content(identified)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "retrieval/test-projection/" + identified.ID.String(), Contents: []artifact.Content{content},
		Lineage: agentRetrievalProjectionLineage(identified),
	}); err != nil {
		t.Fatal(err)
	}
	return identified
}

func agentRetrievalStoredBytes(t *testing.T, store *overgodb.Store, id artifact.ID) uint64 {
	t.Helper()
	descriptor, found, err := store.Artifact(t.Context(), id)
	if err != nil || !found {
		t.Fatalf("stored retrieval descriptor=(%+v, %t, %v)", descriptor, found, err)
	}
	return descriptor.Size
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
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "retrieval/fixture", Artifacts: []artifact.Descriptor{
			{ID: datasetID}, {ID: modelID}, {ID: rerankID},
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
