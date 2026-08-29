package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/sequencescore"
	"overgo/internal/testutil"
)

// TestPromptTemplateAuthority pins the store declaration: a template
// publishes under its model's alias, reads back intact, re-declaration
// replaces it, and an undeclared model reports absent.
func TestPromptTemplateAuthority(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	model := testutil.ArtifactID(t, artifact.KindModel, "template-model")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/template/model", Artifacts: []artifact.Descriptor{{ID: model}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, declared, err := LoadPromptTemplate(ctx, store, model); err != nil || declared {
		t.Fatalf("undeclared template = (%t, %v)", declared, err)
	}
	published, err := PublishPromptTemplate(ctx, store, PromptTemplate{
		Model: model, Source: "tokenizer-metadata", ScoringPrefix: "<dna>",
	})
	if err != nil {
		t.Fatal(err)
	}
	loaded, declared, err := LoadPromptTemplate(ctx, store, model)
	if err != nil || !declared || loaded.ID != published.ID || loaded.ScoringPrefix != "<dna>" {
		t.Fatalf("loaded template = (%+v, %t, %v)", loaded, declared, err)
	}
	if _, err := PublishPromptTemplate(ctx, store, PromptTemplate{
		Model: model, Source: "gguf-chat-template", ChatTemplate: "{{messages}}",
	}); err != nil {
		t.Fatal(err)
	}
	if replaced, _, _ := LoadPromptTemplate(ctx, store, model); replaced.ChatTemplate != "{{messages}}" || replaced.ScoringPrefix != "" {
		t.Fatalf("replaced template = %+v", replaced)
	}
}

type prefixedFixtureScorer struct {
	sequenceFixtureScorer
	prefix string
}

func (s *prefixedFixtureScorer) ScoringPrefix() string { return s.prefix }

// TestSequenceScoringUsesDeclaredPrefix pins the shaping contract: a
// runtime declaring a scoring prefix scores every sequence inside it,
// and one without scores from the empty context.
func TestSequenceScoringUsesDeclaredPrefix(t *testing.T) {
	compiled, err := CompileSequenceScoring(SequenceScoringSuite{
		Kind: SequenceScoringKind, Schema: "carbon/dna-corpus/v1", Source: "store/dna",
		Cases: []SequenceScoringCase{{Name: "seq-1", Group: "mrna_evo2", Text: "ACGTACGT"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	authorities := ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"),
		RuntimeRecipe:   planID(t, artifact.KindRecipe, "recipe"), CodeCommit: planTestCommit,
		Environment: planID(t, artifact.KindEvidence, "environment"),
		Execution:   ExecutionPolicy{Lifecycle: LifecycleResident},
	}
	plan, err := BindSequenceScoring(compiled, authorities)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	publishPlanFixtureAuthorities(t, store, plan)
	scorer := &prefixedFixtureScorer{prefix: "<dna>"}
	scorer.scores = []sequencescore.Score{{LogProbability: -4, Tokens: 4}}
	if _, err := EvaluateSequenceScoring(t.Context(), store, scorer, compiled, plan); err != nil {
		t.Fatal(err)
	}
	if len(scorer.prompts) != 1 || scorer.prompts[0] != "<dna>" {
		t.Fatalf("prompts = %v, want the declared prefix", scorer.prompts)
	}

	bare := &sequenceFixtureScorer{scores: []sequencescore.Score{{LogProbability: -4, Tokens: 4}}}
	if _, err := EvaluateSequenceScoring(t.Context(), store, bare, compiled, plan); err != nil {
		t.Fatal(err)
	}
	if len(bare.prompts) != 1 || bare.prompts[0] != "" {
		t.Fatalf("bare prompts = %v, want the empty context", bare.prompts)
	}
}
