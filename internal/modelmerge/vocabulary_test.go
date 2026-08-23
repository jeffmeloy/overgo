package modelmerge

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/tensor"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

func TestVocabularyArtifactEmbeddingProvenance(t *testing.T) {
	fixture := vocabularyArtifactFixture(t)
	result, err := (Compiler{}).VocabularyArtifact(
		fixture.source, fixture.sourceVocabulary, fixture.targetVocabulary,
		fixture.targetDefinition, []string{"token_embd.weight", "output.weight"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Model.Base != fixture.source.ID || len(result.Provenance.Rows) != 4 {
		t.Fatalf("unexpected vocabulary artifact lineage: %+v", result.Provenance)
	}
	if result.Provenance.Rows[0].Tensor != "output.weight" ||
		result.Provenance.Rows[0].OutputRow != 0 || result.Provenance.Rows[0].SourceRow != 2 {
		t.Fatalf("unexpected first provenance row: %+v", result.Provenance.Rows[0])
	}
	if err := result.Provenance.ValidateIdentity(); err != nil {
		t.Fatal(err)
	}
	if _, err := result.Provenance.Content(); err != nil {
		t.Fatal(err)
	}
}

func TestVocabularyArtifactGeneration(t *testing.T) {
	fixture := vocabularyArtifactFixture(t)
	result, err := (Compiler{}).VocabularyArtifact(
		fixture.source, fixture.sourceVocabulary, fixture.targetVocabulary,
		fixture.targetDefinition, []string{"token_embd.weight", "output.weight"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := result.ValidateGeneratedArtifact(fixture.source); err != nil {
		t.Fatal(err)
	}
	if got := result.Model.Tensors["token_embd.weight"]; got.Layout != tensor.MustShape(2, 2) ||
		!slices.Equal(got.Values, []float32{5, 6, 1, 2}) {
		t.Fatalf("generated embedding = %+v", got)
	}
	corrupt := result
	corrupt.Model.Tensors = cloneWeights(result.Model.Tensors)
	weight := corrupt.Model.Tensors["token_embd.weight"]
	weight.Values[0]++
	corrupt.Model.Tensors["token_embd.weight"] = weight
	if err := corrupt.ValidateGeneratedArtifact(fixture.source); err == nil {
		t.Fatal("generated artifact validation accepted changed embedding bytes")
	}
}

type vocabularyArtifactTestFixture struct {
	source           Snapshot
	sourceVocabulary tokenizer.VocabularyArtifact
	targetVocabulary tokenizer.VocabularyArtifact
	targetDefinition artifact.ID
}

func vocabularyArtifactFixture(t *testing.T) vocabularyArtifactTestFixture {
	t.Helper()
	compiler := Compiler{}
	source, err := compiler.Seal(
		testutil.ArtifactID(t, artifact.KindModelDefinition, "vocabulary source definition"),
		artifact.ID{}, map[string]Weight{
			"token_embd.weight": {Layout: tensor.MustShape(2, 3), Values: []float32{1, 2, 3, 4, 5, 6}},
			"output.weight":     {Layout: tensor.MustShape(2, 3), Values: []float32{10, 20, 30, 40, 50, 60}},
			"block.weight":      {Layout: tensor.MustShape(2, 1), Values: []float32{7, 8}},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	semantics := tokenizer.VocabularySemantics{Model: "llama", AddPrefix: true}
	sourceVocabulary, err := tokenizer.NewVocabularyArtifact(
		testutil.ArtifactID(t, artifact.KindTokenizer, "vocabulary source tokenizer"), semantics,
		[]tokenizer.VocabularyToken{
			{Text: "alpha", Type: tokenizer.TokenNormal},
			{Text: "beta", Type: tokenizer.TokenNormal},
			{Text: "gamma", Type: tokenizer.TokenNormal},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	targetVocabulary, err := tokenizer.NewVocabularyArtifact(
		testutil.ArtifactID(t, artifact.KindTokenizer, "vocabulary target tokenizer"), semantics,
		[]tokenizer.VocabularyToken{
			{Text: "gamma", Type: tokenizer.TokenNormal},
			{Text: "alpha", Type: tokenizer.TokenNormal},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return vocabularyArtifactTestFixture{
		source: source, sourceVocabulary: sourceVocabulary, targetVocabulary: targetVocabulary,
		targetDefinition: testutil.ArtifactID(t, artifact.KindModelDefinition, "vocabulary target definition"),
	}
}
