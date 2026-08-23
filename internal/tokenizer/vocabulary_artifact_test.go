package tokenizer

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestVocabularyArtifactExactMapping(t *testing.T) {
	tokenizerID := testutil.ArtifactID(t, artifact.KindTokenizer, "source vocabulary")
	semantics := VocabularySemantics{Model: "llama", AddPrefix: true}
	source, err := NewVocabularyArtifact(tokenizerID, semantics, []VocabularyToken{
		{Text: "<s>", Type: TokenControl, Roles: []VocabularyRole{VocabularyRoleBOS}},
		{Text: "red", Type: TokenNormal},
		{Text: "blue", Type: TokenNormal},
	})
	if err != nil {
		t.Fatal(err)
	}
	target, err := NewVocabularyArtifact(
		testutil.ArtifactID(t, artifact.KindTokenizer, "target vocabulary"), semantics,
		[]VocabularyToken{
			{Text: "blue", Type: TokenNormal},
			{Text: "<s>", Type: TokenControl, Roles: []VocabularyRole{VocabularyRoleBOS}},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	mapping, err := ExactVocabularyMapping(source, target)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(mapping.Rows, []TokenID{2, 0}) {
		t.Fatalf("mapping rows = %v", mapping.Rows)
	}
	if err := mapping.ValidateIdentity(); err != nil {
		t.Fatal(err)
	}
	if _, err := source.Content(); err != nil {
		t.Fatal(err)
	}
	if _, err := mapping.Content(); err != nil {
		t.Fatal(err)
	}
}

func TestVocabularyArtifactSemanticRefusal(t *testing.T) {
	semantics := VocabularySemantics{Model: "llama"}
	source, err := NewVocabularyArtifact(
		testutil.ArtifactID(t, artifact.KindTokenizer, "semantic source"), semantics,
		[]VocabularyToken{{Text: "same", Type: TokenNormal}},
	)
	if err != nil {
		t.Fatal(err)
	}
	wrongPolicy, err := NewVocabularyArtifact(
		testutil.ArtifactID(t, artifact.KindTokenizer, "wrong policy"),
		VocabularySemantics{Model: "llama", Lowercase: true},
		[]VocabularyToken{{Text: "same", Type: TokenNormal}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ExactVocabularyMapping(source, wrongPolicy); err == nil {
		t.Fatal("mapping accepted different normalization semantics")
	}
	wrongRole, err := NewVocabularyArtifact(
		testutil.ArtifactID(t, artifact.KindTokenizer, "wrong role"), semantics,
		[]VocabularyToken{{Text: "same", Type: TokenNormal, Roles: []VocabularyRole{VocabularyRoleBOS}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ExactVocabularyMapping(source, wrongRole); err == nil {
		t.Fatal("mapping accepted different special-token semantics")
	}
	missing, err := NewVocabularyArtifact(
		testutil.ArtifactID(t, artifact.KindTokenizer, "missing token"), semantics,
		[]VocabularyToken{{Text: "absent", Type: TokenNormal}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ExactVocabularyMapping(source, missing); err == nil {
		t.Fatal("mapping invented a missing token embedding")
	}
}
