package modelartifact

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestModelConfigArtifactContract pins the declaration document: at least
// one typed component, a validated sequence-extension declaration, source
// provenance with digests, order-canonical identity, and lineage to the
// model it describes. Nothing in the document or its validation names any
// model -- the components are the format.
func TestModelConfigArtifactContract(t *testing.T) {
	model := testutil.ArtifactID(t, artifact.KindModel, "config-model")
	digest := strings.Repeat("ab", 32)
	sequence := &SequenceExtensionConfig{
		K: 2, StartID: 100, Vocabulary: 19,
		SpecialTokens: []string{"<seq>", "</seq>", "<oov>"},
	}
	generation := &GenerationEssentials{EOSTokens: []int64{7}, ContextLength: 4096}
	sources := []ConfigSource{
		{Name: "generation_config.json", SHA256: digest},
		{Name: "dna_config.json", SHA256: digest},
	}
	document, err := NewModelConfigDocument(model, sequence, generation, sources)
	if err != nil {
		t.Fatal(err)
	}
	if document.Sources[0].Name != "dna_config.json" {
		t.Fatalf("sources not sorted: %+v", document.Sources)
	}
	replay, err := NewModelConfigDocument(model, sequence, generation, []ConfigSource{sources[1], sources[0]})
	if err != nil || replay.ID != document.ID {
		t.Fatalf("identity not order-canonical: (%v, %v)", replay.ID, err)
	}
	content, err := document.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseModelConfigDocument(content.Data)
	if err != nil || parsed.ID != document.ID || parsed.Sequence.K != 2 ||
		parsed.Generation.ContextLength != 4096 {
		t.Fatalf("roundtrip = (%+v, %v)", parsed, err)
	}
	batch, err := document.Batch("model-config/" + document.ID.String())
	if err != nil || len(batch.Lineage) != 1 || batch.Lineage[0].Parent != model {
		t.Fatalf("batch lineage = (%+v, %v), want the model as sole parent", batch.Lineage, err)
	}

	type variant struct {
		sequence   *SequenceExtensionConfig
		generation *GenerationEssentials
		sources    []ConfigSource
	}
	for name, broken := range map[string]variant{
		"no components":    {nil, nil, sources},
		"no sources":       {sequence, generation, nil},
		"bad digest":       {sequence, generation, []ConfigSource{{Name: "config.json", SHA256: "zz"}}},
		"pathed source":    {sequence, generation, []ConfigSource{{Name: "dir/config.json", SHA256: digest}}},
		"duplicate source": {sequence, generation, []ConfigSource{sources[0], sources[0]}},
	} {
		if _, err := NewModelConfigDocument(model, broken.sequence, broken.generation, broken.sources); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
	if _, err := NewModelConfigDocument(
		testutil.ArtifactID(t, artifact.KindEvidence, "not-a-model"), sequence, generation, sources,
	); err == nil {
		t.Fatal("non-model identity accepted")
	}
}

// TestModelConfigArtifactFromFiles pins extraction: declared components come
// from the checkpoint directory's config files with content digests, absent
// files are absent components, scalar and list token IDs both parse, and
// malformed declarations error rather than silently vanish.
func TestModelConfigArtifactFromFiles(t *testing.T) {
	directory := t.TempDir()
	write := func(name, content string) string {
		t.Helper()
		if err := os.WriteFile(filepath.Join(directory, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		digest := sha256.Sum256([]byte(content))
		return hex.EncodeToString(digest[:])
	}
	dnaDigest := write("dna_config.json",
		`{"k":6,"dna_start_id":151669,"dna_vocab_size":4107,"dna_special_tokens":["<dna>","</dna>","<oov>"],"auto_dna_tags":false}`)
	write("generation_config.json", `{"bos_token_id":null,"eos_token_id":[151645,151643]}`)
	write("config.json", `{"architectures":["LlamaForCausalLM"],"max_position_embeddings":8192}`)

	sequence, generation, sources, err := ReadModelConfigComponents(directory)
	if err != nil {
		t.Fatal(err)
	}
	if sequence == nil || sequence.K != 6 || sequence.StartID != 151669 ||
		sequence.Vocabulary != 4107 || len(sequence.SpecialTokens) != 3 || sequence.AutoTags {
		t.Fatalf("sequence = %+v", sequence)
	}
	if generation == nil || len(generation.BOSTokens) != 0 ||
		len(generation.EOSTokens) != 2 || generation.EOSTokens[0] != 151645 ||
		generation.ContextLength != 8192 {
		t.Fatalf("generation = %+v", generation)
	}
	if len(sources) != 3 {
		t.Fatalf("sources = %+v", sources)
	}
	model := testutil.ArtifactID(t, artifact.KindModel, "extracted-model")
	document, err := NewModelConfigDocument(model, sequence, generation, sources)
	if err != nil {
		t.Fatal(err)
	}
	for _, source := range document.Sources {
		if source.Name == "dna_config.json" && source.SHA256 != dnaDigest {
			t.Fatalf("dna_config digest = %s, want %s", source.SHA256, dnaDigest)
		}
	}

	empty := t.TempDir()
	sequence, generation, sources, err = ReadModelConfigComponents(empty)
	if err != nil || sequence != nil || generation != nil || len(sources) != 0 {
		t.Fatalf("empty directory = (%v, %v, %v, %v), want absent components", sequence, generation, sources, err)
	}

	if err := os.WriteFile(filepath.Join(empty, "generation_config.json"), []byte(`{"eos_token_id":"seven"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := ReadModelConfigComponents(empty); err == nil {
		t.Fatal("malformed token declaration accepted")
	}
}
