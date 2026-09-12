package evaluation

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
)

type interleavedCatalogRepository struct {
	artifact.Repository
	afterRead func()
}

func (repository *interleavedCatalogRepository) ResolveAlias(ctx context.Context, name string) (artifact.ID, bool, error) {
	id, bound, err := repository.Repository.ResolveAlias(ctx, name)
	if name == benchmarkCatalogAlias && repository.afterRead != nil {
		apply := repository.afterRead
		repository.afterRead = nil
		apply()
	}
	return id, bound, err
}

func TestIFEvalParameterCatalogAcceptance(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	storePath := filepath.Join(root, "store")
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	raw := []byte(`{"prompt":"Count letters.","instruction_id_list":["keywords:letter_frequency"],"kwargs":[{"letter":"#","let_frequency":2,"let_relation":"at least"}]}` + "\n")
	path := filepath.Join(root, "source.jsonl")
	if err := os.WriteFile(path, raw, 0600); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(raw)
	spec := dataset.BenchmarkImportSpec{Source: "parameter-fixture", Revision: "v1", SHA256: hex.EncodeToString(digest[:]), Split: "train", Format: dataset.BenchmarkFormatJSONL, Conversion: "lm-eval/ifeval/v1", Fields: []dataset.FieldBinding{{Source: "prompt", Target: "prompt"}, {Source: "instruction_id_list", Target: "instruction_id_list"}, {Source: "kwargs", Target: "kwargs"}}}
	imported, err := dataset.ImportBenchmark(ctx, store, path, spec)
	if err != nil {
		t.Fatal(err)
	}
	write := func(contract artifact.DocumentContract, value any) artifact.ID {
		t.Helper()
		raw, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		content, err := contract.ContentBytes(raw)
		if err != nil {
			t.Fatal(err)
		}
		batch, err := artifact.NewDocumentBatch("fixture/"+content.Descriptor.ID.String(), []artifact.Content{content}, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
		return content.Descriptor.ID
	}
	reference := write(artifact.DocumentContract{Kind: artifact.KindEvidence, MediaType: "application/json", Schema: "test/native-parameters/v1"}, map[string]string{"source": "fixed independent strict/loose operands"})
	makeProfile := func() ifevalParameters {
		return ifevalParameters{Dataset: imported.ID, Reference: reference, Bindings: []ifevalParameterBinding{{Record: imported.Records[0], Instruction: "keywords:letter_frequency", Strict: rawFields(t, map[string]any{"letter": "a", "let_frequency": 2, "let_relation": "at least"}), Loose: rawFields(t, map[string]any{"letter": "b", "let_frequency": 2, "let_relation": "at least"})}}}
	}
	parameters := write(ifevalParameterContract, makeProfile())
	declaration := benchmarkDeclaration{Name: "ifeval/default/train", Dataset: imported.ID, Parameters: parameters}
	manifestPath := filepath.Join(root, "manifest.json")
	catalogue := func(declaration benchmarkDeclaration) (artifact.ID, error) {
		raw, err := json.Marshal(benchmarkManifest{Datasets: []benchmarkDeclaration{declaration}})
		if err != nil {
			return artifact.ID{}, err
		}
		if err := os.WriteFile(manifestPath, raw, 0600); err != nil {
			return artifact.ID{}, err
		}
		return CatalogLocalBenchmarks(ctx, store, manifestPath)
	}
	catalogID, err := catalogue(declaration)
	if err != nil {
		t.Fatal(err)
	}
	checkDerived := func() {
		t.Helper()
		suites, dropped, err := DeriveStoreSuiteFamily(ctx, store, ExactAuthorities{ModelDefinition: planID(t, artifact.KindModelDefinition, "model"), RuntimeRecipe: planID(t, artifact.KindRecipe, "recipe"), CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "environment"), Execution: ExecutionPolicy{Lifecycle: LifecycleResident}}, "ifeval")
		if err != nil || len(suites) != 1 || dropped["ifeval"] != 0 {
			t.Fatalf("derive: %d %v %v", len(suites), dropped, err)
		}
		loose := InstructionRule{Name: "keywords:letter_frequency", Kind: ifevalLetters, Values: []string{"b"}, Count: &CountRule{Relation: RelationAtLeast, Value: 2}}
		strict := InstructionRule{Name: loose.Name, Kind: ifevalLetters, Values: []string{"a"}, Count: &CountRule{Relation: RelationAtLeast, Value: 2}, Loose: &loose}
		want, err := CompileInstructionRules(InstructionRulesSuite{Kind: InstructionRulesKind, Schema: "lm-eval/ifeval/v4.0", Source: "store/ifeval", Cases: []InstructionRulesCase{{Name: "ifeval/default/train/0", Prompt: "Count letters.", MaxTokens: ifevalDerivedMaxTokens, Rules: []InstructionRule{strict}}}})
		if err != nil || suites[0].plan.body.CaseProfile != want.identity {
			t.Fatalf("production derivation did not consume frozen views: %v", err)
		}
		for _, sample := range []struct {
			response      string
			strict, loose bool
		}{{"aa", true, false}, {"bb", false, true}} {
			a, b, err := evaluateInstructionViews(sample.response, want.rules[0])
			if err != nil {
				t.Fatal(err)
			}
			if a[0] != sample.strict || b[0] != sample.loose {
				t.Fatal("view binding changed")
			}
		}
	}
	checkDerived()
	// An unchanged physical reimport must retain its frozen parameter binding.
	again, err := catalogue(benchmarkDeclaration{Name: declaration.Name, Path: "source.jsonl", Spec: spec})
	if err != nil || again != catalogID {
		t.Fatalf("same-dataset import lost parameters: %v", err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store, err = overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	again, err = catalogue(declaration)
	if err != nil || again != catalogID {
		t.Fatalf("restart required source reacquisition: %v", err)
	}
	checkDerived()
	for _, mutation := range []string{"wrong dataset", "wrong record", "missing instruction", "duplicate", "missing strict", "missing loose", "unresolved strict", "unresolved loose", "missing reference", "unused operand"} {
		t.Run(mutation, func(t *testing.T) {
			profile := makeProfile()
			switch mutation {
			case "wrong dataset":
				profile.Dataset, _ = artifact.JSONID(artifact.KindDataset, "unrelated")
			case "wrong record":
				profile.Bindings[0].Record, _ = artifact.JSONID(artifact.KindDatasetShard, "unrelated")
			case "missing instruction":
				profile.Bindings[0].Instruction = "keywords:frequency"
			case "duplicate":
				profile.Bindings = append(profile.Bindings, profile.Bindings[0])
			case "missing strict":
				profile.Bindings[0].Strict = nil
			case "missing loose":
				profile.Bindings[0].Loose = nil
			case "unresolved strict":
				profile.Bindings[0].Strict["letter"] = json.RawMessage(`"!"`)
			case "unresolved loose":
				profile.Bindings[0].Loose["letter"] = json.RawMessage(`"!"`)
			case "missing reference":
				profile.Reference, _ = artifact.JSONID(artifact.KindEvidence, "absent")
			case "unused operand":
				profile.Bindings[0].Loose["ignored"] = json.RawMessage(`true`)
			}
			bad := declaration
			bad.Parameters = write(ifevalParameterContract, profile)
			if _, err := catalogue(bad); err == nil {
				t.Fatal("invalid parameters accepted")
			}
			active, bound, err := store.ResolveAlias(ctx, benchmarkCatalogAlias)
			if err != nil || !bound || active != catalogID {
				t.Fatal("refused binding changed active catalog")
			}
		})
	}
	for _, mutation := range []string{"path", "spec"} {
		bad := declaration
		if mutation == "path" {
			bad.Path = "source.jsonl"
		} else {
			bad.Spec.Limit = 1
		}
		if _, err := catalogue(bad); err == nil {
			t.Fatal("ambiguous reuse/import inputs accepted")
		}
	}
	t.Run("concurrent publication preserves foreign entries", func(t *testing.T) {
		var foreignID artifact.ID
		interleaved := &interleavedCatalogRepository{Repository: store, afterRead: func() {
			foreign := declaration
			foreign.Name = "foreign/default/train"
			var err error
			foreignID, err = catalogBenchmarkDeclarations(ctx, store, "", []benchmarkDeclaration{foreign})
			if err != nil {
				t.Fatal(err)
			}
		}}
		addition := declaration
		addition.Name = "new/default/train"
		if _, err := catalogBenchmarkDeclarations(ctx, interleaved, "", []benchmarkDeclaration{addition}); err == nil {
			t.Fatal("stale catalog publication overwrote a concurrent writer")
		}
		active, _, err := store.ResolveAlias(ctx, benchmarkCatalogAlias)
		if err != nil || active != foreignID {
			t.Fatal("concurrent catalog update was lost")
		}
		merged, err := catalogBenchmarkDeclarations(ctx, store, "", []benchmarkDeclaration{addition})
		if err != nil {
			t.Fatal(err)
		}
		catalog, found, err := benchmarkCatalogCodec.Read(ctx, store, merged)
		if err != nil || !found || len(catalog.Entries) != 3 {
			t.Fatal("retry did not preserve both additions")
		}
	})
}
