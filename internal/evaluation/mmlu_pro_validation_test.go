package evaluation

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
)

// TestMMLUProImportsItsValidationSplit holds MMLU-Pro's validation split imported beside its test split
// (mmlu-pro-lm-eval/validation-split), the split lm_eval's leaderboard_mmlu_pro draws its five demonstrations from. A
// Hugging Face cache holding both derives a seed of each, and a catalog holding both derives the very MMLU-Pro suite the
// test split alone derives: no reading scores a validation row, and no earlier reading's identity moves.
func TestMMLUProImportsItsValidationSplit(t *testing.T) {
	t.Parallel()
	family := hfCacheFamilies["TIGER-Lab___mmlu-pro"]
	cache := t.TempDir()
	for _, split := range []string{"test", "validation"} {
		path := filepath.Join(cache, "TIGER-Lab___mmlu-pro", "default", "0.0.0", "9126953e", "mmlu-pro-"+split+".arrow")
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
		testcheck.NoError(t, os.WriteFile(path, []byte(split), 0o644))
	}
	var derived []string
	for _, seed := range testcheck.Must(deriveHFCacheBenchmarks(cache))(t) {
		derived = append(derived, seed.Name+" of the "+seed.Spec.Split+" split")
	}
	if want := []string{"mmlu-pro/default/test of the test split", "mmlu-pro/default/validation of the validation split"}; !slices.Equal(derived, want) {
		t.Fatalf("the cache derives %q, want %q", derived, want)
	}

	root := t.TempDir()
	declare := func(split string, rows int) benchmarkDeclaration {
		var data bytes.Buffer
		for row := range rows {
			fmt.Fprintf(&data, `{"question":"%s question %d","options":["one","two","three"],"answer_index":%d,"category":"math"}`+"\n", split, row, row%3)
		}
		testcheck.NoError(t, os.WriteFile(filepath.Join(root, split+".jsonl"), data.Bytes(), 0o600))
		digest := sha256.Sum256(data.Bytes())
		return benchmarkDeclaration{Name: "mmlu-pro/default/" + split, Path: split + ".jsonl", Spec: dataset.BenchmarkImportSpec{
			Source: "mmlu-pro:default", Revision: "0.0.0/9126953e", SHA256: hex.EncodeToString(digest[:]), Split: split,
			Format: dataset.BenchmarkFormatJSONL, Conversion: family.conversion, Fields: family.fields,
		}}
	}
	test, validation := declare("test", 4), declare("validation", 5)
	authorities := mmluProProtocols()[PromptingRawCompletion.Label()].authorities
	plan := func(declarations ...benchmarkDeclaration) artifact.ID {
		store := overgodbtest.Open(t, t.TempDir())
		testcheck.Must(catalogBenchmarkDeclarations(t.Context(), store, root, declarations))(t)
		suites, _, err := DeriveStoreSuiteFamily(t.Context(), store, authorities, "mmlu-pro"+wholeFamily)
		if err != nil || len(suites) != 1 {
			t.Fatalf("the catalog of %d entries derives %d MMLU-Pro suites: %v", len(declarations), len(suites), err)
		}
		return suites[0].Plan().Identity()
	}
	if alone, beside := plan(test), plan(test, validation); alone != beside {
		t.Fatalf("with the validation split beside it, the MMLU-Pro suite binds plan %s, where the test split alone binds %s", beside, alone)
	}
}
