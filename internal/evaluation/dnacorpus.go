package evaluation

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
)

// dnaCorpusConversion labels corpus-slice imports of the DNA
// pretraining corpus; the dna suite family assembles them into
// sequence-scoring suites.
const dnaCorpusConversion = "carbon/dna-corpus/v1"

// CatalogDNACorpus imports a bounded slice of every corpus subset and
// merges the entries into the active benchmark catalog: each subset's
// lexically first parquet file contributes its first limit sequences,
// hashed and recorded like every other benchmark import. The slice
// overlaps training data by construction -- it measures in-domain fit,
// and comparisons across models on the same slice stay meaningful.
func CatalogDNACorpus(ctx context.Context, repository artifact.Repository, root string, limit int) (artifact.ID, int, error) {
	if limit <= 0 {
		return artifact.ID{}, 0, errors.New("evaluation: DNA corpus slice limit must be positive")
	}
	subsets, err := dnaCorpusSubsets(root)
	if err != nil {
		return artifact.ID{}, 0, err
	}
	declarations := make([]benchmarkDeclaration, 0, len(subsets))
	for _, subset := range subsets {
		digest, err := dataset.HashFile(subset.file)
		if err != nil {
			return artifact.ID{}, 0, err
		}
		declarations = append(declarations, benchmarkDeclaration{
			Name: "dna/" + subset.name + "/corpus-slice",
			Path: subset.file,
			Spec: dataset.BenchmarkImportSpec{
				Source:   "carbon-pretraining-corpus:" + subset.name,
				Revision: filepath.Base(subset.file), SHA256: digest, Split: "corpus-slice",
				Format: dataset.BenchmarkFormatParquetText, Conversion: dnaCorpusConversion,
				Fields: []dataset.FieldBinding{{Target: "text", Source: dataset.BenchmarkColumnAutoText}},
				Limit:  uint64(limit),
			},
		})
	}
	id, err := catalogBenchmarkDeclarations(ctx, repository, "", declarations)
	return id, len(declarations), err
}

type dnaSubset struct {
	name string
	file string
}

// dnaCorpusSubsets finds every directory (up to two levels deep) that
// directly holds parquet files, naming it by its corpus-relative path.
func dnaCorpusSubsets(root string) ([]dnaSubset, error) {
	var subsets []dnaSubset
	visit := func(relative, directory string) error {
		entries, err := os.ReadDir(directory)
		if err != nil {
			return err
		}
		files := make([]string, 0, len(entries))
		for _, entry := range entries {
			if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".parquet") {
				files = append(files, entry.Name())
			}
		}
		if len(files) == 0 {
			return nil
		}
		sort.Strings(files)
		subsets = append(subsets, dnaSubset{
			name: strings.ReplaceAll(relative, "/", "-"),
			file: filepath.Join(directory, files[0]),
		})
		return nil
	}
	top, err := os.ReadDir(root)
	if err != nil {
		return nil, err
	}
	for _, entry := range top {
		if !entry.IsDir() {
			continue
		}
		if err := visit(entry.Name(), filepath.Join(root, entry.Name())); err != nil {
			return nil, err
		}
		nested, err := os.ReadDir(filepath.Join(root, entry.Name()))
		if err != nil {
			return nil, err
		}
		for _, inner := range nested {
			if !inner.IsDir() {
				continue
			}
			if err := visit(entry.Name()+"/"+inner.Name(), filepath.Join(root, entry.Name(), inner.Name())); err != nil {
				return nil, err
			}
		}
	}
	if len(subsets) == 0 {
		return nil, errors.New("evaluation: the corpus root holds no parquet subsets")
	}
	sort.Slice(subsets, func(i, j int) bool { return subsets[i].name < subsets[j].name })
	return subsets, nil
}

// assembleDNASuite renders corpus-slice records as one sequence-scoring
// suite: each sequence scores full-text, grouped by corpus subset.
func assembleDNASuite(cases []storeCase) (any, int, error) {
	suite := SequenceScoringSuite{
		Kind: SequenceScoringKind, Schema: dnaCorpusConversion, Source: "store/dna",
	}
	for _, entry := range cases {
		text, err := caseString(entry.fields, "text")
		if err != nil {
			return nil, 0, err
		}
		suite.Cases = append(suite.Cases, SequenceScoringCase{
			Name:  fmt.Sprintf("%s/%d", entry.entry, entry.ordinal),
			Group: entry.subset, Text: text,
		})
	}
	return suite, 0, nil
}
