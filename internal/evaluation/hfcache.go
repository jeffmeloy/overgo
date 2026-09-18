package evaluation

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
)

// hfCacheFamily declares how one lm_eval benchmark family in the
// HuggingFace dataset cache maps onto benchmark imports: which split
// files carry the evaluation cases, the field bindings the suite
// compilers consume, and the conversion label the import records. The
// table is data about public benchmark layouts, not about any model.
type hfCacheFamily struct {
	family     string
	domain     string
	splitBase  string
	splits     []string
	conversion string
	fields     []dataset.FieldBinding
}

var hfCacheFamilies = map[string]hfCacheFamily{
	// The DNA corpus is not an lm_eval cache family -- its entries come
	// from CatalogDNACorpus -- but the conversion-to-family mapping that
	// routes suite assembly lives in this table for every family.
	"carbon-pretraining-corpus": {
		family: "dna", domain: "dna", conversion: dnaCorpusConversion,
	},
	"cais___mmlu": {
		family: "mmlu", domain: DomainText, splitBase: "mmlu", splits: []string{"test"},
		conversion: "lm-eval/mmlu/v1",
		fields: []dataset.FieldBinding{
			{Target: "question", Source: "question"},
			{Target: "choices", Source: "choices"},
			{Target: "answer", Source: "answer"},
		},
	},
	"TIGER-Lab___mmlu-pro": {
		family: "mmlu-pro", domain: DomainText, splitBase: "mmlu-pro", splits: []string{"test"},
		conversion: "lm-eval/mmlu-pro/v1",
		fields: []dataset.FieldBinding{
			{Target: "question", Source: "question"},
			{Target: "options", Source: "options"},
			{Target: "answer_index", Source: "answer_index"},
			{Target: "category", Source: "category"},
		},
	},
	"SaylorTwift___bbh": {
		family: "bbh", domain: DomainText, splitBase: "bbh", splits: []string{"test"},
		conversion: "lm-eval/bbh/v1",
		fields: []dataset.FieldBinding{
			{Target: "input", Source: "input"},
			{Target: "target", Source: "target"},
		},
	},
	"TAUR-Lab___mu_sr": {
		family: "musr", domain: DomainText, splitBase: "mu_sr",
		splits:     []string{"murder_mysteries", "object_placements", "team_allocation"},
		conversion: "lm-eval/musr/v1",
		fields: []dataset.FieldBinding{
			{Target: "narrative", Source: "narrative"},
			{Target: "question", Source: "question"},
			{Target: "choices", Source: "choices"},
			{Target: "answer_index", Source: "answer_index"},
		},
	},
	"wis-k___instruction-following-eval": {
		family: "ifeval", domain: DomainText, splitBase: "instruction-following-eval", splits: []string{"train"},
		conversion: "lm-eval/ifeval/v1",
		fields: []dataset.FieldBinding{
			{Target: "key", Source: "key"},
			{Target: "prompt", Source: "prompt"},
			{Target: "instruction_id_list", Source: "instruction_id_list"},
			{Target: "kwargs", Source: "kwargs"},
		},
	},
	"DigitalLearningGmbH___math-lighteval": {
		family: "math", domain: DomainText, splitBase: "math-lighteval", splits: []string{"test"},
		conversion: "lm-eval/math/v1",
		fields: []dataset.FieldBinding{
			{Target: "problem", Source: "problem"},
			{Target: "solution", Source: "solution"},
			{Target: "level", Source: "level"},
			{Target: "type", Source: "type"},
		},
	},
}

// BenchmarkSeed is one derived benchmark declaration: a split file in
// the cache bound to the import spec that lands it in the store.
type BenchmarkSeed struct {
	Name string
	Path string
	Spec dataset.BenchmarkImportSpec
}

// DeriveHFCacheBenchmarks scans a HuggingFace dataset cache root and
// derives one benchmark seed per recognized family, subset, and split
// file: the layout is the source of truth for what benchmarks are
// locally held, so nothing is hand-listed. Unrecognized directories
// are skipped -- the cache also holds processing scratch -- but a
// recognized family with no importable split refuses loudly.
func DeriveHFCacheBenchmarks(root string) ([]BenchmarkSeed, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, err
	}
	seeds := make([]BenchmarkSeed, 0, len(entries))
	for _, entry := range entries {
		family, recognized := hfCacheFamilies[entry.Name()]
		if !entry.IsDir() || !recognized {
			continue
		}
		familySeeds, err := deriveFamilySeeds(filepath.Join(root, entry.Name()), family)
		if err != nil {
			return nil, fmt.Errorf("evaluation: derive %s benchmarks: %w", family.family, err)
		}
		seeds = append(seeds, familySeeds...)
	}
	if len(seeds) == 0 {
		return nil, errors.New("evaluation: the cache holds no recognized benchmark families")
	}
	sort.Slice(seeds, func(i, j int) bool { return seeds[i].Name < seeds[j].Name })
	return seeds, nil
}

func deriveFamilySeeds(familyRoot string, family hfCacheFamily) ([]BenchmarkSeed, error) {
	subsets, err := os.ReadDir(familyRoot)
	if err != nil {
		return nil, err
	}
	var seeds []BenchmarkSeed
	for _, subset := range subsets {
		if !subset.IsDir() {
			continue
		}
		for _, split := range family.splits {
			path, revision, found, err := findSplitFile(
				filepath.Join(familyRoot, subset.Name()), family.splitBase, split,
			)
			if err != nil {
				return nil, err
			}
			if !found {
				continue
			}
			digest, err := dataset.HashFile(path)
			if err != nil {
				return nil, err
			}
			name := family.family + "/" + subset.Name() + "/" + split
			seeds = append(seeds, BenchmarkSeed{
				Name: name, Path: path,
				Spec: dataset.BenchmarkImportSpec{
					Source:   family.family + ":" + subset.Name(),
					Revision: revision, SHA256: digest, Split: split,
					Format: dataset.BenchmarkFormatArrow, Conversion: family.conversion,
					Fields: family.fields,
				},
			})
		}
	}
	if len(seeds) == 0 {
		return nil, errors.New("no importable split files")
	}
	return seeds, nil
}

// findSplitFile locates <base>-<split>.arrow under the subset's
// version and fingerprint directories, picking the lexically first
// fingerprint that holds it so repeated scans derive the same seed.
func findSplitFile(subsetRoot, base, split string) (path, revision string, found bool, err error) {
	versions, err := os.ReadDir(subsetRoot)
	if err != nil {
		return "", "", false, err
	}
	wanted := base + "-" + split + ".arrow"
	for _, version := range versions {
		if !version.IsDir() {
			continue
		}
		fingerprints, err := os.ReadDir(filepath.Join(subsetRoot, version.Name()))
		if err != nil {
			return "", "", false, err
		}
		names := make([]string, 0, len(fingerprints))
		for _, fingerprint := range fingerprints {
			if fingerprint.IsDir() {
				names = append(names, fingerprint.Name())
			}
		}
		slices.Sort(names)
		for _, fingerprint := range names {
			candidate := filepath.Join(subsetRoot, version.Name(), fingerprint, wanted)
			if _, err := os.Stat(candidate); err == nil {
				return candidate, version.Name() + "/" + fingerprint, true, nil
			}
		}
	}
	return "", "", false, nil
}

// CatalogHFCacheBenchmarks imports every derived seed and publishes
// the active benchmark catalog -- the cache scan replaces the
// hand-written manifest as the declaration of what benchmarks the
// store serves.
func CatalogHFCacheBenchmarks(ctx context.Context, repository artifact.Repository, root string) (artifact.ID, int, error) {
	seeds, err := DeriveHFCacheBenchmarks(root)
	if err != nil {
		return artifact.ID{}, 0, err
	}
	declarations := make([]benchmarkDeclaration, len(seeds))
	for index, seed := range seeds {
		declarations[index] = benchmarkDeclaration{Name: seed.Name, Path: seed.Path, Spec: seed.Spec}
	}
	id, err := catalogBenchmarkDeclarations(ctx, repository, "", declarations)
	return id, len(seeds), err
}

// DomainText is the default domain for natural-language evaluation suites.
const DomainText = "text"

// SuiteDomain names the domain a derived suite belongs to; every
// lm_eval family is text today, and new families declare theirs in the
// cache table.
func SuiteDomain(source string) string {
	suffix := strings.TrimPrefix(source, "store/")
	for _, family := range hfCacheFamilies {
		if family.family == suffix {
			return family.domain
		}
	}
	return DomainText
}

// FilterSuitesForDomains keeps the suites whose domain the model
// declares. An undeclared model (declared=false) keeps everything.
func FilterSuitesForDomains(suites []CompiledSuite, domains []string, declared bool) []CompiledSuite {
	if !declared {
		return suites
	}
	kept := make([]CompiledSuite, 0, len(suites))
	for _, suite := range suites {
		if slices.Contains(domains, SuiteDomain(suite.Descriptor().Source)) {
			kept = append(kept, suite)
		}
	}
	return kept
}
