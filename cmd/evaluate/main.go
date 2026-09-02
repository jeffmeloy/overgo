// Command evaluate runs benchmark suites against local models: a
// hand-written manifest names models and suite files, or -all derives
// both sides from the store -- the servable model listing and the
// suites compiled from the active benchmark catalog -- and campaigns
// each model in its own worker process.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/strictjson"
)

type manifest struct {
	Repository string         `json:"repository"`
	Catalog    string         `json:"catalog"`
	CodeCommit string         `json:"code_commit"`
	Device     int            `json:"device"`
	Models     []modelRequest `json:"models"`
	// ChatProtocol scores multiple-choice prompts through each model's
	// declared chat template instead of the raw base-style completion.
	// The raw protocol stays the recorded anchor: template presence does
	// not distinguish base from instruct artifacts, so the shaped
	// protocol is an explicit measurement choice.
	ChatProtocol bool `json:"chat_protocol,omitzero"`
}

type modelRequest struct {
	Path   string   `json:"path"`
	Suites []string `json:"suites"`
}

type workerLauncher func(context.Context, int) error

func main() {
	clioptions.Main(run)
}

func run() error {
	manifestPath := flag.String("manifest", "", "evaluation manifest")
	worker := flag.Bool("worker", false, "run one model worker")
	modelIndex := flag.Int("model-index", -1, "worker model index")
	modelPath := flag.String("model-path", "", "worker model path for -all (the parent lists the catalog once and hands each worker its model)")
	budget := flag.Duration("budget", evaluationBudget, "wall-clock ceiling on one -all pass, shared as equal slices across the models still to run; a model cut at its slice is recorded budget-exceeded")
	importCache := flag.String("import-hf-cache", "", "scan a HuggingFace dataset cache root, import every recognized benchmark, and publish the active catalog")
	listSuites := flag.Bool("list-derived-suites", false, "compile the store's benchmark catalog into suites and list their descriptors")
	repository := flag.String("repo", "overgodb-store", "OvergoDB root for -import-hf-cache, -list-derived-suites, and -all")
	allModels := flag.Bool("all", false, "evaluate every servable local model against the store's derived suites")
	chatProtocol := flag.Bool("chat-protocol", false, "score multiple-choice suites through each model's declared chat template; the raw completion protocol stays the recorded anchor")
	device := flag.Int("device", 0, "CUDA device ordinal for -all")
	family := flag.String("family", "", "restrict -all and -list-derived-suites to one derived suite family (e.g. mmlu); only that family's records are read and compiled")
	catalogLimit := flag.Int("catalog-limit", 256, "servable model listing bound for -all")
	declareDomains := flag.String("declare-domain", "", "comma-separated eval domains to declare for the positional model path (e.g. dna)")
	declareReferences := flag.String("declare-references", "", "JSON spec of published or externally measured reference scores per model location ({declarations:[{model, references:[{suite, metric, value, protocol, source}]}]})")
	importDNA := flag.String("import-dna-corpus", "", "import a bounded slice of every parquet subset under this corpus root and merge the entries into the active benchmark catalog")
	dnaLimit := flag.Int("dna-limit", 16, "sequences imported per corpus subset for -import-dna-corpus")
	flag.Parse()
	if root := strings.TrimSpace(*importDNA); root != "" {
		if flag.NArg() != 0 {
			return errors.New("usage: evaluate -import-dna-corpus <root> [-dna-limit N] [-repo <store>]")
		}
		store, err := overgodb.Open(*repository)
		if err != nil {
			return err
		}
		defer store.Close()
		catalog, count, err := evaluation.CatalogDNACorpus(context.Background(), store, root, *dnaLimit)
		if err != nil {
			return err
		}
		fmt.Printf("benchmark catalog %s merged %d DNA corpus slice(s)\n", catalog, count)
		return nil
	}
	if spec := strings.TrimSpace(*declareReferences); spec != "" {
		if flag.NArg() != 0 {
			return errors.New("usage: evaluate -declare-references <spec.json> [-repo <store>]")
		}
		return declareReferenceScores(context.Background(), *repository, spec, *catalogLimit)
	}
	if csv := strings.TrimSpace(*declareDomains); csv != "" {
		if flag.NArg() != 1 {
			return errors.New("usage: evaluate -declare-domain <domains-csv> [-repo <store>] <model-path>")
		}
		return declareEvalDomain(context.Background(), *repository, flag.Arg(0), csv, *catalogLimit)
	}
	if *allModels {
		if strings.TrimSpace(*manifestPath) != "" {
			return errors.New("usage: evaluate -all [-repo <store>] [-device N] [-family name]")
		}
		if *worker {
			return runAllWorker(context.Background(), *repository, *device, *family, *modelPath, *chatProtocol)
		}
		return runAllParent(context.Background(), *repository, *device, *family, *catalogLimit, *chatProtocol, *budget)
	}
	if *listSuites {
		if flag.NArg() != 0 || *worker || strings.TrimSpace(*manifestPath) != "" {
			return errors.New("usage: evaluate -list-derived-suites [-repo <store>]")
		}
		store, err := overgodb.OpenReadOnly(*repository)
		if err != nil {
			return err
		}
		defer store.Close()
		// Placeholder authorities admit compilation for listing; running a
		// suite still binds the real model, recipe, and environment.
		suites, skipped, err := evaluation.DeriveStoreSuiteFamily(context.Background(), store, evaluation.ListingAuthorities(), strings.TrimSpace(*family))
		if err != nil {
			return err
		}
		for _, suite := range suites {
			descriptor := suite.Descriptor()
			fmt.Printf("%s cases=%d source=%s\n", descriptor.Kind, descriptor.Cases, descriptor.Source)
		}
		for family, dropped := range skipped {
			fmt.Printf("skipped %d %s case(s) outside the exact vocabulary\n", dropped, family)
		}
		return nil
	}
	if cache := strings.TrimSpace(*importCache); cache != "" {
		if flag.NArg() != 0 || *worker || strings.TrimSpace(*manifestPath) != "" {
			return errors.New("usage: evaluate -import-hf-cache <root> [-repo <store>]")
		}
		store, err := overgodb.Open(*repository)
		if err != nil {
			return err
		}
		defer store.Close()
		catalog, count, err := evaluation.CatalogHFCacheBenchmarks(context.Background(), store, cache)
		if err != nil {
			return err
		}
		fmt.Printf("benchmark catalog %s published from %d imported dataset(s)\n", catalog, count)
		return nil
	}
	if strings.TrimSpace(*manifestPath) == "" || flag.NArg() != 0 {
		return errors.New("usage: evaluate -manifest manifest.json")
	}
	compiled, err := readManifest(*manifestPath)
	if err != nil {
		return err
	}
	if *worker {
		if *modelIndex < 0 || *modelIndex >= len(compiled.Models) {
			return errors.New("evaluate: worker model index is invalid")
		}
		return executeModel(context.Background(), compiled, compiled.Models[*modelIndex], openEvaluationSession)
	}
	if err := catalogBenchmarks(context.Background(), compiled); err != nil {
		return err
	}
	executable, err := os.Executable()
	if err != nil {
		return err
	}
	absoluteManifest, err := filepath.Abs(*manifestPath)
	if err != nil {
		return err
	}
	return runParent(context.Background(), compiled, func(ctx context.Context, index int) error {
		receipt, err := processcontrol.Run(ctx, processcontrol.Command{
			Path:   executable,
			Args:   []string{"-worker", "-manifest", absoluteManifest, "-model-index", strconv.Itoa(index)},
			Stdout: os.Stdout, Stderr: os.Stderr,
		})
		if err != nil {
			return err
		}
		if receipt.ExitCode != 0 {
			return fmt.Errorf("evaluate: isolated worker exited with status %d", receipt.ExitCode)
		}
		return nil
	})
}

func readManifest(path string) (manifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return manifest{}, err
	}
	var value manifest
	if err := strictjson.DecodeBytes(data, &value); err != nil {
		return manifest{}, fmt.Errorf("evaluate: decode manifest: %w", err)
	}
	base, err := filepath.Abs(filepath.Dir(path))
	if err != nil {
		return manifest{}, err
	}
	if !filepath.IsAbs(value.Repository) {
		value.Repository = filepath.Join(base, value.Repository)
	}
	if !filepath.IsAbs(value.Catalog) {
		value.Catalog = filepath.Join(base, value.Catalog)
	}
	for modelIndex := range value.Models {
		model := &value.Models[modelIndex]
		if !filepath.IsAbs(model.Path) {
			model.Path = filepath.Join(base, model.Path)
		}
		for suiteIndex := range model.Suites {
			if !filepath.IsAbs(model.Suites[suiteIndex]) {
				model.Suites[suiteIndex] = filepath.Join(base, model.Suites[suiteIndex])
			}
		}
	}
	return compileManifest(value)
}

func compileManifest(value manifest) (manifest, error) {
	value.Repository = strings.TrimSpace(value.Repository)
	value.Catalog = filepath.Clean(strings.TrimSpace(value.Catalog))
	value.CodeCommit = strings.TrimSpace(value.CodeCommit)
	if value.Repository == "" || value.Catalog == "." || value.CodeCommit == "" || value.Device < 0 || len(value.Models) == 0 {
		return manifest{}, errors.New("evaluate: incomplete manifest")
	}
	models := make([]modelRequest, 0, len(value.Models))
	indexByPath := make(map[string]int, len(value.Models))
	for _, request := range value.Models {
		request.Path = filepath.Clean(strings.TrimSpace(request.Path))
		if request.Path == "." || len(request.Suites) == 0 {
			return manifest{}, errors.New("evaluate: model path or suites are absent")
		}
		request.Suites = slices.Clone(request.Suites)
		for index := range request.Suites {
			request.Suites[index] = filepath.Clean(strings.TrimSpace(request.Suites[index]))
			if request.Suites[index] == "." {
				return manifest{}, errors.New("evaluate: suite path is absent")
			}
		}
		if index, exists := indexByPath[request.Path]; exists {
			models[index].Suites = append(models[index].Suites, request.Suites...)
			continue
		}
		indexByPath[request.Path] = len(models)
		models = append(models, request)
	}
	value.Models = models
	return value, nil
}

func catalogBenchmarks(ctx context.Context, value manifest) error {
	store, err := overgodb.Open(value.Repository)
	if err != nil {
		return err
	}
	if _, err := evaluation.CatalogLocalBenchmarks(ctx, store, value.Catalog); err != nil {
		return errors.Join(err, store.Close())
	}
	return store.Close()
}

// evaluationBudget is the owner ceiling on one full evaluation pass
// (2026-09-01): a pass that cannot finish inside it must fail loudly with
// the models it never reached named, so the operator scales suites or
// hardware -- exceeding the ceiling silently is not an option, and neither
// is reporting a truncated pass as complete.
const evaluationBudget = 6 * time.Hour

func runParent(ctx context.Context, value manifest, launch workerLauncher) error {
	if ctx == nil || launch == nil || len(value.Models) == 0 {
		return errors.New("evaluate: incomplete parent execution")
	}
	deadline := time.Now().Add(evaluationBudget)
	var failures []error
	for index, model := range value.Models {
		if time.Now().After(deadline) {
			failures = append(failures, fmt.Errorf(
				"model %q: unevaluated; the %s evaluation budget elapsed", model.Path, evaluationBudget))
			continue
		}
		if err := launch(ctx, index); err != nil {
			failures = append(failures, fmt.Errorf("model %q: %w", model.Path, err))
		}
	}
	return errors.Join(failures...)
}
