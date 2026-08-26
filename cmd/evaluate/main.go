package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/strictjson"
)

type manifest struct {
	Repository string         `json:"repository"`
	Catalog    string         `json:"catalog"`
	CodeCommit string         `json:"code_commit"`
	Device     int            `json:"device"`
	Models     []modelRequest `json:"models"`
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
	importCache := flag.String("import-hf-cache", "", "scan a HuggingFace dataset cache root, import every recognized benchmark, and publish the active catalog")
	repository := flag.String("repo", "overgodb-store", "OvergoDB root for -import-hf-cache")
	flag.Parse()
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
		command := exec.CommandContext(
			ctx, executable, "-worker", "-manifest", absoluteManifest, "-model-index", strconv.Itoa(index),
		)
		command.Stdout, command.Stderr = os.Stdout, os.Stderr
		return command.Run()
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

func runParent(ctx context.Context, value manifest, launch workerLauncher) error {
	if ctx == nil || launch == nil || len(value.Models) == 0 {
		return errors.New("evaluate: incomplete parent execution")
	}
	var failures []error
	for index, model := range value.Models {
		if err := launch(ctx, index); err != nil {
			failures = append(failures, fmt.Errorf("model %q: %w", model.Path, err))
		}
	}
	return errors.Join(failures...)
}
