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

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/repodb"
	"overgo/internal/strictjson"
	"overgo/internal/trainingprogram"
)

type manifest struct {
	Repository string           `json:"repository"`
	Catalog    string           `json:"catalog"`
	CodeCommit string           `json:"code_commit"`
	Device     int              `json:"device"`
	Models     []modelRequest   `json:"models"`
	SFTViews   []sftViewRequest `json:"sft_views,omitempty"`
}

type modelRequest struct {
	Path   string   `json:"path"`
	Suites []string `json:"suites"`
}

type sftViewRequest struct {
	Objective           artifact.ID                           `json:"objective"`
	Training            artifact.ID                           `json:"training_membership"`
	Heldout             artifact.ID                           `json:"heldout_membership"`
	TextTargets         *evaluation.TextTargetSuite           `json:"text_targets,omitempty"`
	TextObservations    []evaluation.TextTargetObservation    `json:"text_observations,omitempty"`
	NumericTargets      *evaluation.NumericTargetSuite        `json:"numeric_targets,omitempty"`
	NumericObservations []evaluation.NumericTargetObservation `json:"numeric_observations,omitempty"`
	MediaTargets        *evaluation.MediaTargetSuite          `json:"media_targets,omitempty"`
	MediaObservations   []evaluation.MediaTargetObservation   `json:"media_observations,omitempty"`
}

type workerLauncher func(context.Context, int) error

type batchDocument interface {
	Batch(string) (artifact.Batch, error)
}

func main() {
	clioptions.Main(run)
}

func run() error {
	manifestPath := flag.String("manifest", "", "evaluation manifest")
	worker := flag.Bool("worker", false, "run one model worker")
	modelIndex := flag.Int("model-index", -1, "worker model index")
	flag.Parse()
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
	for _, view := range value.SFTViews {
		if view.Objective.Kind() != artifact.KindProfile || view.Training.Kind() != artifact.KindDatasetShard ||
			view.Heldout.Kind() != artifact.KindDatasetShard || view.Training == view.Heldout ||
			(view.TextTargets == nil) != (len(view.TextObservations) == 0) ||
			(view.NumericTargets == nil) != (len(view.NumericObservations) == 0) ||
			(view.MediaTargets == nil) != (len(view.MediaObservations) == 0) {
			return manifest{}, errors.New("evaluate: invalid SFT evaluation view")
		}
	}
	return value, nil
}

func catalogBenchmarks(ctx context.Context, value manifest) error {
	store, err := repodb.Open(value.Repository)
	if err != nil {
		return err
	}
	if _, err := evaluation.CatalogLocalBenchmarks(ctx, store, value.Catalog); err != nil {
		return errors.Join(err, store.Close())
	}
	for _, request := range value.SFTViews {
		objective, err := trainingprogram.LoadObjective(ctx, store, request.Objective)
		if err != nil {
			return errors.Join(err, store.Close())
		}
		training, ok, err := dataset.LoadMembership(ctx, store, request.Training)
		if err != nil || !ok {
			return errors.Join(err, errors.New("evaluate: training membership is absent"), store.Close())
		}
		heldout, ok, err := dataset.LoadMembership(ctx, store, request.Heldout)
		if err != nil || !ok {
			return errors.Join(err, errors.New("evaluate: held-out membership is absent"), store.Close())
		}
		view, err := evaluation.CompileSFTEvaluationView(objective, training, heldout)
		if err != nil {
			return errors.Join(err, store.Close())
		}
		if err := publishDocument(ctx, store, "evaluation/sft-view/"+view.ID.String(), view); err != nil {
			return errors.Join(err, store.Close())
		}
		if request.TextTargets != nil {
			plan, err := evaluation.CompileTextTargetPlan(view, *request.TextTargets)
			if err != nil {
				return errors.Join(err, store.Close())
			}
			if err := publishDocument(ctx, store, "evaluation/text-target/"+plan.ID.String(), plan); err != nil {
				return errors.Join(err, store.Close())
			}
			report, err := evaluation.ScoreTextTargets(plan, request.TextObservations)
			if err != nil {
				return errors.Join(err, store.Close())
			}
			if err := publishDocument(ctx, store, "evaluation/text-target-report/"+report.ID.String(), report); err != nil {
				return errors.Join(err, store.Close())
			}
		}
		if request.NumericTargets != nil {
			plan, err := evaluation.CompileNumericTargetPlan(view, *request.NumericTargets)
			if err != nil {
				return errors.Join(err, store.Close())
			}
			if err := publishDocument(ctx, store, "evaluation/numeric-target/"+plan.ID.String(), plan); err != nil {
				return errors.Join(err, store.Close())
			}
			report, err := evaluation.ScoreNumericTargets(plan, request.NumericObservations)
			if err != nil {
				return errors.Join(err, store.Close())
			}
			if err := publishDocument(ctx, store, "evaluation/numeric-target-report/"+report.ID.String(), report); err != nil {
				return errors.Join(err, store.Close())
			}
		}
		if request.MediaTargets != nil {
			plan, err := evaluation.CompileMediaTargetPlan(view, *request.MediaTargets)
			if err != nil {
				return errors.Join(err, store.Close())
			}
			if err := publishDocument(ctx, store, "evaluation/media-target/"+plan.ID.String(), plan); err != nil {
				return errors.Join(err, store.Close())
			}
			report, err := evaluation.ScoreMediaTargets(plan, request.MediaObservations)
			if err != nil {
				return errors.Join(err, store.Close())
			}
			if err := publishDocument(ctx, store, "evaluation/media-target-report/"+report.ID.String(), report); err != nil {
				return errors.Join(err, store.Close())
			}
		}
	}
	return store.Close()
}

func publishDocument(ctx context.Context, repository artifact.Repository, key string, document batchDocument) error {
	batch, err := document.Batch(key)
	if err != nil {
		return err
	}
	_, err = artifact.CommitBatch(ctx, repository, batch)
	return err
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
