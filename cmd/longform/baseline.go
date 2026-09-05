package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"reflect"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/longform"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
)

func readCorpus(options options, limit int) (string, error) {
	if options.corpusText != "" {
		return options.corpusText, nil
	}
	if options.Corpus == "" {
		return longform.Corpus(options.Root, limit)
	}
	data, err := os.ReadFile(options.Corpus)
	if err != nil {
		return "", err
	}
	if len(data) == 0 {
		return "", errors.New("longform: fixed corpus is empty")
	}
	return string(data), nil
}

func bindBaselines(ctx context.Context, options options, targets []target) error {
	selected, err := readBaselines(ctx, options)
	if err != nil {
		return err
	}
	return bindSelectedBaselines(options, targets, selected)
}

func readBaselines(ctx context.Context, options options) (map[artifact.ID]longform.Summary, error) {
	corpus, err := readCorpus(options, 0)
	if err != nil {
		return nil, err
	}
	digest := sha256.Sum256([]byte(corpus))
	store, err := overgodb.OpenReadOnly(options.Repository)
	if err != nil {
		return nil, err
	}
	defer store.Close()
	selected := make(map[artifact.ID]longform.Summary)
	for _, text := range options.Baselines {
		id, err := artifact.ParseID(text)
		if err != nil {
			return nil, err
		}
		baseline, err := longform.ReadBaseline(ctx, store, id)
		if err != nil {
			return nil, err
		}
		if baseline.Result.Inputs.CorpusDigest != hex.EncodeToString(digest[:]) || baseline.Result.Floors != longform.DeclaredFloors() {
			return nil, errors.New("longform: baseline corpus or floors differ from this experiment")
		}
		model := baseline.Result.Inputs.Model
		if _, duplicate := selected[model]; duplicate {
			return nil, fmt.Errorf("longform: multiple baselines for %s", model)
		}
		selected[model] = baseline
	}
	return selected, nil
}

func bindSelectedBaselines(options options, targets []target, selected map[artifact.ID]longform.Summary) error {
	for index := range targets {
		baseline, found := selected[targets[index].weights]
		if !found {
			return fmt.Errorf("longform: no explicit baseline for %s", targets[index].entry.Location)
		}
		if options.Guard || options.ValidateBaselines {
			if err := validateGuard(baseline.Result); err != nil {
				return fmt.Errorf("longform: baseline %s: %w", baseline.Record, err)
			}
			if baseline.Result.Program.Recipe != targets[index].entry.Recipe || baseline.Result.Program.Model != targets[index].entry.Model {
				return errors.New("longform: baseline recipe or serving model differs from the selected active binding")
			}
		}
		targets[index].record = baseline
		delete(selected, targets[index].weights)
	}
	if len(selected) != 0 {
		return errors.New("longform: baseline supplied for an unselected model")
	}
	return nil
}

// validateGuard checks the measured denominator, not merely a stored PASS bit.
// A legacy admission record cannot silently become a memory regression guard.
func validateGuard(result longform.Result) error {
	floors := longform.DeclaredFloors()
	if result.Inputs.Protocol != longform.GuardContinuation {
		return errors.New("guard: requires the explicit fixed-budget generation protocol")
	}
	if result.Floors != floors || !result.Verdict.Passed || len(result.Verdict.Reasons) != 0 {
		return errors.New("guard: failed verdict or changed declared floors")
	}
	program := result.Program
	if !program.Model.Valid() || program.Model.Kind() != artifact.KindModel || !program.Profile.Valid() || program.Profile.Kind() != artifact.KindProfile ||
		!program.Definition.Valid() || program.Definition.Kind() != artifact.KindModelDefinition || program.Recipe.Kind() != artifact.KindRecipe ||
		!program.Recipe.Valid() || program.RecipeVersion == 0 || program.Placement == "" || program.Residency == "" || program.Runtime != modelrecipe.RuntimeInference ||
		result.ContextLength == 0 || result.Device.Name == "" || result.Device.TotalMemoryBytes == 0 {
		return errors.New("guard: missing recipe, execution, device or context identity")
	}
	if result.BudgetNS <= 0 || result.WallNS <= 0 || result.WallNS > result.BudgetNS || result.ModelBudgetNS < 0 ||
		result.ModelBudgetNS > 0 && result.WallNS > result.ModelBudgetNS {
		return errors.New("guard: measurement lacks a completed bounded execution")
	}
	if result.Shape.PromptTokens != floors.ShortPromptTokens || result.Shape.Measure.PromptTokens != floors.ShortPromptTokens ||
		math.IsNaN(result.Shape.NLL) || math.IsInf(result.Shape.NLL, 0) || result.Shape.NLL < 0 {
		return errors.New("guard: short prompt or quality evidence is incomplete")
	}
	for _, rate := range []float64{result.Short.PromptTokensPerSecond, result.Short.DecodeTokensPerSecond} {
		if math.IsNaN(rate) || math.IsInf(rate, 0) || rate <= 0 {
			return errors.New("guard: short benchmark rate is absent or non-finite")
		}
	}
	required := longform.LadderRungs(result.ContextLength, int(result.ContextLength), floors, floors.CheckRungCeiling)
	if !slices.Contains(required, floors.PromptTokens) {
		return errors.New("guard: declared context does not cover the required judged rung")
	}
	climbed := make(map[int]bool)
	shapes := append([]longform.Rung{{Measure: result.Shape.Measure, OutputIDs: result.Shape.OutputIDs}}, result.Rungs...)
	var previousPeak uint64
	for index, shape := range shapes {
		measure := shape.Measure
		expectedOutput := floors.OutputTokens
		if index == 0 {
			expectedOutput = floors.ShortOutputTokens
		}
		if measure.OutputTokens != len(shape.OutputIDs) || measure.OutputTokens != expectedOutput || measure.StoppedEarly ||
			len(shape.OutputIDs) < max(floors.IdenticalTokens, floors.MinimumOutputTokens) {
			return fmt.Errorf("guard: prompt %d lacks its declared output fingerprint", measure.PromptTokens)
		}
		memory := measure.Memory
		if memory.CurrentBytes == 0 || memory.PeakBytes < memory.CurrentBytes || memory.PeakBytes < previousPeak || memory.Allocations == 0 ||
			memory.PeakBytes > result.Device.TotalMemoryBytes {
			return fmt.Errorf("guard: prompt %d has absent, inconsistent or over-capacity allocation evidence", measure.PromptTokens)
		}
		previousPeak = memory.PeakBytes
		if index == 0 {
			continue
		}
		if climbed[measure.PromptTokens] || measure.Score.ScoreTokens != floors.ScoreTokens {
			return fmt.Errorf("guard: prompt %d is duplicated or lacks the required score denominator", measure.PromptTokens)
		}
		for _, nll := range []float64{measure.Score.LongContextNLL, measure.Score.ShortContextNLL} {
			if math.IsNaN(nll) || math.IsInf(nll, 0) || nll < 0 {
				return fmt.Errorf("guard: prompt %d has invalid quality evidence", measure.PromptTokens)
			}
		}
		if measure.Score.ContextGain != measure.Score.ShortContextNLL-measure.Score.LongContextNLL {
			return fmt.Errorf("guard: prompt %d has inconsistent context gain", measure.PromptTokens)
		}
		climbed[measure.PromptTokens] = true
		if measure.PromptTokens == floors.PromptTokens && !reflect.DeepEqual(measure, result.Measure) {
			return errors.New("guard: judged measure differs from its recorded rung")
		}
	}
	for _, length := range required {
		if !climbed[length] {
			return fmt.Errorf("guard: required rung %d is missing (%s)", length, result.LadderStop)
		}
	}
	if verdict := longform.Judge(result.Measure, result.Short, floors); !verdict.Passed {
		return fmt.Errorf("guard: measured quality or rate floors fail: %s", verdict)
	}
	if verdict := longform.Compare(result, result, floors, floors.CheckRungCeiling); !verdict.Passed {
		return fmt.Errorf("guard: incomplete comparison evidence: %s", verdict)
	}
	return nil
}

func validateSelectedBaselines(output io.Writer, targets []target, surface string) error {
	if len(targets) == 0 || surface == "" {
		return errors.New("longform: guard acceptance needs selected records and a source surface")
	}
	rungs := 0
	for _, target := range targets {
		baseline := target.record
		if !baseline.Record.Valid() || baseline.Result.Surface != surface {
			return fmt.Errorf("longform: guard record %s does not cover the current source surface", baseline.Record)
		}
		if err := validateGuard(baseline.Result); err != nil {
			return err
		}
		rungs += len(baseline.Result.Rungs)
		fmt.Fprintf(output, "validated guard: model=%s recipe=%s record=%s short=1 rungs=%d\n", baseline.Result.Inputs.Model, baseline.Result.Program.Recipe, baseline.Record, len(baseline.Result.Rungs))
	}
	fmt.Fprintf(output, "guard audit: selected_models=%d explicit_records=%d short_shapes=%d rungs=%d; quality/rate floors and owned peak/retained allocation evidence checked; read-only, no models loaded or measured, no GPU tests or benchmark suites run\n", len(targets), len(targets), len(targets), rungs)
	return nil
}
