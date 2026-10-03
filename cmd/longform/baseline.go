//overgo:runtime-inputs caller

package main

import (
	"errors"
	"fmt"
	"math"
	"os"
	"reflect"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/longform"
	"overgo/internal/modelrecipe"
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
	if !validShortRates(result.Short) {
		return errors.New("guard: short benchmark rate is absent or non-finite")
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

func validShortRates(short longform.ShortRates) bool {
	for _, rate := range []float64{short.PromptTokensPerSecond, short.DecodeTokensPerSecond} {
		if math.IsNaN(rate) || math.IsInf(rate, 0) || rate <= 0 {
			return false
		}
	}
	return true
}
