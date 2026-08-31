package main

import (
	"encoding/json"
	"errors"
	"flag"
	"os"
	"strings"

	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
)

// attributeEvaluation judges one controlled recipe comparison: the typed
// attribution prints when the delta genuinely attributes to recipe behavior,
// and a confounded comparison exits nonzero with the exact refusal.
func attributeEvaluation(arguments []string) error {
	flags := flag.NewFlagSet("recipe attribute", flag.ContinueOnError)
	input := flags.String("input", "", "comparison JSON path ({candidate, baseline})")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if strings.TrimSpace(*input) == "" || flags.NArg() != 0 {
		return errors.New("usage: recipe attribute -input <comparison.json>")
	}
	var comparison struct {
		Candidate modelrecipe.RecipeEvaluationSide `json:"candidate"`
		Baseline  modelrecipe.RecipeEvaluationSide `json:"baseline"`
	}
	if err := jsonfile.DecodeStrict(*input, &comparison); err != nil {
		return err
	}
	attribution, err := modelrecipe.AttributeRecipeEvaluation(comparison.Candidate, comparison.Baseline)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetEscapeHTML(false)
	return encoder.Encode(attribution)
}
