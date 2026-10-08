package scorecard

import (
	_ "embed"
	"fmt"
	"strings"

	"overgo/internal/recipe"
	"overgo/internal/scorecard/metric"
	"overgo/internal/strictjson"
)

// modelUsesJSON says, for a model whose held-out reading measures something
// other than what the model is for, what it is for by its own documentation,
// what the reading measures instead, and what a measure of its use needs. A
// reading can be true and beside the point: Carbon-500M is a drafter for a
// larger model and was read on perplexity, an OCR model was read on general
// knowledge with no picture, and each stood on the scorecard as that model's
// quality with nothing to say it was not.
//
//go:embed model_uses.json
var modelUsesJSON []byte

// ModelUse is one model and task's use, as its documentation states it, set
// against what its held-out reading measures. Source says where the
// documentation states the use, or that none ships with the model. Needs says
// what a measure of the use would take: data that is the owner's to approve,
// a path that is not served, or work that is planned.
type ModelUse struct {
	Model    string      `json:"model"`
	Task     recipe.Task `json:"task"`
	Use      string      `json:"use"`
	Source   string      `json:"source"`
	Measured string      `json:"measured"`
	Needs    string      `json:"needs"`
}

// ModelUses reads the embedded uses. Each names a model and a task once and
// states all four things of it.
func ModelUses() ([]ModelUse, error) {
	var document struct {
		Uses []ModelUse `json:"uses"`
	}
	if err := strictjson.DecodeBytes(modelUsesJSON, &document); err != nil {
		return nil, fmt.Errorf("scorecard: model uses: %w", err)
	}
	seen := map[string]bool{}
	for _, entry := range document.Uses {
		key := entry.Model + "\n" + string(entry.Task)
		said := func(text string) bool { return strings.TrimSpace(text) != "" }
		if !said(entry.Model) || !entry.Task.Valid() || seen[key] || !said(entry.Use) || !said(entry.Source) || !said(entry.Measured) || !said(entry.Needs) {
			return nil, fmt.Errorf("scorecard: the use of %q %s is repeated or incomplete", entry.Model, entry.Task)
		}
		seen[key] = true
	}
	return document.Uses, nil
}

// markUses states, on each measured held-out quality whose model and task a
// use is recorded for, what the reading measures in place of that use. A cell
// that reads nothing is left as it is: its own reason says why.
func markUses(rows []Row, uses []ModelUse) {
	for index, row := range rows {
		cell := row.Cells[metric.Quality]
		if !cell.Measured() {
			continue
		}
		for _, entry := range uses {
			if entry.Model == row.Name && entry.Task == row.Task {
				cell.OutsideUse = entry.Measured
			}
		}
		rows[index].Cells[metric.Quality] = cell
	}
}
