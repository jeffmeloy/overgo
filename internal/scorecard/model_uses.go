package scorecard

import (
	_ "embed"
	"fmt"
	"slices"
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
// against what its held-out reading measures. Measure names the measure that
// reading was scored on, a suite by its name or an evaluated set by the
// source its builder gave it, and is empty where the model has no reading: a
// reading on any other measure is not spoken of. Source says where the
// documentation states the use, or that none ships with the model. Needs says
// what a measure of the use would take: data that is the owner's to approve,
// a path that is not served, or work that is planned. Protocol, where it is
// given, narrows the reading spoken of to the one asked that way: the same
// questions can be beside a model's use asked as a conversation and of its
// use put as plain text, as they are for a base language model its maker
// says is not for conversations.
type ModelUse struct {
	Model    string      `json:"model"`
	Task     recipe.Task `json:"task"`
	Measure  string      `json:"measure,omitzero"`
	Protocol string      `json:"protocol,omitzero"`
	Use      string      `json:"use"`
	Source   string      `json:"source"`
	Measured string      `json:"measured"`
	Needs    string      `json:"needs"`
}

// speaksOf reports whether the use speaks of a reading on measure asked
// under protocol: the measure it names, or a set built for it, whose name
// carries its cases after the source, and the protocol it names where it
// names one.
func (use ModelUse) speaksOf(measure, protocol string) bool {
	return use.Measure != "" && (measure == use.Measure || strings.HasPrefix(measure, use.Measure+" ")) &&
		(use.Protocol == "" || use.Protocol == protocol)
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

// markUses states, on each measured held-out quality of a model and task a
// use is recorded for, what the reading measures in place of that use, where
// it was scored on the measure the use names. A row's reading on another
// measure is left unmarked, and is the one a user reads first: where the
// headline is marked and another of the row's readings is not, the two change
// places. A cell that reads nothing is left as it is: its own reason says why.
func markUses(rows []Row, uses []ModelUse) {
	for index := range rows {
		row := &rows[index]
		at := slices.IndexFunc(uses, func(entry ModelUse) bool { return entry.Model == row.Name && entry.Task == row.Task })
		if at < 0 {
			continue
		}
		mark := func(cell Cell) Cell {
			if cell.Measured() && uses[at].speaksOf(cell.Measure, cell.Contract.Protocol) {
				cell.OutsideUse = uses[at].Measured
			}
			return cell
		}
		headline := mark(row.Cells[metric.Quality])
		for other := range row.Heldout {
			row.Heldout[other] = mark(row.Heldout[other])
		}
		ofUse := slices.IndexFunc(row.Heldout, func(cell Cell) bool { return cell.Measured() && cell.OutsideUse == "" })
		if headline.OutsideUse != "" && ofUse >= 0 {
			headline, row.Heldout[ofUse] = row.Heldout[ofUse], headline
		}
		row.Cells[metric.Quality] = headline
	}
}
