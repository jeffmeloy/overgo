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

// servedModelsJSON gives the words the README's served-model table names each row and measure by: what the record
// holds is figures and identities, and a reader reads a model by its published name, the kind of data it works in and
// what a request gives it and gets back, and a measure by the benchmark's name.
//
//go:embed served_models.json
var servedModelsJSON []byte

// ServedModel is the words of one row name of the record: the name the table shows, its modality and its input and
// output.
type ServedModel struct {
	Row         string `json:"row"`
	Shown       string `json:"shown"`
	Modality    string `json:"modality"`
	InputOutput string `json:"input_output"`
}

// MeasureWords names one measure a held-out quality is scored on (Cell.Measure, less the cases its set was built over).
type MeasureWords struct {
	Measure string `json:"measure"`
	Words   string `json:"words"`
}

// ServedWords is the declaration the served-model table is written by: its models in the order the table lists them,
// and its measures.
type ServedWords struct {
	Models   []ServedModel  `json:"models"`
	Measures []MeasureWords `json:"measures"`
}

// DeclaredServedWords reads the embedded declaration. Each row and each measure is named once, with every word said.
func DeclaredServedWords() (ServedWords, error) {
	var words ServedWords
	if err := strictjson.DecodeBytes(servedModelsJSON, &words); err != nil {
		return ServedWords{}, fmt.Errorf("scorecard: served models: %w", err)
	}
	said := func(text string) bool { return strings.TrimSpace(text) != "" }
	rows, measures := map[string]bool{}, map[string]bool{}
	for _, model := range words.Models {
		if rows[model.Row] || !said(model.Row) || !said(model.Shown) || !said(model.Modality) || !said(model.InputOutput) {
			return ServedWords{}, fmt.Errorf("scorecard: served model %q is repeated or incomplete", model.Row)
		}
		rows[model.Row] = true
	}
	for _, measure := range words.Measures {
		if measures[measure.Measure] || !said(measure.Measure) || !said(measure.Words) {
			return ServedWords{}, fmt.Errorf("scorecard: measure %q is repeated or unnamed", measure.Measure)
		}
		measures[measure.Measure] = true
	}
	return words, nil
}

// MeasureOf is the words of the measure a held-out quality was scored on, and false where none are declared. A set the
// evaluation built carries its cases after its source (Cell.Measure); its words are its source's.
func (words ServedWords) MeasureOf(cell Cell) (string, bool) {
	source, _, _ := strings.Cut(cell.Measure, " ")
	at := slices.IndexFunc(words.Measures, func(measure MeasureWords) bool { return measure.Measure == source })
	if at < 0 {
		return "", false
	}
	return words.Measures[at].Words, true
}

// gibibyte is the bytes of the unit peak device memory is shown in.
const gibibyte = 1 << 30

// ServedTable is the README's table of the models the record serves, in the order words declares them: a line for each
// declared name some row of which serves a task (any row but a training one, and not stale), with its modality and
// input and output, each measured held-out quality of its rows in a reader's words (Cell.Brief), and the most device
// memory a measured attempt of its rows held, "host" where it held none. A name with no quality or memory measured in
// this record shows unreadSpeed there, or says no held-out measure applies where none does. A served row whose name or
// measure words do not declare is refused, so no row is left out of the table and none is shown by an identity.
func (record Record) ServedTable(words ServedWords) (string, error) {
	served := map[string][]Row{}
	var undeclared []string
	for _, row := range record.Rows {
		if row.Task == recipe.TaskTraining || row.Stale != "" {
			continue
		}
		if !slices.ContainsFunc(words.Models, func(model ServedModel) bool { return model.Row == row.Name }) && !slices.Contains(undeclared, row.Name) {
			undeclared = append(undeclared, row.Name)
		}
		served[row.Name] = append(served[row.Name], row)
	}
	if len(undeclared) != 0 {
		return "", fmt.Errorf("scorecard: %s serves rows %s that internal/scorecard/served_models.json names no words for", record.Name, strings.Join(undeclared, ", "))
	}
	var table strings.Builder
	table.WriteString("| Model | Modality | Input → output | Held-out quality | Peak device memory |\n| --- | --- | --- | --- | --- |\n")
	for _, model := range words.Models {
		rows := served[model.Row]
		if len(rows) == 0 {
			continue
		}
		var qualities []string
		applies, measured := false, false
		var peak float64
		for _, row := range rows {
			quality := row.Cells[metric.Quality]
			applies = applies || !quality.NotApplicable
			if quality.Measured() {
				measure, named := words.MeasureOf(quality)
				if !named {
					return "", fmt.Errorf("scorecard: %s's %s quality is scored on %q, which internal/scorecard/served_models.json names no words for", row.Name, row.Task, quality.Measure)
				}
				qualities = append(qualities, quality.Brief(measure))
			}
			if memory := row.Cells[metric.PeakMemory]; memory.Measured() && (!measured || memory.Value > peak) {
				peak, measured = memory.Value, true
			}
		}
		quality, memory := strings.Join(qualities, "; "), unreadSpeed
		switch {
		case quality == "" && !applies:
			quality = "no held-out measure applies"
		case quality == "":
			quality = unreadSpeed
		}
		switch {
		case measured && peak == 0:
			memory = "host"
		case measured:
			memory = fmt.Sprintf("%.1f GiB", peak/gibibyte)
		}
		fmt.Fprintf(&table, "| %s | %s | %s | %s | %s |\n", model.Shown, model.Modality, model.InputOutput, quality, memory)
	}
	return table.String(), nil
}

// Brief is the cell's held-out quality as the README reads it, named by measure: its value and standard error, its
// base's value where a verdict judged it against one, the cases it rests on, the answers it counted wrong because they
// named no option and why, where it does not clear what a constant answer scores, and that it is not a measure of what
// the model is for where it is not.
func (c Cell) Brief(measure string) string {
	text := fmt.Sprintf("%s %.3g", measure, c.Value)
	if c.Noise > 0 {
		text += fmt.Sprintf(" ±%.2g", c.Noise)
	}
	if c.Base != nil {
		text += fmt.Sprintf(" against its base's %.3g", *c.Base)
	}
	if c.Cases > 0 {
		text += " on " + thousands(int(c.Cases)) + " cases"
	}
	if c.Unread > 0 {
		text += fmt.Sprintf(", %d named no option", c.Unread)
		if causes := c.UnreadCauses.String(); causes != "" {
			text += " (" + causes + ")"
		}
	}
	if c.Developed > 0 {
		text += fmt.Sprintf(", %d of them development evidence", c.Developed)
	}
	switch standing, has := c.againstConstant(); {
	case has && standing == noBetter:
		text += ", no better than a constant answer"
	case has && standing == withinNoise:
		text += ", within its noise of a constant answer"
	}
	if c.OutsideUse != "" {
		text += ", not a measure of what the model is for"
	}
	return text
}

// constantStanding is how a held-out quality stands to what a constant answer scores on its measure.
type constantStanding int

const (
	// beatsConstant is a reading better than a constant answer's by more than its noise.
	beatsConstant constantStanding = iota
	// withinNoise is a reading better than a constant answer's by less than its own noise, which tells nothing from it.
	withinNoise
	// noBetter is a reading no better than a constant answer's.
	noBetter
)

// againstConstant is how the cell stands to Constant, and false where its measure has no constant answer.
func (c Cell) againstConstant() (constantStanding, bool) {
	if c.Constant == nil {
		return beatsConstant, false
	}
	advantage, directed := c.Direction.Advantage(c.Value, *c.Constant)
	switch {
	case !worse(c.Direction, *c.Constant, c.Value, 0):
		return noBetter, true
	case directed && advantage < noiseDeviates*c.Noise:
		return withinNoise, true
	}
	return beatsConstant, true
}
