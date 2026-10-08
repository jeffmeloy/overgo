package scorecard

import (
	_ "embed"
	"fmt"
	"strings"

	"overgo/internal/recipe"
	"overgo/internal/scorecard/metric"
	"overgo/internal/strictjson"
)

// pathComparisonsJSON names, for a model and task, the test that compares the
// path a request takes with a reference output, or why none does yet. A
// held-out reading says how a served path scored; only that comparison says
// the path is the model's. LiveEdit read 12 of 64 with four faults in its
// served path, and nothing on the scorecard said that path had never met a
// reference.
//
//go:embed path_comparisons.json
var pathComparisonsJSON []byte

// PathComparison is one model and task's standing against a reference
// output: the test that compares the path a request takes with one, and what
// it holds to what; or, where no comparison is made, why, and what would
// reopen the question. A deferral is a decision recorded, not a comparison
// passed.
type PathComparison struct {
	Model    string      `json:"model"`
	Task     recipe.Task `json:"task"`
	Package  string      `json:"package,omitzero"`
	Test     string      `json:"test,omitzero"`
	Holds    string      `json:"holds,omitzero"`
	Deferred string      `json:"deferred,omitzero"`
	Reopens  string      `json:"reopens,omitzero"`
}

// PathComparisons reads the embedded comparisons. Each names a model and a
// task once, and either a test with its package and what it holds, or a
// deferral with what reopens it, never both and never neither.
func PathComparisons() ([]PathComparison, error) {
	var document struct {
		Comparisons []PathComparison `json:"comparisons"`
	}
	if err := strictjson.DecodeBytes(pathComparisonsJSON, &document); err != nil {
		return nil, fmt.Errorf("scorecard: path comparisons: %w", err)
	}
	seen := map[string]bool{}
	for _, entry := range document.Comparisons {
		key := entry.Model + "\n" + string(entry.Task)
		compared := entry.Package != "" && entry.Test != "" && entry.Holds != ""
		deferred := entry.Deferred != "" && entry.Reopens != ""
		named := entry.Package != "" || entry.Test != "" || entry.Holds != ""
		excused := entry.Deferred != "" || entry.Reopens != ""
		if strings.TrimSpace(entry.Model) == "" || !entry.Task.Valid() || seen[key] || compared == deferred || named != compared || excused != deferred {
			return nil, fmt.Errorf("scorecard: the path comparison for %q %s is repeated or incomplete", entry.Model, entry.Task)
		}
		seen[key] = true
	}
	return document.Comparisons, nil
}

// uncompared is what a held-out reading's gap says when nothing compares its
// served path with a reference.
const uncompared = "no comparison of the path a request takes with a reference output is named for this reading, so a fault in that path would read as the model's score; name the test that makes the comparison in internal/scorecard/path_comparisons.json, write it where none exists, or record there why none is made and what would reopen it"

// comparePaths states each row's standing against a reference on its
// held-out quality: the comparison named for its model and task, or the
// deferral. A training row's quality is a held-out verdict against its base
// model on the same records, which no served path produces, and states none.
func comparePaths(rows []Row) error {
	comparisons, err := PathComparisons()
	if err != nil {
		return err
	}
	for index, row := range rows {
		cell := row.Cells[metric.Quality]
		if row.Task == recipe.TaskTraining || !cell.Measured() {
			continue
		}
		for _, entry := range comparisons {
			if entry.Model == row.Name && entry.Task == row.Task {
				cell.Compared, cell.Uncompared = strings.TrimSpace(entry.Package+" "+entry.Test), entry.Deferred
			}
		}
		rows[index].Cells[metric.Quality] = cell
	}
	return nil
}

// PathUnchecked reports whether a row's held-out quality is a served path's
// reading that neither a named comparison nor a recorded deferral stands
// behind.
func (row Row) PathUnchecked() bool {
	cell := row.Cells[metric.Quality]
	return row.Task != recipe.TaskTraining && cell.Measured() && cell.Compared == "" && cell.Uncompared == ""
}
