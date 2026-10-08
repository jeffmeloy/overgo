package scorecard

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestScorecardReadingStatesItsStanding holds what the scorecard says of the
// path behind a held-out reading. Every comparison the embedded document
// names is a test that exists in the package it names, so the document cannot
// claim a comparison nothing makes. A served reading whose model and task the
// document names carries that test; one it does not name is unchecked, and
// files the row that asks for the comparison, by its task. A reading with a
// recorded deferral files none, and neither does a training reading, which is
// a verdict against a base model and not a served path's, nor a cell no
// record measures.
func TestScorecardReadingStatesItsStanding(t *testing.T) {
	t.Parallel()
	comparisons, err := PathComparisons()
	if err != nil || len(comparisons) == 0 {
		t.Fatalf("the embedded path comparisons read %d entries: %v", len(comparisons), err)
	}
	root := testutil.RepoRoot(t)
	for _, entry := range comparisons {
		if entry.Test == "" {
			continue
		}
		files, err := filepath.Glob(filepath.Join(root, filepath.FromSlash(entry.Package), "*_test.go"))
		if err != nil {
			t.Fatal(err)
		}
		if !slices.ContainsFunc(files, func(file string) bool {
			source, err := os.ReadFile(file)
			return err == nil && strings.Contains(string(source), "\nfunc "+entry.Test+"(t *testing.T)")
		}) {
			t.Errorf("%s %s is compared by %s in %s, and no test file of that package declares it", entry.Model, entry.Task, entry.Test, entry.Package)
		}
	}

	measured := Cell{Value: 0.25, Label: "accuracy, set (protocol)", Source: testutil.ArtifactID(t, artifact.KindEvaluation, "reading")}
	row := func(name string, task recipe.Task, quality Cell) Row {
		cells := map[metric.Name]Cell{}
		for _, column := range metric.Columns {
			cells[column] = inapplicable("not measured in this fixture")
		}
		cells[metric.Quality] = quality
		return Row{Model: testutil.ArtifactID(t, artifact.KindModel, name+string(task)), Name: name, Task: task, Cells: cells}
	}
	named := comparisons[slices.IndexFunc(comparisons, func(entry PathComparison) bool { return entry.Test != "" })]
	deferred := measured
	deferred.Uncompared = "its reference is not on this machine"
	rows := []Row{
		row(named.Model, named.Task, measured),
		row("a model the document does not name", named.Task, measured),
		row(named.Model, recipe.TaskTraining, measured),
		row("an unmeasured model", named.Task, gap("no held-out evaluation of this recipe")),
		row("a deferred model", recipe.TaskImageGen, deferred),
		row("another unnamed model", recipe.TaskImageGen, measured),
	}
	if err := comparePaths(rows); err != nil {
		t.Fatal(err)
	}
	if got, want := rows[0].Cells[metric.Quality].Compared, named.Package+" "+named.Test; got != want || rows[0].PathUnchecked() {
		t.Fatalf("the named model's reading is compared by %q, want %q", got, want)
	}
	var unchecked []string
	for _, row := range rows {
		if row.PathUnchecked() {
			unchecked = append(unchecked, row.Name)
		}
	}
	if !slices.Equal(unchecked, []string{"a model the document does not name", "another unnamed model"}) {
		t.Fatalf("unchecked readings are %q, want the two served readings the document does not name", unchecked)
	}
	if rows[2].Cells[metric.Quality].Compared != "" {
		t.Fatal("a training reading was given a served path's comparison")
	}
	var filed []string
	for _, gap := range (Record{Commit: "b", Rows: rows}).Gaps(&Record{Commit: "a"}) {
		if strings.Contains(gap.Key, "-"+gapUncompared+"-") {
			if !KnownGapKey(gap.Key) || !strings.Contains(gap.Detail, " held-out-quality: "+uncompared) {
				t.Fatalf("gap %+v is not one the scorecard can verify, or does not say what it asks", gap)
			}
			filed = append(filed, gap.Key+": "+strings.Join(strings.Fields(strings.ReplaceAll(gap.Detail, " held-out-quality: "+uncompared, "")), " "))
		}
	}
	want := []string{
		gapKey(gapUncompared, recipe.TaskImageGen) + ": - another unnamed model",
		gapKey(gapUncompared, named.Task) + ": - a model the document does not name",
	}
	slices.Sort(want)
	if !slices.Equal(filed, want) {
		t.Fatalf("the unchecked readings file %q, want %q", filed, want)
	}
}
