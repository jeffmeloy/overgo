package main

import (
	"bytes"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestScorecardReadingStatesItsStanding holds what the printed scorecard says
// of the path behind each held-out reading. A reading whose model and task the
// embedded comparisons name is listed with the test that compares its served
// path with a reference output and what that test holds; a served reading
// nothing compares is listed as one a fault in the path would pass for the
// model's score; a training reading, a verdict against a base model, is in
// neither list. A scorecard with no held-out reading says nothing of paths.
func TestScorecardReadingStatesItsStanding(t *testing.T) {
	t.Parallel()
	comparisons, err := scorecard.PathComparisons()
	if err != nil {
		t.Fatal(err)
	}
	at := slices.IndexFunc(comparisons, func(entry scorecard.PathComparison) bool { return entry.Test != "" })
	if at < 0 {
		t.Fatal("the embedded comparisons name no test")
	}
	named := comparisons[at]
	row := func(name string, task recipe.Task, compared string) scorecard.Row {
		return scorecard.Row{Name: name, Task: task, Cells: map[metric.Name]scorecard.Cell{metric.Quality: {
			Value: 0.25, Label: "accuracy, set (protocol)", Source: testutil.ArtifactID(t, artifact.KindEvaluation, name), Compared: compared,
		}}}
	}
	text, err := pathStanding([]scorecard.Row{
		row(named.Model, named.Task, named.Package+" "+named.Test),
		row("an unchecked model", recipe.TaskImageGen, ""),
		row("a trained model", recipe.TaskTraining, ""),
	})
	if err != nil {
		t.Fatal(err)
	}
	compared := "- " + named.Model + " " + string(named.Task) + ": " + named.Package + " " + named.Test + " holds " + named.Holds
	before, after, found := strings.Cut(text, "no comparison with a reference output is named")
	if !found || !strings.Contains(before, compared) || !strings.Contains(after, "- an unchecked model image-gen\n") || strings.Contains(before, "an unchecked model") || strings.Contains(text, "a trained model") {
		t.Fatalf("the standing reads:\n%s\nwant %q under the compared readings, the unchecked model under the readings no comparison is named for, and the trained model under neither", text, compared)
	}
	var output bytes.Buffer
	if err := runArgs(t.Context(), []string{"-scorecard", "-repo", scorecardFixture(t)}, &output); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(output.String(), "served path") {
		t.Fatalf("a scorecard with no held-out reading speaks of served paths:\n%s", output.String())
	}
}
