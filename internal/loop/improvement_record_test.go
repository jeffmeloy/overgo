package loop

import (
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/scorecard"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestTheSmallestModelsImprovementRecordIsComplete reads the record of the run that took a training gap from the scorecard
// to a judged and measured candidate with no person, and holds it to being whole. Its gap is the row the declared target
// files for its model, and its command is the row's own. Its verdict names a checkpoint, a session and a run of measured
// cells, and its figures agree with each other: the gain is the trained reading less the base's, supported only clear of
// what its cases can tell from noise, and the gap is closed only by a supported gain of the target. A record of a supported
// loss would be as complete: the test holds the record to the rule, not to a good result.
func TestTheSmallestModelsImprovementRecordIsComplete(t *testing.T) {
	t.Parallel()
	raw, err := os.ReadFile(filepath.Join(testutil.RepoRoot(t), "docs", "verification", "improvement-run-qwen25-05b.txt"))
	testcheck.NoError(t, err)
	fields := map[string]string{}
	for line := range strings.Lines(string(raw)) {
		if key, value, found := strings.Cut(strings.TrimSpace(line), ": "); found && !strings.Contains(key, " ") {
			fields[key] = value
		}
	}
	identity := func(key string, kind artifact.Kind) artifact.ID {
		t.Helper()
		id, err := artifact.ParseID(fields[key])
		if err != nil || id.Kind() != kind {
			t.Fatalf("the record's %s is %q (%v), want a %s", key, fields[key], err, kind)
		}
		return id
	}
	number := func(key string) float64 {
		t.Helper()
		value, err := strconv.ParseFloat(fields[key], 64)
		if err != nil {
			t.Fatalf("the record's %s: %v", key, err)
		}
		return value
	}
	model := identity("model", artifact.KindModel)
	identity("observation", artifact.KindEvidence)
	identity("checkpoint", artifact.KindCheckpoint)
	identity("verdict", artifact.KindEvaluation)
	gaps := scorecard.Record{Rows: []scorecard.Row{{Model: model, Task: recipe.TaskTraining}}}.Gaps(&scorecard.Record{})
	at := slices.IndexFunc(gaps, func(gap scorecard.Gap) bool { return gap.Key == fields["gap"] })
	if at < 0 {
		t.Fatalf("no declared target files %q for the record's model: it files %+v", fields["gap"], gaps)
	}
	_, flags, found := strings.Cut(strings.SplitN(gaps[at].Detail, "\n", 2)[0], "<a new directory> ")
	if !found || !strings.HasSuffix(fields["command"], flags) {
		t.Fatalf("the record ran %q, and the row's command ends %q", fields["command"], flags)
	}
	const tolerance = 1e-9
	base, trained, gain, detectable := number("base"), number("trained"), number("gain"), number("detectable")
	if difference := trained - base - gain; difference > tolerance || difference < -tolerance {
		t.Fatalf("the record gains %g from %g to %g", gain, base, trained)
	}
	if supported := fields["supported"] == "true"; supported != (gain > detectable) || number("cases") < 2 {
		t.Fatalf("the record says supported %q of a gain of %g over %g detectable on %q cases", fields["supported"], gain, detectable, fields["cases"])
	}
	closed := fields["supported"] == "true" && gain >= number("target-gain")
	if closed != (fields["gap-after"] == "closed") {
		t.Fatalf("the record says the gap is %q after a gain of %g, supported %q, on a target of %q", fields["gap-after"], gain, fields["supported"], fields["target-gain"])
	}
	if number("measured-cells") < 1 || fields["measured-again"] != "nothing to re-acquire" {
		t.Fatalf("the record measured %q cells, and a second run %q: want a measurement, once", fields["measured-cells"], fields["measured-again"])
	}
}
