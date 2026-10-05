package evaluation

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// TestScoredCommitIsUnknownForAnUnreadableReceipt holds the commit a carried quality names as
// the one that scored it to a chain that can be read all the way: a run that reports a replay
// receipt the store cannot read is not a run that scored its own cases, so the chain through
// it names no commit, where it used to stop at that run and name its own; an intact chain still
// names the first run, and a run that scored its own cases names itself.
func TestScoredCommitIsUnknownForAnUnreadableReceipt(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	platform, err := runrecord.NewEnvironment(runrecord.Environment{Host: "fixture", OS: "windows", Arch: "amd64", Device: "cpu", Backend: "host", Driver: "fixture"})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := platform.Batch("scored/environment")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	recipe, dataset := planID(t, artifact.KindRecipe, "scored commit recipe"), planID(t, artifact.KindDataset, "scored commit dataset")

	// evaluated commits a run at the commit that reports output, a report of its own when it is
	// zero, and the evaluation of it.
	plan := planID(t, artifact.KindProfile, "scored commit plan")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "scored/plan", Artifacts: []artifact.Descriptor{{ID: plan}, {ID: recipe}, {ID: dataset}}}); err != nil {
		t.Fatal(err)
	}
	evaluated := func(commit string, output artifact.ID) artifact.ID {
		t.Helper()
		if !output.Valid() {
			output = planID(t, artifact.KindEvaluation, commit+" own report")
			if _, err := store.Commit(ctx, artifact.Batch{Key: "scored/report/" + commit, Artifacts: []artifact.Descriptor{{ID: output}}}); err != nil {
				t.Fatal(err)
			}
		}
		run, err := runrecord.NewBoundRun(recipe, runrecord.OutcomeSucceeded, []artifact.ID{plan}, []artifact.ID{output}, "", strings.Repeat(commit, 40), platform.ID, 1, nil)
		if err != nil {
			t.Fatal(err)
		}
		content, err := run.Content()
		if err != nil {
			t.Fatal(err)
		}
		batch, err := artifact.NewDocumentBatch("scored/run/"+commit, []artifact.Content{content}, run.Lineage(), nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
		record, err := runrecord.NewEvaluation(recipe, run.ID, dataset, []runrecord.Metric{{Name: "accuracy", Value: 0.5, Direction: runrecord.DirectionMaximize}})
		if err != nil {
			t.Fatal(err)
		}
		if batch, err = record.Batch("scored/evaluation/" + commit); err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
		return record.ID
	}
	// A readable receipt carries the record it names, and an unreadable one is a descriptor of the
	// receipt schema whose content the store does not hold.
	readable := func(label string, carried artifact.ID) artifact.ID {
		t.Helper()
		document, err := replayReceiptCodec.NewInitial(ReplayReceipt{
			Version: artifact.InitialDocumentVersion, Plan: plan, Carried: carried,
			Report: planID(t, artifact.KindEvaluation, label+" report"), Replay: planID(t, artifact.KindEvaluation, label+" replay"), Cases: []int{0, 3},
		})
		if err != nil {
			t.Fatal(err)
		}
		batch, err := replayReceiptCodec.Batch("scored/receipt/"+label, document, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
		return document.ID
	}
	unreadable := func(label string) artifact.ID {
		t.Helper()
		id := planID(t, artifact.KindEvaluation, label+" unreadable receipt")
		if _, err := store.Commit(ctx, artifact.Batch{Key: "scored/receipt/" + label, Artifacts: []artifact.Descriptor{{ID: id, Schema: ReplayReceiptSchema}}}); err != nil {
			t.Fatal(err)
		}
		return id
	}

	first := evaluated("a", artifact.ID{})
	second := evaluated("b", readable("b", first))
	third := evaluated("c", readable("c", second))
	broken := evaluated("d", unreadable("d"))
	beyond := evaluated("e", readable("e", broken))

	for name, test := range map[string]struct {
		carried artifact.ID
		want    string
	}{
		"a run that scored its own cases":       {first, strings.Repeat("a", 40)},
		"a chain of two carries":                {third, strings.Repeat("a", 40)},
		"a run whose receipt cannot be read":    {broken, ""},
		"a chain through an unreadable receipt": {beyond, ""},
	} {
		if got := scoredCommit(ctx, store, test.carried); got != test.want {
			t.Errorf("%s: scored at %q, want %q", name, got, test.want)
		}
	}
}
