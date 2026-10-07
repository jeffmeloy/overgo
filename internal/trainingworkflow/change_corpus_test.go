package trainingworkflow

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// TestChangeCorpusRecordListsEachRunOnce reads the record of the corpus
// collected from the installed text model: one line a run, each a change with
// its verdict. The record is a corpus as the document type takes one: its
// lines canonicalize under the corpus it names, the order seeds count up from
// one with none repeated, every run was judged against one base reading, and
// every run improved on it.
func TestChangeCorpusRecordListsEachRunOnce(t *testing.T) {
	t.Parallel()
	record, err := os.ReadFile(filepath.Join(testutil.RepoRoot(t), "docs", "verification", "change-corpus-qwen25-05b.txt"))
	if err != nil {
		t.Fatal(err)
	}
	var corpus ChangeCorpus
	for line := range strings.Lines(string(record)) {
		line = strings.TrimSpace(line)
		if id, found := strings.CutPrefix(line, "corpus: "); found {
			if corpus.ID, err = artifact.ParseID(id); err != nil {
				t.Fatal(err)
			}
		}
		if !strings.HasPrefix(line, "{") {
			continue
		}
		var change WeightChange
		if err := strictjson.DecodeBytes([]byte(line), &change); err != nil {
			t.Fatalf("record line %q: %v", line, err)
		}
		corpus.Changes = append(corpus.Changes, change)
	}
	if corpus.ID.Kind() != artifact.KindEvidence || len(corpus.Changes) == 0 {
		t.Fatalf("the record names corpus %s and %d runs", corpus.ID, len(corpus.Changes))
	}
	named := corpus.ID
	corpus.Version, corpus.Metric = artifact.InitialDocumentVersion, trainingprogram.MetricTokenAccuracy
	corpus.Objective, corpus.Plan, corpus.Inputs = testutil.ArtifactID(t, artifact.KindProfile, "objective"), testutil.ArtifactID(t, artifact.KindProfile, "plan"), testutil.ArtifactID(t, artifact.KindProfile, "inputs")
	corpus.Model = testutil.ArtifactID(t, artifact.KindModel, "base")
	if err := canonicalizeChangeCorpus(&corpus); err != nil {
		t.Fatalf("the record's runs are not a corpus: %v", err)
	}
	for index, change := range corpus.Changes {
		if change.Order != uint64(index+1) || change.Base != corpus.Changes[0].Base || !(change.Changed > change.Base) {
			t.Fatalf("run %d of the record has order %d and reads %v from a base of %v", index+1, change.Order, change.Changed, change.Base)
		}
	}
	t.Logf("corpus %s: %d runs from a base of %.4f", named, len(corpus.Changes), corpus.Changes[0].Base)
}

// TestChangeCorpusBindsEachChangeToItsVerdict fine-tunes one tiny model
// under one text objective in runs that differ only in the seed their
// training records are read by, a few updates each so every seed trains on
// other text, and collects them as a change corpus. Each change is the
// checkpoint its run published, bound to the verdict that judged exactly
// that checkpoint; every verdict read the same base on the same held-out
// inputs; a seed's run repeated is the same change; and a corpus asked to
// grow trains only the runs it lacks. A verdict put beside another run's
// checkpoint, a change counted twice and a run of another base model are
// each refused.
func TestChangeCorpusBindsEachChangeToItsVerdict(t *testing.T) {
	var documents []string
	for index, pattern := range []string{"abcd", "dcba", "acbd", "bdac", "cadb", "dbca", "abdc", "cdab", "badc", "dacb", "cbad", "adcb", "bcda", "cabd", "dbac", "acdb"} {
		// Distinct lengths keep every file its own record, and distinct
		// patterns make what a run learns depend on the records it reads.
		documents = append(documents, strings.Repeat(pattern, 6)[:8+index])
	}
	text := registerTextObjective(t, documents, trainingprogram.MetricTokenAccuracy)
	ctx := t.Context()
	const (
		window = 8
		steps  = 3
		first  = 1
		count  = 3
	)
	request := TextHeldoutRequest{
		Alias: text.alias, ModelDirectory: text.model, OutputDirectory: filepath.Join(text.root, "corpus"), Steps: steps, MaximumSequence: window, Host: true,
	}
	corpus, err := CollectChangeCorpus(ctx, text.store, request, first, count)
	if err != nil {
		t.Fatal(err)
	}
	committed, err := changeCorpusCodec.Require(ctx, text.store, corpus.ID)
	if err != nil || len(committed.Changes) != count || corpus.Objective != text.objective.ID || corpus.Metric != trainingprogram.MetricTokenAccuracy {
		t.Fatalf("committed corpus = (%d changes of objective %s by %s, %v), want %d of %s", len(committed.Changes), corpus.Objective, corpus.Metric, err, count, text.objective.ID)
	}
	runs := make([]changeRun, count)
	for index, change := range corpus.Changes {
		session, err := runrecord.RequireServingObservation(ctx, text.store, change.Observation)
		if err != nil || session.Model != corpus.Model {
			t.Fatalf("the change of order %d was trained from %s (%v), want the corpus's base %s", change.Order, session.Model, err, corpus.Model)
		}
		checkpoint, err := trainingprogram.LoadCheckpoint(filepath.Join(request.OutputDirectory, fmt.Sprintf("order-%04d", change.Order)))
		if err != nil || checkpoint.ID() != change.Trained || change.Order != uint64(first+index) {
			t.Fatalf("the change of order %d names checkpoint %s, its directory holds %s (%v)", change.Order, change.Trained, checkpoint.ID(), err)
		}
		if change.Base != corpus.Changes[0].Base {
			t.Fatalf("the change of order %d read its base at %v, the first at %v: one base on the same held-out inputs reads once", change.Order, change.Base, corpus.Changes[0].Base)
		}
		runs[index] = changeRun{Order: change.Order, Observation: change.Observation, Trained: change.Trained, Verdict: change.Verdict}
		t.Logf("order %d: %s %.4f -> %.4f", change.Order, corpus.Metric, change.Base, change.Changed)
	}

	// The same seed is the same change, and other seeds are other changes.
	again := request
	again.Order, again.OutputDirectory = first, filepath.Join(text.root, "again")
	repeated, err := RunTextHeldout(ctx, text.store, again)
	if err != nil || repeated.Trained != runs[0].Trained {
		t.Fatalf("order %d trained again = (%s, %v), want the same checkpoint %s", first, repeated.Trained, err, runs[0].Trained)
	}
	membership := request
	membership.OutputDirectory = filepath.Join(text.root, "membership")
	unseeded, err := RunTextHeldout(ctx, text.store, membership)
	if err != nil || slices.ContainsFunc(runs, func(run changeRun) bool { return run.Trained == unseeded.Trained }) {
		t.Fatalf("membership order trained %s (%v), want a checkpoint no seeded run published", unseeded.Trained, err)
	}

	// A corpus asked to grow keeps the runs it has.
	grown, err := CollectChangeCorpus(ctx, text.store, request, first, count+1)
	if err != nil || len(grown.Changes) != count+1 || grown.ID == corpus.ID {
		t.Fatalf("grown corpus = (%d changes, %v), want %d in a corpus of its own", len(grown.Changes), err, count+1)
	}
	for index, change := range corpus.Changes {
		if grown.Changes[index] != change {
			t.Fatalf("growing the corpus trained order %d again: %+v, was %+v", change.Order, grown.Changes[index], change)
		}
	}

	// What a corpus refuses.
	swapped := slices.Clone(runs)
	swapped[0].Verdict, swapped[1].Verdict = swapped[1].Verdict, swapped[0].Verdict
	if _, err := publishChangeCorpus(ctx, text.store, swapped); err == nil || !strings.Contains(err.Error(), "another run's checkpoint") {
		t.Fatalf("a verdict beside another run's checkpoint = %v, want it refused", err)
	}
	twice := append(slices.Clone(runs), changeRun{Order: first + count, Observation: runs[0].Observation, Trained: runs[0].Trained, Verdict: runs[0].Verdict})
	if _, err := publishChangeCorpus(ctx, text.store, twice); err == nil || !strings.Contains(err.Error(), "twice") {
		t.Fatalf("one change under two orders = %v, want it refused", err)
	}
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 6,
	})
	other := request
	other.ModelDirectory, other.OutputDirectory, other.Order = filepath.Join(text.root, "other-model"), filepath.Join(text.root, "other"), first+count+1
	writeModel(t, other.ModelDirectory, weights, shapes)
	if err := os.WriteFile(filepath.Join(other.ModelDirectory, "generation_config.json"), []byte(`{"eos_token_id":0}`), 0o600); err != nil {
		t.Fatal(err)
	}
	stranger, err := RunTextHeldout(ctx, text.store, other)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := publishChangeCorpus(ctx, text.store, append(slices.Clone(runs), changeRun{Order: other.Order, Observation: stranger.Observation, Trained: stranger.Trained, Verdict: stranger.Verdict})); err == nil || !strings.Contains(err.Error(), "another model") {
		t.Fatalf("a run of another base model = %v, want it refused", err)
	}
	if _, err := CollectChangeCorpus(ctx, text.store, request, 0, count); err == nil {
		t.Fatal("a corpus counted up from membership order, which is no seed")
	}
	var none artifact.ID
	if _, err := publishChangeCorpus(ctx, text.store, []changeRun{{Order: first, Verdict: none}}); err == nil {
		t.Fatal("a run with no verdict entered a corpus")
	}
}
