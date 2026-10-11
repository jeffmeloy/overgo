package trainingworkflow

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
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
	text := registerTextObjective(t, changeDocuments(), trainingprogram.MetricTokenAccuracy)
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

// changeDocuments is the text a change corpus's tests train on: sixteen
// documents, each its own pattern over the tiny model's four letters.
func changeDocuments() []string {
	var documents []string
	for index, pattern := range []string{"abcd", "dcba", "acbd", "bdac", "cadb", "dbca", "abdc", "cdab", "badc", "dacb", "cbad", "adcb", "bcda", "cabd", "dbac", "acdb"} {
		// Distinct lengths keep every file its own record, and distinct
		// patterns make what a run learns depend on the records it reads.
		documents = append(documents, strings.Repeat(pattern, 6)[:8+index])
	}
	return documents
}

// otherPolicyRecipe publishes the text objective's default recipe with
// another optimizer policy in place of its own: a recipe of the same model
// and objective whose runs take other updates. It returns the recipe.
func otherPolicyRecipe(t *testing.T, text registeredText) artifact.ID {
	t.Helper()
	ctx := t.Context()
	weights, err := IdentifyModelWeights(text.model)
	if err != nil {
		t.Fatal(err)
	}
	defaultRecipe, err := BootstrapObjectiveRecipe(ctx, text.store, weights, text.alias)
	if err != nil {
		t.Fatal(err)
	}
	definition, err := recipe.RequireDefinition(ctx, text.store, defaultRecipe)
	if err != nil {
		t.Fatal(err)
	}
	policy := trainingprogram.BuiltinOptimizerPolicy()
	policy.ParameterExponent, policy.MomentumEffectiveSamples = -0.5, 7
	if policy.ID, err = artifact.JSONID(artifact.KindProfile, policy); err != nil {
		t.Fatal(err)
	}
	content, err := policy.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := text.store.Commit(ctx, artifact.Batch{Key: "other-policy", Contents: []artifact.Content{content}}); err != nil {
		t.Fatal(err)
	}
	for index := range definition.Dependencies {
		if definition.Dependencies[index].Role == recipe.DependencyOptimizer {
			definition.Dependencies[index].Artifact = policy.ID
		}
	}
	selected, err := recipe.NewDefinitionWithDependencies(definition.Task, definition.Dependencies, definition.Nodes, definition.Edges, definition.Inputs, definition.Outputs)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, text.store, "other-policy-recipe", selected); err != nil {
		t.Fatal(err)
	}
	return selected.ID
}

// TestChangeCorpusCreditsOnlyRunsOfItsRequest collects a small change corpus
// and asks for it again in the same directory, counting every update any run
// trains. The same request trains nothing and names the same corpus, and the
// same request for one more run trains that run alone. A request that differs
// in its update count, its window, its base model, its objective, its recipe
// or how its trainer reads and trains is refused before any update and leaves
// every finished run's record as it was, and one that asks the corpus to grow
// under another update count starts no run. What a request asks is named
// apart for adapters of two ranks and for averages of two decays, not only
// by whether one is asked, and alike for two equal adapters. A record that is
// another order's, and one written before runs recorded what they were
// asked, are refused the same way. An objective alias that moves while a
// corpus collects ends the collection at its next run, which trains nothing.
func TestChangeCorpusCreditsOnlyRunsOfItsRequest(t *testing.T) {
	text := registerTextObjective(t, changeDocuments(), trainingprogram.MetricTokenAccuracy)
	ctx := t.Context()
	const (
		window = 8
		steps  = 3
		first  = 1
		count  = 2
		// refusal is the phrase of the collector's own refusal: resolving
		// another model, objective or recipe has errors of its own, and none
		// of those may pass for it.
		refusal = "not this request's run"
	)
	updates := 0
	request := TextHeldoutRequest{
		Alias: text.alias, ModelDirectory: text.model, OutputDirectory: filepath.Join(text.root, "corpus"), Steps: steps, MaximumSequence: window, Host: true,
		Step: func(int, int, float64) { updates++ },
	}
	record := func(order uint64) string {
		return filepath.Join(runDirectory(request.OutputDirectory, order), changeRunFile)
	}
	records := func() []string {
		t.Helper()
		var read []string
		for order := uint64(first); order < first+count; order++ {
			data, err := os.ReadFile(record(order))
			if err != nil {
				t.Fatal(err)
			}
			read = append(read, string(data))
		}
		return read
	}

	// Another base model, another objective over the same records, and
	// another recipe of the same model and objective, all in the store
	// before the corpus is collected, so that the request alone tells them
	// from its own.
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 6,
	})
	otherModel := filepath.Join(text.root, "other-model")
	writeModel(t, otherModel, weights, shapes)
	if err := os.WriteFile(filepath.Join(otherModel, "generation_config.json"), []byte(`{"eos_token_id":0}`), 0o600); err != nil {
		t.Fatal(err)
	}
	spec := text.objective.ObjectiveSpec
	spec.Name = "texts-renamed"
	renamed, err := trainingprogram.NewObjective(spec)
	if err != nil {
		t.Fatal(err)
	}
	content, err := renamed.Content()
	if err != nil {
		t.Fatal(err)
	}
	const otherAlias = "objective.registered.texts-renamed"
	if _, err := artifact.CommitBatch(ctx, text.store, artifact.Batch{
		Key: "renamed-objective", Contents: []artifact.Content{content},
		Aliases: []artifact.AliasBinding{{Name: otherAlias, Target: renamed.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	otherRecipe := otherPolicyRecipe(t, text)

	corpus, err := CollectChangeCorpus(ctx, text.store, request, first, count)
	if err != nil || updates != count*steps {
		t.Fatalf("collecting %d runs of %d updates took %d updates (%v)", count, steps, updates, err)
	}
	finished := records()
	again, err := CollectChangeCorpus(ctx, text.store, request, first, count)
	if err != nil || again.ID != corpus.ID || updates != count*steps {
		t.Fatalf("the same request again = (corpus %s after %d updates, %v), want corpus %s and the %d updates it had", again.ID, updates, err, corpus.ID, count*steps)
	}

	// A finished corpus asked again under another request is refused before
	// any update, and its records stay as they were.
	for _, other := range []struct {
		what   string
		change func(*TextHeldoutRequest)
	}{
		{"another update count", func(asked *TextHeldoutRequest) { asked.Steps++ }},
		{"another window", func(asked *TextHeldoutRequest) { asked.MaximumSequence -= 2 }},
		{"another base model", func(asked *TextHeldoutRequest) { asked.ModelDirectory = otherModel }},
		{"another objective", func(asked *TextHeldoutRequest) { asked.Alias = otherAlias }},
		{"another recipe", func(asked *TextHeldoutRequest) { asked.Recipe = otherRecipe }},
		{"the device's trainer", func(asked *TextHeldoutRequest) { asked.Host = false }},
		{"a frozen lexical table", func(asked *TextHeldoutRequest) { asked.FreezeLexical = true }},
		{"records read as chat text", func(asked *TextHeldoutRequest) { asked.Chat = true }},
		{"records read as conversations", func(asked *TextHeldoutRequest) { asked.Conversations = true }},
		{"a moving average of the weights", func(asked *TextHeldoutRequest) { asked.Average = 0.5 }},
		{"an adapter in place of the model", func(asked *TextHeldoutRequest) { asked.LoRA = &LoRA{Rank: 2} }},
	} {
		asked := request
		other.change(&asked)
		_, err := CollectChangeCorpus(ctx, text.store, asked, first, count)
		if kept := slices.Equal(records(), finished); err == nil || !strings.Contains(err.Error(), refusal) || updates != count*steps || !kept {
			t.Errorf("the finished corpus asked again with %s = (%d updates, records as they were %t, %v), want it refused before any update past %d", other.what, updates, kept, err, count*steps)
		}
	}

	// The requests above tell an adapter and an average only from none asked.
	// The name itself also tells one rank from another and one decay from
	// another, and reads an adapter's settings, not where the request holds it.
	named := func(average float64, adapter *LoRA) artifact.ID {
		t.Helper()
		asked := request
		asked.Recipe, asked.Average, asked.LoRA = otherRecipe, average, adapter
		name, err := askedChange(asked)
		if err != nil {
			t.Fatal(err)
		}
		return name
	}
	adapter, averaged := named(0, &LoRA{Rank: 2}), named(0.5, nil)
	if alike, ranked, decayed := named(0, &LoRA{Rank: 2}) == adapter, named(0, &LoRA{Rank: 4}) != adapter, named(0.25, nil) != averaged; !alike || !ranked || !decayed {
		t.Errorf("what a request asks is named alike for two equal adapters %t, apart for adapters of two ranks %t, apart for averages of two decays %t, want all three", alike, ranked, decayed)
	}

	// A corpus asked to grow under another update count starts no run: the
	// runs it has and the one it lacks would be of two budgets.
	budget := request
	budget.Steps++
	_, err = CollectChangeCorpus(ctx, text.store, budget, first, count+1)
	if _, started := os.Stat(runDirectory(request.OutputDirectory, first+count)); err == nil || !strings.Contains(err.Error(), refusal) || updates != count*steps || !errors.Is(started, os.ErrNotExist) {
		t.Errorf("the corpus asked to grow under another update count = (%d updates, the new run's directory %v, %v), want it refused before any update past %d", updates, started, err, count*steps)
	}
	if t.Failed() {
		// What follows reads the directory, which a request that was not
		// refused has trained into.
		t.FailNow()
	}

	// The same request for one more run trains that run alone.
	grown, err := CollectChangeCorpus(ctx, text.store, request, first, count+1)
	if err != nil || len(grown.Changes) != count+1 || !slices.Equal(grown.Changes[:count], corpus.Changes) || updates != (count+1)*steps {
		t.Fatalf("the same request for one more run = (%d changes after %d updates, %v), want the %d it had and one more after %d", len(grown.Changes), updates, err, count, (count+1)*steps)
	}

	// A record that is another order's, and one that does not say what its
	// run was asked, as a record written before records said so does not.
	var fields map[string]json.RawMessage
	if err := json.Unmarshal([]byte(finished[0]), &fields); err != nil || fields["asked"] == nil {
		t.Fatalf("the record of order %d does not say what its run was asked (%v): %s", first, err, finished[0])
	}
	delete(fields, "asked")
	unasked, err := json.Marshal(fields)
	if err != nil {
		t.Fatal(err)
	}
	for _, other := range [][2]string{{"another order's record", finished[1]}, {"a record that does not say what was asked", string(unasked)}} {
		if err := os.WriteFile(record(first), []byte(other[1]), 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := CollectChangeCorpus(ctx, text.store, request, first, count+1); err == nil || !strings.Contains(err.Error(), refusal) || updates != (count+1)*steps {
			t.Fatalf("order %d holding %s = (%d updates, %v), want it refused before any update past %d", first, other[0], updates, err, (count+1)*steps)
		}
	}
	if err := os.WriteFile(record(first), []byte(finished[0]), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := CollectChangeCorpus(ctx, text.store, request, first, count+1); err != nil || updates != (count+1)*steps {
		t.Fatalf("the corpus with its own record back = (%d updates, %v), want it credited as before", updates, err)
	}

	// An alias that moves while a corpus collects: the next run is asked
	// under the recipe the corpus resolved before its first, which is no
	// recipe of the objective the alias names now, and trains nothing.
	moving := request
	moving.OutputDirectory = filepath.Join(text.root, "moving")
	trained := 0
	moving.Step = func(_, total int, _ float64) {
		if trained++; trained != total {
			return
		}
		if _, err := artifact.CommitBatch(ctx, text.store, artifact.Batch{
			Key: "moved-alias", Aliases: []artifact.AliasBinding{artifact.AliasMove(text.alias, renamed.ID, text.objective.ID)},
		}); err != nil {
			t.Error(err)
		}
	}
	if _, err := CollectChangeCorpus(ctx, text.store, moving, first, count); err == nil || !strings.Contains(err.Error(), "recipe objective differs") || trained != steps {
		t.Fatalf("a corpus whose objective alias moved after its first run = (%d updates, %v), want its second run refused after %d", trained, err, steps)
	}
}
