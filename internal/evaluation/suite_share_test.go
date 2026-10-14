package evaluation

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
)

// TestEvaluationSuitesAreSizedForTheirDecisions holds the part of a family a
// measurement run scores.
//
// The draw: of each group of a family's cases, one case in so many is kept,
// rounded up, so a group of thirty keeps three at one in ten, one of seven
// keeps one and one of one keeps itself; a group is the records' field where
// the family names one and the catalog entry's subset where it does not. The
// cases kept are the same however the cases are ordered and whatever other
// groups hold, since a case's place in the draw is its own name's.
//
// The assemblers: a case the draw left out is not scored, and is still read
// for what its task's cases share, so a task whose scored cases choose two
// answers still offers the four its whole task has.
//
// The store: a family a run scores a part of is derived as that part, the
// same part each time, whole when its name ends in /whole, and as N cases
// taken evenly through the whole when it ends in /N; the part's cases are
// cases of the whole, and it is a suite of its own, so its readings are
// compared with each other and not with the whole suite's. A count the family
// cannot give, or no count, is refused.
func TestEvaluationSuitesAreSizedForTheirDecisions(t *testing.T) {
	t.Parallel()
	sizes := map[string]int{"physics": 30, "law": 7, "history": 1, "math": 100}
	var cases []storeCase
	for _, category := range slices.Sorted(func(yield func(string) bool) {
		for category := range sizes {
			if !yield(category) {
				return
			}
		}
	}) {
		for ordinal := range sizes[category] {
			cases = append(cases, storeCase{
				entry: "mmlu-pro/" + category + "/test", subset: "one-subset", ordinal: ordinal,
				fields: rawFields(t, map[string]any{"category": category, "question": "q", "options": []string{"a", "b"}, "answer_index": 0}),
			})
		}
	}
	kept := func(cases []storeCase, field string) map[string][]string {
		drawn := slices.Clone(cases)
		for index := range drawn {
			drawn[index].undrawn = false
		}
		drawShare(drawn, field)
		names := map[string][]string{}
		for _, entry := range drawn {
			if !entry.undrawn {
				category, _ := caseString(entry.fields, "category")
				names[category] = append(names[category], fmt.Sprintf("%s/%d", entry.entry, entry.ordinal))
			}
		}
		for _, group := range names {
			slices.Sort(group)
		}
		return names
	}
	tenth := suiteShares["mmlu-pro"]
	first := kept(cases, tenth)
	for category, size := range sizes {
		if want := (size + suiteShare - 1) / suiteShare; len(first[category]) != want {
			t.Fatalf("%s keeps %d of its %d cases at one in %d, want %d", category, len(first[category]), size, suiteShare, want)
		}
	}
	reversed := slices.Clone(cases)
	slices.Reverse(reversed)
	onlyPhysics := slices.DeleteFunc(slices.Clone(cases), func(entry storeCase) bool { return !strings.Contains(entry.entry, "physics") })
	if again := kept(reversed, tenth); !slices.Equal(again["physics"], first["physics"]) || !slices.Equal(again["math"], first["math"]) {
		t.Fatal("the cases kept changed with the order the cases came in")
	}
	if alone := kept(onlyPhysics, tenth); !slices.Equal(alone["physics"], first["physics"]) {
		t.Fatal("the cases a group keeps changed with what the other groups hold")
	}
	// With no field named the group is the entry's subset, which these cases share: one group of 138.
	var whole int
	for _, group := range kept(cases, "") {
		whole += len(group)
	}
	if want := (len(cases) + suiteShare - 1) / suiteShare; whole != want {
		t.Fatalf("one subset of %d cases keeps %d at one in %d, want %d", len(cases), whole, suiteShare, want)
	}

	drawn := slices.Clone(cases)
	drawShare(drawn, tenth)
	assembled, _, err := assembleMMLUProSuite(drawn)
	if err != nil {
		t.Fatal(err)
	}
	if scored := len(assembled.(GroupedChoiceSuite).Cases); scored != len(first["physics"])+len(first["law"])+len(first["history"])+len(first["math"]) {
		t.Fatalf("the MMLU-Pro suite scores %d cases, and the draw kept %d", scored, len(first["physics"])+len(first["law"])+len(first["history"])+len(first["math"]))
	}
	// A task of forty cases whose answers are four: the four scored cases cannot hold more answers than the task.
	var task []storeCase
	answers := []string{"north", "south", "east", "west"}
	for ordinal := range 40 {
		task = append(task, storeCase{entry: "bbh/navigate/test", subset: "navigate", ordinal: ordinal,
			fields: rawFields(t, map[string]any{"input": fmt.Sprintf("turn %d", ordinal), "target": answers[ordinal%len(answers)]})})
	}
	drawShare(task, suiteShares["bbh"])
	grouped, _, err := assembleChoiceGroups("bbh", "lm-eval/leaderboard-bbh/v1.0", task)
	if err != nil {
		t.Fatal(err)
	}
	if scored := grouped.(GroupedChoiceSuite).Cases; len(scored) != 4 || len(scored[0].Candidates) != len(answers) {
		t.Fatalf("a task of 40 cases and %d answers scores %d cases, each among %d answers, want 4 cases among all %d", len(answers), len(scored), len(scored[0].Candidates), len(answers))
	}

	// Through the store: twenty-five prompts of IFEval, of which a run scores a tenth, rounded up.
	ctx := t.Context()
	root := t.TempDir()
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	var raw []byte
	const prompts = 25
	for prompt := range prompts {
		line, err := json.Marshal(map[string]any{"prompt": fmt.Sprintf("Write about item %d.", prompt), "instruction_id_list": []string{"punctuation:no_comma"}, "kwargs": []map[string]any{{}}})
		if err != nil {
			t.Fatal(err)
		}
		raw = append(append(raw, line...), '\n')
	}
	path := filepath.Join(root, "source.jsonl")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(raw)
	imported, err := dataset.ImportBenchmark(ctx, store, path, dataset.BenchmarkImportSpec{
		Source: "share-fixture", Revision: "v1", SHA256: hex.EncodeToString(digest[:]), Split: "train", Format: dataset.BenchmarkFormatJSONL, Conversion: "lm-eval/ifeval/v1",
		Fields: []dataset.FieldBinding{{Source: "prompt", Target: "prompt"}, {Source: "instruction_id_list", Target: "instruction_id_list"}, {Source: "kwargs", Target: "kwargs"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := json.Marshal(benchmarkManifest{Datasets: []benchmarkDeclaration{{Name: "ifeval/default/train", Dataset: imported.ID}}})
	if err != nil {
		t.Fatal(err)
	}
	manifestPath := filepath.Join(root, "manifest.json")
	if err := os.WriteFile(manifestPath, manifest, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := CatalogLocalBenchmarks(ctx, store, manifestPath); err != nil {
		t.Fatal(err)
	}
	// A benchmark's import is a dataset a pilot's groups are divided from, and records no content for
	// its cases: a confirmation on it counts exposure by record and group, and is not refused for it.
	if contents, err := recordContents(ctx, store, imported.ID); err != nil || len(contents) != 0 {
		t.Fatalf("a benchmark import's records have %d contents (%v), want none and no refusal", len(contents), err)
	}
	if _, err := recordContents(ctx, store, planID(t, artifact.KindDataset, "a dataset nobody committed")); err == nil {
		t.Fatal("a dataset that is not committed has contents, want it refused")
	}
	authorities := ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"), RuntimeRecipe: planID(t, artifact.KindRecipe, "recipe"),
		CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "environment"), Execution: ExecutionPolicy{Lifecycle: LifecycleResident},
	}
	derive := func(family string) (SuiteDescriptor, []string) {
		t.Helper()
		suites, dropped, err := DeriveStoreSuiteFamily(ctx, store, authorities, family)
		if err != nil || len(suites) != 1 || len(dropped) != 0 {
			t.Fatalf("%s: derived %d suites, dropped %v (%v)", family, len(suites), dropped, err)
		}
		var suite instructionRulesSuite
		// The first authority a plan binds is its case profile, which for this kind is the suite's document.
		if err := json.Unmarshal(suites[0].plan.authorities[0].Data, &suite); err != nil {
			t.Fatal(err)
		}
		names := make([]string, len(suite.Cases))
		for index, entry := range suite.Cases {
			names[index] = entry.Name
		}
		return suites[0].Descriptor(), names
	}
	part, partCases := derive("ifeval")
	again, againCases := derive("ifeval")
	all, allCases := derive("ifeval" + wholeFamily)
	if want := (prompts + suiteShare - 1) / suiteShare; part.Cases != uint64(want) || len(partCases) != want || all.Cases != prompts {
		t.Fatalf("a run scores %d of IFEval's %d prompts and the whole family is %d, want %d and %d", part.Cases, prompts, all.Cases, want, prompts)
	}
	if part.Dataset != again.Dataset || !slices.Equal(partCases, againCases) {
		t.Fatal("the part of a family a run scores was drawn otherwise the second time")
	}
	if part.Dataset == all.Dataset || part.Source != all.Source {
		t.Fatalf("the part is suite %s of %s and the whole %s of %s: want two suites of one family", part.Dataset, part.Source, all.Dataset, all.Source)
	}
	for _, name := range partCases {
		if !slices.Contains(allCases, name) {
			t.Fatalf("the part scores %s, which is no case of the whole family", name)
		}
	}
	// A count of a family's cases is taken evenly through the whole: five of twenty-five are every fifth.
	if _, spacedCases := derive("ifeval/5"); !slices.Equal(spacedCases, []string{allCases[0], allCases[5], allCases[10], allCases[15], allCases[20]}) {
		t.Fatalf("five cases taken evenly through the family are %v, want every fifth of %v", spacedCases, allCases)
	}
	for _, refused := range []string{"ifeval/26", "ifeval/0", "ifeval/some"} {
		if _, _, err := DeriveStoreSuiteFamily(ctx, store, authorities, refused); err == nil {
			t.Errorf("family %s was derived, want it refused", refused)
		}
	}
}
