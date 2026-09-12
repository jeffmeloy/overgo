package evaluation

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"slices"
	"testing"
	"unicode/utf8"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/sequencescore"
	"overgo/internal/testutil"
)

func TestBBHNativeProtocolAcceptance(t *testing.T) {
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	parse := func(text string) artifact.ID {
		t.Helper()
		id, err := artifact.ParseID(text)
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	var oracle struct {
		Cases   int
		Profile string `json:"profile_sha256"`
		Groups  []struct {
			Name    string
			Dataset artifact.ID
			Cases   []struct {
				Prompt string `json:"prompt_sha256"`
				Answer int
			}
		}
		Controls []struct {
			Group  string
			Index  int
			Prompt string `json:"prompt_sha256"`
		}
		Scoring []struct {
			Group    string
			Values   []float64 `json:"log_likelihoods"`
			Selected int
		}
	}
	readRetainedEvidence(t, store, parse("evidence:sha256:5a9fa4544ea25896c32996e29c96f6532d9c253667e84e02148482645f37cf74"), &oracle)
	if oracle.Cases != 5761 || len(oracle.Groups) != 24 || len(oracle.Controls) != 96 || len(oracle.Scoring) != 72 || oracle.Profile != fmt.Sprintf("%x", sha256.Sum256(bbhNativeProfileData)) {
		t.Fatal("native BBH reference denominator or profile changed")
	}
	groups, err := compiledNativeBBH()
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 24 {
		t.Fatal("native BBH task declarations changed")
	}
	var cases []storeCase
	var wantPrompts []string
	var wantAnswers []int
	seen := map[string]bool{}
	for _, group := range oracle.Groups {
		if seen[group.Name] {
			t.Fatal("duplicate native group")
		}
		seen[group.Name] = true
		imported, found, err := dataset.ReadBenchmarkImport(t.Context(), store, group.Dataset)
		if err != nil || !found || len(imported.Records) != len(group.Cases) {
			t.Fatalf("BBH corpus changed: %s %v", group.Name, err)
		}
		for index, id := range imported.Records {
			record, found, err := dataset.ReadBenchmarkRecord(t.Context(), store, id)
			if err != nil || !found {
				t.Fatalf("BBH record missing: %v", err)
			}
			fields := map[string]json.RawMessage{}
			for _, field := range record.Fields {
				fields[field.Name] = field.Value
			}
			cases = append(cases, storeCase{entry: "bbh/" + group.Name + "/test", subset: group.Name, ordinal: index, fields: fields})
			wantPrompts = append(wantPrompts, group.Cases[index].Prompt)
			wantAnswers = append(wantAnswers, group.Cases[index].Answer)
		}
	}
	if len(cases) != 5761 {
		t.Fatal("native BBH corpus incomplete")
	}
	assembled, dropped, err := assembleNativeBBH(cases)
	if err != nil || dropped != 0 {
		t.Fatalf("native BBH assembly: %v dropped=%d", err, dropped)
	}
	suite := assembled.(GroupedChoiceSuite)
	if suite.Normalization != sequencescore.NormalizationCharacters || suite.TieBreak != TieBreakFirst || len(suite.Cases) != 5761 {
		t.Fatal("native BBH scoring contract changed")
	}
	for index, c := range suite.Cases {
		if fmt.Sprintf("%x", sha256.Sum256([]byte(c.Prompt))) != wantPrompts[index] || c.Answer != wantAnswers[index] {
			t.Fatalf("native context or target differs: %s", c.Name)
		}
		group := groups[c.Group]
		if len(c.Candidates) != len(group.Choices) || len(c.CandidateCharacters) != len(group.Choices) {
			t.Fatalf("native option extent differs: %s", c.Name)
		}
		for j, choice := range group.Choices {
			if c.Candidates[j] != targetDelimiter+choice || c.CandidateCharacters[j] != uint64(utf8.RuneCountInString(choice)) {
				t.Fatalf("native option order or denominator differs: %s", c.Name)
			}
		}
	}
	for _, control := range oracle.Controls {
		group := groups[control.Group]
		doc := bbhNativeExample{Input: "protocol probe\nΩ", Target: group.Choices[0]}
		if control.Index > 0 {
			doc = group.Examples[control.Index-1]
		}
		value, _, err := assembleNativeBBH([]storeCase{{entry: "bbh/" + control.Group + "/test", subset: control.Group, fields: rawFields(t, map[string]any{"input": doc.Input, "target": doc.Target})}})
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256([]byte(value.(GroupedChoiceSuite).Cases[0].Prompt))) != control.Prompt {
			t.Fatalf("native example exclusion control differs: %s/%d", control.Group, control.Index)
		}
	}
	var extra struct {
		Profile  string `json:"profile_sha256"`
		Controls []struct {
			Group  string
			Prompt string `json:"prompt_sha256"`
		}
	}
	readRetainedEvidence(t, store, parse("evidence:sha256:2af6af0c69ac77915c3ec8d06d1d9b72e0b6a86a4073ba9a3c95ad4cf8be9a67"), &extra)
	if extra.Profile != oracle.Profile || len(extra.Controls) != 24 {
		t.Fatal("native extra-field controls changed")
	}
	for _, control := range extra.Controls {
		example := groups[control.Group].Examples[0]
		value, _, err := assembleNativeBBH([]storeCase{{entry: "bbh/" + control.Group + "/test", subset: control.Group, fields: rawFields(t, map[string]any{"input": example.Input, "target": example.Target, "_probe_extra": true})}})
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256([]byte(value.(GroupedChoiceSuite).Cases[0].Prompt))) != control.Prompt {
			t.Fatalf("native complete-document equality differs: %s", control.Group)
		}
	}
	for _, control := range oracle.Scoring {
		group := groups[control.Group]
		scores := make([]sequencescore.Score, len(group.Choices))
		if len(control.Values) != len(scores) {
			t.Fatal("native score extent changed")
		}
		for index, choice := range group.Choices {
			scores[index] = sequencescore.Score{LogProbability: control.Values[index], Tokens: 1, Characters: uint64(utf8.RuneCountInString(choice))}
		}
		selected, err := sequencescore.Select(scores, suite.Normalization)
		if err != nil || selected.Index != control.Selected {
			t.Fatalf("native normalized selection differs: %s %v", control.Group, err)
		}
		if !choiceCorrect(ChoiceObservation{Selected: selected.Index, Answer: control.Selected, Tied: selected.Tied}, suite.TieBreak) {
			t.Fatal("native first-option tie was rejected")
		}
	}
	legacy, _, err := assembleChoiceGroups("bbh", "lm-eval/leaderboard-bbh/v1.0", cases)
	if err != nil {
		t.Fatal(err)
	}
	old, err := CompileGroupedChoice(legacy.(GroupedChoiceSuite))
	if err != nil {
		t.Fatal(err)
	}
	if old.choice.identity.String() != "profile:sha256:15e6dbe88da49d3bd37f49542e2d4743528f7400e40e07f8330adf62d505c356" {
		t.Fatal("retained generated-letter case identity changed")
	}
	changedPrompts, changedOptions := 0, 0
	for index, prior := range legacy.(GroupedChoiceSuite).Cases {
		if prior.Prompt != suite.Cases[index].Prompt {
			changedPrompts++
		}
		if !slices.Equal(prior.Candidates, suite.Cases[index].Candidates) {
			changedOptions++
		}
	}
	native, err := CompileGroupedChoice(suite)
	if err != nil {
		t.Fatal(err)
	}
	if native.choice.identity == old.choice.identity {
		t.Fatal("native raw protocol reused the generated case identity")
	}
	authorities := ExactAuthorities{ModelDefinition: planID(t, artifact.KindModelDefinition, "model"), RuntimeRecipe: planID(t, artifact.KindRecipe, "recipe"), Environment: planID(t, artifact.KindEvidence, "environment"), CodeCommit: planTestCommit, Execution: ExecutionPolicy{Lifecycle: LifecycleResident}}
	for _, prompting := range []Prompting{PromptingRawCompletion, PromptingChatTemplate} {
		authorities.Execution.Prompting = prompting
		derived, dropped, err := DeriveStoreSuiteFamily(t.Context(), store, authorities, "bbh")
		if err != nil || len(derived) != 1 || len(dropped) != 0 || derived[0].Descriptor().Cases != 5761 {
			t.Fatalf("BBH protocol dispatch failed: %v drops=%v", err, dropped)
		}
		want := native.choice.identity
		if prompting == PromptingChatTemplate {
			want = old.choice.identity
		}
		if derived[0].Plan().body.CaseProfile != want {
			t.Fatal("BBH dispatch selected the wrong protocol")
		}
	}
	for _, change := range []func(*storeCase){func(c *storeCase) { c.subset = "unknown" }, func(c *storeCase) { c.fields = rawFields(t, map[string]any{"input": "input", "target": "unknown"}) }} {
		c := cases[0]
		change(&c)
		if _, _, err := assembleNativeBBH([]storeCase{c}); err == nil {
			t.Fatal("unresolved native task or target admitted")
		}
	}
	changed := slices.Clone(bbhNativeProfileData)
	changed[len(changed)-1] = ' '
	if _, err := decodeNativeBBH(changed); err == nil {
		t.Fatal("changed native profile admitted")
	}
	t.Log("5761 native raw context/target pairs, 24 fixed option sets, 120 prompt controls and 72 scoring controls; retained generated-letter case identity unchanged; no model loads")
	t.Logf("native raw requests differ from the retained generated protocol in %d contexts and %d option sets; no acquisition invalidation outside this protocol comparison", changedPrompts, changedOptions)
}
