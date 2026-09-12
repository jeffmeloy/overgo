package evaluation

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"slices"
	"testing"

	"github.com/dlclark/regexp2/v2"
	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

func TestIFEvalTokenizerAcceptance(t *testing.T) {
	tokenizer, err := compiledIFEvalTokenizer()
	if err != nil {
		t.Fatal(err)
	}
	parse := func(text string) artifact.ID {
		t.Helper()
		id, err := artifact.ParseID(text)
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	var selection struct {
		Oracle   artifact.ID
		Native   artifact.ID `json:"native_scores"`
		Profile  string      `json:"compiled_profile_sha256"`
		Boundary int         `json:"boundary_cases"`
		Views    int         `json:"retained_views"`
		Distinct int         `json:"distinct_retained"`
	}
	readIFEvalEvidence(t, store, parse("evidence:sha256:00c1a0e3c5c7080967bf434df2e418b6c511c9d2d3b896248d901f16da7568e9"), &selection)
	if selection.Native.DigestHex() != "78a1241f82ce179d229248564ccffb66f291d922eba864d1a261fc94d386c839" || selection.Profile != fmt.Sprintf("%x", sha256.Sum256(ifevalNLTKProfile)) || selection.Boundary != 48 || selection.Views != 512 || selection.Distinct != 225 {
		t.Fatal("native tokenizer selection changed")
	}
	type observation struct {
		Sentences, Words []string
		Capitals         int
	}
	var oracle struct {
		Boundary []struct {
			Response string
			observation
		}
		Retained []struct {
			Name string
			View int
			Hash string `json:"response_sha256"`
		}
		Observations map[string]observation
	}
	readIFEvalEvidence(t, store, selection.Oracle, &oracle)
	if len(oracle.Boundary) != selection.Boundary || len(oracle.Retained) != selection.Views || len(oracle.Observations) != selection.Distinct {
		t.Fatal("native tokenization denominator changed")
	}
	compare := func(name, raw string, want observation) {
		t.Helper()
		sentences, err := tokenizer.sentences(raw)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(sentences, want.Sentences) {
			t.Errorf("%s sentence tokens differ", name)
		}
		words, err := tokenizer.words(raw)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(words, want.Words) {
			t.Errorf("%s word tokens differ", name)
		}
		count := 0
		for _, word := range words {
			if nltkAllUpper(word) {
				count++
			}
		}
		if count != want.Capitals {
			t.Errorf("%s capital words=%d/%d", name, count, want.Capitals)
		}
	}
	for index, c := range oracle.Boundary {
		compare(fmt.Sprintf("boundary/%d", index), c.Response, c.observation)
	}
	var native struct{ Selection, Profile artifact.ID }
	readIFEvalEvidence(t, store, selection.Native, &native)
	responses := retainedIFEvalResponses(t, store, native.Selection, native.Profile)
	seen := map[string]bool{}
	viewKeys := map[string]bool{}
	for _, row := range oracle.Retained {
		key := fmt.Sprintf("%s/%d", row.Name, row.View)
		raw, present := responses[row.Name]
		views := looseInstructionViews(raw)
		if !present || row.View < 0 || row.View >= len(views) || viewKeys[key] {
			t.Fatal("missing or duplicate native response view")
		}
		viewKeys[key] = true
		raw = views[row.View]
		if fmt.Sprintf("%x", sha256.Sum256([]byte(raw))) != row.Hash {
			t.Fatal("native response view identity differs")
		}
		want, present := oracle.Observations[row.Hash]
		if !present {
			t.Fatal("native tokens missing")
		}
		if !seen[row.Hash] {
			compare(key, raw, want)
			seen[row.Hash] = true
		}
	}
	if len(seen) != selection.Distinct {
		t.Fatal("unconsumed native tokenization")
	}
	t.Run("profile identity and unresolved operands", func(t *testing.T) {
		for _, data := range [][]byte{nil, bytes.Replace(ifevalNLTKProfile, []byte("3.9.2"), []byte("3.9.1"), 1)} {
			if _, err := loadIFEvalTokenizer(data); err == nil {
				t.Fatal("changed profile accepted")
			}
		}
		for _, kind := range []string{ifevalSentences, ifevalCapitalWords} {
			for _, count := range []*CountRule{nil, {Relation: RelationAtLeast, Value: 0}, {Relation: RelationEqual, Value: 2}, {Relation: ifevalLessThan, Value: -1}} {
				if _, err := compileInstructionRule(InstructionRule{Name: "invalid", Kind: kind, Count: count}); err == nil {
					t.Fatal("unresolved native count accepted")
				}
			}
		}
	})
	t.Run("tokenizer errors refuse evaluation publication", func(t *testing.T) {
		broken := *tokenizer
		broken.period, err = regexp2.Compile(`(?:^){100}`, regexp2.OptionMaxBacktrackingStackSize(32))
		if err != nil {
			t.Fatal(err)
		}
		rule, err := compileInstructionRule(InstructionRule{Name: "sentences", Kind: ifevalSentences, Count: &CountRule{Relation: RelationAtLeast, Value: 1}})
		if err != nil {
			t.Fatal(err)
		}
		rule.tokenizer = &broken
		strict, loose, err := evaluateInstructionViews("x", []compiledInstructionRule{rule})
		if !errors.Is(err, regexp2.ErrBacktrackingStackLimit) || strict != nil || loose != nil {
			t.Fatalf("tokenizer error became a verdict: %v %v %v", strict, loose, err)
		}
		compiled, err := CompileInstructionRules(InstructionRulesSuite{Kind: InstructionRulesKind, Schema: "tokenizer-error", Source: "fixture", Cases: []InstructionRulesCase{{Name: "sentence", Prompt: "Write a sentence.", MaxTokens: 1, Rules: []InstructionRule{rule.source}}}})
		if err != nil {
			t.Fatal(err)
		}
		compiled.rules[0][0].tokenizer = &broken
		plan, err := BindInstructionRules(compiled, ExactAuthorities{ModelDefinition: planID(t, artifact.KindModelDefinition, "model"), RuntimeRecipe: planID(t, artifact.KindRecipe, "recipe"), CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "environment"), Execution: ExecutionPolicy{Lifecycle: LifecycleResident}})
		if err != nil {
			t.Fatal(err)
		}
		local, err := overgodb.Open(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		defer local.Close()
		publishPlanFixtureAuthorities(t, local, plan)
		report, err := EvaluateInstructionRules(t.Context(), local, exactGenerator{pieces: []string{"x"}}, compiled, plan)
		if !errors.Is(err, regexp2.ErrBacktrackingStackLimit) || report.ID.Valid() {
			t.Fatalf("tokenizer error published a report: %s %v", report.ID, err)
		}
		compiled.rules[0][0].tokenizer = tokenizer
		report, err = EvaluateInstructionRules(t.Context(), local, exactGenerator{pieces: []string{"One sentence."}}, compiled, plan)
		if err != nil || !report.ID.Valid() || report.PromptStrict != 1 {
			t.Fatalf("evaluation did not recover: %s %v", report.ID, err)
		}
	})
	t.Logf("native sentence and word tokens match: %d boundaries, %d view references, %d distinct retained responses", selection.Boundary, selection.Views, selection.Distinct)
}
