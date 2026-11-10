package evaluation

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
)

// TestAReaskPublishesWhatItAsksBeforeAskingIt records a chat-template reading of three questions whose answers name a
// letter, end in a working the budget cut, and decline every option, then re-asks it (SQA-361). A variant that cannot
// be asked and an id that names no evaluation are refused with nothing published. Prepared, the re-ask admits the one
// budget-ended answer and publishes what its case asks as a development part of the reading's dataset, so a held-out
// part holding it is refused a confirmation, while the other cases stay unexposed; only then is the model asked, after
// the variant's opener.
func TestAReaskPublishesWhatItAsksBeforeAskingIt(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store := testcheck.Must(overgodb.Open(t.TempDir()))(t)
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Error(err)
		}
	})
	identity := modelrecipe.ProgramIdentity{
		Model:      planID(t, artifact.KindModel, "reask model"),
		Definition: planID(t, artifact.KindModelDefinition, "reask definition"),
		Recipe:     planID(t, artifact.KindRecipe, "reask recipe"),
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "fixture/reask", Artifacts: []artifact.Descriptor{{ID: identity.Model}, {ID: identity.Definition}, {ID: identity.Recipe}}}); err != nil {
		t.Fatal(err)
	}
	environment := testcheck.Must(runrecord.NewEnvironment(runrecord.Environment{Host: "fixture", OS: "fixture", Arch: "fixture", Device: "fixture", Backend: "fixture", Driver: "fixture"}))(t)
	suite := chatLetterSuite()
	reading := &countedChat{answers: []writtenAnswer{{"B) two", 2}, {"11.9, which corresponds to", chatAnswerTokens}, {"None of the options fits", chatAnswerTokens}}}
	campaign := testcheck.Must(newCampaign(store, reading, identity, environment, planTestCommit, LifecycleIsolated))(t)
	compiled := testcheck.Must(CompileSuite(testcheck.Must(json.Marshal(suite))(t), campaign.WithPrompting(PromptingChatTemplate).Authorities()))(t)
	result := testcheck.Must(campaign.Evaluate(ctx, compiled))(t)

	exposedBy := func(testCase MultipleChoiceCase) []artifact.ID {
		t.Helper()
		named := testcheck.Must(artifact.IdentifyBytes(artifact.KindEvidence, []byte(poolContent+testcheck.Must(choiceContent(testCase))(t))))(t)
		var parts []artifact.ID
		for _, edge := range testcheck.Must(store.Parents(ctx, named))(t) {
			if part, found, err := dataset.LoadMembership(ctx, store, edge.Parent); err == nil && found && part.Partition == PoolParts[poolDevelopment] {
				parts = append(parts, edge.Parent)
			}
		}
		return parts
	}
	paren := []AnswerVariant{{Opener: "The answer is (", Tokens: 8}}
	for name, refused := range map[string]func() error{
		"a variant with no opener": func() error {
			_, err := PrepareReask(ctx, store, result.Evaluation, []AnswerVariant{{Tokens: 8}})
			return err
		},
		"an id that names no evaluation": func() error {
			_, err := PrepareReask(ctx, store, compiled.Descriptor().Dataset, paren)
			return err
		},
	} {
		if err := refused(); err == nil {
			t.Errorf("%s was prepared", name)
		}
	}
	if parts := exposedBy(suite.Cases[1]); len(parts) != 0 {
		t.Fatalf("a refused re-ask published the case it would ask in %v", parts)
	}
	developed := func() uint64 {
		t.Helper()
		summaries := LatestEvidence(ctx, store, map[artifact.ID]string{compiled.Descriptor().Dataset: suite.Source}, 0).EvaluationsByRecipe[identity.Recipe]
		if len(summaries) != 1 {
			t.Fatalf("the evidence index reads %+v, want the one reading", summaries)
		}
		return summaries[0].Developed
	}
	if before := developed(); before != 0 {
		t.Fatalf("before any re-ask the reading has %d cases of development evidence, want none", before)
	}

	prepared := testcheck.Must(PrepareReask(ctx, store, result.Evaluation, paren))(t)
	if after := developed(); after != 1 {
		t.Errorf("after the re-ask is prepared the reading says %d of its cases are development evidence, want the one it will ask", after)
	}
	if parts := exposedBy(suite.Cases[1]); len(parts) != 1 {
		t.Fatalf("the budget-ended case is held by development parts %v before it is asked, want one", parts)
	}
	for _, unasked := range []MultipleChoiceCase{suite.Cases[0], suite.Cases[2]} {
		if parts := exposedBy(unasked); len(parts) != 0 {
			t.Errorf("case %s, which the re-ask does not ask, is held by development parts %v", unasked.Name, parts)
		}
	}
	runtime := &askedChat{chatFixture: &chatFixture{answers: []string{"A) one"}}}
	reasked := testcheck.Must(prepared.Ask(ctx, runtime, paren[0]))(t)
	if reasked != (Reasked{Asked: 1, Lettered: 1}) || len(runtime.asked) != 1 || !strings.HasSuffix(runtime.asked[0], "<model>The answer is (") ||
		!slices.ContainsFunc(runtime.asked, func(asked string) bool { return strings.Contains(asked, suite.Cases[1].Prompt) }) {
		t.Errorf("the re-ask found %+v and asked %q, want the budget-ended question asked once after its variant's opener and its letter read", reasked, runtime.asked)
	}
}
