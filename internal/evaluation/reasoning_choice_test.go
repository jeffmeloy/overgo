package evaluation

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/sequencescore"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// thoughtClose is the marker the reasoning fixture's template closes a thought with.
const thoughtClose = "</think>"

// reasoningFixture stands in for a templated model that thinks before it
// answers: each generation is its thought, then the marker and what it says
// after, or the thought alone when what it says is empty, as a thought cut by
// the limit is. It keeps what it was asked and under which options.
type reasoningFixture struct {
	thought  string
	said     []string
	shaped   []string
	thinking []bool
	options  []inference.GenerateOptions
	calls    int
}

func (f *reasoningFixture) ShapeChatTurn(prompt string, thinking bool) (string, error) {
	f.shaped, f.thinking = append(f.shaped, prompt), append(f.thinking, thinking)
	return "<user>" + prompt + "<model><think>", nil
}

func (f *reasoningFixture) Generate(_ context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	if !strings.HasPrefix(prompt, "<user>") || !strings.HasSuffix(prompt, "<model><think>") {
		panic("generation received a prompt the template did not shape")
	}
	f.options = append(f.options, options)
	text := f.thought
	if said := f.said[f.calls%len(f.said)]; said != "" {
		text += thoughtClose + said
	}
	f.calls++
	options.OnPromptEvaluated(inference.PromptEvaluation{Tokens: 1})
	if err := options.OnToken(inference.TokenEvent{Piece: text}); err != nil {
		return nil, "", err
	}
	return []tokenizer.TokenID{0, 1}, text, nil
}

func (f *reasoningFixture) AfterThought(text string) (string, bool) {
	_, said, closed := strings.Cut(text, thoughtClose)
	return said, closed
}

func (f *reasoningFixture) ScoreContinuations(context.Context, string, []string) ([]sequencescore.Score, error) {
	panic("the reasoning protocol must not score likelihoods")
}

// reasoningCampaign is a campaign over store under the reasoning protocol, its model's identities committed.
func reasoningCampaign(t *testing.T, store *overgodb.Store, runtime Runtime) (*Campaign, modelrecipe.ProgramIdentity) {
	t.Helper()
	identity := modelrecipe.ProgramIdentity{
		Model:      planID(t, artifact.KindModel, "reasoning model"),
		Definition: planID(t, artifact.KindModelDefinition, "reasoning definition"),
		Recipe:     planID(t, artifact.KindRecipe, "reasoning recipe"),
	}
	if _, err := artifact.Publish(t.Context(), store, artifact.Batch{Key: "fixture/reasoning", Artifacts: []artifact.Descriptor{{ID: identity.Model}, {ID: identity.Definition}, {ID: identity.Recipe}}}); err != nil {
		t.Fatal(err)
	}
	environment, err := runrecord.NewEnvironment(runrecord.Environment{Host: "fixture", OS: "fixture", Arch: "fixture", Device: "fixture", Backend: "fixture", Driver: "fixture"})
	if err != nil {
		t.Fatal(err)
	}
	campaign, err := newCampaign(store, runtime, identity, environment, planTestCommit, LifecycleIsolated)
	if err != nil {
		t.Fatal(err)
	}
	return campaign.WithPrompting(PromptingReasoning), identity
}

// TestEvaluationProtocolCapturesAllPolicyInputs holds the exploratory request
// protocol to being a protocol the evaluation command names and records whole.
//
// The direct protocols. A fixed case set binds, under raw completion, the chat
// template and hosted chat, the plan, comparison and answer identities it
// bound before the reasoning protocol was added, byte for byte.
//
// The identity. Each setting the reasoning protocol asks under, changed
// alone, binds another plan and another comparison: the thinking switch, the
// wording, every sampler setting, the seed, the limit and the parser. All but
// the parser name other answers too, since the model is asked otherwise; the
// parser is the scorer's and reads the same answers. The model and the cases
// bind other plans as well, and the protocol's scorer is neither direct one.
// The settings each name a field of the policy, so a setting added to it
// without a case here fails.
//
// The parser. It reads what the model said after its thought closed: the
// letter of the last "answer is" there, in any case and dressing, else the
// last option letter standing alone. A letter of the thought is not read, a
// thought that never closed reads nothing, and with thinking off the whole
// answer is read. The other parser reads the first letter after the thought.
//
// A run. Under the protocol the question is asked with its wording and
// thinking on, sampled with the policy's limit, and the store records the
// execution and the scorer that hold every setting; the evidence index reads
// the record as the protocol's, with the answer that never closed its thought
// counted as naming no option.
//
// Exposure. The cases a run asks are published as development evidence. A
// promotion part that holds what one of them asks, though it is registered
// under another dataset and name, is refused a confirmation, with nothing
// charged and nothing read; an acceptance over the same documents that holds
// none of them confirms.
func TestEvaluationProtocolCapturesAllPolicyInputs(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	suite := chatLetterSuite()
	compiled, err := CompileMultipleChoice(suite)
	if err != nil {
		t.Fatal(err)
	}
	asked := func(suite MultipleChoiceSuite) []askedChoice {
		listed := make([]askedChoice, len(suite.Cases))
		for index, testCase := range suite.Cases {
			listed[index] = askedChoice{Name: testCase.Name, Prompt: testCase.Prompt, Candidates: testCase.Candidates}
		}
		return listed
	}
	// identities binds a suite and returns its plan with the plan's comparison and answer identities.
	identities := func(compiled multipleChoicePlan, authorities ExactAuthorities) (Plan, [3]artifact.ID) {
		t.Helper()
		plan, err := BindMultipleChoice(compiled, authorities)
		if err != nil {
			t.Fatal(err)
		}
		comparison, err := plan.comparisonIdentity()
		if err != nil {
			t.Fatal(err)
		}
		observed, err := plan.observed(asked(compiled.suite))
		if err != nil {
			t.Fatal(err)
		}
		return plan, [3]artifact.ID{plan.Identity(), comparison, observed}
	}

	// The direct protocols: the identities trunk 01c937185 bound for this case set.
	for label, pinned := range map[prompting][3]string{
		PromptingRawCompletion: {"957dbf9acd984322bcc2f3cf7225824448832fbfb3e74444fdb8aeda5e12cd92", "2e5db8ab90a9c5d621d71a5bbf6c28b30e70934c30f1d61a1c0f3ed22c28e16a", "e55c3a691b2baf1482aef22c0068554991b3b408a22f2f857b64e2c57fd31202"},
		PromptingChatTemplate:  {"cdeddf7e8339b7c63bd00b38adc62d578d7a473f05aa2fb953f12a7aa23b66cd", "280fc72b317bed02414d9d539183763ed1fa2f2e80ec375aaef70f1b8bff9fbe", "643d1abecdbabad7e9a9f7bd2a4b4df836521fcf2cd7c3e6f139f3124af2e41d"},
		PromptingHostedChat:    {"e134c1753b6bca87dfeb6750cbe4332d79604e3921e7aebd64eec56b43f38fb0", "4b7e4e1b3978797e943c0d3a9de779a51ac5e504980b9c71ea07d71b61e7843f", "55d6d397faf464197d85ef8aa396d2766d9c45fffbe2c2f9a95235b752da09d2"},
	} {
		authorities := ListingAuthorities()
		authorities.Execution.Prompting = label
		_, bound := identities(compiled, authorities)
		for at, name := range []string{"plan", "comparison", "answers"} {
			if want := "profile:sha256:" + pinned[at]; bound[at].String() != want {
				t.Errorf("under %s the %s identity is %s, want %s as before", label.Label(), name, bound[at], want)
			}
		}
	}

	// The identity.
	reasoned := ListingAuthorities()
	reasoned.Execution.Prompting, reasoned.Execution.Reasoning = PromptingReasoning, exploratoryReasoning()
	base, bound := identities(compiled, reasoned)
	changes := map[string]func(*reasoningPolicy){
		"Thinking":        func(p *reasoningPolicy) { p.Thinking = false },
		"Wording":         func(p *reasoningPolicy) { p.Wording = chatChoiceInstruction },
		"Temperature":     func(p *reasoningPolicy) { p.Temperature = 0.6 },
		"TopP":            func(p *reasoningPolicy) { p.TopP = 0.8 },
		"TopK":            func(p *reasoningPolicy) { p.TopK = 40 },
		"MinP":            func(p *reasoningPolicy) { p.MinP = 0.05 },
		"PresencePenalty": func(p *reasoningPolicy) { p.PresencePenalty = 0 },
		"PenaltyWindow":   func(p *reasoningPolicy) { p.PenaltyWindow = -1 },
		"Seed":            func(p *reasoningPolicy) { p.Seed = 7 },
		"MaxTokens":       func(p *reasoningPolicy) { p.MaxTokens = 8192 },
		"Parser":          func(p *reasoningPolicy) { p.Parser = reasonedFirstLetter },
	}
	fields := reflect.TypeFor[reasoningPolicy]()
	for field := range fields.Fields() {
		if _, held := changes[field.Name]; !held {
			t.Errorf("setting %s of the reasoning protocol is not changed alone here", field.Name)
		}
	}
	for name, change := range changes {
		changed := reasoned
		change(&changed.Execution.Reasoning)
		plan, other := identities(compiled, changed)
		parser := name == "Parser"
		if other[0] == bound[0] || other[1] == bound[1] || (other[2] == bound[2]) != parser ||
			(plan.body.Execution == base.body.Execution) != parser || (plan.body.Scorer == base.body.Scorer) == parser {
			t.Errorf("%s changed alone binds plan %t, comparison %t, answers %t, execution %t and scorer %t apart; want the plan and the comparison apart, and the scorer apart for the parser and the answers and execution for every other setting",
				name, other[0] != bound[0], other[1] != bound[1], other[2] != bound[2], plan.body.Execution != base.body.Execution, plan.body.Scorer != base.body.Scorer)
		}
	}
	modelled := reasoned
	modelled.ModelDefinition = planID(t, artifact.KindModelDefinition, "another model")
	reworded := suite
	reworded.Cases = slices.Clone(suite.Cases)
	reworded.Cases[0].Prompt = "Q1 asked otherwise\nAnswer:"
	recompiled, err := CompileMultipleChoice(reworded)
	if err != nil {
		t.Fatal(err)
	}
	if _, other := identities(compiled, modelled); other[0] == bound[0] || other[2] == bound[2] {
		t.Error("another model binds the same plan or answers under the reasoning protocol")
	}
	if _, other := identities(recompiled, reasoned); other[0] == bound[0] || other[1] == bound[1] || other[2] == bound[2] {
		t.Error("another case binds the same plan, comparison or answers under the reasoning protocol")
	}
	for _, direct := range []prompting{PromptingRawCompletion, PromptingChatTemplate} {
		authorities := ListingAuthorities()
		authorities.Execution.Prompting = direct
		if plan, _ := identities(compiled, authorities); plan.body.Scorer == base.body.Scorer {
			t.Errorf("the reasoning protocol binds the %s scorer", direct.Label())
		}
	}
	unknown := reasoned
	unknown.Execution.Reasoning.Parser = "first-letter-anywhere"
	if _, err := BindMultipleChoice(compiled, unknown); err == nil {
		t.Error("a parser this code does not hold was bound")
	}
	if _, err := bindKindPlan(compiled.dataset, compiled.split, compiled.identity, compiled.suite, "exact-generation", reasoned); err == nil {
		t.Error("a suite that is not a choice suite was bound under the reasoning protocol")
	}
	if label, err := PromptingNamed(PromptingReasoning.Label()); err != nil || label != PromptingReasoning {
		t.Errorf("the protocol's own label names %q (%v)", label, err)
	}
	if _, err := PromptingNamed(PromptingHostedChat.Label()); err == nil {
		t.Error("a local model was asked under the hosted protocol")
	}

	// The parser.
	letters := []string{" A", " B", " C", " D"}
	fourCase := MultipleChoiceCase{Name: "four", Prompt: "Q\nAnswer:", Candidates: letters}
	for _, reading := range []struct {
		text     string
		thinking bool
		parser   string
		selected int
	}{
		{"B is tempting.</think>The answer is (C).", true, reasonedAnswerIs, 2},
		{"A, B.</think>THE ANSWER IS **D**", true, reasonedAnswerIs, 3},
		{"</think>The answer is: A. On reflection the answer is (B)", true, reasonedAnswerIs, 1},
		{"Surely the answer is (A).</think>Not D; it must be C.", true, reasonedAnswerIs, 2},
		{"So the answer is (A), but wait", true, reasonedAnswerIs, unmatchedChoice},
		{"</think>The answer is Apple", true, reasonedAnswerIs, unmatchedChoice},
		{"B, then the answer is (A)", false, reasonedAnswerIs, 0},
		{"A.</think>B, though the answer is (C)", true, reasonedFirstLetter, 1},
	} {
		policy := exploratoryReasoning()
		policy.Thinking, policy.Parser = reading.thinking, reading.parser
		observation, err := readReasonedChoice(fourCase, choiceAnswer{Written: &ExactResult{Text: reading.text}}, policy, &reasoningFixture{})
		if err != nil || observation.Selected != reading.selected {
			t.Errorf("%q with thinking %t by %s selects %d (%v), want %d", reading.text, reading.thinking, reading.parser, observation.Selected, err, reading.selected)
		}
	}

	// A run, in the store a held-out pool is divided in.
	store, objective, training, heldout := poolObjective(t)
	model := &reasoningFixture{thought: "A first, then B.", said: []string{"So the answer is (B).", "", "Both fit, but A."}}
	campaign, identity := reasoningCampaign(t, store, model)
	source, err := json.Marshal(suite)
	if err != nil {
		t.Fatal(err)
	}
	evaluated, err := CompileSuite(source, campaign.Authorities())
	if err != nil {
		t.Fatal(err)
	}
	result, err := campaign.Evaluate(ctx, evaluated)
	if err != nil {
		t.Fatal(err)
	}
	if len(model.options) != len(suite.Cases) || !slices.Equal(model.thinking, []bool{true, true, true}) ||
		!strings.HasSuffix(model.shaped[0], exploratoryWording) || model.options[0].Sampler == nil || model.options[0].DeviceGreedy || model.options[0].MaxNewTokens != exploratoryMaxTokens {
		t.Fatalf("the model was asked %q with thinking %v under %d generations; want each question with the wording and thinking on, sampled up to %d tokens", model.shaped, model.thinking, len(model.options), exploratoryMaxTokens)
	}
	recorded, err := ReadExecutionPolicy(ctx, store, evaluated.Plan().Execution())
	want := exploratoryReasoning()
	want.Parser = ""
	if err != nil || recorded.Prompting != PromptingReasoning || recorded.Reasoning != want {
		t.Fatalf("the store records execution %+v (%v), want every setting but the parser %+v", recorded, err, want)
	}
	scorer, _, err := artifact.ReadContent(ctx, store, evaluated.Plan().body.Scorer)
	if err != nil || !strings.Contains(string(scorer.Data), `"method":"`+reasonedAnswerIs+`"`) {
		t.Fatalf("the store records scorer %s (%v), want the parser %s as its method", scorer.Data, err, reasonedAnswerIs)
	}
	summaries := LatestEvidence(ctx, store, map[artifact.ID]string{evaluated.Descriptor().Dataset: suite.Source}, 0).EvaluationsByRecipe[identity.Recipe]
	if len(summaries) != 1 || summaries[0].Prompting != PromptingReasoning.Label() || summaries[0].Metrics["accuracy"] != 2.0/3 || summaries[0].Unread != 1 || len(result.Metrics) == 0 {
		t.Fatalf("the evidence index reads %+v; want one %s reading of two right in three with one answer naming no option", summaries, PromptingReasoning.Label())
	}

	// Exposure.
	exposed, err := choiceContent(suite.Cases[0])
	if err != nil {
		t.Fatal(err)
	}
	const combinations = 140
	var cases []InstructionCase
	for index, record := range slices.Concat(training.Records, heldout.Records) {
		rules := []string{fmt.Sprintf("length:%d", index%combinations), fmt.Sprintf("keywords:%d", index%combinations)}
		cases = append(cases, InstructionCase{Record: record.ID, Rules: rules, Content: fmt.Sprintf("exposure fixture case %d", index)})
	}
	request := InstructionAcceptanceRequest{
		Dataset: objective.Dataset, Base: testutil.ArtifactID(t, artifact.KindModel, "exposure fixture base"), Training: testutil.ArtifactID(t, artifact.KindProfile, "exposure fixture training"),
		Cases: cases, Protocol: InstructionProtocol{Scorer: InstructionScorer, Template: "chat-template", Thinking: "off", Sampling: "greedy", Stop: "end of turn", Window: 4096, MaxTokens: 1280},
		Seed: 7, Weights: [poolPartCount]uint64{1, 1, 4, 1}, Gain: 0.2,
		Comparisons: []InstructionComparison{{Name: "reward over base", Candidate: "reward", Baseline: "base"}},
		Pairs:       200, Discordant: 10, Basis: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvaluation, "exposure fixture development reading")},
	}
	if _, err := artifact.Publish(ctx, store, artifact.Batch{Key: "exposure-fixture/training", Artifacts: []artifact.Descriptor{{ID: request.Training}}}); err != nil {
		t.Fatal(err)
	}
	parts, err := InstructionGroups(request)
	if err != nil {
		t.Fatal(err)
	}
	holding := request
	holding.Cases = slices.Clone(cases)
	holding.Cases[slices.IndexFunc(holding.Cases, func(listed InstructionCase) bool { return listed.Record == parts[poolPromotion][0] })].Content = exposed
	clean := request
	clean.Seed = 8
	read := 0
	for _, confirmed := range []struct {
		request InstructionAcceptanceRequest
		refused bool
	}{{holding, true}, {clean, false}} {
		acceptance, batch, err := BuildInstructionAcceptance(confirmed.request)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.Publish(ctx, store, batch); err != nil {
			t.Fatal(err)
		}
		before := read
		_, _, err = ConfirmInstruction(ctx, store, acceptance.ID, request.Protocol, "reward over base", testutil.ArtifactID(t, artifact.KindModel, "reward"), request.Base,
			func(prompts []string) ([]bool, []bool, error) {
				read++
				return make([]bool, len(prompts)), make([]bool, len(prompts)), nil
			})
		content, _, partitionErr := artifact.ReadContent(ctx, store, acceptance.Split)
		partition, parseErr := runrecord.ParseSplitPartition(content.Data)
		granted, budgetErr := runrecord.NewBudget(poolUnit, partition.Promotion, 1, acceptance.ID)
		_, charges, chargesErr := runrecord.BudgetCharges(ctx, store, granted.ID)
		if partitionErr != nil || parseErr != nil || budgetErr != nil || chargesErr != nil {
			t.Fatal(partitionErr, parseErr, budgetErr, chargesErr)
		}
		refused := err != nil && strings.Contains(err.Error(), "development evidence")
		if refused != confirmed.refused || (len(charges) == 0) != confirmed.refused || (read == before) != confirmed.refused {
			t.Errorf("a promotion part that holds an exposed case %t is refused %t (%v) with %d charges and %d readings; want it refused with nothing charged or read exactly when it holds one",
				confirmed.refused, refused, err, len(charges), read-before)
		}
	}
	// The development part names the cases the run asked.
	named, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte(poolContent+exposed))
	if err != nil {
		t.Fatal(err)
	}
	holders, err := store.Parents(ctx, named)
	if err != nil {
		t.Fatal(err)
	}
	developed := slices.ContainsFunc(holders, func(edge artifact.Lineage) bool {
		part, found, err := dataset.LoadMembership(ctx, store, edge.Parent)
		return err == nil && found && part.Partition == PoolParts[poolDevelopment] && len(part.Records) == len(suite.Cases)
	})
	if !developed {
		t.Errorf("what case %s asks is held by %+v, want the development part of the %d cases the run asked", suite.Cases[0].Name, holders, len(suite.Cases))
	}
}
