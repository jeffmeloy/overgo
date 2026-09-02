package evaluation

import "testing"

func TestPromptingIsAPlanAuthority(t *testing.T) {
	raw := ListingAuthorities()
	templated := raw
	templated.Execution.Prompting = PromptingChatTemplate
	suite := MultipleChoiceSuite{
		Kind: MultipleChoiceKind, Schema: "test/mmlu/v1", Source: "store/mmlu",
		Normalization: "sum", Aggregation: AggregationAccuracy,
		Cases: []MultipleChoiceCase{{Name: "case", Prompt: "Q: one\nA:", Candidates: []string{" A", " B"}, Answer: 1}},
	}
	compiled, err := CompileMultipleChoice(suite)
	if err != nil {
		t.Fatal(err)
	}
	rawPlan, err := BindMultipleChoice(compiled, raw)
	if err != nil {
		t.Fatal(err)
	}
	templatedPlan, err := BindMultipleChoice(compiled, templated)
	if err != nil {
		t.Fatal(err)
	}
	if rawPlan.Identity() == templatedPlan.Identity() || rawPlan.Execution() == templatedPlan.Execution() {
		t.Fatal("prompting protocol did not change the plan identity")
	}
	invalid := raw
	invalid.Execution.Prompting = "few-shot"
	if _, err := BindMultipleChoice(compiled, invalid); err == nil {
		t.Fatal("unknown prompting protocol was bound")
	}
	if PromptingRawCompletion.Label() != "raw-completion" || PromptingChatTemplate.Label() != "chat-template" {
		t.Fatalf("labels = %q, %q", PromptingRawCompletion.Label(), PromptingChatTemplate.Label())
	}
}
