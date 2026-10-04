package main

import (
	"testing"

	"overgo/internal/evaluation"
)

// mmluProOwed is what one model's MMLU-Pro records come to.
type mmluProOwed int

const (
	// mmluProComplete: both protocols; the model counts toward the floor.
	mmluProComplete mmluProOwed = iota
	// mmluProAnchorOnly: a model served without its chat template carries
	// the raw anchor, all it owes; reported, not counted.
	mmluProAnchorOnly
	// mmluProIncomplete: a protocol the model owes is missing.
	mmluProIncomplete
)

// mmluProCoverage judges one model scored on MMLU-Pro by the protocols that
// scored it and whether its declared verification serves it through its own
// chat template. A chat model owes both protocols: its users meet it through
// the template, and the raw anchor compares it with every other model. A
// model served without the template, a base model or Unlimited-OCR's
// document decoder, owes the raw anchor alone; a chat-template score measures
// nothing its users meet.
func mmluProCoverage(chat bool, protocols map[string]bool) mmluProOwed {
	raw, template := protocols[evaluation.PromptingRawCompletion.Label()], protocols[evaluation.PromptingChatTemplate.Label()]
	switch {
	case raw && template:
		return mmluProComplete
	case raw && !chat:
		return mmluProAnchorOnly
	default:
		return mmluProIncomplete
	}
}

// TestMMLUProCoverageOwesChatModelsBothProtocols holds the judgment the live
// acceptance applies: a chat model with one protocol is incomplete whichever
// it lacks, a model served without its template owes the raw anchor alone,
// and only a model carrying both protocols counts toward the floor.
func TestMMLUProCoverageOwesChatModelsBothProtocols(t *testing.T) {
	raw, template := evaluation.PromptingRawCompletion.Label(), evaluation.PromptingChatTemplate.Label()
	for _, test := range []struct {
		name      string
		chat      bool
		protocols []string
		want      mmluProOwed
	}{
		{"a chat model with both protocols counts", true, []string{raw, template}, mmluProComplete},
		{"a base model with both protocols counts", false, []string{raw, template}, mmluProComplete},
		{"a chat model with the raw anchor alone is incomplete", true, []string{raw}, mmluProIncomplete},
		{"a chat model with the template pass alone is incomplete", true, []string{template}, mmluProIncomplete},
		{"a model served without its template owes the raw anchor alone", false, []string{raw}, mmluProAnchorOnly},
		{"a model served without its template still owes the raw anchor", false, []string{template}, mmluProIncomplete},
	} {
		protocols := map[string]bool{}
		for _, protocol := range test.protocols {
			protocols[protocol] = true
		}
		if got := mmluProCoverage(test.chat, protocols); got != test.want {
			t.Errorf("%s: judged %d, want %d", test.name, got, test.want)
		}
	}
}
