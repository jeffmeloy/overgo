package evaluation

import (
	"strings"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
)

// TestTheChatProtocolOpensTheAnswerAsItWasMeasured asks a choice suite under the chat-template protocol as a
// measurement run does and holds that every question opens the model's turn with "The answer is", the opener ten
// models' paired MMLU-Pro readings answered 38 more cases right under than under "The answer is (" (answer-evidence/
// earlier-opener), that a letter written after it is read and an answer naming none is counted, and that the
// protocol's words name it.
func TestTheChatProtocolOpensTheAnswerAsItWasMeasured(t *testing.T) {
	t.Parallel()
	runtime := &askedChat{chatFixture: &chatFixture{answers: []string{" (B) two", " **A**.", " 11.9, which corresponds to"}}}
	reading := indexedChatReading(t, runtime, chatLetterSuite())
	if len(runtime.asked) != len(chatLetterSuite().Cases) {
		t.Fatalf("the model was asked %d questions, want one for each of the suite's %d", len(runtime.asked), len(chatLetterSuite().Cases))
	}
	for _, asked := range runtime.asked {
		if !strings.HasSuffix(asked, "<model>The answer is") {
			t.Errorf("a question opens the model's turn as %q, want it to end with The answer is", asked)
		}
	}
	if reading.Metrics["accuracy"] != 2.0/3 || reading.Unread != 1 {
		t.Errorf("the letters written after the opener read %+v, want two of three right and the answer naming none counted", reading)
	}
	if words, _ := ChoiceProtocol(PromptingChatTemplate.Label()); !strings.Contains(words, `"The answer is"`) {
		t.Errorf("the chat-template protocol is said as %q, which does not name its opener", words)
	}
}

// TestAnAnswerAfterAnotherOpenerIsNeverReadAgain holds the two openers apart in one store. A plan recorded under the
// opener that ends at the letter's place binds under its own method and asks with that opener; the plan bound now is
// another plan, with another scorer, and its questions are put to the model again rather than read from the other
// opener's answers, whose plan still reads them without asking. A method that names no opener is refused, and the
// hosted protocol asks with the protocol's opener under its own method.
func TestAnAnswerAfterAnotherOpenerIsNeverReadAgain(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	suite := chatLetterSuite()
	compiled := testcheck.Must(CompileMultipleChoice(suite))(t)
	authorities := ListingAuthorities()
	authorities.Execution.Prompting = PromptingChatTemplate
	paren := testcheck.Must(bindMultipleChoice(compiled, authorities, parenChatMethod))(t)
	current := testcheck.Must(BindMultipleChoice(compiled, authorities))(t)
	if paren.opener != parenChatOpener || current.opener != chatAnswerOpener || paren.Identity() == current.Identity() || paren.body.Scorer == current.body.Scorer {
		t.Fatalf("the paren plan opens with %q and the current with %q, the two plans or scorers the same: %v", paren.opener, current.opener, paren.Identity() == current.Identity())
	}
	if _, err := bindMultipleChoice(compiled, authorities, "generated-letter-unknown"); err == nil {
		t.Fatal("a chat-template plan was bound under a method that names no opener")
	}
	hosted := authorities
	hosted.Execution.Prompting = PromptingHostedChat
	if plan := testcheck.Must(BindMultipleChoice(compiled, hosted))(t); plan.opener != chatAnswerOpener {
		t.Fatalf("the hosted protocol opens with %q, want the protocol's opener", plan.opener)
	}
	store := testcheck.Must(overgodb.Open(t.TempDir()))(t)
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Error(err)
		}
	})
	asked := func(plan Plan) []string {
		runtime := &askedChat{chatFixture: &chatFixture{answers: []string{" B", " A", " A"}}}
		if _, _, _, err := scoreMultipleChoice(ctx, store, plan, runtime, suite); err != nil {
			t.Fatal(err)
		}
		return runtime.asked
	}
	if first := asked(paren); len(first) != len(suite.Cases) || !strings.HasSuffix(first[0], "<model>"+parenChatOpener) {
		t.Fatalf("the paren plan asked %q, want each question opened with %q", first, parenChatOpener)
	}
	if again := asked(current); len(again) != len(suite.Cases) || !strings.HasSuffix(again[0], "<model>"+chatAnswerOpener) {
		t.Fatalf("the current plan asked %q over the paren plan's answers, want each question asked again after %q", again, chatAnswerOpener)
	}
	if held := asked(paren); len(held) != 0 {
		t.Fatalf("the paren plan asked %d questions again, want its answers read from the store", len(held))
	}
}
