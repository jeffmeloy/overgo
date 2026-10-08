package evaluation

import (
	"slices"
	"strings"
	"testing"
)

// TestDrawnTextSuiteAsksForTheTextItWasToldToDraw holds the measure of what
// an infographic model is for: text it was asked to draw, read back. Every
// brief of the fixed list makes a prompt that gives five texts in quotation
// marks, a title, three lines with a figure and a line of small print, and one
// question for each. A question offers four different texts, the one asked for
// among three near it, and no near text is in the prompt: a picture that holds
// what it was asked for holds exactly one of the four, and nothing about the
// subject tells which. The same seed picks the same briefs and the same order
// of choices, and another seed another.
//
// The numbers near a number are as long as it, differ from it and from each
// other, and never open with a zero where it has more than one digit. The
// suite names each case by its brief and question, shows the picture, asks
// for a letter and accepts the letter of the text asked for; a brief with no
// picture, a suite with no judge and more briefs than the list holds are
// refused.
func TestDrawnTextSuiteAsksForTheTextItWasToldToDraw(t *testing.T) {
	t.Parallel()
	const seed = 17
	briefs, err := drawnBriefs()
	if err != nil {
		t.Fatal(err)
	}
	listed := len(briefs)
	items, err := PickDrawnText(listed, seed)
	if err != nil {
		t.Fatal(err)
	}
	again, err := PickDrawnText(listed, seed)
	if err != nil {
		t.Fatal(err)
	}
	other, err := PickDrawnText(listed, seed+1)
	if err != nil {
		t.Fatal(err)
	}
	same := func(left, right []TIFAItem) bool {
		return slices.EqualFunc(left, right, func(a, b TIFAItem) bool {
			return a.ID == b.ID && a.Caption == b.Caption && slices.EqualFunc(a.Questions, b.Questions, func(p, q TIFAQuestion) bool {
				return p.Answer == q.Answer && slices.Equal(p.Choices, q.Choices)
			})
		})
	}
	if !same(items, again) || same(items, other) {
		t.Fatal("a seed does not fix the briefs and the order of their choices, or two seeds give the same")
	}
	asked := 1 + nearTexts + 1
	identities := map[string]bool{}
	for _, item := range items {
		if identities[item.ID] || len(item.Questions) != asked {
			t.Fatalf("brief %s is picked twice or asks %d questions, want %d", item.ID, len(item.Questions), asked)
		}
		identities[item.ID] = true
		for _, question := range item.Questions {
			choices := slices.Clone(question.Choices)
			slices.Sort(choices)
			if len(question.Choices) != 1+nearTexts || len(slices.Compact(choices)) != 1+nearTexts || !slices.Contains(question.Choices, question.Answer) {
				t.Fatalf("brief %s offers %q for %q, want four different texts with the one asked for", item.ID, question.Choices, question.Answer)
			}
			for _, choice := range question.Choices {
				if quoted := strings.Contains(item.Caption, `"`+choice+`"`); quoted != (choice == question.Answer) {
					t.Fatalf("brief %s: the prompt holds %q in quotation marks: %t, and it is the text asked for: %t", item.ID, choice, quoted, choice == question.Answer)
				}
			}
		}
	}
	if len(identities) != listed {
		t.Fatalf("%d briefs were picked of %d", len(identities), listed)
	}

	for number, want := range map[string][]string{
		"9":    {"2", "0", "5"},
		"44":   {"47", "54", "40"},
		"80":   {"83", "90", "86"},
		"412":  {"421", "415", "512"},
		"9400": {"9403", "1400", "9406"},
		"2024": {"2042", "2027", "3024"},
	} {
		near, err := nearNumbers(number)
		if err != nil || !slices.Equal(near, want) {
			t.Errorf("the numbers near %s are %q (%v), want %q", number, near, err, want)
		}
	}
	for _, text := range []string{"", "4 2", "twelve"} {
		if _, err := nearNumbers(text); err == nil {
			t.Errorf("%q was read as a number", text)
		}
	}

	few := items[:2]
	images := map[string]string{few[0].ID: "first.png", few[1].ID: "second.png"}
	suite, err := BuildDrawnText(few, images, "a judge")
	if err != nil {
		t.Fatal(err)
	}
	if len(suite.Cases) != len(few)*asked || suite.Source != drawnTextSource || suite.Judge != "a judge" {
		t.Fatalf("the suite holds %d cases of %s judged by %q", len(suite.Cases), suite.Source, suite.Judge)
	}
	first, question := suite.Cases[0], few[0].Questions[0]
	letter := string(tifaLetters[slices.Index(question.Choices, question.Answer)])
	if first.Name != drawnTextSource+"/"+few[0].ID+"/0" || !slices.Equal(first.Answers, []string{letter}) || first.MaxTokens != tifaAnswerTokens ||
		!strings.Contains(first.Prompt, "first.png") || !strings.Contains(first.Prompt, drawnTextQuestion+"\nA. "+question.Choices[0]+"\n") ||
		!strings.HasSuffix(first.Prompt, "Answer with the letter only.") {
		t.Fatalf("the first case is %+v, want the first brief's title asked of its picture and answered %s", first, letter)
	}
	if _, err := CompileGeneratedAnswer(suite); err != nil {
		t.Fatalf("the suite does not compile: %v", err)
	}
	if _, err := BuildDrawnText(few, map[string]string{few[0].ID: "first.png"}, "a judge"); err == nil {
		t.Error("a suite was built with a brief that has no picture")
	}
	if _, err := BuildDrawnText(few, images, " "); err == nil {
		t.Error("a suite was built with no judge")
	}
	if _, err := BuildDrawnText(nil, images, "a judge"); err == nil {
		t.Error("a suite was built that asks about no picture")
	}
	for _, count := range []int{0, listed + 1} {
		if _, err := PickDrawnText(count, seed); err == nil {
			t.Errorf("%d briefs were picked of %d", count, listed)
		}
	}
}
