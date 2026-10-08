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
// The numbers near a number are as long as it and differ from it and from
// each other in one digit alone, its last that is not a zero, turned forward
// or back as the draw says; none opens with a zero. The titles near a
// title are its two swaps made one at a time and together, and a swap that
// names a word the title does not hold once, or puts in a word it holds, is
// refused. The suite names each case by its brief and question, shows the
// picture, asks for a letter and accepts the letter of the text asked for; a
// brief with no picture, a suite with no judge and more briefs than the list
// holds are refused.
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

	forward, backward := func() bool { return false }, func() bool { return true }
	for number, want := range map[string][2][]string{
		"9":    {{"1", "2", "3"}, {"8", "7", "6"}},
		"1":    {{"2", "3", "4"}, {"9", "8", "7"}},
		"80":   {{"90", "10", "20"}, {"70", "60", "50"}},
		"412":  {{"413", "414", "415"}, {"411", "410", "419"}},
		"2024": {{"2025", "2026", "2027"}, {"2023", "2022", "2021"}},
		"5280": {{"5290", "5200", "5210"}, {"5270", "5260", "5250"}},
	} {
		for way, back := range []func() bool{forward, backward} {
			near, err := nearNumbers(number, back)
			if err != nil || !slices.Equal(near, want[way]) {
				t.Errorf("the numbers near %s are %q (%v), want %q", number, near, err, want[way])
			}
		}
	}
	for _, text := range []string{"", "4 2", "twelve", "0", "000"} {
		if _, err := nearNumbers(text, forward); err == nil {
			t.Errorf("%q was read as a number that has numbers near it", text)
		}
	}
	if near, err := nearTitles("THE NEW HILL ROUTE", [][]string{{"NEW", "OLD"}, {"HILL", "MILL"}}); err != nil ||
		!slices.Equal(near, []string{"THE OLD HILL ROUTE", "THE NEW MILL ROUTE", "THE OLD MILL ROUTE"}) {
		t.Errorf("the titles near a title are %q (%v)", near, err)
	}
	for name, swaps := range map[string][][]string{
		"one swap":                    {{"NEW", "OLD"}},
		"a word the title lacks":      {{"NEW", "OLD"}, {"ROAD", "MILL"}},
		"a word the title holds":      {{"NEW", "HILL"}, {"ROUTE", "ROAD"}},
		"one word swapped twice":      {{"NEW", "OLD"}, {"NEW", "LAST"}},
		"a swap that names one word":  {{"NEW"}, {"HILL", "MILL"}},
		"a swap that puts nothing in": {{"NEW", " "}, {"HILL", "MILL"}},
	} {
		if _, err := nearTitles("THE NEW HILL ROUTE", swaps); err == nil {
			t.Errorf("a title was given titles near it by %s", name)
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

// TestDrawnTextChoicesDoNotGiveTheAnswerAway holds the four texts of a
// question to saying nothing of which was asked for. A judge that cannot read
// the picture has the four texts and nothing else, and three ways of guessing
// from them alone are tried over many seeds: the text nearest all the others,
// which found the answer every time when each wrong text was one change from
// the right one; the smallest number; and the largest. None is right more
// often than one time in four and a fifth of that again. The fourth is the
// titles' own: where a title falls among its four in alphabetical order is
// fixed by the briefs, and it is first, or last, in no more than half of them.
//
// What this cannot hold is what a judge knows of the world: that a title
// reads more naturally than a title near it, or that a source is dated in a
// past year. That is measured, by asking the judge about pictures with no
// text in them.
func TestDrawnTextChoicesDoNotGiveTheAnswerAway(t *testing.T) {
	t.Parallel()
	const seeds = 40
	briefs, err := drawnBriefs()
	if err != nil {
		t.Fatal(err)
	}
	// apart counts the places two texts of one length differ at, or their words where the lengths differ.
	apart := func(left, right string) int {
		differ := 0
		if len(left) == len(right) {
			for index := range left {
				if left[index] != right[index] {
					differ++
				}
			}
			return differ
		}
		leftWords, rightWords := strings.Fields(left), strings.Fields(right)
		for index := range min(len(leftWords), len(rightWords)) {
			if leftWords[index] != rightWords[index] {
				differ++
			}
		}
		return differ + max(len(leftWords), len(rightWords)) - min(len(leftWords), len(rightWords))
	}
	number := func(text string) int {
		value := 0
		for _, r := range text {
			if strings.ContainsRune(decimalDigits, r) {
				value = value*len(decimalDigits) + strings.IndexRune(decimalDigits, r)
			}
		}
		return value
	}
	var questions, numbers, nearest, smallest, largest int
	first, last := 0, 0
	for seed := range uint64(seeds) {
		items, err := PickDrawnText(len(briefs), seed)
		if err != nil {
			t.Fatal(err)
		}
		for _, item := range items {
			for asked, question := range item.Questions {
				questions++
				central := slices.MinFunc(question.Choices, func(left, right string) int {
					total := func(text string) int {
						sum := 0
						for _, other := range question.Choices {
							sum += apart(text, other)
						}
						return sum
					}
					return total(left) - total(right)
				})
				if central == question.Answer {
					nearest++
				}
				if asked == 0 {
					if seed == 0 {
						sorted := slices.Sorted(slices.Values(question.Choices))
						if sorted[0] == question.Answer {
							first++
						}
						if sorted[len(sorted)-1] == question.Answer {
							last++
						}
					}
					continue
				}
				numbers++
				if slices.MinFunc(question.Choices, func(left, right string) int { return number(left) - number(right) }) == question.Answer {
					smallest++
				}
				if slices.MaxFunc(question.Choices, func(left, right string) int { return number(left) - number(right) }) == question.Answer {
					largest++
				}
			}
		}
	}
	chance := 1 / float64(1+nearTexts)
	for name, right := range map[string]float64{
		"the text nearest all the others": float64(nearest) / float64(questions),
		"the smallest number":             float64(smallest) / float64(numbers),
		"the largest number":              float64(largest) / float64(numbers),
	} {
		// Over some thousands of questions a guess at chance is within a twentieth of it. A way of guessing
		// that does worse than chance gives a judge nothing it could use without knowing which way that is.
		if right > chance+chance/5 {
			t.Errorf("guessing %s is right %.3f of the time, and chance is %.2f", name, right, chance)
		}
	}
	if half := len(briefs) / 2; first > half || last > half {
		t.Errorf("the title asked for is the first of its four in alphabetical order in %d of %d briefs and the last in %d", first, len(briefs), last)
	}
	t.Logf("%d questions over %d seeds: the nearest text is right %.3f of the time, the smallest number %.3f, the largest %.3f; a title is first of its four in %d briefs and last in %d",
		questions, seeds, float64(nearest)/float64(questions), float64(smallest)/float64(numbers), float64(largest)/float64(numbers), first, last)
}
