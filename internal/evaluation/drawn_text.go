package evaluation

import (
	_ "embed"
	"errors"
	"fmt"
	"math/rand/v2"
	"slices"
	"strings"

	"overgo/internal/strictjson"
)

const (
	drawnTextSource = "overgo/drawn-text"
	drawnTextSchema = "overgo/drawn-text-questions/v2"
	// drawnTextQuestion asks a judge to read a picture: the texts it is offered
	// differ by a word or a digit, so what the picture shows of its subject does
	// not answer it.
	drawnTextQuestion = "Which of these is written in the picture?"
	// drawnTextPrompt asks for an infographic that holds five stated texts at
	// three sizes: a title, three lines of figures, and a line of small print.
	// It is worded as the infographic model's own examples word a prompt, the
	// style and the layout described and each text given in quotation marks.
	drawnTextPrompt = `An infographic about %s, in a flat vector illustration style on a light cream background. ` +
		`At the top is the main title in large bold dark blue letters: "%s". ` +
		`Below the title are three panels in a row, each with a simple icon above one line of large text. ` +
		`The left panel reads "%s", the middle panel reads "%s", and the right panel reads "%s". ` +
		`At the bottom, in small grey letters, is the line "%s". ` +
		`The layout is clean, the lettering is clear and legible, and there is no other text.`
	// nearTexts is how many texts near the one asked for a question offers
	// beside it, and how many lines with a figure a brief holds.
	nearTexts = 3
	// titleSwaps is how many words of its title a brief names another word
	// for, and swapWords the words a swap holds, the title's and the one put
	// in its place. Two swaps make three titles near a title: the one made,
	// the other made, and both.
	titleSwaps, swapWords = 2, 2
	// decimalDigits are the digits a number is written in, in order, so that a
	// digit is turned to another by its place among them.
	decimalDigits = "0123456789"
	// smallTurn and largeTurn are how far one digit of a number is turned to
	// make the three numbers near it: by the one, by the other, and by both.
	// No two of the four digits that gives are the same, among ten digits or
	// among nine, and none is far from the digit asked for: a year stays a
	// year someone might have written.
	smallTurn, largeTurn = 1, 2
)

// drawnBriefsJSON holds the infographics the measure asks for: everyday
// subjects, each with figures no one could guess, so that a picture is read
// or its question is not answered.
//
//go:embed drawn_briefs.json
var drawnBriefsJSON []byte

// drawnLine is one line of an infographic that states a figure: the words
// before the number, the number as it is written, and the words after it.
type drawnLine struct {
	Before string `json:"before,omitzero"`
	Number string `json:"number"`
	After  string `json:"after,omitzero"`
}

// with writes the line with another number in the number's place.
func (line drawnLine) with(number string) string { return line.Before + number + line.After }

// drawnBrief is an infographic to ask for: what it is about, its title with
// two swaps of one of its words for another, three lines that each state a
// figure, and a line of small print that states a year.
type drawnBrief struct {
	ID      string      `json:"id"`
	Subject string      `json:"subject"`
	Title   string      `json:"title"`
	Swaps   [][]string  `json:"swaps"`
	Lines   []drawnLine `json:"lines"`
	Print   drawnLine   `json:"print"`
}

// drawnBriefs reads the embedded briefs. Each is named once, and holds a
// subject, a title with two swaps that make three titles near it, and three
// lines.
func drawnBriefs() ([]drawnBrief, error) {
	var document struct {
		Briefs []drawnBrief `json:"briefs"`
	}
	if err := strictjson.DecodeBytes(drawnBriefsJSON, &document); err != nil {
		return nil, fmt.Errorf("evaluation: drawn-text briefs: %w", err)
	}
	seen := map[string]bool{}
	for _, brief := range document.Briefs {
		if brief.ID == "" || seen[brief.ID] || brief.Subject == "" || len(brief.Lines) != nearTexts {
			return nil, fmt.Errorf("evaluation: drawn-text brief %q is repeated or incomplete", brief.ID)
		}
		if _, err := nearTitles(brief.Title, brief.Swaps); err != nil {
			return nil, fmt.Errorf("evaluation: drawn-text brief %s: %w", brief.ID, err)
		}
		seen[brief.ID] = true
	}
	return document.Briefs, nil
}

// A question must not answer itself. Three texts each made by one change to
// the text asked for point at it: it is the one text a single change away
// from every other, and a judge shown a picture with no text in it found it
// two times in three. So the four texts a question offers are the corners of
// a square: two changes, each made or not, which leaves every text one change
// from two of the others and two changes from the third, and none standing
// apart. The changes say nothing of the picture's subject either, or a judge
// would read the answer from what the picture shows.

// nearTitles gives the three titles near a title: the first of its two swaps
// made, the second, and both. A swap names a word the title holds once and a
// word it does not hold to put in its place, and the two swap different
// words.
func nearTitles(title string, swaps [][]string) ([]string, error) {
	words := strings.Fields(title)
	if len(words) == 0 || title != strings.Join(words, " ") || len(swaps) != titleSwaps {
		return nil, fmt.Errorf("evaluation: the title %q is not words one space apart with %d swaps", title, titleSwaps)
	}
	held := func(word string) int {
		return len(words) - len(slices.DeleteFunc(slices.Clone(words), func(other string) bool { return other == word }))
	}
	swapped := func(made ...[]string) string {
		changed := slices.Clone(words)
		for _, swap := range made {
			changed[slices.Index(words, swap[0])] = swap[1]
		}
		return strings.Join(changed, " ")
	}
	for _, swap := range swaps {
		if len(swap) != swapWords || held(swap[0]) != 1 || strings.TrimSpace(swap[1]) == "" || held(swap[1]) != 0 {
			return nil, fmt.Errorf("evaluation: the title %q does not hold the word a swap names once, or holds the word it puts in: %q", title, swap)
		}
	}
	if swaps[0][0] == swaps[1][0] {
		return nil, fmt.Errorf("evaluation: the title %q swaps one word twice", title)
	}
	return []string{swapped(swaps[0]), swapped(swaps[1]), swapped(swaps[0], swaps[1])}, nil
}

// nearNumbers gives the three numbers near one written in digits, each as
// long as it: one of its digits turned by a small step, by a large one, and by
// both. Each step goes forward or back as back draws, so the number asked for
// is as likely to be any corner of the four as another, and nothing in the
// four says which it was. The digit turned is the last that is not a zero, so
// a round number's four are all as round as it, and a judge who expects a
// round figure learns nothing. A number's first digit is turned among the
// digits from one to nine, so no number opens with a zero or is one; any
// other digit among all ten.
func nearNumbers(number string, back func() bool) ([]string, error) {
	if number == "" || strings.Trim(number, decimalDigits) != "" {
		return nil, fmt.Errorf("evaluation: %q is not a number written in digits", number)
	}
	place := strings.LastIndexAny(number, decimalDigits[1:])
	if place < 0 {
		return nil, fmt.Errorf("evaluation: %q has no digit to turn that is not a zero", number)
	}
	digits := decimalDigits
	if place == 0 {
		digits = decimalDigits[1:]
	}
	at := strings.IndexByte(digits, number[place])
	turned := func(by int) string {
		return number[:place] + string(digits[((at+by)%len(digits)+len(digits))%len(digits)]) + number[place+1:]
	}
	small, large := smallTurn, largeTurn
	if back() {
		small = -small
	}
	if back() {
		large = -large
	}
	return []string{turned(small), turned(large), turned(small + large)}, nil
}

// PickDrawnText picks count infographics to ask an image model for, by seed,
// from a fixed list of briefs. Each item's caption is the prompt, which gives
// the five texts the picture is to hold in quotation marks: a title, three
// lines that each state a figure, and a line of small print. Each of the five
// is one question, offering the text asked for among three near it, a title
// with one or both of two words swapped or a line with its number's last
// digit turned, in an order the seed draws. No text near an asked one is in
// any prompt, so a picture that holds what it was asked for holds one of the
// four, and the four do not say which.
func PickDrawnText(count int, seed uint64) ([]TIFAItem, error) {
	briefs, err := drawnBriefs()
	if err != nil {
		return nil, err
	}
	if count <= 0 || count > len(briefs) {
		return nil, fmt.Errorf("evaluation: a drawn-text suite draws between 1 and %d infographics, not %d", len(briefs), count)
	}
	rng := rand.New(rand.NewPCG(seed, seed))
	rng.Shuffle(len(briefs), func(left, right int) { briefs[left], briefs[right] = briefs[right], briefs[left] })
	back := func() bool { return rng.Uint64()&1 == 1 }
	items := make([]TIFAItem, 0, count)
	for _, brief := range briefs[:count] {
		// The prompt names its subject and then the five texts, in the order asked.
		named := []any{brief.Subject}
		item := TIFAItem{ID: brief.ID}
		ask := func(text string, near []string) {
			named = append(named, text)
			choices := append([]string{text}, near...)
			rng.Shuffle(len(choices), func(left, right int) { choices[left], choices[right] = choices[right], choices[left] })
			item.Questions = append(item.Questions, TIFAQuestion{Question: drawnTextQuestion, Choices: choices, Answer: text})
		}
		titles, err := nearTitles(brief.Title, brief.Swaps)
		if err != nil {
			return nil, fmt.Errorf("evaluation: drawn-text brief %s: %w", brief.ID, err)
		}
		ask(brief.Title, titles)
		for _, line := range append(slices.Clone(brief.Lines), brief.Print) {
			numbers, err := nearNumbers(line.Number, back)
			if err != nil {
				return nil, fmt.Errorf("evaluation: drawn-text brief %s: %w", brief.ID, err)
			}
			for position, number := range numbers {
				numbers[position] = line.with(number)
			}
			ask(line.with(line.Number), numbers)
		}
		item.Caption = fmt.Sprintf(drawnTextPrompt, named...)
		items = append(items, item)
	}
	return items, nil
}

// BuildDrawnText builds the held-out suite that asks a judge to read each
// infographic an image model drew: one case for each text the picture was to
// hold, naming the picture by its file name, which images maps the brief's
// identity to, and asking which of four texts is written in it. The judge is
// named in the suite, since the score depends on what it can read.
func BuildDrawnText(items []TIFAItem, images map[string]string, judge string) (GeneratedAnswerSuite, error) {
	if len(items) == 0 {
		return GeneratedAnswerSuite{}, errors.New("evaluation: a drawn-text suite asks about at least one picture")
	}
	return buildPictureQuestions(drawnTextSource, drawnTextSchema, items, images, judge)
}
