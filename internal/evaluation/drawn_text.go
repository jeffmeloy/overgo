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
	drawnTextSchema = "overgo/drawn-text-questions/v1"
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
	// decimalDigits are the digits a number is written in, in order, so that a
	// digit is turned to another by its place among them.
	decimalDigits = "0123456789"
	// nearTurn and farTurn are how far a number's last digit is turned to make
	// two numbers near it, and firstTurn how far its first.
	nearTurn, farTurn, firstTurn = 3, 6, 1
	// twoDigits is the least length a number has two digits to exchange at.
	twoDigits = 2
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
// three titles that differ from it by a word, three lines that each state a
// figure, and a line of small print that states a year.
type drawnBrief struct {
	ID      string      `json:"id"`
	Subject string      `json:"subject"`
	Title   string      `json:"title"`
	Titles  []string    `json:"titles"`
	Lines   []drawnLine `json:"lines"`
	Print   drawnLine   `json:"print"`
}

// drawnBriefs reads the embedded briefs. Each is named once, and holds a
// subject, a title with three other titles, and three lines.
func drawnBriefs() ([]drawnBrief, error) {
	var document struct {
		Briefs []drawnBrief `json:"briefs"`
	}
	if err := strictjson.DecodeBytes(drawnBriefsJSON, &document); err != nil {
		return nil, fmt.Errorf("evaluation: drawn-text briefs: %w", err)
	}
	seen := map[string]bool{}
	for _, brief := range document.Briefs {
		if brief.ID == "" || seen[brief.ID] || brief.Subject == "" || brief.Title == "" || len(brief.Titles) != nearTexts || len(brief.Lines) != nearTexts ||
			slices.Contains(brief.Titles, brief.Title) {
			return nil, fmt.Errorf("evaluation: drawn-text brief %q is repeated or incomplete", brief.ID)
		}
		seen[brief.ID] = true
	}
	return document.Briefs, nil
}

// nearNumbers gives three numbers near one written in digits, each as long
// as it and none of them it: its last two digits exchanged, its last digit
// turned, its first digit turned, its last digit turned further, in that order
// and passing over any that repeats one before it or would open with a zero.
// They are the numbers a careless reading of the picture gives.
func nearNumbers(number string) ([]string, error) {
	turned := func(at, by int) string {
		digit := decimalDigits[(strings.IndexByte(decimalDigits, number[at])+by)%len(decimalDigits)]
		// A number of several digits does not open with a zero.
		if at == 0 && len(number) > 1 && digit == decimalDigits[0] {
			digit = decimalDigits[firstTurn]
		}
		return number[:at] + string(digit) + number[at+1:]
	}
	if number == "" || strings.Trim(number, decimalDigits) != "" {
		return nil, fmt.Errorf("evaluation: %q is not a number written in digits", number)
	}
	last := len(number) - 1
	var candidates, near []string
	if len(number) >= twoDigits && (len(number) > twoDigits || number[last] != decimalDigits[0]) {
		candidates = append(candidates, number[:last-1]+string(number[last])+string(number[last-1]))
	}
	for _, candidate := range append(candidates, turned(last, nearTurn), turned(0, firstTurn), turned(last, farTurn)) {
		if len(near) < nearTexts && candidate != number && !slices.Contains(near, candidate) {
			near = append(near, candidate)
		}
	}
	if len(near) < nearTexts {
		return nil, fmt.Errorf("evaluation: %q has fewer than %d numbers near it", number, nearTexts)
	}
	return near, nil
}

// PickDrawnText picks count infographics to ask an image model for, by seed,
// from a fixed list of briefs. Each item's caption is the prompt, which gives
// the five texts the picture is to hold in quotation marks: a title, three
// lines that each state a figure, and a line of small print. Each of the five
// is one question, offering the text asked for among three near it, a title
// with one word changed or a line with its number misread, in an order the
// seed draws. No text near an asked one is in any prompt, so a picture that
// holds what it was asked for holds one of the four.
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
		ask(brief.Title, brief.Titles)
		for _, line := range append(slices.Clone(brief.Lines), brief.Print) {
			numbers, err := nearNumbers(line.Number)
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
