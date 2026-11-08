package evaluation

import (
	_ "embed"
	"fmt"
	"math/rand/v2"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/strictjson"
)

// Composed cases keep a picture suite off its ceiling: on 2026-11-08 SenseNova read 100 of 100 drawn texts and
// Krea-2-Turbo 0.94 of TIFA's questions, where a worse picture cannot lower the reading. GenEval 2 keeps strong models
// off the ceiling with prompts of 3 to 10 atoms (objects, counts, colours, positions, text); a suite composes such
// cases from its own atoms, one question an atom, drawn by seed so a suite built again holds the same ones.
const (
	fewestAtoms, mostAtoms = 3, 10
	// mostCount is the largest count an object is composed in, which a count question offers every count up to.
	mostCount = 4
	// atomsPerObject is what a composed object adds at most: its count, its colour and its place beside the one before.
	atomsPerObject = 3
	// composedPrompt asks for an infographic of a title and figures at two sizes, worded as drawnTextPrompt is.
	composedPrompt = `An infographic of %s, in a flat vector illustration style on a light cream background. ` +
		`At the top is the main title in large bold dark blue letters: "%s". ` +
		`Below the title are %d panels in a grid, each with a simple icon above one line of large text, reading in order %s. ` +
		`The layout is clean, the lettering is clear and legible, and there is no other text.`
)

// countWords write a count up to mostCount as a caption says it.
var countWords = []string{"one", "two", "three", "four"}

// composedAtomsJSON holds the atoms of TIFA's own questions a caption composes: countable objects, each as one and as
// many, and plain colours.
//
//go:embed composed_atoms.json
var composedAtomsJSON []byte

// ComposeTIFA composes count captions by seed, each of 3 to 10 atoms from TIFA's countable objects and plain colours:
// two to four objects from left to right, each in a count of one to four and some in a colour. Each atom is one
// question: how many of an object there are, what colour it is, and which of two neighbours is further left, the two
// named in an order the seed draws so the question does not say.
func ComposeTIFA(count int, seed uint64) ([]TIFAItem, error) {
	var atoms struct {
		Objects [][]string `json:"objects"`
		Colours []string   `json:"colours"`
	}
	if err := strictjson.DecodeBytes(composedAtomsJSON, &atoms); err != nil {
		return nil, fmt.Errorf("evaluation: composed atoms: %w", err)
	}
	rng := rand.New(rand.NewPCG(seed, seed))
	shuffled := func(choices []string) []string {
		rng.Shuffle(len(choices), func(left, right int) { choices[left], choices[right] = choices[right], choices[left] })
		return choices
	}
	items := make([]TIFAItem, count)
	for index := range items {
		total := fewestAtoms + rng.IntN(mostAtoms-fewestAtoms+1)
		// n objects make n counts and n-1 places beside a neighbour, and the colours make up the rest, n at most.
		objects := (total + atomsPerObject) / atomsPerObject
		coloured := rng.Perm(objects)[:total-objects-(objects-1)]
		picked := rng.Perm(len(atoms.Objects))[:objects]
		item := TIFAItem{ID: fmt.Sprintf("composed_%d", index)}
		var phrases []string
		for place, at := range picked {
			name, plural, many := atoms.Objects[at][0], atoms.Objects[at][1], 1+rng.IntN(mostCount)
			noun, asked := plural, "what color are the "+plural+"?"
			if many == 1 {
				noun, asked = name, "what color is the "+name+"?"
			}
			counts := make([]string, mostCount)
			for value := range counts {
				counts[value] = strconv.Itoa(value + 1)
			}
			item.Questions = append(item.Questions, TIFAQuestion{Question: "how many " + plural + " are there?", Choices: counts, Answer: strconv.Itoa(many)})
			if slices.Contains(coloured, place) {
				colours := shuffled(slices.Clone(atoms.Colours))[:mostCount]
				colour := colours[0]
				noun = colour + " " + noun
				item.Questions = append(item.Questions, TIFAQuestion{Question: asked, Choices: shuffled(colours), Answer: colour})
			}
			if place > 0 {
				neighbours := shuffled([]string{atoms.Objects[picked[place-1]][0], name})
				item.Questions = append(item.Questions, TIFAQuestion{Question: "which is further to the left, the " + neighbours[0] + " or the " + neighbours[1] + "?",
					Choices: neighbours, Answer: atoms.Objects[picked[place-1]][0]})
			}
			phrases = append(phrases, countWords[many-1]+" "+noun)
		}
		item.Caption = "From left to right: " + strings.Join(phrases[:len(phrases)-1], ", ") + " and " + phrases[len(phrases)-1] + "."
		items[index] = item
	}
	return items, nil
}

// ComposeDrawnText composes count infographics by seed, each of 3 to 10 texts: a brief's title and two to nine lines
// of figures drawn from every brief's lines and small print. Each text is one question, offering it among three near
// it as PickDrawnText's do.
func ComposeDrawnText(count int, seed uint64) ([]TIFAItem, error) {
	briefs, err := drawnBriefs()
	if err != nil {
		return nil, err
	}
	var lines []drawnLine
	for _, brief := range briefs {
		lines = append(append(lines, brief.Lines...), brief.Print)
	}
	rng := rand.New(rand.NewPCG(seed, seed))
	back := func() bool { return rng.Uint64()&1 == 1 }
	items := make([]TIFAItem, count)
	for index := range items {
		titled := briefs[rng.IntN(len(briefs))]
		item := TIFAItem{ID: fmt.Sprintf("composed_%d", index)}
		ask := func(text string, near []string) {
			choices := append([]string{text}, near...)
			rng.Shuffle(len(choices), func(left, right int) { choices[left], choices[right] = choices[right], choices[left] })
			item.Questions = append(item.Questions, TIFAQuestion{Question: drawnTextQuestion, Choices: choices, Answer: text})
		}
		titles, err := nearTitles(titled.Title, titled.Swaps)
		if err != nil {
			return nil, err
		}
		ask(titled.Title, titles)
		var quoted []string
		for _, at := range rng.Perm(len(lines))[:fewestAtoms-1+rng.IntN(mostAtoms-fewestAtoms+1)] {
			numbers, err := nearNumbers(lines[at].Number, back)
			if err != nil {
				return nil, err
			}
			for position, number := range numbers {
				numbers[position] = lines[at].with(number)
			}
			ask(lines[at].with(lines[at].Number), numbers)
			quoted = append(quoted, strconv.Quote(lines[at].with(lines[at].Number)))
		}
		item.Caption = fmt.Sprintf(composedPrompt, titled.Subject, titled.Title, len(quoted), strings.Join(quoted, ", "))
		items[index] = item
	}
	return items, nil
}
