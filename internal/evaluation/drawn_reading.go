package evaluation

import (
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/runrecord"
)

// CharacterMetric is the mean over a suite's pictures of the share of the characters each was asked to hold that its
// reader read back (PictureReadings), beside the judge's choice of text.
const CharacterMetric = "characters-read"

// PictureReadings is what an OCR reader read off each picture of a suite of drawn text, by the picture's name as its
// cases' prompts give it, with the reader's model, recipe and projection: a second instrument beside the judge, which
// chooses among near texts and so cannot say how many characters a picture got right. A picture is named and not
// grouped since suites built before cases had groups ask about many pictures under none.
type PictureReadings struct {
	Reader MediaRuntime      `json:"reader"`
	Read   map[string]string `json:"read"`
}

// Asked is the picture the case asks about and the text it asks whether that picture holds: the choice its answer
// letters, read from the lettered listing its question holds ("A. text"); false where the case names no picture or
// lists no such text.
func (c GeneratedAnswerCase) Asked() (picture, text string, asked bool) {
	kind, picture, question, framed := ParseMediaPrompt(c.Prompt)
	if !framed || kind != MediaImage || len(c.Answers) == 0 {
		return "", "", false
	}
	for line := range strings.Lines(question) {
		if text, listed := strings.CutPrefix(strings.TrimRight(line, "\r\n"), c.Answers[0]+". "); listed {
			return picture, text, true
		}
	}
	return "", "", false
}

// askedTexts are the texts the cases ask about, by the picture each case asks about.
func askedTexts(cases []GeneratedAnswerCase) (map[string][]string, error) {
	asked := map[string][]string{}
	for _, testCase := range cases {
		picture, text, listed := testCase.Asked()
		if !listed {
			return nil, fmt.Errorf("evaluation: case %s asks about no picture's listed text a reading can be scored by", testCase.Name)
		}
		asked[picture] = append(asked[picture], text)
	}
	return asked, nil
}

// validate refuses readings with no reader, of a picture no case asks about, or missing a picture a case asks about.
func (r PictureReadings) validate(cases []GeneratedAnswerCase) error {
	if !r.Reader.Model.Valid() || !r.Reader.Recipe.Valid() || !r.Reader.Projection.Valid() {
		return errors.New("evaluation: a suite's readings name the reader's model, recipe and projection")
	}
	asked, err := askedTexts(cases)
	if err != nil {
		return err
	}
	for picture := range asked {
		if _, read := r.Read[picture]; !read {
			return fmt.Errorf("evaluation: the suite's readings hold none of %s", picture)
		}
	}
	for picture := range r.Read {
		if _, held := asked[picture]; !held {
			return fmt.Errorf("evaluation: the suite's readings read %s, which no case asks about", picture)
		}
	}
	return nil
}

// Metrics are the reading metrics a judged reading of the suite records: the mean over its pictures of each one's
// characterShare of the texts its cases asked about.
func (r PictureReadings) Metrics(cases []GeneratedAnswerCase) []runrecord.Metric {
	asked, _ := askedTexts(cases)
	var mean float64
	for picture, texts := range asked {
		mean += characterShare(texts, r.Read[picture]) / float64(len(asked))
	}
	return []runrecord.Metric{{Name: CharacterMetric, Value: mean, Direction: runrecord.DirectionMaximize}}
}

// characterShare is the share of the characters of asked that read holds: each text counts its characters less the
// fewest edits that make it a stretch of read, none below zero, read and asked alike in lower case with each run of
// space one space, since a reader breaks a picture's text into rows where the picture breaks it.
func characterShare(asked []string, read string) float64 {
	fold := func(text string) []rune { return []rune(strings.ToLower(strings.Join(strings.Fields(text), " "))) }
	page := fold(read)
	var held, total int
	for _, text := range asked {
		want := fold(text)
		total += len(want)
		held += max(len(want)-stretchDistance(want, page), 0)
	}
	if total == 0 {
		return 0
	}
	return float64(held) / float64(total)
}

// stretchDistance is the fewest insertions, deletions and substitutions that turn want into some stretch of page:
// the edit distance with page's start and end free (Sellers, 1980).
func stretchDistance(want, page []rune) int {
	previous := make([]int, len(page)+1)
	for i, x := range want {
		current := make([]int, len(page)+1)
		current[0] = i + 1
		for j, y := range page {
			substitute := previous[j]
			if x != y {
				substitute++
			}
			current[j+1] = min(previous[j+1]+1, current[j]+1, substitute)
		}
		previous = current
	}
	return slices.Min(previous)
}
