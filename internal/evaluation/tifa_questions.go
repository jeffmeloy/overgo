package evaluation

import (
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"os"
	"slices"
	"strings"
)

const (
	tifaSource = "tifa-v1.0/image-questions"
	tifaSchema = "overgo/tifa-v1.0-image-questions/v1"
	// tifaAnswerTokens is the generated length of an answer: its letter.
	tifaAnswerTokens = 1
	// tifaMinimumChoices is the fewest answers a question offers.
	tifaMinimumChoices = 2
	// tifaLetters letter the choices of a question, in the order the file gives them.
	tifaLetters = "ABCDEFGH"
)

// TIFAQuestion is one multiple-choice question about what a picture of a caption shows.
type TIFAQuestion struct {
	Question string   `json:"question"`
	Choices  []string `json:"choices"`
	Answer   string   `json:"answer"`
}

// TIFAItem is a caption from TIFA v1.0 and the questions that check whether a
// picture drawn from it shows what it says.
type TIFAItem struct {
	ID        string
	Caption   string
	Questions []TIFAQuestion
}

// tifaRow is one row of the TIFA question-answer file.
type tifaRow struct {
	ID       string   `json:"id"`
	Caption  string   `json:"caption"`
	Question string   `json:"question"`
	Choices  []string `json:"choices"`
	Answer   string   `json:"answer"`
}

// PickTIFA reads the TIFA v1.0 question-answer file and picks count captions
// by seed, each with the questions the file holds for it, in the file's order.
// A caption none of whose questions is well formed is not a candidate.
func PickTIFA(path string, count int, seed uint64) ([]TIFAItem, error) {
	if count <= 0 {
		return nil, errors.New("evaluation: a TIFA suite draws at least one caption")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var rows []tifaRow
	if err := json.Unmarshal(data, &rows); err != nil {
		return nil, fmt.Errorf("evaluation: read %s: %w", path, err)
	}
	var items []TIFAItem
	index := map[string]int{}
	for _, row := range rows {
		if strings.TrimSpace(row.ID) == "" || strings.TrimSpace(row.Caption) == "" || strings.TrimSpace(row.Question) == "" ||
			len(row.Choices) < tifaMinimumChoices || len(row.Choices) > len(tifaLetters) || !slices.Contains(row.Choices, row.Answer) {
			continue
		}
		at, known := index[row.ID]
		if !known {
			at = len(items)
			index[row.ID] = at
			items = append(items, TIFAItem{ID: row.ID, Caption: row.Caption})
		}
		items[at].Questions = append(items[at].Questions, TIFAQuestion{Question: row.Question, Choices: row.Choices, Answer: row.Answer})
	}
	if len(items) < count {
		return nil, fmt.Errorf("evaluation: %d usable captions in %s, fewer than the %d asked", len(items), path, count)
	}
	rng := rand.New(rand.NewPCG(seed, seed))
	rng.Shuffle(len(items), func(left, right int) { items[left], items[right] = items[right], items[left] })
	return items[:count], nil
}

// BuildTIFAQuestions builds the held-out suite that asks a judge about each
// picture drawn from a TIFA caption: one case for each question, naming the
// picture by its file name, which images maps the caption's identity to, and
// asking which choice, lettered in the file's order, the picture shows. The
// judge is named in the suite, since the score depends on it.
func BuildTIFAQuestions(items []TIFAItem, images map[string]string, judge string) (GeneratedAnswerSuite, error) {
	if strings.TrimSpace(judge) == "" {
		return GeneratedAnswerSuite{}, errors.New("evaluation: a suite of questions about pictures names its judge")
	}
	suite := GeneratedAnswerSuite{
		Kind: GeneratedAnswerKind, Schema: tifaSchema, Source: tifaSource, Judge: judge,
		Transforms: []string{TransformTrimSpace, transformLowercase},
	}
	for _, item := range items {
		name, drawn := images[item.ID]
		if !drawn {
			return GeneratedAnswerSuite{}, fmt.Errorf("evaluation: caption %s has no picture", item.ID)
		}
		for number, question := range item.Questions {
			var listing []string
			for position, choice := range question.Choices {
				listing = append(listing, fmt.Sprintf("%c. %s", tifaLetters[position], choice))
			}
			suite.Cases = append(suite.Cases, GeneratedAnswerCase{
				Name:      fmt.Sprintf("%s/%s/%d", tifaSource, item.ID, number),
				Prompt:    mediaPrompt(MediaImage, name, fmt.Sprintf("%s\n%s\nAnswer with the letter only.", question.Question, strings.Join(listing, "\n"))),
				MaxTokens: tifaAnswerTokens,
				Answers:   []string{string(tifaLetters[slices.Index(question.Choices, question.Answer)])},
			})
		}
	}
	return suite, nil
}
