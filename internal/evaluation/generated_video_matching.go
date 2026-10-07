package evaluation

import (
	"cmp"
	"errors"
	"fmt"
	"math/rand/v2"
	"slices"
	"strings"
)

const (
	generatedVideoSource = "vidgen-1m/generated-video-caption-matching"
	generatedVideoSchema = "overgo/vidgen-generated-video-caption-matching/v2"
	// generatedVideoQuestion asks which of the captions a clip drawn from one of them shows.
	generatedVideoQuestion = "Which description matches the video?\n%s\nAnswer with the letter only."
	// generatedVideoRounds is how many questions a clip is asked, each with its own distractors:
	// a clip is cheap to ask about and costly to draw.
	generatedVideoRounds = 2
	// nearMissPool is how many captions of other videos, in the seed's order, a caption's
	// near misses are sought among: enough to hold captions of the same subject.
	nearMissPool = 20000
)

// VideoCaptionQuestion is one question about a clip: the captions the judge chooses among,
// in the order they are lettered, and which of them the clip was drawn from.
type VideoCaptionQuestion struct {
	Options []string
	Answer  int
}

// VideoCaptionItem is a caption a video model is asked to draw, and the questions the
// judge is asked when it is shown the clip.
type VideoCaptionItem struct {
	ID        string
	Caption   string
	Questions []VideoCaptionQuestion
}

// captionField is what near-miss questions about a caption are drawn from: every caption
// by its identity, the scenes of each source video, and the sample of captions, with their
// words, that a caption's nearest in wording are sought among.
type captionField struct {
	path      string
	captions  map[string]string
	bySource  map[string][]string
	pool      []string
	poolWords []map[string]int
}

// newCaptionField groups the sorted identities of path's captions by source video and
// counts the words of the pool's captions.
func newCaptionField(path string, captions map[string]string, identities, pool []string) captionField {
	field := captionField{path: path, captions: captions, bySource: map[string][]string{}, pool: pool, poolWords: make([]map[string]int, len(pool))}
	for _, identity := range identities {
		source, _, _ := strings.Cut(identity, sceneMarker)
		field.bySource[source] = append(field.bySource[source], identity)
	}
	for index, candidate := range pool {
		field.poolWords[index] = wordCounts(captions[candidate])
	}
	return field
}

// questions asks rounds questions about the caption of identity. A question's distractors
// are near misses: captions of other scenes of the same source video, which describe the
// same people and places doing something else, and, where the source video has too few,
// the captions of other videos nearest the true one in wording. A round takes captions the
// round before did not.
func (field captionField) questions(rng *rand.Rand, identity string, rounds int) ([]VideoCaptionQuestion, error) {
	captions := field.captions
	source, _, _ := strings.Cut(identity, sceneMarker)
	// The sibling scenes whose captions differ from this one's and from each other's,
	// then the captions of other videos nearest this one in wording, nearest first.
	var siblings []string
	for _, sibling := range field.bySource[source] {
		if captions[sibling] != captions[identity] && !slices.ContainsFunc(siblings, func(kept string) bool { return captions[kept] == captions[sibling] }) {
			siblings = append(siblings, sibling)
		}
	}
	rng.Shuffle(len(siblings), func(left, right int) { siblings[left], siblings[right] = siblings[right], siblings[left] })
	words := wordCounts(captions[identity])
	nearest := make([]int, 0, len(field.pool))
	scores := make([]float64, len(field.pool))
	for index, candidate := range field.pool {
		if candidateSource, _, _ := strings.Cut(candidate, sceneMarker); candidateSource != source {
			scores[index] = wordOverlap(words, field.poolWords[index])
			nearest = append(nearest, index)
		}
	}
	slices.SortFunc(nearest, func(left, right int) int {
		return cmp.Or(cmp.Compare(scores[right], scores[left]), strings.Compare(field.pool[left], field.pool[right]))
	})
	var questions []VideoCaptionQuestion
	var offered []string
	for range rounds {
		options := []string{identity}
		taken := min(len(siblings), captionChoices-1)
		options, siblings = append(options, siblings[:taken]...), siblings[taken:]
		for len(options) < captionChoices {
			if len(nearest) == 0 {
				return nil, fmt.Errorf("evaluation: %s holds too few distinct captions to offer %d choices", field.path, captionChoices)
			}
			other := field.pool[nearest[0]]
			nearest = nearest[1:]
			distinct := func(chosen string) bool { return captions[chosen] == captions[other] }
			if !slices.ContainsFunc(options, distinct) && !slices.ContainsFunc(offered, distinct) {
				options = append(options, other)
			}
		}
		offered = append(offered, options...)
		rng.Shuffle(len(options), func(left, right int) { options[left], options[right] = options[right], options[left] })
		question := VideoCaptionQuestion{Answer: slices.Index(options, identity)}
		for _, option := range options {
			question.Options = append(question.Options, captions[option])
		}
		questions = append(questions, question)
	}
	return questions, nil
}

// PickVideoCaptions reads the VidGen caption file and picks count captions by seed, each
// with two questions whose distractors are near misses, the sample of other videos'
// captions being the ones the seed orders after the picked. Captions of unrelated videos
// are told apart by any clip of the right subject, and a suite of them reads full marks
// for every generator that works.
func PickVideoCaptions(path string, count int, seed uint64) ([]VideoCaptionItem, error) {
	if count <= 0 {
		return nil, errors.New("evaluation: a generated-video suite draws at least one caption")
	}
	captions, err := readCaptions(path)
	if err != nil {
		return nil, err
	}
	identities := make([]string, 0, len(captions))
	for identity := range captions {
		identities = append(identities, identity)
	}
	slices.Sort(identities)
	if len(identities) < count+captionChoices {
		return nil, fmt.Errorf("evaluation: %d captions in %s, too few to draw %d and offer %d choices each", len(identities), path, count, captionChoices)
	}
	rng := rand.New(rand.NewPCG(seed, seed))
	picked := slices.Clone(identities)
	rng.Shuffle(len(picked), func(left, right int) { picked[left], picked[right] = picked[right], picked[left] })
	field := newCaptionField(path, captions, identities, picked[count:min(len(picked), count+nearMissPool)])
	var items []VideoCaptionItem
	for _, identity := range picked[:count] {
		questions, err := field.questions(rng, identity, generatedVideoRounds)
		if err != nil {
			return nil, err
		}
		items = append(items, VideoCaptionItem{ID: identity, Caption: captions[identity], Questions: questions})
	}
	return items, nil
}

// lettered lists a question's options, each on a line under its letter.
func lettered(options []string) string {
	listing := make([]string, len(options))
	for position, option := range options {
		listing[position] = fmt.Sprintf("%c. %s", captionLetters[position], option)
	}
	return strings.Join(listing, "\n")
}

// BuildGeneratedVideoMatching builds the held-out suite that asks a judge which of four
// captions a clip drawn from one of them shows: one case for each question of each
// caption, naming the clip by its file name, which videos maps the caption's identity
// to, answered by the letter of the caption it was drawn from. The judge is named in the
// suite, since the score depends on it.
func BuildGeneratedVideoMatching(items []VideoCaptionItem, videos map[string]string, judge string) (GeneratedAnswerSuite, error) {
	if strings.TrimSpace(judge) == "" {
		return GeneratedAnswerSuite{}, errors.New("evaluation: a suite of questions about clips names its judge")
	}
	suite := GeneratedAnswerSuite{
		Kind: GeneratedAnswerKind, Schema: generatedVideoSchema, Source: generatedVideoSource, Judge: judge,
		Transforms: []string{TransformTrimSpace, transformLowercase},
	}
	for _, item := range items {
		name, drawn := videos[item.ID]
		if !drawn {
			return GeneratedAnswerSuite{}, fmt.Errorf("evaluation: caption %s has no clip", item.ID)
		}
		for round, question := range item.Questions {
			suite.Cases = append(suite.Cases, GeneratedAnswerCase{
				Name:      fmt.Sprintf("%s/%s/%d", generatedVideoSource, item.ID, round),
				Prompt:    mediaPrompt(MediaVideo, name, fmt.Sprintf(generatedVideoQuestion, lettered(question.Options))),
				MaxTokens: captionAnswerTokens,
				Answers:   []string{string(captionLetters[question.Answer])},
			})
		}
	}
	return suite, nil
}
