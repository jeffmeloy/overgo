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

// PickVideoCaptions reads the VidGen caption file and picks count captions by seed, each
// with two questions. A question's distractors are near misses: captions of other scenes
// of the same source video, which describe the same people and places doing something
// else, and, where the source video has too few, which is most of the file, the captions
// of other videos nearest the true one in wording, from a sample the seed fixes. A round
// takes captions the round before did not. Captions of unrelated videos are told apart by
// any clip of the right subject, and a suite of them reads full marks for every generator
// that works.
func PickVideoCaptions(path string, count int, seed uint64) ([]VideoCaptionItem, error) {
	if count <= 0 {
		return nil, errors.New("evaluation: a generated-video suite draws at least one caption")
	}
	captions, err := readCaptions(path)
	if err != nil {
		return nil, err
	}
	identities := make([]string, 0, len(captions))
	bySource := map[string][]string{}
	for identity := range captions {
		identities = append(identities, identity)
	}
	slices.Sort(identities)
	for _, identity := range identities {
		source, _, _ := strings.Cut(identity, sceneMarker)
		bySource[source] = append(bySource[source], identity)
	}
	if len(identities) < count+captionChoices {
		return nil, fmt.Errorf("evaluation: %d captions in %s, too few to draw %d and offer %d choices each", len(identities), path, count, captionChoices)
	}
	rng := rand.New(rand.NewPCG(seed, seed))
	picked := slices.Clone(identities)
	rng.Shuffle(len(picked), func(left, right int) { picked[left], picked[right] = picked[right], picked[left] })
	// The captions after the picked ones, in the seed's order, are the sample the near
	// misses of other videos are found in.
	pool := picked[count:min(len(picked), count+nearMissPool)]
	poolWords := make([]map[string]int, len(pool))
	for index, candidate := range pool {
		poolWords[index] = wordCounts(captions[candidate])
	}
	var items []VideoCaptionItem
	for _, identity := range picked[:count] {
		source, _, _ := strings.Cut(identity, sceneMarker)
		// The sibling scenes whose captions differ from this one's and from each other's,
		// then the captions of other videos nearest this one in wording, nearest first.
		var siblings []string
		for _, sibling := range bySource[source] {
			if captions[sibling] != captions[identity] && !slices.ContainsFunc(siblings, func(kept string) bool { return captions[kept] == captions[sibling] }) {
				siblings = append(siblings, sibling)
			}
		}
		rng.Shuffle(len(siblings), func(left, right int) { siblings[left], siblings[right] = siblings[right], siblings[left] })
		words := wordCounts(captions[identity])
		nearest := make([]int, 0, len(pool))
		scores := make([]float64, len(pool))
		for index, candidate := range pool {
			if candidateSource, _, _ := strings.Cut(candidate, sceneMarker); candidateSource != source {
				scores[index] = wordOverlap(words, poolWords[index])
				nearest = append(nearest, index)
			}
		}
		slices.SortFunc(nearest, func(left, right int) int {
			return cmp.Or(cmp.Compare(scores[right], scores[left]), strings.Compare(pool[left], pool[right]))
		})
		item := VideoCaptionItem{ID: identity, Caption: captions[identity]}
		var offered []string
		for range generatedVideoRounds {
			options := []string{identity}
			taken := min(len(siblings), captionChoices-1)
			options, siblings = append(options, siblings[:taken]...), siblings[taken:]
			for len(options) < captionChoices {
				if len(nearest) == 0 {
					return nil, fmt.Errorf("evaluation: %s holds too few distinct captions to offer %d choices", path, captionChoices)
				}
				other := pool[nearest[0]]
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
			item.Questions = append(item.Questions, question)
		}
		items = append(items, item)
	}
	return items, nil
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
			var listing []string
			for position, option := range question.Options {
				listing = append(listing, fmt.Sprintf("%c. %s", captionLetters[position], option))
			}
			suite.Cases = append(suite.Cases, GeneratedAnswerCase{
				Name:      fmt.Sprintf("%s/%s/%d", generatedVideoSource, item.ID, round),
				Prompt:    mediaPrompt(MediaVideo, name, fmt.Sprintf(generatedVideoQuestion, strings.Join(listing, "\n"))),
				MaxTokens: captionAnswerTokens,
				Answers:   []string{string(captionLetters[question.Answer])},
			})
		}
	}
	return suite, nil
}
