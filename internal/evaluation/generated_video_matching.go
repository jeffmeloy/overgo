package evaluation

import (
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
// with two questions. A question's distractors are captions of other scenes of the same
// source video, which describe the same people and places doing something else and so
// can be told from the true one only by what the clip shows; a round takes scenes the
// round before did not, and where the source video has too few, captions of other
// videos fill it. Captions of unrelated videos alone are told apart by any clip of the
// right subject, and a suite of them reads full marks for every generator that works.
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
	var items []VideoCaptionItem
	for _, identity := range picked[:count] {
		source, _, _ := strings.Cut(identity, sceneMarker)
		// The sibling scenes whose captions differ from this one's and from each other's.
		var siblings []string
		for _, sibling := range bySource[source] {
			if captions[sibling] != captions[identity] && !slices.ContainsFunc(siblings, func(kept string) bool { return captions[kept] == captions[sibling] }) {
				siblings = append(siblings, sibling)
			}
		}
		rng.Shuffle(len(siblings), func(left, right int) { siblings[left], siblings[right] = siblings[right], siblings[left] })
		item := VideoCaptionItem{ID: identity, Caption: captions[identity]}
		for range generatedVideoRounds {
			options := []string{identity}
			taken := min(len(siblings), captionChoices-1)
			options, siblings = append(options, siblings[:taken]...), siblings[taken:]
			for draws := 0; len(options) < captionChoices; draws++ {
				if draws > len(identities) {
					return nil, fmt.Errorf("evaluation: %s holds too few distinct captions to offer %d choices", path, captionChoices)
				}
				other := identities[rng.IntN(len(identities))]
				otherSource, _, _ := strings.Cut(other, sceneMarker)
				if otherSource != source && !slices.ContainsFunc(options, func(chosen string) bool { return captions[chosen] == captions[other] }) {
					options = append(options, other)
				}
			}
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
