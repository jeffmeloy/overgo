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
	generatedVideoSchema = "overgo/vidgen-generated-video-caption-matching/v1"
	// generatedVideoQuestion asks which of the captions a clip drawn from one of them shows.
	generatedVideoQuestion = "Which description matches the video?\n%s\nAnswer with the letter only."
)

// VideoCaptionItem is a caption a video model is asked to draw, and the captions the
// judge chooses among when it is shown the clip: the caption itself and others, in the
// order they are lettered.
type VideoCaptionItem struct {
	ID      string
	Caption string
	Options []string
	Answer  int
}

// PickVideoCaptions reads the VidGen caption file and picks count captions by seed, each
// with captionChoices-1 captions of other source videos as the distractors the judge must
// tell it from, and the options in a shuffled order. A caption is a candidate only if
// it differs from every other the case offers.
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
	var items []VideoCaptionItem
	for _, identity := range picked[:count] {
		source, _, _ := strings.Cut(identity, sceneMarker)
		options := []string{identity}
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
		item := VideoCaptionItem{ID: identity, Caption: captions[identity], Answer: slices.Index(options, identity)}
		for _, option := range options {
			item.Options = append(item.Options, captions[option])
		}
		items = append(items, item)
	}
	return items, nil
}

// BuildGeneratedVideoMatching builds the held-out suite that asks a judge which of four
// captions a clip drawn from one of them shows: one case for each caption, naming the
// clip by its file name, which videos maps the caption's identity to, answered by the
// letter of the caption it was drawn from. The judge is named in the suite, since the
// score depends on it.
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
		var listing []string
		for position, option := range item.Options {
			listing = append(listing, fmt.Sprintf("%c. %s", captionLetters[position], option))
		}
		suite.Cases = append(suite.Cases, GeneratedAnswerCase{
			Name:      generatedVideoSource + "/" + item.ID,
			Prompt:    mediaPrompt(MediaVideo, name, fmt.Sprintf(generatedVideoQuestion, strings.Join(listing, "\n"))),
			MaxTokens: captionAnswerTokens,
			Answers:   []string{string(captionLetters[item.Answer])},
		})
	}
	return suite, nil
}
