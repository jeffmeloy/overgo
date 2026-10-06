package evaluation

import (
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

const (
	captionMatchingSource = "vidgen-1m/caption-matching"
	captionMatchingSchema = "overgo/vidgen-caption-matching/v1"
	// captionChoices is how many captions a case offers: the true one and the
	// distractors, lettered in the order they are shown.
	captionChoices = 4
	captionLetters = "ABCD"
	// captionAnswerTokens is the generated length of an answer: its letter.
	captionAnswerTokens = 1
	clipExtension       = ".mp4"
	// sceneMarker joins a source video's identity to a scene number in a clip's name.
	sceneMarker     = "-Scene-"
	captionQuestion = "Which description matches the video?\n%s\nAnswer with the letter only."
)

// CaptionMatchingSpec names what a caption-matching suite is built from: a
// directory of clips named by their captions' identities, the caption file,
// how many clips the suite asks about and the seed that picks them.
type CaptionMatchingSpec struct {
	Clips    string
	Captions string
	Cases    int
	Seed     uint64
}

type captionRow struct {
	Vid     string `json:"vid"`
	Caption string `json:"caption"`
}

// captionRound is one clip a suite asks about and the captions it offers for
// it, in the order they are shown, the clip's own among them.
type captionRound struct {
	vid     string
	options []string
}

// captionRounds picks the clips of a spec and, for each, its true caption and
// three captions of other scenes of the same source video where the caption
// file has them, which a model can only tell apart by looking, and of other
// clips otherwise. The same spec picks the same rounds.
func captionRounds(spec CaptionMatchingSpec) ([]captionRound, map[string]string, error) {
	if spec.Cases <= 0 {
		return nil, nil, errors.New("evaluation: a caption-matching suite asks about at least one clip")
	}
	entries, err := os.ReadDir(spec.Clips)
	if err != nil {
		return nil, nil, err
	}
	captions, err := readCaptions(spec.Captions)
	if err != nil {
		return nil, nil, err
	}
	var clips []string
	for _, entry := range entries {
		vid, isClip := strings.CutSuffix(entry.Name(), clipExtension)
		if _, captioned := captions[vid]; isClip && captioned {
			clips = append(clips, vid)
		}
	}
	slices.Sort(clips)
	if len(clips) < spec.Cases {
		return nil, nil, fmt.Errorf("evaluation: %d captioned clips in %s, fewer than the %d cases asked", len(clips), spec.Clips, spec.Cases)
	}
	identities := make([]string, 0, len(captions))
	bySource := map[string][]string{}
	for vid := range captions {
		identities = append(identities, vid)
	}
	slices.Sort(identities)
	for _, vid := range identities {
		source, _, _ := strings.Cut(vid, sceneMarker)
		bySource[source] = append(bySource[source], vid)
	}
	rng := rand.New(rand.NewPCG(spec.Seed, spec.Seed))
	rng.Shuffle(len(clips), func(left, right int) { clips[left], clips[right] = clips[right], clips[left] })
	var rounds []captionRound
	for _, vid := range clips[:spec.Cases] {
		options := []string{vid}
		source, _, _ := strings.Cut(vid, sceneMarker)
		siblings := slices.DeleteFunc(slices.Clone(bySource[source]), func(other string) bool { return other == vid })
		rng.Shuffle(len(siblings), func(left, right int) { siblings[left], siblings[right] = siblings[right], siblings[left] })
		distinct := func(other string) bool {
			return !slices.ContainsFunc(options, func(chosen string) bool { return captions[chosen] == captions[other] })
		}
		for _, sibling := range siblings {
			if len(options) < captionChoices && distinct(sibling) {
				options = append(options, sibling)
			}
		}
		for draws := 0; len(options) < captionChoices; draws++ {
			if draws > len(identities) {
				return nil, nil, fmt.Errorf("evaluation: %s holds too few distinct captions to offer %d choices", filepath.Base(spec.Captions), captionChoices)
			}
			if other := identities[rng.IntN(len(identities))]; distinct(other) {
				options = append(options, other)
			}
		}
		rng.Shuffle(len(options), func(left, right int) { options[left], options[right] = options[right], options[left] })
		rounds = append(rounds, captionRound{vid: vid, options: options})
	}
	return rounds, captions, nil
}

// BuildCaptionMatching builds the held-out video caption-matching suite: each
// case shows one clip and asks which of four captions describes it. The answer
// is the letter, scored exactly.
func BuildCaptionMatching(spec CaptionMatchingSpec) (GeneratedAnswerSuite, error) {
	rounds, captions, err := captionRounds(spec)
	if err != nil {
		return GeneratedAnswerSuite{}, err
	}
	suite := GeneratedAnswerSuite{
		Kind: GeneratedAnswerKind, Schema: captionMatchingSchema, Source: captionMatchingSource,
		Transforms: []string{TransformTrimSpace, transformLowercase},
	}
	for _, round := range rounds {
		var listing []string
		for index, option := range round.options {
			listing = append(listing, fmt.Sprintf("%c. %s", captionLetters[index], captions[option]))
		}
		suite.Cases = append(suite.Cases, GeneratedAnswerCase{
			Name:      suite.Source + "/" + round.vid,
			Prompt:    mediaPrompt(MediaVideo, round.vid+clipExtension, fmt.Sprintf(captionQuestion, strings.Join(listing, "\n"))),
			MaxTokens: captionAnswerTokens,
			Answers:   []string{string(captionLetters[slices.Index(round.options, round.vid)])},
		})
	}
	return suite, nil
}

// readCaptions reads the caption file, a JSON array of {vid, caption}, by
// identity; an identity named twice with different captions is refused.
func readCaptions(path string) (map[string]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	if _, err := decoder.Token(); err != nil {
		return nil, fmt.Errorf("evaluation: read captions %s: %w", filepath.Base(path), err)
	}
	captions := map[string]string{}
	for decoder.More() {
		var row captionRow
		if err := decoder.Decode(&row); err != nil {
			return nil, fmt.Errorf("evaluation: read captions %s: %w", filepath.Base(path), err)
		}
		if row.Vid == "" || row.Caption == "" || captions[row.Vid] != "" && captions[row.Vid] != row.Caption {
			return nil, fmt.Errorf("evaluation: caption row %q of %s is empty or repeats an identity with another caption", row.Vid, filepath.Base(path))
		}
		captions[row.Vid] = row.Caption
	}
	return captions, nil
}
