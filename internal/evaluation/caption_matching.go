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
	// frameMatchingSource and frameMatchingSchema name the suite whose cases show
	// a still frame cut from the clip, for a model that reads one image.
	frameMatchingSource = "vidgen-1m/frame-caption-matching"
	frameMatchingSchema = "overgo/vidgen-frame-caption-matching/v1"
	frameExtension      = ".png"
	// captionChoices is how many captions a case offers: the true one and the
	// distractors, lettered in the order they are shown.
	captionChoices = 4
	captionLetters = "ABCD"
	// captionAnswerTokens is the generated length of an answer: its letter.
	captionAnswerTokens = 1
	clipExtension       = ".mp4"
	// sceneMarker joins a source video's identity to a scene number in a clip's name.
	sceneMarker     = "-Scene-"
	captionQuestion = "Which description matches the %s?\n%s\nAnswer with the letter only."
)

// CaptionMatchingSpec names what a caption-matching suite is built from: a
// directory of clips named by their captions' identities, the caption file,
// how many clips the suite asks about and the seed that picks them.
type CaptionMatchingSpec struct {
	Clips    string
	Captions string
	Cases    int
	Seed     uint64
	// Kind is what each case shows: the clip itself, or a frame cut from it,
	// named by the clip's identity and the extension of a frame. The frames
	// are the caller's to write.
	Kind MediaKind
}

type captionRow struct {
	Vid     string `json:"vid"`
	Caption string `json:"caption"`
}

// BuildCaptionMatching builds the held-out video caption-matching suite: each
// case shows one clip and asks which of four captions describes it, the true
// caption and three captions of other scenes of the same source video where
// the caption file has them, which the model can only tell apart by looking,
// and of other clips otherwise. The answer is the letter, scored exactly.
func BuildCaptionMatching(spec CaptionMatchingSpec) (GeneratedAnswerSuite, error) {
	if spec.Cases <= 0 {
		return GeneratedAnswerSuite{}, errors.New("evaluation: a caption-matching suite asks about at least one clip")
	}
	entries, err := os.ReadDir(spec.Clips)
	if err != nil {
		return GeneratedAnswerSuite{}, err
	}
	captions, err := readCaptions(spec.Captions)
	if err != nil {
		return GeneratedAnswerSuite{}, err
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
		return GeneratedAnswerSuite{}, fmt.Errorf("evaluation: %d captioned clips in %s, fewer than the %d cases asked", len(clips), spec.Clips, spec.Cases)
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
	// A model that reads one image answers in a sentence that opens with its
	// letter, so its answer is read by that letter.
	suite := GeneratedAnswerSuite{
		Kind: GeneratedAnswerKind, Schema: captionMatchingSchema, Source: captionMatchingSource,
		Transforms: []string{TransformTrimSpace, transformLowercase},
	}
	kind, extension, noun := MediaVideo, clipExtension, "video"
	if spec.Kind == MediaImage {
		kind, extension, noun = MediaImage, frameExtension, "image"
		suite.Schema, suite.Source = frameMatchingSchema, frameMatchingSource
		suite.Transforms = []string{TransformTrimSpace, transformLeadingLetter, transformLowercase}
	}
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
				return GeneratedAnswerSuite{}, fmt.Errorf("evaluation: %s holds too few distinct captions to offer %d choices", filepath.Base(spec.Captions), captionChoices)
			}
			if other := identities[rng.IntN(len(identities))]; distinct(other) {
				options = append(options, other)
			}
		}
		rng.Shuffle(len(options), func(left, right int) { options[left], options[right] = options[right], options[left] })
		var listing []string
		for index, option := range options {
			listing = append(listing, fmt.Sprintf("%c. %s", captionLetters[index], captions[option]))
		}
		suite.Cases = append(suite.Cases, GeneratedAnswerCase{
			Name:      suite.Source + "/" + vid,
			Prompt:    mediaPrompt(kind, vid+extension, fmt.Sprintf(captionQuestion, noun, strings.Join(listing, "\n"))),
			MaxTokens: captionAnswerTokens,
			Answers:   []string{string(captionLetters[slices.Index(options, vid)])},
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
