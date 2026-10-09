package evaluation

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"os"
	"slices"
	"strings"
)

const (
	localEditSource = "vidgen-1m/generated-video-local-edit"
	localEditSchema = "overgo/vidgen-generated-video-local-edit/v1"
	// localEditInstruction tells an editor to change one thing a clip shows and
	// nothing else: the thing by the colour its caption gives it, and the colour
	// it is to have.
	localEditInstruction = "Change the %s %s to a %s %s."
)

// wornThings are things a caption says someone wears, each named by one word a
// colour stands before: what an editor of one thing in a clip is asked to
// recolour.
var wornThings = []string{"shirt", "t-shirt", "jacket", "dress", "sweater", "hoodie", "cap", "hat", "coat"}

// recolours gives, for a colour a caption names, the colour an edit is asked
// to change it to: one far from it, so that a judge tells the two apart at a
// glance. A colour that takes "an" before it is left out, since the caption
// with another colour in its place would not read.
var recolours = map[string]string{
	"red": "blue", "blue": "yellow", "green": "red", "yellow": "blue", "white": "red", "black": "yellow",
	"pink": "green", "purple": "yellow", "brown": "blue", "gray": "red", "grey": "red",
}

// colourJoiners are words that, standing before a colour, make it part of a
// longer one, "black and white" or "light blue": such a thing is not named by
// the one colour, and is not asked about.
var colourJoiners = []string{"and", "light", "dark", "bright", "pale", "navy", "sky", "deep"}

// wornColour finds the one worn thing a caption names by a colour: the colour
// and the thing. A caption that names no worn thing by one colour, names two,
// or already holds the colour the thing would be changed to gives none: an
// instruction about it would not say which thing, or the caption with the new
// colour would be true of something else in the clip.
func wornColour(caption string) (colour, thing string, found bool) {
	lowered := strings.ToLower(caption)
	if len(lowered) != len(caption) {
		return "", "", false
	}
	words := strings.FieldsFunc(lowered, func(char rune) bool { return char != '-' && (char < 'a' || char > 'z') })
	for at := 0; at+1 < len(words); at++ {
		if _, coloured := recolours[words[at]]; !coloured || !slices.Contains(wornThings, words[at+1]) {
			continue
		}
		if found && (words[at] != colour || words[at+1] != thing) || at > 0 && slices.Contains(colourJoiners, words[at-1]) {
			return "", "", false
		}
		colour, thing, found = words[at], words[at+1], true
	}
	if !found || slices.Contains(words, recolours[colour]) {
		return "", "", false
	}
	return colour, thing, true
}

// recoloured writes a caption with the thing's colour changed wherever the
// caption names the thing by it, the rest as it is.
func recoloured(caption, colour, thing, changed string) string {
	lowered, phrase := strings.ToLower(caption), colour+" "+thing
	var written strings.Builder
	for from := 0; ; {
		at := strings.Index(lowered[from:], phrase)
		if at < 0 {
			written.WriteString(caption[from:])
			return written.String()
		}
		at += from
		written.WriteString(caption[from:at])
		written.WriteString(changed)
		from = at + len(colour)
	}
}

// PickLocalVideoEdits picks count captioned clips of a directory by seed whose
// caption names one worn thing by a colour, and gives each the instruction to
// change that thing's colour and nothing else: what an editor that preserves
// what it is not told to change is for. The one question about the edited clip
// offers four captions: the clip's own with the new colour, which is true of a
// right edit; its own as it is, true of a clip given back unchanged; and those
// of two other videos nearest it in wording, true of a clip that no longer
// shows what it showed. An editor that changes nothing is wrong on every clip,
// and so is one that recolours the thing and redraws the scene. The other
// captions are never of another scene of the clip's own source video, which
// shows the same people in the same place and could be true of a right edit.
//
// The count of right answers is the reading; which wrong caption was chosen
// does not sort the fault. A judge takes the old colour's caption whenever the
// new one is false of the thing named, whether the editor gave the clip back
// as it was, recoloured something else in it, or replaced the thing with
// another. And a caption says nothing of whose face it is: an edit that draws
// the wearer as another person in the same place is right by this measure.
func PickLocalVideoEdits(clips, path string, count int, seed uint64) ([]VideoEditItem, error) {
	if count <= 0 {
		return nil, errors.New("evaluation: a video-edit suite edits at least one clip")
	}
	entries, err := os.ReadDir(clips)
	if err != nil {
		return nil, err
	}
	captions, err := readCaptions(path)
	if err != nil {
		return nil, err
	}
	var local []string
	for _, entry := range entries {
		identity, isClip := strings.CutSuffix(entry.Name(), clipExtension)
		if _, _, worn := wornColour(captions[identity]); isClip && worn {
			local = append(local, identity)
		}
	}
	slices.Sort(local)
	if len(local) < count {
		return nil, fmt.Errorf("evaluation: %d captioned clips in %s name one worn thing by a colour, fewer than the %d asked", len(local), clips, count)
	}
	pool := make([]string, 0, len(captions))
	for identity := range captions {
		pool = append(pool, identity)
	}
	slices.Sort(pool)
	rng := rand.New(rand.NewPCG(seed, seed))
	rng.Shuffle(len(local), func(left, right int) { local[left], local[right] = local[right], local[left] })
	rng.Shuffle(len(pool), func(left, right int) { pool[left], pool[right] = pool[right], pool[left] })
	// A field that knows no source video's scenes offers only other videos' captions.
	field := newCaptionField(path, captions, nil, pool[:min(len(pool), nearMissPool)])
	var items []VideoEditItem
	for _, identity := range local[:count] {
		own := captions[identity]
		colour, thing, _ := wornColour(own)
		changed := recolours[colour]
		content, err := field.questions(rng, identity, 1)
		if err != nil {
			return nil, err
		}
		// The right answer first, then the clip unchanged, then two other videos.
		statements := []string{recoloured(own, colour, thing, changed), own}
		for _, option := range content[0].Options {
			if option != own && len(statements) < editChoices {
				statements = append(statements, option)
			}
		}
		order := rng.Perm(len(statements))
		question := VideoCaptionQuestion{Options: make([]string, len(statements))}
		for position, statement := range order {
			question.Options[position] = statements[statement]
			if statement == 0 {
				question.Answer = position
			}
		}
		items = append(items, VideoEditItem{
			ID: identity, Caption: own, Instruction: fmt.Sprintf(localEditInstruction, colour, thing, changed, thing), Question: question,
		})
	}
	return items, nil
}

// BuildGeneratedVideoLocalEdit builds the held-out suite that asks a judge one
// question about each clip an editor was told to change one thing in: which of
// four captions, the clip's own with the thing's new colour and with its old,
// and two other videos', is true of it. videos maps a clip's identity to the
// file name of its edited clip. The judge is named in the suite, since the
// score depends on it.
func BuildGeneratedVideoLocalEdit(items []VideoEditItem, videos map[string]string, judge string) (GeneratedAnswerSuite, error) {
	return buildVideoEdit(localEditSource, localEditSchema, items, videos, judge)
}
