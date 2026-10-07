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
	generatedEditSource = "vidgen-1m/generated-video-edit"
	generatedEditSchema = "overgo/vidgen-generated-video-edit/v2"
	// generatedEditQuestion asks what an edited clip shows and whether it was changed.
	generatedEditQuestion = "Which statement is true of the video?\n%s\nAnswer with the letter only."
	// editChoices is how many statements the one question about an edited clip offers:
	// the clip's own caption and one near miss, each with the change made and without.
	editChoices = 4
)

// videoChange is one change an editor is told to make: the instruction it is given, the
// statement that is true of a clip once it is made, the statement that is true of a clip
// it was not made to, and the caption words that mean a clip shows it already, which
// leaves nothing to tell apart.
type videoChange struct {
	instruction, statement, absence string
	words                           []string
}

// videoChanges are changes of a whole clip that a viewer sees at a glance whatever the
// clip shows, so that one list serves every source clip.
var videoChanges = []videoChange{
	{"Make it snow heavily.", "Heavy snow is falling.", "No snow is falling.", []string{"snow", "snowing", "snowy", "snowfall"}},
	{"Make it rain heavily.", "Heavy rain is falling.", "No rain is falling.", []string{"rain", "raining", "rainy"}},
	{"Change the time to night.", "It is night.", "It is not night.", []string{"night", "nighttime", "dark"}},
	{"Cover the scene in thick fog.", "Thick fog covers the scene.", "There is no fog.", []string{"fog", "foggy", "mist", "misty"}},
	{"Turn the video into a black-and-white pencil sketch.", "It is a black-and-white pencil sketch.", "It is ordinary footage, not a sketch.", []string{"sketch", "drawing", "drawn", "pencil"}},
	{"Turn the video into a colorful cartoon.", "It is a colorful cartoon.", "It is ordinary footage, not a cartoon.", []string{"cartoon", "animated", "animation"}},
}

// VideoEditItem is a clip a video editor is asked to change: the clip's identity and
// caption, the instruction the editor is given, and the one question the judge is asked
// about the result.
type VideoEditItem struct {
	ID          string
	Caption     string
	Instruction string
	Question    VideoCaptionQuestion
}

// PickVideoEdits picks count captioned clips of a directory by seed and gives each an
// instruction from a fixed list of whole-clip changes. A clip whose caption says it shows
// one of the changes already is passed over. An edit is right only when the change was
// made and the clip still shows what it showed, so the one question about it offers four
// statements: the clip's own caption with the change made, the same caption without it,
// which is true of a clip given back as it was, and a near-miss caption each way, which
// are true of a clip that no longer shows what it showed. An editor that changes nothing
// is wrong on every clip, and so is one that draws snow over another scene.
func PickVideoEdits(clips, path string, count int, seed uint64) ([]VideoEditItem, error) {
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
	shown := func(caption string) bool {
		words := wordCounts(caption)
		return slices.ContainsFunc(videoChanges, func(change videoChange) bool {
			return slices.ContainsFunc(change.words, func(word string) bool { return words[word] > 0 })
		})
	}
	var local []string
	for _, entry := range entries {
		identity, isClip := strings.CutSuffix(entry.Name(), clipExtension)
		if caption, captioned := captions[identity]; isClip && captioned && !shown(caption) {
			local = append(local, identity)
		}
	}
	slices.Sort(local)
	if len(local) < count {
		return nil, fmt.Errorf("evaluation: %d captioned clips in %s show none of the changes already, fewer than the %d asked", len(local), clips, count)
	}
	identities := make([]string, 0, len(captions))
	for identity := range captions {
		identities = append(identities, identity)
	}
	slices.Sort(identities)
	rng := rand.New(rand.NewPCG(seed, seed))
	rng.Shuffle(len(local), func(left, right int) { local[left], local[right] = local[right], local[left] })
	pool := slices.Clone(identities)
	rng.Shuffle(len(pool), func(left, right int) { pool[left], pool[right] = pool[right], pool[left] })
	field := newCaptionField(path, captions, identities, pool[:min(len(pool), nearMissPool)])
	var items []VideoEditItem
	for _, identity := range local[:count] {
		// The seed's draws stay in the order the first form of this suite made them, so
		// the same seed gives every clip the instruction it had and clips already edited
		// for it are asked the new question: the change, the order of four statements,
		// then the near misses of the clip's caption.
		change := videoChanges[rng.Perm(len(videoChanges))[0]]
		order := make([]int, editChoices)
		for position := range order {
			order[position] = position
		}
		rng.Shuffle(len(order), func(left, right int) { order[left], order[right] = order[right], order[left] })
		content, err := field.questions(rng, identity, 1)
		if err != nil {
			return nil, err
		}
		own := captions[identity]
		other := content[0].Options[(content[0].Answer+1)%len(content[0].Options)]
		statements := []string{own + " " + change.statement, own + " " + change.absence, other + " " + change.statement, other + " " + change.absence}
		question := VideoCaptionQuestion{Options: make([]string, editChoices)}
		for position, statement := range order {
			question.Options[position] = statements[statement]
			if statement == 0 {
				question.Answer = position
			}
		}
		items = append(items, VideoEditItem{ID: identity, Caption: own, Instruction: change.instruction, Question: question})
	}
	return items, nil
}

// BuildGeneratedVideoEdit builds the held-out suite that asks a judge one question about
// each edited clip: which of four statements, of what it shows and whether the change
// was made, is true of it. videos maps a clip's identity to the file name of its edited
// clip. The judge is named in the suite, since the score depends on it.
func BuildGeneratedVideoEdit(items []VideoEditItem, videos map[string]string, judge string) (GeneratedAnswerSuite, error) {
	if strings.TrimSpace(judge) == "" {
		return GeneratedAnswerSuite{}, errors.New("evaluation: a suite of questions about clips names its judge")
	}
	suite := GeneratedAnswerSuite{
		Kind: GeneratedAnswerKind, Schema: generatedEditSchema, Source: generatedEditSource, Judge: judge,
		Transforms: []string{TransformTrimSpace, transformLowercase},
	}
	for _, item := range items {
		name, edited := videos[item.ID]
		if !edited {
			return GeneratedAnswerSuite{}, fmt.Errorf("evaluation: clip %s has no edited clip", item.ID)
		}
		suite.Cases = append(suite.Cases, GeneratedAnswerCase{
			Name:      fmt.Sprintf("%s/%s", generatedEditSource, item.ID),
			Prompt:    mediaPrompt(MediaVideo, name, fmt.Sprintf(generatedEditQuestion, lettered(item.Question.Options))),
			MaxTokens: captionAnswerTokens,
			Answers:   []string{string(captionLetters[item.Question.Answer])},
		})
	}
	return suite, nil
}
