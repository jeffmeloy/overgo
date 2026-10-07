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
	generatedEditSchema = "overgo/vidgen-generated-video-edit/v1"
	// generatedEditQuestion asks which change an edited clip shows.
	generatedEditQuestion = "Which statement is true of the video?\n%s\nAnswer with the letter only."
	// unchangedStatement is the choice that is true of a clip the editor gave back as it
	// was: an editor that changes nothing answers every change question wrong.
	unchangedStatement = "It is an ordinary video, and none of the other statements is true of it."
	// editChangeCase and editContentCase name the two questions asked of an edited clip.
	editChangeCase  = "change"
	editContentCase = "content"
)

// videoChange is one change an editor is told to make: the instruction it is given, the
// statement that is true of a clip once it is made, and the caption words that mean a
// clip shows it already, which leaves nothing to tell apart.
type videoChange struct {
	instruction, statement string
	words                  []string
}

// videoChanges are changes of a whole clip that a viewer sees at a glance whatever the
// clip shows, so that one list serves every source clip.
var videoChanges = []videoChange{
	{"Make it snow heavily.", "Heavy snow is falling.", []string{"snow", "snowing", "snowy", "snowfall"}},
	{"Make it rain heavily.", "Heavy rain is falling.", []string{"rain", "raining", "rainy"}},
	{"Change the time to night.", "It is night.", []string{"night", "nighttime", "dark"}},
	{"Cover the scene in thick fog.", "Thick fog covers the scene.", []string{"fog", "foggy", "mist", "misty"}},
	{"Turn the video into a black-and-white pencil sketch.", "It is a black-and-white pencil sketch.", []string{"sketch", "drawing", "drawn", "pencil"}},
	{"Turn the video into a colorful cartoon.", "It is a colorful cartoon.", []string{"cartoon", "animated", "animation"}},
}

// VideoEditItem is a clip a video editor is asked to change: the clip's identity and
// caption, the instruction the editor is given, and the two questions the judge is asked
// about the result, which change it shows and which caption still describes it.
type VideoEditItem struct {
	ID          string
	Caption     string
	Instruction string
	Change      VideoCaptionQuestion
	Content     VideoCaptionQuestion
}

// PickVideoEdits picks count captioned clips of a directory by seed and gives each an
// instruction from a fixed list of whole-clip changes. A clip whose caption says it shows
// one of the changes already is passed over. The change question offers the statement of
// the change made, those of two other changes, and that the clip is unchanged; the content
// question offers the clip's caption among near misses, as the generators' suite does: an
// editor is held both to making the change and to keeping what the clip showed.
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
		// The change made, then two others to tell it from, in the seed's order.
		order := rng.Perm(len(videoChanges))
		change := videoChanges[order[0]]
		statements := []string{change.statement, unchangedStatement}
		for _, other := range order[1 : captionChoices-len(statements)+1] {
			statements = append(statements, videoChanges[other].statement)
		}
		rng.Shuffle(len(statements), func(left, right int) { statements[left], statements[right] = statements[right], statements[left] })
		content, err := field.questions(rng, identity, 1)
		if err != nil {
			return nil, err
		}
		items = append(items, VideoEditItem{
			ID: identity, Caption: captions[identity], Instruction: change.instruction,
			Change:  VideoCaptionQuestion{Options: statements, Answer: slices.Index(statements, change.statement)},
			Content: content[0],
		})
	}
	return items, nil
}

// BuildGeneratedVideoEdit builds the held-out suite that asks a judge about each edited
// clip: which change it shows, and which caption describes it. videos maps a clip's
// identity to the file name of its edited clip. The judge is named in the suite, since the
// score depends on it.
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
		for _, asked := range []struct {
			name, question string
			choices        VideoCaptionQuestion
		}{
			{editChangeCase, generatedEditQuestion, item.Change},
			{editContentCase, generatedVideoQuestion, item.Content},
		} {
			suite.Cases = append(suite.Cases, GeneratedAnswerCase{
				Name:      fmt.Sprintf("%s/%s/%s", generatedEditSource, item.ID, asked.name),
				Prompt:    mediaPrompt(MediaVideo, name, fmt.Sprintf(asked.question, lettered(asked.choices.Options))),
				MaxTokens: captionAnswerTokens,
				Answers:   []string{string(captionLetters[asked.choices.Answer])},
			})
		}
	}
	return suite, nil
}
