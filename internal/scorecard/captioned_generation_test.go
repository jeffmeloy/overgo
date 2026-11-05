package scorecard

import (
	"strings"
	"testing"

	"overgo/internal/recipe"
	"overgo/internal/scorecard/metric"
)

// TestOnlyAPromptTakingGeneratorIsAskedAboutItsCaption holds the rule that
// decides which image and video recipes the held-out evaluation of the
// generators can be asked of: one with an input of prompt conditioning, and
// not one conditioned on a class, on an image, on nothing, or on a video it edits.
func TestOnlyAPromptTakingGeneratorIsAskedAboutItsCaption(t *testing.T) {
	t.Parallel()
	input := func(kind recipe.DataKind) recipe.Input { return recipe.Input{Name: "condition", Data: kind} }
	for name, test := range map[string]struct {
		inputs []recipe.Input
		want   bool
	}{
		"a prompt":                {[]recipe.Input{input(recipe.DataPromptConditioning)}, true},
		"a prompt beside a video": {[]recipe.Input{input(recipe.DataPromptConditioning), input(recipe.DataVideo)}, false},
		"a class":                 {[]recipe.Input{input(recipe.DataClassConditioning)}, false},
		"an image tensor":         {[]recipe.Input{input(recipe.DataImageTensor)}, false},
		"no input":                {nil, false},
	} {
		if got := acceptsCaption(recipe.Definition{Inputs: test.inputs}); got != test.want {
			t.Errorf("%s: acceptsCaption = %t, want %t", name, got, test.want)
		}
	}
}

// TestAnEditorIsProposedItsOwnEvaluation holds what an unmeasured image or
// video recipe's quality reads: a generator keeps the proposal of its task; an
// editor, which takes an instruction and a video, is proposed the edit
// evaluation and so files a row until it is measured; and a model that takes
// no words is not applicable and files none. The Proposed flag decides what
// files a row: a proposed held-out evaluation does, and a quality declared not
// applicable with its reason does not, though both read not applicable.
func TestAnEditorIsProposedItsOwnEvaluation(t *testing.T) {
	t.Parallel()
	input := func(kind recipe.DataKind) recipe.Input { return recipe.Input{Name: recipe.PortName(kind), Data: kind} }
	proposed, _ := qualityCell(recipe.TaskVideoGen, nil, nil)
	generator := unmeasuredGeneration(proposed, recipe.Definition{Inputs: []recipe.Input{input(recipe.DataPromptConditioning)}})
	editor := unmeasuredGeneration(proposed, recipe.Definition{Inputs: []recipe.Input{input(recipe.DataPromptConditioning), input(recipe.DataVideo)}})
	classed := unmeasuredGeneration(proposed, recipe.Definition{Inputs: []recipe.Input{input(recipe.DataClassConditioning)}})
	if generator.Reason != proposed.Reason || !generator.Proposed {
		t.Fatalf("a generator's unmeasured quality = %+v, want the proposal of its task", generator)
	}
	if !editor.Proposed || !editor.NotApplicable || editor.Reason != editorProposal || editor.Reason == proposed.Reason {
		t.Fatalf("an editor's unmeasured quality = %+v, want the edit evaluation proposed", editor)
	}
	if classed.Proposed || !classed.NotApplicable || classed.Reason != uncaptionedGeneration {
		t.Fatalf("a class-conditioned model's unmeasured quality = %+v, want not applicable with no proposal", classed)
	}
	captioned, _ := qualityCell(recipe.TaskImageGen, nil, nil)
	declared := inapplicable(uncaptionedGeneration)
	if !captioned.Proposed || !captioned.NotApplicable || declared.Proposed || !declared.NotApplicable {
		t.Fatalf("an image generator's proposed quality %+v, a declared inapplicable one %+v", captioned, declared)
	}
	record := Record{Rows: []Row{
		{Name: "editor", Task: recipe.TaskVideoGen, Cells: map[metric.Name]Cell{metric.Quality: editor}},
		{Name: "class-conditioned", Task: recipe.TaskVideoGen, Cells: map[metric.Name]Cell{metric.Quality: classed}},
		{Name: "captioned", Task: recipe.TaskImageGen, Cells: map[metric.Name]Cell{metric.Quality: captioned}},
		{Name: "declared-inapplicable", Task: recipe.TaskVideoGen, Cells: map[metric.Name]Cell{metric.Quality: declared}},
	}}
	var detail []string
	for _, gap := range record.Gaps(nil) {
		detail = append(detail, gap.Detail)
	}
	if joined := strings.Join(detail, "\n"); !strings.Contains(joined, "editor held-out-quality") || !strings.Contains(joined, "captioned held-out-quality") ||
		strings.Contains(joined, "class-conditioned") || strings.Contains(joined, "declared-inapplicable") {
		t.Fatalf("gaps filed:\n%s\nwant the editor's and the captioned rows only", joined)
	}
}
