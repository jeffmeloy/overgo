package scorecard

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestAMeasureBuiltAgainIsReadFromItsNewestBuild holds which held-out reading
// heads a recipe that was measured more than once. A measure built again is
// another set its descriptor names as the same measure, as an edit evaluation
// is each time its model draws the clips again: of its builds the one
// evaluated last is read, though the earlier build's set is named first, and
// the earlier reading stays among the others. Building a measure again does
// not move it among the recipe's other measures: it stands where its first
// build's set stood. Two measures keep the order of their sets' names, whichever
// was evaluated last, as do sets no descriptor names.
func TestAMeasureBuiltAgainIsReadFromItsNewestBuild(t *testing.T) {
	t.Parallel()
	summary := func(set string, value float64) evaluation.EvalSummary {
		return evaluation.EvalSummary{
			Suite: set, Prompting: "chat-template", Record: testutil.ArtifactID(t, artifact.KindEvaluation, set),
			Metrics: map[string]float64{"accuracy": value}, Directions: map[string]runrecord.Direction{"accuracy": runrecord.DirectionMaximize},
		}
	}
	read := func(summaries []evaluation.EvalSummary, descriptors map[string]evaluation.SuiteDescriptor) []string {
		cell, others := qualityCell(recipe.TaskVideoGen, summaries, descriptors)
		order := []string{cell.Contract.Suite}
		for _, other := range others {
			order = append(order, other.Contract.Suite)
		}
		return order
	}
	equal := func(got []string, want ...string) bool { return slices.Equal(got, want) }
	const earlier, again, captions = "dataset:m-earlier-build", "dataset:p-built-again", "dataset:k-captions"
	edit, caption := evaluation.SuiteDescriptor{Source: "clips/edit asked-these"}, evaluation.SuiteDescriptor{Source: "clips/caption asked-these"}
	// Summaries arrive newest first: the build made again was evaluated last.
	rebuilt := []evaluation.EvalSummary{summary(again, 0.25), summary(earlier, 0.1875)}
	if order := read(rebuilt, map[string]evaluation.SuiteDescriptor{earlier: edit, again: edit}); !equal(order, again, earlier) {
		t.Fatalf("a measure built again reads %v, want its newest build and then the earlier one", order)
	}
	// A build counted another way carries its counting in its protocol's name. It is still the build read,
	// though that name sorts after the earlier build's.
	renamed := []evaluation.EvalSummary{summary(again, 0.25), summary(earlier, 0.1875)}
	renamed[0].Prompting, renamed[1].Prompting = "raw-completion, whisper-english", "raw-completion"
	if order := read(renamed, map[string]evaluation.SuiteDescriptor{earlier: edit, again: edit}); !equal(order, again, earlier) {
		t.Fatalf("a measure built again under a protocol named later reads %v, want its newest build first", order)
	}
	// One set's own alternatives keep the order of their protocols' names, whichever was evaluated last.
	alternatives := []evaluation.EvalSummary{summary(earlier, 0.25), summary(earlier, 0.5)}
	alternatives[0].Prompting, alternatives[1].Prompting = "raw-completion", "chat-template"
	if cell, _ := qualityCell(recipe.TaskVideoGen, alternatives, map[string]evaluation.SuiteDescriptor{earlier: edit}); cell.Contract.Protocol != "chat-template" {
		t.Fatalf("one set's alternatives read %q first, want the protocol named first whichever ran last", cell.Contract.Protocol)
	}
	// A build whose set is named before every other set does not carry its measure to the head.
	const before = "dataset:a-built-again"
	moved := []evaluation.EvalSummary{summary(before, 0.25), summary(captions, 0.5), summary(earlier, 0.1875)}
	if order := read(moved, map[string]evaluation.SuiteDescriptor{earlier: edit, before: edit, captions: caption}); !equal(order, captions, before, earlier) {
		t.Fatalf("beside another measure the builds read %v, want the measure whose set was named first, then the newest build, then the earlier", order)
	}
	// Two measures of one builder are not one built again, and unnamed sets are not either.
	variant := evaluation.SuiteDescriptor{Source: "clips/edit asked-others"}
	if order := read(rebuilt, map[string]evaluation.SuiteDescriptor{earlier: edit, again: variant}); !equal(order, earlier, again) {
		t.Fatalf("two measures read %v, want the set named first whichever was evaluated last", order)
	}
	if order := read(rebuilt, nil); !equal(order, earlier, again) {
		t.Fatalf("two sets no descriptor names read %v, want the set named first", order)
	}
}
