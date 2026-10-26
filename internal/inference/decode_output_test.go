package inference

import (
	"testing"

	"overgo/internal/recipe"
	"overgo/internal/tensor"
)

func TestCompiledOutputPolicyOwnsAllDecodePaths(t *testing.T) {
	const (
		vocabulary     = uint32(32)
		candidateCount = uint32(4)
	)
	graph := deviceBatchGraph{
		logits: &tensor.Tensor{}, selection: &tensor.Tensor{}, candidates: &tensor.Tensor{},
	}
	tests := []struct {
		mode deviceOutputMode
		topK uint32
		want *tensor.Tensor
	}{
		{mode: deviceOutputLogits, want: graph.logits},
		{mode: deviceOutputGreedy, want: graph.selection},
		{mode: deviceOutputGreedySpan, want: graph.selection},
		{mode: deviceOutputLogitsSpan, want: graph.logits},
		{mode: deviceOutputTopK, topK: candidateCount, want: graph.candidates},
	}
	for _, test := range tests {
		plan, err := compileDeviceOutputPlan(test.mode, test.topK, vocabulary)
		if err != nil {
			t.Fatal(err)
		}
		if plan.graphOutput(graph) != test.want {
			t.Fatalf("mode %d selected the wrong graph output", test.mode)
		}
	}
	for _, invalid := range []struct {
		mode deviceOutputMode
		topK uint32
	}{
		{mode: deviceOutputLogits, topK: candidateCount},
		{mode: deviceOutputGreedySpan, topK: candidateCount},
		{mode: deviceOutputLogitsSpan, topK: candidateCount},
		{mode: deviceOutputTopK},
		{mode: deviceOutputTopK, topK: vocabulary + 1},
		{mode: deviceOutputLogitsSpan + 1},
	} {
		if _, err := compileDeviceOutputPlan(invalid.mode, invalid.topK, vocabulary); err == nil {
			t.Fatalf("invalid output policy accepted: %+v", invalid)
		}
	}
}

// TestPromptReadsMarkTheirExperts holds a prompt read of more than one token by a runner whose experts are offloaded to
// the prompt's mark on its routed nodes, whose small groups take the grouped products, and a read of one token, a
// decode step's append, or any read by a runner holding every expert on the card, to none.
func TestPromptReadsMarkTheirExperts(t *testing.T) {
	for _, residency := range []recipe.RealizedResidency{recipe.RealizedExpertOffloaded, recipe.RealizedDeviceNativeBF16} {
		runner := &Runner{}
		runner.spec.VocabularySize = 32
		runner.realizedResidency = residency
		for _, c := range []struct {
			tokens int
			marked bool
		}{{1, false}, {2, true}, {31, true}, {8192, true}} {
			plan, err := runner.promptReadPlan(deviceOutputGreedy, c.tokens)
			if err != nil {
				t.Fatal(err)
			}
			marked := c.marked && residency == recipe.RealizedExpertOffloaded
			if plan.promptExperts != marked || !plan.is(deviceOutputGreedy) {
				t.Errorf("a read of %d tokens over %s marks its experts %t in mode %v, want %t", c.tokens, residency, plan.promptExperts, plan.mode, marked)
			}
		}
	}
}
