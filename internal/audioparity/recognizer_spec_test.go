package audioparity

import (
	"encoding/json"
	"testing"

	"overgo/internal/speechrecognition"
	"overgo/internal/strictjson"
)

// TestReviewedRecognizerResolvesEachDeclaredForm holds the reviewed registry
// to resolving a checkpoint by the type its own declaration names, and to
// handing the runtime the decoder that declaration states: a recurrent
// checkpoint carries its decoder binding and reads ungrouped frames, and a
// connectionist one carries none and states its grouping instead. A
// checkpoint no declaration answers for is reported unknown rather than
// loaded against another family's bindings.
func TestReviewedRecognizerResolvesEachDeclaredForm(t *testing.T) {
	t.Parallel()
	var recurrent recurrentDeclaration
	if err := json.Unmarshal(transducerExecutionJSON, &recurrent); err != nil {
		t.Fatal(err)
	}
	if recurrent.ModelType == "" {
		t.Fatal("the recurrent declaration names no model type")
	}
	reviewed, known, err := RecognizerSpecFor(recurrent.ModelType)
	if err != nil || !known {
		t.Fatalf("recurrent checkpoint known = %t: %v", known, err)
	}
	var binding speechrecognition.TransducerBinding
	if err := strictjson.DecodeBytes(reviewed.Transducer, &binding); err != nil {
		t.Fatal(err)
	}
	if binding.Blank < 0 || len(binding.Recurrent) == 0 || binding.Bands == 0 {
		t.Fatalf("recurrent binding is incomplete: %+v", binding)
	}
	var declaration speechrecognition.Declaration
	if err := strictjson.DecodeBytes(reviewed.Execution, &declaration); err != nil {
		t.Fatal(err)
	}
	if reviewed.Grouping.StackFrames != 0 || reviewed.Grouping.DeltaRadius != 0 {
		t.Fatalf("a recurrent checkpoint declared a grouping: %+v", reviewed.Grouping)
	}
	recipe, _, err := graniteRecipe()
	if err != nil {
		t.Fatal(err)
	}
	connectionist, known, err := RecognizerSpecFor(recipe.ModelType)
	if err != nil || !known {
		t.Fatalf("connectionist checkpoint known = %t: %v", known, err)
	}
	if len(connectionist.Transducer) != 0 {
		t.Fatal("a connectionist checkpoint carries a decoder binding")
	}
	if connectionist.Grouping.StackFrames == 0 {
		t.Fatal("a connectionist checkpoint declares no grouping")
	}
	if _, known, err := RecognizerSpecFor(recurrent.ModelType + recipe.ModelType); err != nil || known {
		t.Fatalf("an unreviewed model type resolved: known = %t: %v", known, err)
	}
}
