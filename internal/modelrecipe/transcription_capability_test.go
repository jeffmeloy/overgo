package modelrecipe

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestTranscriptionCapabilityCompiles holds the generic capability builder to
// knowing transcription: the onboarding path resolves a task through it, and
// a task it does not know refuses before a model is ever read. The stage is
// the audio-in transcript-out module the audio contract declares, and it
// carries a session lifetime, because the recognizer stays loaded across
// requests as the synthesizer does; without one the component session plan is
// empty and every verify refuses before execution.
func TestTranscriptionCapabilityCompiles(t *testing.T) {
	t.Parallel()
	model := testutil.ArtifactID(t, artifact.KindModel, "speech recognizer")
	definition, err := CapabilityDefinition(recipe.TaskTranscription, model)
	if err != nil {
		t.Fatal(err)
	}
	if definition.Task != recipe.TaskTranscription || len(definition.Nodes) != 1 {
		t.Fatalf("definition = %+v", definition)
	}
	node := definition.Nodes[0]
	if node.Module != ModuleTranscribeAudio || node.Session != recipe.SessionCapacity {
		t.Fatalf("node = %+v", node)
	}
	program, err := CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	if len(program.Stages()) == 0 {
		t.Fatal("compiled program has no stages")
	}
}
