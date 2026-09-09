package audioparity

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/speechrecognition"
)

// publishTranscriptionBase binds the same real model and processor for training,
// segmented recognition and serving fixtures, without running a training step.
func (p *audioPublication) publishTranscriptionBase(t *testing.T, fixture ctcTrainingFixture) (recipe.Definition, speechrecognition.ExecutionProfile) {
	t.Helper()
	inventory, err := modelartifact.FromHFPath(fixture.modelRoot)
	if err != nil || inventory.Manifest.ID != fixture.election.Model.ID {
		t.Fatalf("base inventory: %v", err)
	}
	batch, err := inventory.Batch("adapter/base")
	p.commit(t, batch, err)
	profile, err := speechrecognition.NewExecutionProfile(speechrecognition.ExecutionProfile{Frontend: fixture.frontend, Grouping: fixture.grouping, Encoder: fixture.declaration, BlankToken: fixture.blank, Language: "en"})
	if err != nil {
		t.Fatal(err)
	}
	batch, err = profile.Batch("adapter/profile")
	p.commit(t, batch, err)
	contract, err := modelrecipe.NewAudioContract(recipecontract.AudioFormat{SampleRate: uint64(fixture.frontend.SampleRate), Channels: 1, Encoding: "pcm-f32le"}, fixture.frontend.Geometry, artifact.ID{}, artifact.ID{})
	if err != nil {
		t.Fatal(err)
	}
	batch, err = contract.Batch("adapter/contract")
	p.commit(t, batch, err)
	var tokenizerID artifact.ID
	for _, component := range inventory.Manifest.Components {
		if component.Role == artifact.ComponentTokenizer && component.Name == "tokenizer.json" {
			tokenizerID = component.Artifact
		}
	}
	base, err := modelrecipe.TranscriptionDefinition(inventory.Manifest.ID, contract.ID, profile.ID, tokenizerID, inventory.TensorInventory.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(t.Context(), p.store, "adapter/base-recipe", base); err != nil {
		t.Fatal(err)
	}

	return base, profile
}
