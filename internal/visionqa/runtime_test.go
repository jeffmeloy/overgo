package visionqa

import (
	"context"
	"testing"

	"overgo/internal/modelrecipetest"
	"overgo/internal/recipe"
	"overgo/internal/workflowruntime"
)

const (
	questionFixture = "where?"
	answerFixture   = "there"
)

var imageFixture = Image{Data: []byte("img")}

func TestRegisteredRuntimeExecutesVQAProgram(t *testing.T) {
	fixture := modelrecipetest.NewCapability(t, "vqa-model", recipe.TaskVQA)
	if err := RegisterRuntime(fixture.Runtime, fixture.Model,
		func(_ context.Context, image Image, question string) (string, error) {
			if string(image.Data) != string(imageFixture.Data) || question != questionFixture {
				t.Fatalf("input = (%d, %q)", len(image.Data), question)
			}
			return answerFixture, nil
		}); err != nil {
		t.Fatal(err)
	}
	result, err := fixture.Runtime.ExecuteProgram(context.Background(), "vqa/runtime", fixture.Program,
		map[recipe.PortName]workflowruntime.Value{
			"image":    {Kind: recipe.DataImage, Items: []workflowruntime.Datum{{Value: imageFixture}}},
			"question": {Kind: recipe.DataText, Items: []workflowruntime.Datum{{Value: questionFixture}}},
		})
	if err != nil {
		t.Fatal(err)
	}
	datum, one := result.Outputs["answer"].Single()
	answer, typed := datum.Value.(string)
	if !one || !typed || answer != answerFixture || !result.Commit.Valid() {
		t.Fatalf("answer = (%q, %v, %v), commit=%v", answer, one, typed, result.Commit)
	}
}
