package diffusionimage

import (
	"math"
	"testing"

	"overgo/internal/optimizer"
	"overgo/internal/recipecontract"
	"overgo/internal/trainingdata"
)

func TestTrainerConsumesSharedImageBatch(t *testing.T) {
	fixture := loadGolden[uditTinyFixture](t, "udit_tiny")
	model := compileTiny(t, fixture)
	trainer, err := NewTrainer(model, optimizer.Config{BaseLearningRate: 0.02, Momentum: 0.95, Schedule: optimizer.ScheduleConstant})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := trainer.Close(); err != nil {
			t.Fatal(err)
		}
	}()
	input, target := fixedFlowPath(t, fixture.X, fixture.B, 0.001)
	perExample := len(input) / fixture.B
	examples := make([]trainingdata.Example, fixture.B)
	for index := range examples {
		start := index * perExample
		end := start + perExample
		examples[index] = trainingdata.Example{
			ID: "image", Group: "train",
			Values: []trainingdata.Value{
				imageValue(trainingdata.RoleInput, fixture.InChannels, fixture.H, fixture.W, input[start:end]),
				imageValue(trainingdata.RoleTarget, fixture.InChannels, fixture.H, fixture.W, target[start:end]),
			},
		}
	}
	losses, err := trainer.TrainBatch(trainingdata.Batch{Examples: examples})
	if err != nil {
		t.Fatal(err)
	}
	if len(losses) != 1 || math.IsNaN(losses[0]) || math.IsInf(losses[0], 0) {
		t.Fatalf("losses = %v", losses)
	}
}

func imageValue(role trainingdata.ValueRole, channels, height, width int, values []float32) trainingdata.Value {
	return trainingdata.Value{
		Role: role, Modality: recipecontract.ModalityImage, Encoding: trainingdata.EncodingFloat32LE,
		Shape: []int{channels, height, width}, Data: trainingdata.EncodeFloat32(values),
	}
}
