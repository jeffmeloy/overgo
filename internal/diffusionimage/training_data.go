package diffusionimage

import (
	"errors"
	"fmt"

	"overgo/internal/recipecontract"
	"overgo/internal/trainingdata"
)

// TrainBatch applies one update per ordered stream microbatch.
func (trainer *Trainer) TrainBatch(batch trainingdata.Batch) ([]float64, error) {
	if trainer == nil || trainer.model == nil {
		return nil, errors.New("diffusionimage train: trainer is unavailable")
	}
	microbatches := batch.Microbatches()
	if len(microbatches) == 0 {
		return nil, errors.New("diffusionimage train: empty stream batch")
	}
	losses := make([]float64, len(microbatches))
	for index, examples := range microbatches {
		x, target, height, width, err := imageBatch(examples, trainer.model.Cfg.InChannels)
		if err != nil {
			return nil, err
		}
		losses[index], err = trainer.Step(x, target, len(examples), height, width)
		if err != nil {
			return nil, err
		}
	}
	return losses, nil
}

func imageBatch(examples []trainingdata.Example, channels int) ([]float32, []float32, int, int, error) {
	var x, target []float32
	var height, width int
	for _, example := range examples {
		input, output, err := imagePair(example)
		if err != nil {
			return nil, nil, 0, 0, err
		}
		if len(input.Shape) != 3 || input.Shape[0] != channels || len(output.Shape) != 3 ||
			input.Shape[0] != output.Shape[0] || input.Shape[1] != output.Shape[1] || input.Shape[2] != output.Shape[2] {
			return nil, nil, 0, 0, errors.New("diffusionimage train: image pair shape mismatch")
		}
		if height == 0 {
			height, width = input.Shape[1], input.Shape[2]
		} else if height != input.Shape[1] || width != input.Shape[2] {
			return nil, nil, 0, 0, errors.New("diffusionimage train: mixed image shapes in microbatch")
		}
		inputValues, err := trainingdata.Float32(input)
		if err != nil {
			return nil, nil, 0, 0, err
		}
		outputValues, err := trainingdata.Float32(output)
		if err != nil {
			return nil, nil, 0, 0, err
		}
		want := channels * height * width
		if len(inputValues) != want || len(outputValues) != want {
			return nil, nil, 0, 0, fmt.Errorf("diffusionimage train: tensor length differs from shape %v", input.Shape)
		}
		x = append(x, inputValues...)
		target = append(target, outputValues...)
	}
	return x, target, height, width, nil
}

func imagePair(example trainingdata.Example) (trainingdata.Value, trainingdata.Value, error) {
	var input, target trainingdata.Value
	for _, value := range example.Values {
		if value.Modality != recipecontract.ModalityImage {
			continue
		}
		switch value.Role {
		case trainingdata.RoleInput:
			if len(input.Data) != 0 {
				return trainingdata.Value{}, trainingdata.Value{}, errors.New("diffusionimage train: multiple image inputs")
			}
			input = value
		case trainingdata.RoleTarget:
			if len(target.Data) != 0 {
				return trainingdata.Value{}, trainingdata.Value{}, errors.New("diffusionimage train: multiple image targets")
			}
			target = value
		}
	}
	if len(input.Data) == 0 || len(target.Data) == 0 {
		return trainingdata.Value{}, trainingdata.Value{}, errors.New("diffusionimage train: image input and target required")
	}
	return input, target, nil
}
