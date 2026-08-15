package diffusionimage

import (
	"errors"
	"fmt"
	"math/rand"

	"overgo/internal/recipecontract"
	"overgo/internal/trainingdata"
)

type otImageMicrobatch struct {
	x, target   []float32
	batch, h, w int
}

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

// TrainOTBatch derives adaptive_new's seeded OT path from real image targets.
func (trainer *Trainer) TrainOTBatch(batch trainingdata.Batch, sigmaMin float64, seed int64) ([]float64, error) {
	return trainer.eachOTMicrobatch(batch, sigmaMin, seed, func(microbatch otImageMicrobatch) (float64, error) {
		return trainer.Step(microbatch.x, microbatch.target, microbatch.batch, microbatch.h, microbatch.w)
	})
}

// OTLoss evaluates the same seeded OT path without updating parameters.
func (trainer *Trainer) OTLoss(batch trainingdata.Batch, sigmaMin float64, seed int64) ([]float64, error) {
	return trainer.eachOTMicrobatch(batch, sigmaMin, seed, func(microbatch otImageMicrobatch) (float64, error) {
		prediction, err := trainer.model.Forward(microbatch.x, microbatch.batch, microbatch.h, microbatch.w)
		if err != nil {
			return 0, err
		}
		gradient := make([]float32, len(prediction))
		return ScaledMSELossGradInto(gradient, prediction, microbatch.target, 1)
	})
}

func (trainer *Trainer) eachOTMicrobatch(
	batch trainingdata.Batch,
	sigmaMin float64,
	seed int64,
	execute func(otImageMicrobatch) (float64, error),
) ([]float64, error) {
	if trainer == nil || trainer.model == nil || sigmaMin <= 0 || sigmaMin >= 1 {
		return nil, errors.New("diffusionimage train: invalid OT batch")
	}
	examples := batch.Microbatches()
	if len(examples) == 0 || execute == nil {
		return nil, errors.New("diffusionimage train: empty stream batch")
	}
	rng := rand.New(rand.NewSource(seed))
	losses := make([]float64, len(examples))
	for index, group := range examples {
		x1, height, width, err := targetImageBatch(group, trainer.model.Cfg.InChannels)
		if err != nil {
			return nil, err
		}
		x0 := make([]float32, len(x1))
		times := make([]float32, len(group))
		for offset := range x0 {
			x0[offset] = float32(rng.NormFloat64())
		}
		for offset := range times {
			times[offset] = float32(rng.Float64())
		}
		x := make([]float32, len(x1))
		target := make([]float32, len(x1))
		if err := OTLinearFlowPathInto(x, target, x1, x0, times, len(group), len(x1)/len(group), sigmaMin); err != nil {
			return nil, err
		}
		losses[index], err = execute(otImageMicrobatch{x: x, target: target, batch: len(group), h: height, w: width})
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

func targetImageBatch(examples []trainingdata.Example, channels int) ([]float32, int, int, error) {
	var target []float32
	var height, width int
	for _, example := range examples {
		var image trainingdata.Value
		for _, value := range example.Values {
			if value.Modality == recipecontract.ModalityImage && value.Role == trainingdata.RoleTarget {
				if len(image.Data) != 0 {
					return nil, 0, 0, errors.New("diffusionimage train: multiple image targets")
				}
				image = value
			}
		}
		if len(image.Shape) != 3 || image.Shape[0] != channels {
			return nil, 0, 0, errors.New("diffusionimage train: image target shape mismatch")
		}
		if height == 0 {
			height, width = image.Shape[1], image.Shape[2]
		} else if height != image.Shape[1] || width != image.Shape[2] {
			return nil, 0, 0, errors.New("diffusionimage train: mixed target shapes")
		}
		values, err := trainingdata.Float32(image)
		if err != nil {
			return nil, 0, 0, err
		}
		if len(values) != channels*height*width {
			return nil, 0, 0, errors.New("diffusionimage train: image target length differs")
		}
		target = append(target, values...)
	}
	return target, height, width, nil
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
