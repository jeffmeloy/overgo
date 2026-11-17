package diffusionimage

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/latentimage"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/workflowruntime"
)

type Request struct {
	Seed   int64 `json:"seed" role:"seed"`
	Steps  int   `json:"steps" role:"steps"`
	Height int   `json:"height" role:"height"`
	Width  int   `json:"width" role:"width"`
	// Batch is how many pictures the network draws at once, each from the next noise the seed gives, one where unset:
	// train.py's sample() draws sixteen, its transformer blocks interleave the batch, and the weights were trained at
	// batches of 16 to 128, so a picture drawn alone is pale and noisy (readme-samples/simplediffusion-fifty). One stays
	// the default while the frozen review cases of docs/image_video_protocol.json hold the pictures drawn alone.
	Batch int `json:"batch,omitzero"`
	// Sheet answers with the whole batch drawn, laid out as the author saves samples, in place of its first picture.
	Sheet bool `json:"sheet,omitzero"`
}

// sampleBatchMost is the most pictures a request draws at once: the batch train.py's sample() draws.
const sampleBatchMost = 16

// batch is how many pictures the request draws at once.
func (request Request) batch() int { return max(request.Batch, 1) }

// sheetColumns and sheetPadding lay a batch out as the author's samples are saved (torchvision's save_image, nrow=4,
// padding=2): four to a row, two black pixels between and around them.
const sheetColumns, sheetPadding = 4, 2

type samplePlan struct{ request Request }

// sampleFeatures are a sampled batch's pixels, picture by picture, with the size of each and the Euler steps that drew
// them, and whether the answer is the whole sheet.
type sampleFeatures struct {
	pixels                      []float32
	height, width, steps, batch int
	sheet                       bool
}

// picture is what features answer with, its planar pixels and size: the batch's first picture, or its sheet.
func (features sampleFeatures) picture() ([]float32, int, int) {
	batch := max(features.batch, 1)
	channels := len(features.pixels) / (batch * features.height * features.width)
	plane := channels * features.height * features.width
	if !features.sheet {
		return features.pixels[:plane], features.height, features.width
	}
	rows := (batch + sheetColumns - 1) / sheetColumns
	width := sheetColumns*features.width + (sheetColumns+1)*sheetPadding
	height := rows*features.height + (rows+1)*sheetPadding
	sheet := make([]float32, channels*height*width)
	// The model's range is -1 to 1, so black, the padding, is -1.
	for index := range sheet {
		sheet[index] = -1
	}
	for picture := range batch {
		top := sheetPadding + picture/sheetColumns*(features.height+sheetPadding)
		left := sheetPadding + picture%sheetColumns*(features.width+sheetPadding)
		for channel := range channels {
			for row := range features.height {
				source := picture*plane + (channel*features.height+row)*features.width
				target := (channel*height+top+row)*width + left
				copy(sheet[target:target+features.width], features.pixels[source:source+features.width])
			}
		}
	}
	return sheet, height, width
}

func ValidateRequest(request Request) error {
	if request.Steps <= 0 || request.Height <= 0 || request.Width <= 0 {
		return errors.New("diffusionimage: positive steps, height, and width required")
	}
	if request.Batch < 0 || request.Batch > sampleBatchMost {
		return fmt.Errorf("diffusionimage: a batch of 1 to %d pictures, as the author's sampler draws at most", sampleBatchMost)
	}
	return nil
}

func (m *Model) prepare(ctx context.Context, request Request) (samplePlan, error) {
	if err := ctx.Err(); err != nil {
		return samplePlan{}, err
	}
	if err := ValidateRequest(request); err != nil {
		return samplePlan{}, err
	}
	if m == nil {
		return samplePlan{}, errors.New("diffusionimage: model is unavailable")
	}
	tile := m.Cfg.PatchSize << uint(m.Cfg.NumLevels-1)
	if request.Height%tile != 0 || request.Width%tile != 0 {
		return samplePlan{}, errors.New("diffusionimage: image size is incompatible with artifact tile")
	}
	return samplePlan{request: request}, nil
}

func (m *Model) integrate(ctx context.Context, plan samplePlan) (sampleFeatures, error) {
	request := plan.request
	pixels, err := m.Sample(ctx, request.batch(), request.Height, request.Width, request.Steps, request.Seed)
	return sampleFeatures{pixels: pixels, height: request.Height, width: request.Width, steps: request.Steps, batch: request.batch(),
		sheet: request.Sheet}, err
}

func (m *Model) decode(ctx context.Context, features sampleFeatures) (latentimage.EncodedImage, error) {
	if err := ctx.Err(); err != nil {
		return latentimage.EncodedImage{}, err
	}
	pixels, height, width := features.picture()
	image, err := latentimage.EncodePlanarPNG(pixels, height, width)
	image.Preset = media.Preset{Width: features.width, Height: features.height, Steps: features.steps, Sampler: "Euler"}
	return image, err
}

type runtimeBinding interface {
	*Model | *ResidentGenerator
	prepare(context.Context, Request) (samplePlan, error)
	integrate(context.Context, samplePlan) (sampleFeatures, error)
	decode(context.Context, sampleFeatures) (latentimage.EncodedImage, error)
}

func RegisterRuntime[T runtimeBinding](runtime *workflowruntime.Runtime, modelID artifact.ID, model T) error {
	if model == nil {
		return errors.New("diffusionimage: incomplete runtime binding")
	}
	return workflowruntime.RegisterContextPipeline(
		runtime, modelID,
		modelrecipe.ModuleDiffusionImagePrepare, model.prepare,
		modelrecipe.ModuleDiffusionImageIntegrate, model.integrate,
		modelrecipe.ModuleDiffusionImageDecode, model.decode,
		latentimage.PNGContent,
	)
}
