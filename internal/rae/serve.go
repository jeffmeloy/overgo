package rae

import (
	"cmp"
	"context"
	_ "embed"
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/strictjson"
	"overgo/internal/workflowruntime"
)

//go:embed recipes/sampler.json
var samplerDeclarationJSON []byte

// samplerDeclaration is the reviewed reference configuration a collection is
// served with: what the collection's files do not state of themselves.
type samplerDeclaration struct {
	Checkpoint string  `json:"checkpoint"`
	Decoder    string  `json:"decoder"`
	Statistics string  `json:"statistics"`
	Encoder    string  `json:"encoder"`
	Steps      int     `json:"num_steps"`
	Guidance   float32 `json:"cfg_scale"`
}

// servedSampler is the sampler declaration, read once.
var servedSampler = sync.OnceValues(func() (samplerDeclaration, error) {
	var document struct {
		Doc     string             `json:"doc"`
		Sampler samplerDeclaration `json:"sampler"`
	}
	if err := strictjson.DecodeBytes(samplerDeclarationJSON, &document); err != nil {
		return samplerDeclaration{}, fmt.Errorf("rae: sampler declaration: %w", err)
	}
	if document.Sampler.Steps < minimumSteps {
		return samplerDeclaration{}, fmt.Errorf("rae: the sampler declares %d grid points, fewer than %d", document.Sampler.Steps, minimumSteps)
	}
	return document.Sampler, nil
})

// Collection is the files a served RAE generation reads from a collection:
// the stage-2 DiT checkpoint, the stage-1 decoder and the latent statistics
// it decodes under. The encoder whose image processor normalized the pixels
// the decoder reproduces sits beside the collection.
type Collection struct {
	Checkpoint, Decoder, Statistics string
	encoder                         string
}

// ProcessorFiles names the file a collection's generation reads that the
// collection's inventory does not hold: the image processor of the encoder
// beside it. A directory that resolves as no collection names none.
func ProcessorFiles(directory string) []string {
	collection, held, err := ResolveCollection(directory)
	if err != nil || !held {
		return nil
	}
	return []string{filepath.Join(collection.encoder, processorConfig)}
}

// ResolveCollection reads directory as an RAE collection served with the
// reviewed sampler. held is false when the directory holds no such DiT
// checkpoint, so the caller resolves it as another kind of model; a held
// collection whose encoder's image processor is absent is refused.
func ResolveCollection(directory string) (collection Collection, held bool, err error) {
	sampler, err := servedSampler()
	if err != nil {
		return Collection{}, false, err
	}
	directory = filepath.Clean(directory)
	checkpoint := filepath.Join(directory, filepath.FromSlash(sampler.Checkpoint))
	if info, err := os.Stat(checkpoint); err != nil || info.IsDir() {
		return Collection{}, false, nil
	}
	collection = Collection{
		Checkpoint: checkpoint,
		Decoder:    filepath.Join(directory, filepath.FromSlash(sampler.Decoder)),
		Statistics: filepath.Join(directory, filepath.FromSlash(sampler.Statistics)),
		encoder:    filepath.Join(filepath.Dir(directory), path.Base(sampler.Encoder)),
	}
	if _, err := os.Stat(filepath.Join(collection.encoder, processorConfig)); err != nil {
		return Collection{}, true, fmt.Errorf("rae: the image processor of the collection's encoder %s: %w", sampler.Encoder, err)
	}
	return collection, true, nil
}

// Request names an ImageNet class and the seed of the CPU torch generator
// the reference draws the starting noise from. A positive Steps replaces the
// sampler's grid points.
type Request struct {
	Class int   `json:"class"`
	Seed  int64 `json:"seed"`
	Steps int   `json:"steps,omitzero"`
}

// ValidateRequest refuses a negative class or a grid too short to integrate
// over; the class's upper bound is the loaded DiT's.
func ValidateRequest(request Request) error {
	if request.Class < 0 {
		return fmt.Errorf("rae: class %d is negative", request.Class)
	}
	if request.Steps != 0 && request.Steps < minimumSteps {
		return fmt.Errorf("rae: a sample needs at least %d grid points, got %d", minimumSteps, request.Steps)
	}
	return nil
}

// Generator is a loaded RAE collection: the DiT that samples a class's
// latent, the decoder that turns it into pixels, and the sampler's grid and
// guidance.
type Generator struct {
	model    *diffusionTransformer
	decoder  *vitDecoder
	steps    int
	guidance float32
}

// Load reads the RAE collection in directory with its reviewed sampler.
func Load(directory string) (*Generator, error) {
	collection, held, err := ResolveCollection(directory)
	if err != nil {
		return nil, err
	}
	if !held {
		return nil, fmt.Errorf("rae: %s holds no DiT checkpoint the reviewed sampler serves", directory)
	}
	sampler, err := servedSampler()
	if err != nil {
		return nil, err
	}
	model, err := loadDiT(collection.Checkpoint)
	if err != nil {
		return nil, err
	}
	decoder, err := loadDecoder(collection.Decoder, collection.Statistics, collection.encoder)
	if err != nil {
		return nil, err
	}
	if model.latent != decoder.latent || model.grid != decoder.grid {
		return nil, fmt.Errorf("rae: the DiT samples %d channels over a %d grid, the decoder reads %d over %d",
			model.latent, model.grid, decoder.latent, decoder.grid)
	}
	return &Generator{model: model, decoder: decoder, steps: sampler.Steps, guidance: sampler.Guidance}, nil
}

// RegisterRuntime binds the generator's stages to the image-gen recipe's
// RAE modules: seeded noise, the sampled latent, and its decoded PNG.
func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, generator *Generator) error {
	if generator == nil {
		return errors.New("rae: incomplete runtime binding")
	}
	return workflowruntime.RegisterContextPipeline(
		runtime, modelID,
		modelrecipe.ModuleRAEImagePrepare, generator.prepare,
		modelrecipe.ModuleRAEImageIntegrate, generator.integrate,
		modelrecipe.ModuleRAEImageDecode, generator.decode,
		latentimage.PNGContent,
	)
}

// samplePlan is one request made ready to sample: the seeded noise, the
// class and the grid points.
type samplePlan struct {
	noise        []float32
	label, steps int
}

// prepare draws the request's seeded noise over the DiT's latent.
func (g *Generator) prepare(ctx context.Context, request Request) (samplePlan, error) {
	if err := ctx.Err(); err != nil {
		return samplePlan{}, err
	}
	if request.Class >= g.model.classes {
		return samplePlan{}, fmt.Errorf("rae: class %d is outside the DiT's %d classes", request.Class, g.model.classes)
	}
	noise, err := torchNormal(request.Seed, g.model.latent*g.model.grid*g.model.grid)
	if err != nil {
		return samplePlan{}, err
	}
	return samplePlan{noise: noise, label: request.Class, steps: cmp.Or(request.Steps, g.steps)}, nil
}

// integrate samples the plan's latent.
func (g *Generator) integrate(ctx context.Context, plan samplePlan) ([]float32, error) {
	return g.model.sample(ctx, plan.noise, plan.label, plan.steps, g.guidance)
}

// signedSpan is the width of [-1, 1], the range the image encoding reads;
// the decoder's pixels lie in [0, 1].
const signedSpan = 2

// decode turns a sampled latent into its PNG.
func (g *Generator) decode(ctx context.Context, latent []float32) (latentimage.EncodedImage, error) {
	if err := ctx.Err(); err != nil {
		return latentimage.EncodedImage{}, err
	}
	pixels, err := g.decoder.decode(latent)
	if err != nil {
		return latentimage.EncodedImage{}, err
	}
	for index, value := range pixels {
		pixels[index] = signedSpan*value - 1
	}
	side := g.decoder.grid * g.decoder.patch
	return latentimage.EncodePlanarPNG(pixels, side, side)
}
