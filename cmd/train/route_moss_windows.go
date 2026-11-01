//go:build windows

package main

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/latentimage"
	"overgo/internal/mosstrain"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/peft"
	"overgo/internal/recipecontract"
	"overgo/internal/safetensors"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

func init() { trainRoutes["moss"] = trainMOSS }

const (
	// mossLossExamples is how many examples the descent check's fixed path reads.
	mossLossExamples = 4
	// mossHeldoutPaths is how many fixed timesteps and noises judge a held-out
	// picture: the loss of one draw moves more with its timestep than with training.
	mossHeldoutPaths = 8
	// mossHeldoutSeed fixes those paths across runs, so two adapters are judged
	// on one set of draws.
	mossHeldoutSeed = 20261101
	// mossDefaultRank is the rank and the alpha the real step's cost was measured at.
	mossDefaultRank = 16
	// A picture value is [channels, height, width].
	mossPictureAxes, mossHeightAxis, mossWidthAxis = 3, 1, 2
	// mossReportMode is a held-out reading's file, kept beside the run's others.
	mossReportMode = 0o644
)

// mossHeldout is a run's held-out reading: for each picture the flow-matching
// loss of the base and of the trained adapter over the same fixed paths.
type mossHeldout struct {
	Pictures []string  `json:"pictures"`
	Paths    int       `json:"paths_per_picture"`
	Base     []float64 `json:"base"`
	Trained  []float64 `json:"trained"`
}

// mossPicture is one line of a route manifest.
type mossPicture struct {
	path, caption string
}

// mossManifest reads a manifest of one picture and its caption per line,
// tab-separated, the picture relative to the manifest.
func mossManifest(manifest string) ([]mossPicture, error) {
	raw, err := os.ReadFile(manifest)
	if err != nil {
		return nil, err
	}
	var pictures []mossPicture
	for line := range bytes.Lines(raw) {
		line = bytes.TrimRight(line, "\r\n")
		if len(line) == 0 {
			continue
		}
		path, caption, found := bytes.Cut(line, []byte{'\t'})
		if !found || len(path) == 0 || len(caption) == 0 {
			return nil, fmt.Errorf("manifest %s: a line needs a picture, a tab and a caption: %q", manifest, line)
		}
		pictures = append(pictures, mossPicture{path: filepath.Join(filepath.Dir(manifest), string(path)), caption: string(caption)})
	}
	if len(pictures) == 0 {
		return nil, fmt.Errorf("manifest %s names no pictures", manifest)
	}
	return pictures, nil
}

// mossPrepare runs the frozen path once over a manifest: each picture through
// the encoder and each caption through the text path, one at a time on the
// device and both gone from it before the trainer places its weights.
func mossPrepare(ctx context.Context, root string, profile latentimage.Profile, pictures []mossPicture, size int) ([]mosstrain.Example, mosstrain.Geometry, error) {
	encoder, err := latentimage.NewPictureEncoder(ctx, root, size, size)
	if err != nil {
		return nil, mosstrain.Geometry{}, err
	}
	defer encoder.Close(context.WithoutCancel(ctx))
	process := trainingdata.ImageProcessor(trainingdata.RoleTarget)
	examples := make([]mosstrain.Example, len(pictures))
	for index, picture := range pictures {
		data, err := os.ReadFile(picture.path)
		if err != nil {
			return nil, mosstrain.Geometry{}, err
		}
		example, err := process(ctx, trainingdata.RawRecord{ID: picture.path, Group: picture.path, Data: data})
		if err != nil {
			return nil, mosstrain.Geometry{}, fmt.Errorf("picture %s: %w", picture.path, err)
		}
		pixels, err := interleavedRows(example.Values[0], size)
		if err != nil {
			return nil, mosstrain.Geometry{}, fmt.Errorf("picture %s: %w", picture.path, err)
		}
		latent, height, width, err := encoder.Encode(ctx, pixels)
		if err != nil {
			return nil, mosstrain.Geometry{}, fmt.Errorf("picture %s: %w", picture.path, err)
		}
		examples[index] = mosstrain.Example{Geometry: mosstrain.Geometry{Height: height, Width: width}, Latent: latent}
	}
	if err := encoder.Close(ctx); err != nil {
		return nil, mosstrain.Geometry{}, err
	}
	conditioner, err := latentimage.NewCaptionConditioner(root, profile)
	if err != nil {
		return nil, mosstrain.Geometry{}, err
	}
	defer conditioner.Close(context.WithoutCancel(ctx))
	var largest mosstrain.Geometry
	for index, picture := range pictures {
		refined, tokens, err := conditioner.Refine(ctx, picture.caption)
		if err != nil {
			return nil, mosstrain.Geometry{}, fmt.Errorf("caption of %s: %w", picture.path, err)
		}
		examples[index].Refined, examples[index].Geometry.Text = refined, tokens
		largest = mosstrain.Geometry{Height: examples[index].Geometry.Height, Width: examples[index].Geometry.Width, Text: max(largest.Text, tokens)}
	}
	return examples, largest, conditioner.Close(ctx)
}

// interleavedRows turns a processor's [3, size, size] picture, already in the
// signed unit range the encoder reads, into the encoder's interleaved rows.
func interleavedRows(value trainingdata.Value, size int) ([]float32, error) {
	if len(value.Shape) != mossPictureAxes || value.Shape[mossHeightAxis] != size || value.Shape[mossWidthAxis] != size {
		return nil, fmt.Errorf("shape %v is not a picture of %d square", value.Shape, size)
	}
	planar, err := trainingdata.Float32(value)
	if err != nil {
		return nil, err
	}
	channels, plane := value.Shape[0], size*size
	rows := make([]float32, len(planar))
	for position := range plane {
		for channel := range channels {
			rows[position*channels+channel] = planar[channel*plane+position]
		}
	}
	return rows, nil
}

// trainMOSS is the moss route: under a supervised session it fine-tunes a
// low-rank adapter of a MOSS-Image transformer on captioned pictures, along the
// source's flow path at a timestep and noise each step draws from the plan's
// seed, refuses a run whose loss on one fixed path does not descend, and
// publishes the factors as a checkpoint, with -adapter as a PEFT adapter and
// with -served as a model directory that serves the merged weights.
func trainMOSS(ctx context.Context, flags *flag.FlagSet, args []string) (err error) {
	model := flags.String("model", "", "MOSS-Image model directory")
	manifest := flags.String("pictures", "", "UTF-8 manifest of one picture, a tab and its caption per line, relative to the manifest; it grounds the recipe dataset")
	size := flags.Int("size", 0, "the square picture extent every manifest picture has, a multiple of the checkpoint's spatial scale")
	rank := flags.Int("rank", mossDefaultRank, "adapter rank")
	alpha := flags.Float64("alpha", mossDefaultRank, "adapter alpha; a module's scale is alpha over rank")
	batch := flags.Int("batch", 1, "examples each step accumulates")
	steps := clioptions.IntOverride(flags, "steps", "required observed Muon steps")
	maxWall := clioptions.DurationOverride(flags, "max-wall", "optional projected-wall bound")
	storePath := flags.String("store", "overgodb-store", "OvergoDB for the session observation, recipe authority and checkpoint record")
	out := flags.String("out", "", "directory the adapter's checkpoint publishes to")
	adapterOut := flags.String("adapter", "", "optional directory the PEFT adapter is written to")
	served := flags.String("served", "", "optional model directory the merged transformer publishes to, beside the base's other components")
	heldout := flags.String("heldout", "", "optional manifest of pictures the run never trains on, judged on fixed paths before and after it")
	report := flags.String("report", "", "with -heldout: the file its reading is written to")
	proof := proofFlag(flags)
	if err := flags.Parse(args); err != nil {
		return err
	}
	name := flags.Name()
	if *model == "" || *manifest == "" || *out == "" || *steps <= 0 || *size <= 0 || *rank <= 0 || *alpha <= 0 || *batch <= 0 {
		return fmt.Errorf("%s: model, pictures, out, size, and positive steps, rank, alpha and batch required", name)
	}
	if (*heldout == "") != (*report == "") {
		return fmt.Errorf("%s: -heldout and -report go together", name)
	}
	pictures, err := mossManifest(*manifest)
	if err != nil {
		return err
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	directory := roots.ResolveModelPath(*model)
	weights, err := trainingworkflow.IdentifyModelWeights(directory)
	if err != nil {
		return err
	}
	store, err := overgodb.Open(*storePath)
	if err != nil {
		return err
	}
	defer store.Close()
	session, err := openSessionAs(ctx, store, weights, *manifest,
		generationContract(trainingprogram.ObjectiveFlowMatching, recipecontract.ModalityText, recipecontract.ModalityImage, trainingprogram.MetricImagePSNRUnit))
	if err != nil {
		return err
	}
	session.proof = *proof
	defer session.settle(ctx, &err)
	profile, err := latentimage.ResolveProfile(directory)
	if err != nil {
		return err
	}
	spec, err := latentimage.ReadRotaryTransformer(directory)
	if err != nil {
		return err
	}
	loadStarted := time.Now()
	examples, largest, err := mossPrepare(ctx, directory, profile, pictures, *size)
	if err != nil {
		return err
	}
	fmt.Printf("prepared %d pictures in %s: %dx%d grid, up to %d text tokens\n", len(examples), time.Since(loadStarted).Round(time.Millisecond), largest.Width, largest.Height, largest.Text)
	var judged []mosstrain.Example
	var reading mossHeldout
	if *heldout != "" {
		held, err := mossManifest(*heldout)
		if err != nil {
			return err
		}
		for _, picture := range held {
			if slices.ContainsFunc(pictures, func(trained mossPicture) bool { return trained.path == picture.path }) {
				return fmt.Errorf("%s: %s is both trained on and held out", name, picture.path)
			}
			reading.Pictures = append(reading.Pictures, filepath.Base(picture.path))
		}
		var bound mosstrain.Geometry
		if judged, bound, err = mossPrepare(ctx, directory, profile, held, *size); err != nil {
			return err
		}
		largest.Text = max(largest.Text, bound.Text)
		reading.Paths = mossHeldoutPaths
	}
	options := mosstrain.Options{Rank: *rank, Alpha: *alpha}
	plan, err := probeSession(trainingprogram.ObjectiveFlowMatching, *steps, mosstrain.ParameterCount(spec, options), *maxWall)
	if err != nil {
		return err
	}
	seed, err := plan.Seed("flow-path")
	if err != nil {
		return err
	}
	source, err := safetensors.OpenSource(filepath.Join(directory, "transformer"))
	if err != nil {
		return err
	}
	defer source.Close()
	tensors := func(name string) ([]float32, error) {
		tensor, known := source.Tensors[name]
		if !known {
			return nil, errors.New("the checkpoint holds no such tensor")
		}
		return safetensors.ReadF32(tensor)
	}
	config := plan.Optimizer()
	trainer, err := mosstrain.New(ctx, spec, options, tensors, mosstrain.InitialAdapter(spec, options, uint64(seed)), config)
	if err != nil {
		return err
	}
	defer trainer.Close()
	if err := trainer.Reserve(ctx, largest); err != nil {
		return err
	}
	fmt.Printf("trainable parameters=%d pictures=%d derived_lr=%.4g derived_momentum=%.4f\n", mosstrain.ParameterCount(spec, options), len(examples), config.BaseLearningRate, config.Momentum)
	timesteps := profile.Sampling.TrainTimesteps
	// path draws an example's timestep and noise from a stream, the source's
	// uniform draw over its training grid.
	path := func(stream *rand.Rand, example mosstrain.Example) (noised, target []float32, timestep float64) {
		timestep = float64(1 + stream.IntN(timesteps))
		noise := make([]float32, len(example.Latent))
		for index := range noise {
			noise[index] = float32(stream.NormFloat64())
		}
		noised, target = mosstrain.FlowPath(example.Latent, noise, timestep/float64(timesteps))
		return noised, target, timestep
	}
	// judge reads each held-out picture on its fixed paths, one stream per picture.
	judge := func() ([]float64, error) {
		losses := make([]float64, len(judged))
		for index, example := range judged {
			stream := rand.New(rand.NewPCG(mossHeldoutSeed, uint64(index)))
			for range mossHeldoutPaths {
				noised, target, timestep := path(stream, example)
				loss, err := trainer.Loss(ctx, example, noised, target, timestep)
				if err != nil {
					return nil, err
				}
				losses[index] += loss / mossHeldoutPaths
			}
		}
		return losses, nil
	}
	if reading.Base, err = judge(); err != nil {
		return err
	}
	return boundedRun{name: name, plan: plan, session: session,
		step: func(step int) (optimizer.ObservedStepResult, error) {
			if err := trainer.Clear(); err != nil {
				return optimizer.ObservedStepResult{}, err
			}
			stream := rand.New(rand.NewPCG(uint64(seed), uint64(step)))
			var total float64
			for range *batch {
				example := examples[stream.IntN(len(examples))]
				noised, target, timestep := path(stream, example)
				loss, err := trainer.Accumulate(ctx, example, noised, target, timestep)
				if err != nil {
					return optimizer.ObservedStepResult{}, err
				}
				total += loss
			}
			norm, err := trainer.Gather(*batch)
			if err != nil {
				return optimizer.ObservedStepResult{}, err
			}
			return trainer.Apply(total/float64(*batch), norm)
		},
		loss: func() (float64, error) {
			stream := rand.New(rand.NewPCG(uint64(seed), ^uint64(0)))
			var total float64
			count := min(len(examples), mossLossExamples)
			for _, example := range examples[:count] {
				noised, target, timestep := path(stream, example)
				loss, err := trainer.Loss(ctx, example, noised, target, timestep)
				if err != nil {
					return 0, err
				}
				total += loss
			}
			return total / float64(count), nil
		},
		publish: func(ctx context.Context) error {
			if err := session.publishPlanned(ctx, *out, trainer, plan.Updates()); err != nil {
				return err
			}
			adapter := trainer.Adapter(peft.Config{BaseModel: directory, Rank: *rank, Alpha: *alpha})
			if *adapterOut != "" {
				if err := peft.Save(*adapterOut, adapter); err != nil {
					return err
				}
			}
			if *served != "" {
				if err := mosstrain.PublishServed(directory, *served, adapter); err != nil {
					return err
				}
			}
			if *heldout == "" {
				return nil
			}
			if reading.Trained, err = judge(); err != nil {
				return err
			}
			return jsonfile.Write(*report, reading, mossReportMode)
		},
	}.train(ctx)
}
