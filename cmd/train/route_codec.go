package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/clioptions"
	"overgo/internal/latentvideo"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/rae"
	"overgo/internal/recipecontract"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
	"overgo/internal/vitencoder"
)

const (
	// codecPictureProfile names how a picture objective's records are read:
	// each one a PNG picture of the pair's size.
	codecPictureProfile = "overgo/codec-picture/png/v1"
	// codecPictureExtension is the file extension of a picture cut for a dataset.
	codecPictureExtension = ".png"
	// codecTokenExtension is the file extension of a picture's cached source tokens.
	codecTokenExtension = ".source"
)

// codecRun is what a codec run was asked: the pair it trains, the store that
// records it, where tokens are kept between runs, where the trained bridge
// goes, and how long it trains.
type codecRun struct {
	name     string
	resolved rae.BridgedPair
	store    *overgodb.Store
	cache    string
	out      string
	pairOut  string
	steps    int
	batch    int
	maxWall  time.Duration
}

// codecParts is the loaded pair a codec run trains: the bridged pair whose
// bridge moves, the source encoder that gives a picture its tokens, the
// collection whose frozen decoder draws, and the directory that encoder's
// tokens are kept in.
type codecParts struct {
	pair       *rae.BridgedAutoencoder
	source     *vitencoder.Encoder
	collection *rae.Autoencoder
	tokens     string
}

// trainCodec is the codec route: under a supervised session it trains the
// part of an installed autoencoder that the existing backward operators
// reach, against the pictures the autoencoder should give back. For a bridged
// pair that part is its bridge: the squared error between each training
// picture and the picture the pair's decoder draws is carried back through
// the frozen decoder, and Muon steps the bridge. A directory whose decoder
// has no backward pass is refused by the name of its first operation.
//
// Given a manifest of clips, the route trains on each clip's first frame
// under a bootstrapped recipe, refuses a run whose loss on its first batch
// does not descend, and publishes the trained bridge as a checkpoint. Given a
// registered picture objective, it trains on the objective's training
// pictures and publishes the held-out verdict of the trained bridge against
// the installed one, or, as a proof, re-verifies the recipe it is given by
// the training run alone. Either can write the trained bridge as a pair of its
// own. With -cut it trains nothing and writes the manifest's first frames as
// the pictures a dataset registers.
func trainCodec(ctx context.Context, flags *flag.FlagSet, args []string) (err error) {
	pairDir := flags.String("pair", "", "the bridged pair to train: its directory, beside the encoder and the collection it joins")
	manifest := flags.String("frames", "", "UTF-8 manifest naming one training clip per line, relative to the manifest; each clip's first frame is a training picture, and the manifest grounds the recipe dataset")
	objective := flags.String("objective", "", "registered picture objective alias, in place of -frames: train on its training pictures and publish the held-out verdict of the trained bridge against the installed one")
	cut := flags.String("cut", "", "with -frames, train nothing: write each clip's first frame, cut to the pair's size, as a PNG picture in this directory, which registers as a picture dataset")
	batch := flags.Int("batch", 0, "training pictures a step is taken on")
	steps := clioptions.IntOverride(flags, "steps", "required observed Muon steps")
	maxWall := clioptions.DurationOverride(flags, "max-wall", "optional projected-wall bound")
	cache := flags.String("cache", "", "directory that keeps each picture's tokens between runs")
	storePath := flags.String("store", "overgodb-store", "OvergoDB for the session observation, recipe authority, checkpoint record and verdict")
	out := flags.String("out", "", "directory the trained bridge's checkpoint publishes to")
	pairOut := flags.String("pair-out", "", "optional directory, beside the encoder and the collection, to write the trained bridge to as a pair of its own")
	recipeText := flags.String("recipe", "", "with -objective, the training recipe artifact ID to train under; omitted trains under the objective's own")
	proof := flags.Bool("proof", false, "with -objective, re-verify the recipe by the training run alone: it reads tokens for the pictures its steps take and keeps none, so it takes no -cache, judges no bridge and publishes no held-out verdict")
	if err := flags.Parse(args); err != nil {
		return err
	}
	name := flags.Name()
	if *pairDir == "" || (*manifest == "") == (*objective == "") || *objective == "" && (*proof || *recipeText != "") {
		return fmt.Errorf("%s: pair and one of frames and objective required, and proof and recipe go with objective", name)
	}
	resolved, bridged, err := rae.ResolveBridgedPair(*pairDir)
	if err != nil {
		return err
	}
	if !bridged {
		return refuseUntrainedDecoder(name, *pairDir)
	}
	if *cut != "" {
		if *manifest == "" {
			return fmt.Errorf("%s: cut reads the clips a frames manifest names", name)
		}
		return cutCodecPictures(ctx, name, resolved, *manifest, *cut)
	}
	if (*cache == "") != *proof || *out == "" || *steps <= 0 || *batch <= 0 {
		return fmt.Errorf("%s: out, positive steps and batch, and a cache, which a proof alone does not take, required", name)
	}
	if *pairOut != "" && filepath.Dir(filepath.Clean(*pairOut)) != filepath.Dir(filepath.Clean(*pairDir)) {
		return fmt.Errorf("%s: the trained pair sits beside the pair it was trained from", name)
	}
	var recipeID artifact.ID
	if *recipeText != "" {
		if recipeID, err = artifact.ParseID(*recipeText); err != nil {
			return fmt.Errorf("training recipe: %w", err)
		}
	}
	store, err := overgodb.Open(*storePath)
	if err != nil {
		return err
	}
	defer store.Close()
	run := codecRun{
		name: name, resolved: resolved, store: store, cache: *cache, out: *out, pairOut: *pairOut,
		steps: *steps, batch: *batch, maxWall: *maxWall,
	}
	if *objective != "" {
		return run.judged(ctx, *objective, recipeID, *proof)
	}
	return run.observed(ctx, *manifest)
}

// load reads the pair's bridge, the encoder it maps and the collection whose
// decoder it feeds, each once, and makes the directory the encoder's tokens
// are kept in when the run was given a cache.
func (run codecRun) load() (codecParts, error) {
	source, err := vitencoder.Load(run.resolved.SourceEncoder)
	if err != nil {
		return codecParts{}, err
	}
	collection, err := rae.LoadAutoencoder(run.resolved.CollectionDirectory)
	if err != nil {
		return codecParts{}, err
	}
	bridge, err := rae.LoadEncoderBridge(run.resolved.Bridge)
	if err != nil {
		return codecParts{}, err
	}
	pair, err := collection.Bridged(source, bridge)
	if err != nil {
		return codecParts{}, err
	}
	// The run keeps the source encoder's tokens alone, so another decoder,
	// bridge or statistics leaves them good. A proof has no cache and keeps none.
	parts := codecParts{pair: pair, source: source, collection: collection}
	if run.cache != "" {
		parts.tokens, err = tokenCache(run.cache, run.resolved.SourceEncoder)
	}
	return parts, err
}

// trainer plans the run's steps over the bridge's weights and starts the
// trainer under the rate the plan derives.
func (run codecRun) trainer(parts codecParts) (trainingprogram.TrainingSessionPlan, *rae.BridgeTrainer, error) {
	plan, err := probeSession(trainingprogram.ObjectiveImageLatent, run.steps, parts.pair.BridgeParameterCount(), run.maxWall)
	if err != nil {
		return trainingprogram.TrainingSessionPlan{}, nil, err
	}
	config := plan.Optimizer()
	trainer, err := rae.NewBridgeTrainer(parts.pair, config)
	if err != nil {
		return trainingprogram.TrainingSessionPlan{}, nil, err
	}
	fmt.Printf("trainable parameters=%d batch=%d derived_lr=%.4g derived_momentum=%.4f\n",
		trainer.TrainableParameterCount(), run.batch, config.BaseLearningRate, config.Momentum)
	return plan, trainer, nil
}

// batchAt is a step's batch: the next run of the pictures, wrapping at their end.
func (run codecRun) batchAt(examples []rae.BridgeExample, step int) []rae.BridgeExample {
	chosen := make([]rae.BridgeExample, run.batch)
	for at := range chosen {
		chosen[at] = examples[(step*run.batch+at)%len(examples)]
	}
	return chosen
}

// savePair writes the trained bridge as a pair of its own when one was asked for.
func (run codecRun) savePair(trainer *rae.BridgeTrainer) error {
	if run.pairOut == "" {
		return nil
	}
	return trainer.SavePair(run.pairOut, run.resolved.SourceEncoder, run.resolved.CollectionDirectory)
}

// observed trains on the first frames of a manifest's clips under a recipe
// bootstrapped over the manifest: the session is admitted before any weight
// is read, the run is held to the loss of its first batch, and the trained
// bridge publishes as a checkpoint.
func (run codecRun) observed(ctx context.Context, manifest string) (err error) {
	clips, err := manifestLines(manifest)
	if err != nil {
		return err
	}
	if len(clips) < run.batch {
		return fmt.Errorf("%s: the manifest names %d clips, fewer than one batch of %d", run.name, len(clips), run.batch)
	}
	// The trainer steps the squared error from frames to the frames the pair
	// gives back through its latent.
	session, err := openSessionAs(ctx, run.store, run.resolved.Bridge, manifest,
		generationContract(trainingprogram.ObjectiveImageLatent, recipecontract.ModalityImage, recipecontract.ModalityImage, trainingprogram.MetricImagePSNRUnit))
	if err != nil {
		return err
	}
	defer session.settle(ctx, &err)
	parts, err := run.load()
	if err != nil {
		return err
	}
	side := parts.collection.Side()
	examples := make([]rae.BridgeExample, len(clips))
	for index, clip := range clips {
		cut, err := clipFirstFrame(ctx, filepath.Join(filepath.Dir(manifest), clip), side)
		if err != nil {
			return err
		}
		picture, err := unitPicture(cut)
		if err != nil {
			return err
		}
		tokens, err := pictureSourceTokens(parts.source, picture, side, filepath.Join(parts.tokens, pictureDigest(picture)+codecTokenExtension))
		if err != nil {
			return err
		}
		examples[index] = rae.BridgeExample{SourceTokens: tokens, Picture: picture}
	}
	plan, trainer, err := run.trainer(parts)
	if err != nil {
		return err
	}
	config := plan.Optimizer()
	return boundedRun{name: run.name, plan: plan, session: session,
		step: func(step int) (optimizer.ObservedStepResult, error) {
			loss, err := trainer.Step(ctx, run.batchAt(examples, step))
			return optimizer.ObservedStepResult{
				Loss: loss, Step: step + tensor.SingletonExtent, LearningRate: config.LearningRate(step + tensor.SingletonExtent), GradientL2: trainer.GradientL2(),
			}, err
		},
		// The run is held to the loss of the first batch, before and after.
		loss: func() (float64, error) { return trainer.Loss(ctx, run.batchAt(examples, 0)) },
		publish: func(ctx context.Context) error {
			if err := run.savePair(trainer); err != nil {
				return err
			}
			return session.publishPlanned(ctx, run.out, trainer, plan.Updates())
		},
	}.train(ctx)
}

// judged trains on the training pictures of a registered picture objective
// and publishes its held-out verdict: the installed bridge and the trained
// one each have the frozen decoder draw every held-out picture from the
// source encoder's tokens, and the verdict compares the mean peak
// signal-to-noise ratio of what it draws. The bridge never sees a held-out
// picture. It trains under recipeID, or under the objective's own recipe when
// that is zero. A proof judges nothing, so it reads the encoder's tokens only
// as its steps take their pictures, keeps none and publishes no verdict.
func (run codecRun) judged(ctx context.Context, alias string, recipeID artifact.ID, proof bool) error {
	objective, err := trainingworkflow.LoadObjectiveAlias(ctx, run.store, alias)
	if err != nil {
		return err
	}
	if objective.Kind != trainingprogram.ObjectiveImageLatent {
		return fmt.Errorf("%s: objective %q trains %q, not a picture through its latent", run.name, alias, objective.Kind)
	}
	loadStart := time.Now()
	parts, err := run.load()
	if err != nil {
		return err
	}
	side := parts.collection.Side()
	data, digests, err := codecPictures(ctx, run.store, objective, side)
	if err != nil {
		return err
	}
	tokens := make(map[string][]float32, len(data.Examples))
	if !proof {
		for record, example := range data.Examples {
			cached := filepath.Join(parts.tokens, digests[record]+codecTokenExtension)
			if tokens[record], err = pictureSourceTokens(parts.source, dtype.ConvertSlice[float32](example.Target), side, cached); err != nil {
				return err
			}
			fmt.Fprintf(os.Stderr, "  tokens %d/%d\n", len(tokens), len(data.Examples))
		}
	}
	plan, trainer, err := run.trainer(parts)
	if err != nil {
		return err
	}
	result, err := trainingworkflow.RunHeldout(ctx, run.store, trainingworkflow.HeldoutRun{
		// The representation autoencoder runs on the host.
		Alias: alias, ModelPath: run.resolved.Bridge, Recipe: recipeID, Data: data, Host: true, Load: time.Since(loadStart), Proof: proof,
		Train: func(ctx context.Context, recipeID artifact.ID, shown []trainingworkflow.HeldoutExample) (trainingprogram.Checkpoint, error) {
			if len(shown) < run.batch {
				return trainingprogram.Checkpoint{}, fmt.Errorf("%s: the training membership holds %d pictures, fewer than one batch of %d", run.name, len(shown), run.batch)
			}
			examples := make([]rae.BridgeExample, len(shown))
			for index, example := range shown {
				examples[index] = rae.BridgeExample{SourceTokens: tokens[example.Record], Picture: dtype.ConvertSlice[float32](example.Target)}
			}
			for step := range plan.Updates() {
				batch := run.batchAt(examples, step)
				if proof {
					// A proof read no tokens before training: it reads them
					// here, for the pictures this step takes.
					for at, example := range batch {
						read, err := rae.BridgeSourceTokens(parts.source, example.Picture, side)
						if err != nil {
							return trainingprogram.Checkpoint{}, err
						}
						batch[at].SourceTokens = read
					}
				}
				loss, err := trainer.Step(ctx, batch)
				if err != nil {
					return trainingprogram.Checkpoint{}, err
				}
				if !checked.Finite64(loss) {
					return trainingprogram.Checkpoint{}, fmt.Errorf("%s: step %d loss is non-finite", run.name, step+tensor.SingletonExtent)
				}
				fmt.Printf("step %d/%d: loss=%.6f grad_l2=%.4g\n", step+tensor.SingletonExtent, plan.Updates(), loss, trainer.GradientL2())
			}
			if err := run.savePair(trainer); err != nil {
				return trainingprogram.Checkpoint{}, err
			}
			// Each update consumed one batch of training pictures.
			return publishPlannedTrained(ctx, run.store, recipeID, run.out, trainer, trainingdata.StreamState{Identity: data.Training, Position: uint64(plan.Updates() * run.batch)}, proof)
		},
		Predict: func(example trainingworkflow.HeldoutExample) (trainingworkflow.HeldoutPrediction, error) {
			mapped, err := trainer.Bridge().Apply(tokens[example.Record])
			if err != nil {
				return trainingworkflow.HeldoutPrediction{}, err
			}
			drawn, err := parts.collection.Decode(ctx, mapped)
			return trainingworkflow.HeldoutPrediction{Values: dtype.ConvertSlice[float64](drawn)}, err
		},
	})
	if err != nil {
		return err
	}
	if proof {
		return reportProof(os.Stdout, result)
	}
	return reportVerdict(os.Stdout, result)
}

// codecPictures reads both memberships of a picture objective's dataset: each
// record is one PNG picture, side by side, which becomes its example's
// target, channel-first in [0, 1]. Beside the examples it returns each
// record's digest, the name its tokens are kept under.
func codecPictures(ctx context.Context, store *overgodb.Store, objective trainingprogram.ObjectiveDocument, side int) (trainingworkflow.ObjectiveData, map[string]string, error) {
	digests := map[string]string{}
	data, err := trainingworkflow.ObjectiveExamples(ctx, store, objective, []byte(codecPictureProfile), recipecontract.ModalityImage,
		trainingdata.Passthrough(trainingdata.RoleInput, recipecontract.ModalityImage, strings.TrimPrefix(codecPictureExtension, ".")),
		func(example trainingdata.Example) (trainingworkflow.HeldoutExample, error) {
			if len(example.Values) != tensor.SingletonExtent {
				return trainingworkflow.HeldoutExample{}, errors.New("not one picture")
			}
			encoded := example.Values[0].Data
			decoded, err := png.Decode(bytes.NewReader(encoded))
			if err != nil {
				return trainingworkflow.HeldoutExample{}, err
			}
			if bounds := decoded.Bounds(); bounds.Dx() != side || bounds.Dy() != side {
				return trainingworkflow.HeldoutExample{}, fmt.Errorf("a picture of %d by %d for a pair that draws %d by %d", bounds.Dx(), bounds.Dy(), side, side)
			}
			picture, err := unitPicture(decoded)
			if err != nil {
				return trainingworkflow.HeldoutExample{}, err
			}
			sum := sha256.Sum256(encoded)
			digests[example.ID] = hex.EncodeToString(sum[:])
			return trainingworkflow.HeldoutExample{Target: dtype.ConvertSlice[float64](picture)}, nil
		})
	return data, digests, err
}

// pictureSourceTokens is the source encoder's tokens for one picture, read
// from cached when a run before this one kept them there.
func pictureSourceTokens(source *vitencoder.Encoder, picture []float32, side int, cached string) ([]float32, error) {
	_, grid, width := source.Shape()
	if tokens, kept := readTokens(cached, grid*grid*width); kept {
		return tokens, nil
	}
	tokens, err := rae.BridgeSourceTokens(source, picture, side)
	if err != nil {
		return nil, err
	}
	return tokens, writeTokens(cached, tokens)
}

// cutCodecPictures writes the first frame of each clip a manifest names, cut
// to the pair's size with its proportions kept, as a PNG picture named after
// the clip: the directory then registers as a picture dataset.
func cutCodecPictures(ctx context.Context, route string, resolved rae.BridgedPair, manifest, directory string) error {
	clips, err := manifestLines(manifest)
	if err != nil {
		return err
	}
	collection, err := rae.LoadAutoencoder(resolved.CollectionDirectory)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(directory, bridgeCacheMode); err != nil {
		return err
	}
	for _, clip := range clips {
		frame, err := clipFirstFrame(ctx, filepath.Join(filepath.Dir(manifest), clip), collection.Side())
		if err != nil {
			return err
		}
		var encoded bytes.Buffer
		if err := png.Encode(&encoded, frame); err != nil {
			return err
		}
		picture := filepath.Join(directory, strings.TrimSuffix(filepath.Base(clip), filepath.Ext(clip))+codecPictureExtension)
		if err := os.WriteFile(picture, encoded.Bytes(), bridgeCacheFileMode); err != nil {
			return err
		}
	}
	fmt.Printf("%s: cut %d pictures of %d by %d into %s\n", route, len(clips), collection.Side(), collection.Side(), directory)
	return nil
}

// refuseUntrainedDecoder says why a directory that is no bridged pair cannot
// be trained by this route: a Wan autoencoder's decoder is refused by the
// name of its first operation, none of whose kind has a backward pass, and
// any other directory as holding nothing the route trains.
func refuseUntrainedDecoder(route, directory string) error {
	if _, err := os.Stat(latentvideo.AutoencoderWeights(directory)); err != nil {
		return fmt.Errorf("%s: %s holds no bridged pair and no autoencoder this route knows", route, directory)
	}
	operations, err := latentvideo.AutoencoderDecoderOperations(directory)
	if err != nil {
		return err
	}
	first, ok := firstOperation(operations)
	if !ok {
		return errors.New(route + ": the autoencoder's decoder compiles to no operations")
	}
	return fmt.Errorf("%s: the autoencoder in %s cannot be trained here: its decoder's first operation, %s, convolves over frames, rows and columns, and none of the decoder's %d operations has a backward pass",
		route, directory, first, len(operations))
}

// firstOperation is the first name of a decoder's operations.
func firstOperation(operations []string) (string, bool) {
	if len(operations) == 0 {
		return "", false
	}
	return operations[0], true
}

// manifestLines reads the names a manifest lists, one per line.
func manifestLines(manifest string) ([]string, error) {
	raw, err := os.ReadFile(manifest)
	if err != nil {
		return nil, err
	}
	var names []string
	for line := range bytes.Lines(raw) {
		if name := string(bytes.TrimSpace(line)); name != "" {
			names = append(names, name)
		}
	}
	if len(names) == 0 {
		return nil, fmt.Errorf("manifest %s names no clips", manifest)
	}
	return names, nil
}
