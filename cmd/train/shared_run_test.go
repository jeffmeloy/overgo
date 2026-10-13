package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/hostoptimizer"
	"overgo/internal/modelrecipe"
	"overgo/internal/optimizer"
	"overgo/internal/oscillatorimage"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/recipecontract"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// TestEveryTrainRoutePublishesACheckpoint holds every route to publishing
// what it trains: each takes a -out checkpoint target it refuses to train
// without, and the shared run it publishes through trains under the shared
// guards and writes a checkpoint that loads with its weights and is
// recorded in the store.
func TestEveryTrainRoutePublishesACheckpoint(t *testing.T) {
	for name := range trainRoutes {
		t.Run(name, func(t *testing.T) {
			if err := runRoute(t.Context(), []string{name, "-out", t.TempDir()}); err == nil || strings.Contains(err.Error(), "not defined") {
				t.Errorf("route %s with only a checkpoint target = %v, want its refusal of the missing flags", name, err)
			}
			if err := runRoute(t.Context(), []string{name}); err == nil || !strings.Contains(err.Error(), "out") {
				t.Errorf("route %s without a checkpoint target = %v, want a refusal requiring -out", name, err)
			}
		})
	}
	t.Run("shared-run", sharedRunPublishes)
}

// sharedRunPublishes trains a host Muon matrix toward a fixed target under a
// bootstrapped session, publishes it as the routes publish a planned
// trainer, and reads the checkpoint back from the directory and the store.
func sharedRunPublishes(t *testing.T) {
	root, store, modelPath, datasetPath := routeFixture(t)
	ctx := t.Context()
	session, err := openSession(ctx, store, modelPath, datasetPath)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := probeSession(session.authority.Objective, 3, testMatrix.End, 0)
	if err != nil {
		t.Fatal(err)
	}
	matrix := newHostMatrix(t, session.authority.Objective, plan.Optimizer())
	target := filepath.Join(root, "checkpoint")
	run := boundedRun{name: "shared-run", plan: plan, session: session, loss: matrix.loss,
		step: func(int) (optimizer.ObservedStepResult, error) { return matrix.step(), nil },
		publish: func(ctx context.Context) error {
			return session.publishPlanned(ctx, target, matrix, plan.Updates())
		},
	}
	if err := run.train(ctx); err != nil {
		t.Fatal(err)
	}
	stream, err := trainingworkflow.TextStream([]byte(routeFixtureDataset), uint64(plan.Updates()))
	if err != nil {
		t.Fatal(err)
	}
	checkpoint, err := trainingprogram.LoadCheckpoint(target)
	if err != nil {
		t.Fatal(err)
	}
	_, recorded, err := store.Artifact(ctx, checkpoint.ID())
	if err != nil || !recorded || checkpoint.Model != session.model || checkpoint.Program != matrix.program.ID() ||
		checkpoint.Stream != trainingprogram.DatasetState(stream) {
		t.Fatalf("checkpoint %s of %s under program %s at %+v is recorded %t (%v), want %s's checkpoint under %s at %+v recorded",
			checkpoint.ID(), checkpoint.Model, checkpoint.Program, checkpoint.Stream, recorded, err, session.model, matrix.program.ID(), stream)
	}
	rerun := filepath.Join(root, "rerun")
	if err := session.publishPlanned(ctx, rerun, matrix, plan.Updates()); err != nil {
		t.Fatal(err)
	}
	if again, err := trainingprogram.LoadCheckpoint(rerun); err != nil || again.ID() != checkpoint.ID() {
		t.Fatalf("a rerun publishing the same checkpoint elsewhere = (%s, %v), want %s recorded again", again.ID(), err, checkpoint.ID())
	}
}

// TestProofWritesOnlyWhatItsClaimReads holds a training proof to writing only
// what its claim reads. A proof re-verifies its recipe on its session's
// observation and on the identity of the checkpoint the session trained, and
// reads nothing else of the checkpoint. So it gives the store the checkpoint's
// document, with what the checkpoint descends from, and keeps no file: nothing
// stands at its -out, no staging directory is left beside it, the store names
// no place the checkpoint is, and a GGUF base's proof registers no trained
// model. The identity is the one a run that is not a proof publishes for the
// same trained state, so the evidence a verification cites is what it was.
//
// That is held for the shared run the routes publish through, for the two
// publications a held-out run's proof makes, the tabular route's and the codec
// route's, and for the dense workflow and a text objective's train -objective
// through the training command's own flags, the child being this test binary
// run as the command on the host; and every route the measurement run proves
// takes -proof. The command refuses -proof with each thing it does that is no
// training run of those two kinds. A run that is not a proof publishes as
// before, which the second run of each case but the text objective's, whose
// full run is another test's, and TestEveryTrainRoutePublishesACheckpoint
// hold.
func TestProofWritesOnlyWhatItsClaimReads(t *testing.T) {
	t.Run("the shared run", func(t *testing.T) {
		root, store, modelPath, datasetPath := routeFixture(t)
		ctx := t.Context()
		session, err := openSession(ctx, store, modelPath, datasetPath)
		if err != nil {
			t.Fatal(err)
		}
		session.proof = true
		plan, err := probeSession(session.authority.Objective, 3, testMatrix.End, 0)
		if err != nil {
			t.Fatal(err)
		}
		matrix := newHostMatrix(t, session.authority.Objective, plan.Optimizer())
		target, full := filepath.Join(root, "proof"), filepath.Join(root, "full")
		if err := (boundedRun{name: "proof", plan: plan, session: session, loss: matrix.loss,
			step: func(int) (optimizer.ObservedStepResult, error) { return matrix.step(), nil },
			publish: func(ctx context.Context) error {
				return session.publishPlanned(ctx, target, matrix, plan.Updates())
			},
		}).train(ctx); err != nil {
			t.Fatal(err)
		}
		proved := *session.published
		requireProofFootprint(t, store, target, proved, session.model)
		// The same trained state, published by a session that is not a proof.
		session.proof = false
		if err := session.publishPlanned(ctx, full, matrix, plan.Updates()); err != nil {
			t.Fatal(err)
		}
		requireFullPublication(t, store, full, proved)
	})
	t.Run("a held-out run's publications", func(t *testing.T) {
		root, store, modelPath, datasetPath := routeFixture(t)
		ctx := t.Context()
		session, err := recipeSession(ctx, store, modelPath, datasetPath, trainingworkflow.TokenContract)
		if err != nil {
			t.Fatal(err)
		}
		config, err := trainingprogram.BuiltinOptimizerPolicy().Config(testMatrix.End)
		if err != nil {
			t.Fatal(err)
		}
		matrix := newHostMatrix(t, session.authority.Objective, config)
		for name, publish := range map[string]func(target string, stream trainingdata.StreamState, proof bool) (trainingprogram.Checkpoint, error){
			"under the trainer's own program, as the tabular route publishes": func(target string, stream trainingdata.StreamState, proof bool) (trainingprogram.Checkpoint, error) {
				return publishTrained(ctx, store, session.recipe, target, matrix, stream, proof)
			},
			"under the program the recipe's objective compiles, as the codec route publishes": func(target string, stream trainingdata.StreamState, proof bool) (trainingprogram.Checkpoint, error) {
				return publishPlannedTrained(ctx, store, session.recipe, target, matrix, stream, proof)
			},
		} {
			// Each publication takes a step of its own, so its checkpoint is no other's.
			stream, err := trainingworkflow.TextStream([]byte(routeFixtureDataset), uint64(matrix.step().Step))
			if err != nil {
				t.Fatal(err)
			}
			target, full := filepath.Join(root, name, "proof"), filepath.Join(root, name, "full")
			proved, err := publish(target, stream, true)
			if err != nil {
				t.Fatalf("a proof's publication %s: %v", name, err)
			}
			requireProofFootprint(t, store, target, proved.ID(), session.model)
			if _, err := publish(full, stream, false); err != nil {
				t.Fatalf("a publication %s that is not a proof's: %v", name, err)
			}
			requireFullPublication(t, store, full, proved.ID())
		}
	})
	t.Run("every route the measurement run proves takes -proof", func(t *testing.T) {
		// cmd/validate names these routes in its training proofs; hybrid-stack
		// is registered where the build carries its device worker.
		for _, name := range []string{"codec", "dit", "hybrid-stack", "mot", "tabular"} {
			if _, registered := trainRoutes[name]; !registered {
				continue
			}
			if err := runRoute(t.Context(), []string{name, "-proof", "-out", t.TempDir()}); err == nil || strings.Contains(err.Error(), "not defined") {
				t.Errorf("route %s given -proof and a checkpoint target = %v, want its refusal of the inputs it lacks", name, err)
			}
		}
	})
	t.Run("the dense workflow and a text objective, by the command's own flags", func(t *testing.T) {
		ctx := t.Context()
		root := t.TempDir()
		base, dataset, storePath := writeServingFixture(t, root), filepath.Join(root, "dataset.txt"), filepath.Join(root, "store")
		if err := os.WriteFile(dataset, bytes.Repeat([]byte("a b c d "), 64), 0o600); err != nil {
			t.Fatal(err)
		}
		store, err := overgodb.Open(storePath)
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		if _, err := modelrecipe.PublishArchitectureProfileCatalog(ctx, store); err != nil {
			t.Fatal(err)
		}
		recipeID, err := trainingworkflow.BootstrapTokenRecipe(ctx, store, base, dataset)
		if err != nil {
			t.Fatal(err)
		}
		model, err := trainingworkflow.WeightsIdentity(base)
		if err != nil {
			t.Fatal(err)
		}
		t.Setenv(commandChildEnvironment, childStandIn)
		// started runs the training command over the arguments and returns
		// what it printed, the status it ended with and the error it wrote for
		// its starter.
		started := func(arguments ...string) (string, int, string) {
			t.Helper()
			var printed bytes.Buffer
			written := filepath.Join(t.TempDir(), "error.txt")
			receipt, err := processcontrol.Run(ctx, processcontrol.Command{
				Path: os.Args[0], Args: arguments, Stdout: &printed, Env: append(os.Environ(), clioptions.ErrorFileEnvironment+"="+written),
			})
			if err != nil {
				t.Fatal(err)
			}
			// A command that ended well wrote no error.
			refusal, _ := os.ReadFile(written)
			return printed.String(), receipt.ExitCode, string(refusal)
		}
		// command trains the fixture's model on the host into out, by the mode
		// and bounds its arguments name, and returns what the command printed;
		// the store is then read afresh, since the command wrote through a
		// handle of its own.
		command := func(out string, arguments ...string) string {
			t.Helper()
			printed, status, refusal := started(append([]string{"-store", storePath, "-model", base, "-host", "-out", out}, arguments...)...)
			if status != 0 {
				t.Fatalf("the training command given %v ended with status %d: %s", arguments, status, refusal)
			}
			if err := store.Refresh(ctx); err != nil {
				t.Fatal(err)
			}
			return printed
		}
		models := func() int {
			registered, err := store.Query(ctx, overgodb.Query{Kind: artifact.KindModel, MaxResults: 100, Projection: overgodb.ProjectArtifacts})
			if err != nil || registered.Truncated {
				t.Fatal(err, registered.Truncated)
			}
			return len(registered.Artifacts)
		}
		// proves runs command as a proof into out, holds what it left to a
		// proof's footprint, and returns the checkpoint the proof named.
		proves := func(out string, arguments ...string) artifact.ID {
			t.Helper()
			before := models()
			printed := command(out, append(arguments, "-proof")...)
			_, named, _ := strings.Cut(printed, "trained=")
			named, _, _ = strings.Cut(named, ";")
			proved, err := artifact.ParseID(named)
			if err != nil {
				t.Fatalf("the proof printed %q, want it to name the checkpoint it trained: %v", printed, err)
			}
			requireProofFootprint(t, store, out, proved, model)
			if after := models(); after != before {
				t.Errorf("a proof over a GGUF base left the store with %d models, and it held %d before: want no trained model registered", after, before)
			}
			return proved
		}
		// A proof is a training run: everything else the command does, and
		// the audio adapter's run, which publishes by a path of its own, is
		// refused one before it starts. The store is named so that a command
		// that was not refused would open the fixture's, not one of its own
		// beside this package.
		for _, other := range []string{"-pilot", "-preview-dataset", "-bootstrap-recipe", "-audio-manifest"} {
			if _, status, refusal := started("-store", storePath, "-proof", other, "named"); status != 1 || !strings.Contains(refusal, "-proof re-verifies") {
				t.Errorf("the training command given -proof with %s ended with status %d and %q, want the two refused together", other, status, refusal)
			}
		}
		// The dense workflow, one step under the recipe it is named.
		dense := []string{"-recipe", recipeID.String(), "-dataset", dataset, "-steps", "1", "-seq", "16"}
		proved, full := proves(filepath.Join(root, "proof"), dense...), filepath.Join(root, "full")
		command(full, dense...)
		requireFullPublication(t, store, full, proved)
		// A text objective, by train -objective: each file of the corpus is one
		// record, of a length of its own.
		corpus := filepath.Join(root, "corpus")
		if err := os.Mkdir(corpus, 0o700); err != nil {
			t.Fatal(err)
		}
		const documents = 8
		for index := range documents {
			text := strings.Repeat("a b c d ", 3)[:8+2*index]
			if err := os.WriteFile(filepath.Join(corpus, fmt.Sprintf("text-%02d.txt", index)), []byte(text), 0o600); err != nil {
				t.Fatal(err)
			}
		}
		source, training := registerSplit(t, ctx, store, "texts", corpus, documents, nil)
		const alias = "objective.registered.texts"
		registerObjective(t, ctx, store, root, alias, trainingprogram.ObjectiveSpec{
			Name: "texts", Kind: trainingprogram.ObjectiveTokenPrediction, Metric: trainingprogram.MetricTokenAccuracy,
			Signature: recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{recipecontract.ModalityText}, Outputs: []recipecontract.Modality{recipecontract.ModalityText}},
			Dataset:   source, Split: training,
		})
		proves(filepath.Join(root, "objective-proof"), "-objective", alias, "-steps", "2", "-seq", "8")
	})
}

// requireProofFootprint holds a proof that trained checkpoint from model, and
// was given target as its output, to what it may leave: nothing at target, no
// staging directory beside it, and in the store the checkpoint's document, its
// descent from model and no location.
func requireProofFootprint(t *testing.T, store *overgodb.Store, target string, checkpoint, model artifact.ID) {
	t.Helper()
	ctx := t.Context()
	if entries, err := os.ReadDir(target); !errors.Is(err, os.ErrNotExist) {
		var names []string
		for _, entry := range entries {
			names = append(names, entry.Name())
		}
		t.Errorf("the proof left %v at %s (%v), want nothing there", names, target, err)
	}
	if staged, err := filepath.Glob(filepath.Join(filepath.Dir(target), ".overgo-checkpoint-*")); err != nil || len(staged) != 0 {
		t.Errorf("the proof left the staging directories %v beside %s (%v), want none", staged, target, err)
	}
	if _, held, err := artifact.ReadContent(ctx, store, checkpoint); err != nil || !held {
		t.Errorf("the store holds the document of the proof's checkpoint %s: %t (%v), want it held", checkpoint, held, err)
	}
	if parents, err := store.Parents(ctx, checkpoint); err != nil || !slices.Contains(parents, artifact.Lineage{Child: checkpoint, Parent: model, Relation: artifact.RelationTrainedFrom}) {
		t.Errorf("the store holds the parents %v of the proof's checkpoint (%v), want its descent from the model %s it trained", parents, err, model)
	}
	if locations, err := store.Locations(ctx, checkpoint); err != nil || len(locations) != 0 {
		t.Errorf("the store records the proof's checkpoint at %v (%v), want no location", locations, err)
	}
}

// requireFullPublication holds a run that is not a proof, publishing at target
// the trained state a proof identified as checkpoint, to publishing as before:
// a checkpoint of that identity loads from target, and the store records
// where it is.
func requireFullPublication(t *testing.T, store *overgodb.Store, target string, checkpoint artifact.ID) {
	t.Helper()
	published, err := trainingprogram.LoadCheckpoint(target)
	if err != nil || published.ID() != checkpoint {
		t.Fatalf("a run that is not a proof published checkpoint (%s, %v) at %s, want the checkpoint %s the proof identified", published.ID(), err, target, checkpoint)
	}
	if locations, err := store.Locations(t.Context(), checkpoint); err != nil || len(locations) == 0 {
		t.Fatalf("the store records the published checkpoint at %v (%v), want where its files are", locations, err)
	}
}

// routeFixtureDataset is the dataset a route fixture's session grounds.
const routeFixtureDataset = "training data"

// routeFixture writes a stand-in model file and a dataset a route's session
// grounds, and opens a store for the session.
func routeFixture(t *testing.T) (root string, store *overgodb.Store, modelPath, datasetPath string) {
	t.Helper()
	root = t.TempDir()
	modelPath, datasetPath = filepath.Join(root, "model.safetensors"), filepath.Join(root, "dataset.txt")
	for path, data := range map[string]string{modelPath: "weights", datasetPath: routeFixtureDataset} {
		if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	return root, store, modelPath, datasetPath
}

// TestRoutePublicationRecordsActualAuthority holds the shared route
// publisher to the facts of what ran. A trainer that steps flow matching to
// video opens its session under that contract, so its checkpoint names the
// flow-matching program it executed and the dataset as the session read it,
// though the file changed after. A bootstrap built from nothing, as the
// oscillatorimage route trains, steps latent L2 toward images from its
// scratch construction, and its checkpoint names that construction as the
// model it trained from. A trainer whose objective the session's recipe
// does not declare is refused before any step and publishes nothing. A run
// whose context ends stops between updates, publishes nothing and is
// recorded as cancelled, and a setup that fails after admission finishes
// its session once.
func TestRoutePublicationRecordsActualAuthority(t *testing.T) {
	flow := generationContract(trainingprogram.ObjectiveFlowMatching, recipecontract.ModalityText, recipecontract.ModalityVideo, trainingprogram.MetricVideoPSNRUnit)
	t.Run("scratch start", func(t *testing.T) {
		root, store, _, datasetPath := routeFixture(t)
		declaration, err := os.ReadFile(datasetPath)
		if err != nil {
			t.Fatal(err)
		}
		construction, processor, err := bootstrapConstruction(oscillatorimage.Config{ModelType: "fixture"}, testMatrix.End, 1, declaration)
		if err != nil {
			t.Fatal(err)
		}
		session, err := scratchSession(t.Context(), store, construction, processor,
			generationContract(trainingprogram.ObjectiveLatentL2, recipecontract.ModalityText, recipecontract.ModalityImage, trainingprogram.MetricImagePSNRUnit))
		if err != nil {
			t.Fatal(err)
		}
		session.data = declaration
		plan, err := probeSession(trainingprogram.ObjectiveLatentL2, 3, testMatrix.End, 0)
		if err != nil {
			t.Fatal(err)
		}
		matrix := newHostMatrix(t, trainingprogram.ObjectiveLatentL2, plan.Optimizer())
		target := filepath.Join(root, "checkpoint")
		if err := (boundedRun{name: "scratch", plan: plan, session: session, loss: matrix.loss,
			step: func(int) (optimizer.ObservedStepResult, error) { return matrix.step(), nil },
			publish: func(ctx context.Context) error {
				return session.publishPlanned(ctx, target, matrix, plan.Updates())
			},
		}).train(t.Context()); err != nil {
			t.Fatal(err)
		}
		checkpoint, err := trainingprogram.LoadCheckpoint(target)
		if err != nil {
			t.Fatal(err)
		}
		scratch := trainingprogram.LineageParent{Artifact: construction.ID(), Relation: artifact.RelationDependsOn}
		if checkpoint.Model != construction.InitializedModel() || checkpoint.Program != matrix.program.ID() || !slices.Contains(checkpoint.Lineage, scratch) {
			t.Fatalf("scratch checkpoint of %s under program %s with lineage %v, want the latent-l2 program %s from construction %s",
				checkpoint.Model, checkpoint.Program, checkpoint.Lineage, matrix.program.ID(), construction.ID())
		}
	})
	t.Run("actual program and data", func(t *testing.T) {
		ran := runUnderContract(t, flow, trainingprogram.ObjectiveFlowMatching, noCancel)
		if ran.err != nil || ran.steps != 3 {
			t.Fatalf("flow-matching run = (%d steps, %v), want all three", ran.steps, ran.err)
		}
		checkpoint, err := trainingprogram.LoadCheckpoint(ran.target)
		if err != nil {
			t.Fatal(err)
		}
		program, err := trainingprogram.CompileObjectiveProgram(trainingprogram.ObjectiveFlowMatching, nil, newHostMatrix(t, trainingprogram.ObjectiveFlowMatching, checkpoint.Optimizer.Config).plan)
		if err != nil {
			t.Fatal(err)
		}
		stream, err := trainingworkflow.TextStream([]byte(routeFixtureDataset), uint64(ran.steps))
		if err != nil {
			t.Fatal(err)
		}
		session := ran.session
		_, recorded, err := session.store.Artifact(t.Context(), checkpoint.ID())
		if err != nil || !recorded || session.authority.Objective != trainingprogram.ObjectiveFlowMatching || checkpoint.Program != program.ID() ||
			checkpoint.Stream != trainingprogram.DatasetState(stream) || checkpoint.Model != session.model || *session.published != checkpoint.ID() {
			t.Fatalf("checkpoint of %s under program %s at %+v recorded %t (%v), want the flow-matching program %s at %+v of %s",
				checkpoint.Model, checkpoint.Program, checkpoint.Stream, recorded, err, program.ID(), stream, session.model)
		}
	})
	t.Run("undeclared objective", func(t *testing.T) {
		ran := runUnderContract(t, trainingworkflow.TokenContract, trainingprogram.ObjectiveLatentL2, noCancel)
		if ran.err == nil || !strings.Contains(ran.err.Error(), "steps latent-l2") || ran.steps != 0 {
			t.Fatalf("latent-l2 run under a token recipe = (%d steps, %v), want it refused before any step", ran.steps, ran.err)
		}
		if _, err := os.Stat(ran.target); !os.IsNotExist(err) {
			t.Fatalf("the refused run published at %s (%v)", ran.target, err)
		}
		if observation, err := ran.session.observer.Finish(t.Context(), ran.session.model, ran.session.recipe, nil, 0); err != nil || observation.Valid() {
			t.Fatalf("a second finish = (%s, %v), want the refused session finished once", observation, err)
		}
	})
	t.Run("cancelled", func(t *testing.T) {
		for name, steps := range map[string]int{"before training": 0, "after one update": 1} {
			ran := runUnderContract(t, flow, trainingprogram.ObjectiveFlowMatching, steps)
			if !errors.Is(ran.err, context.Canceled) || ran.steps != steps {
				t.Fatalf("a run cancelled %s = (%d steps, %v), want %d and its cancellation", name, ran.steps, ran.err, steps)
			}
			if _, err := os.Stat(ran.target); !os.IsNotExist(err) {
				t.Fatalf("a run cancelled %s published at %s (%v)", name, ran.target, err)
			}
		}
	})
	t.Run("setup failure", func(t *testing.T) {
		_, store, modelPath, datasetPath := routeFixture(t)
		session, err := openSession(t.Context(), store, modelPath, datasetPath)
		if err != nil {
			t.Fatal(err)
		}
		failure := errors.New("the model failed to load")
		setupErr := failure
		session.settle(t.Context(), &setupErr)
		if setupErr != failure {
			t.Fatalf("settling a failed setup = %v, want its own failure kept", setupErr)
		}
		if observation, err := session.observer.Finish(t.Context(), session.model, session.recipe, nil, 0); err != nil || observation.Valid() {
			t.Fatalf("a finish after settling = (%s, %v), want the session finished once", observation, err)
		}
	})
}

// noCancel lets a run under contract take every update.
const noCancel = -1

// authorityRun is what a run under contract left: its outcome, the steps
// it took, its session and where it publishes.
type authorityRun struct {
	err     error
	steps   int
	session routeSession
	target  string
}

// runUnderContract trains a host matrix stepping kind for three updates
// under a session opened with contract over a route fixture whose dataset
// file is edited once the session read it. Its context ends once the run
// has taken cancelAt steps, before training for zero, never for noCancel.
func runUnderContract(t *testing.T, contract trainingworkflow.RecipeContract, kind trainingprogram.ObjectiveKind, cancelAt int) authorityRun {
	t.Helper()
	root, store, modelPath, datasetPath := routeFixture(t)
	session, err := openSessionAs(t.Context(), store, modelPath, datasetPath, contract)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(datasetPath, []byte("edited after the session read it"), 0o600); err != nil {
		t.Fatal(err)
	}
	plan, err := probeSession(kind, 3, testMatrix.End, 0)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	if cancelAt == 0 {
		cancel()
	}
	matrix := newHostMatrix(t, kind, plan.Optimizer())
	ran := authorityRun{session: session, target: filepath.Join(root, "checkpoint")}
	ran.err = boundedRun{name: "authority", plan: plan, session: session, loss: matrix.loss,
		step: func(int) (optimizer.ObservedStepResult, error) {
			if ran.steps++; ran.steps == cancelAt {
				cancel()
			}
			return matrix.step(), nil
		},
		publish: func(ctx context.Context) error {
			return session.publishPlanned(ctx, ran.target, matrix, plan.Updates())
		},
	}.train(ctx)
	return ran
}

// testMatrix is the one 2x2 group a hostMatrix trains.
var testMatrix = hostoptimizer.GroupSpec{Name: "adapter.weight", End: 4, Rows: 2, Cols: 2}

// hostMatrix is the smallest checkpointed and planned trainer: a host Muon
// matrix stepped toward all ones under a compiled objective program.
type hostMatrix struct {
	plan               hostoptimizer.Plan
	program            trainingprogram.TrainingProgram
	muon               *hostoptimizer.Optimizer
	weights, gradients []float32
}

func newHostMatrix(t *testing.T, objective trainingprogram.ObjectiveKind, config hostoptimizer.Config) *hostMatrix {
	t.Helper()
	plan, err := hostoptimizer.CompilePlan(testMatrix.End, []hostoptimizer.GroupSpec{testMatrix})
	if err != nil {
		t.Fatal(err)
	}
	matrix := &hostMatrix{plan: plan, weights: make([]float32, testMatrix.End), gradients: make([]float32, testMatrix.End)}
	if matrix.program, err = trainingprogram.CompileObjectiveProgram(objective, nil, plan); err != nil {
		t.Fatal(err)
	}
	if matrix.muon, err = hostoptimizer.New(matrix.weights, matrix.gradients, plan, config); err != nil {
		t.Fatal(err)
	}
	return matrix
}

func (m *hostMatrix) loss() (float64, error) {
	var total float64
	for _, weight := range m.weights {
		total += (float64(weight) - 1) * (float64(weight) - 1)
	}
	return total / float64(len(m.weights)), nil
}

func (m *hostMatrix) step() optimizer.ObservedStepResult {
	before, _ := m.loss()
	for index, weight := range m.weights {
		m.gradients[index] = 2 * (weight - 1) / float32(len(m.weights))
	}
	result := m.muon.Step()
	return optimizer.ObservedStepResult{Loss: before, Step: result.Step, LearningRate: result.LearningRate, GradientL2: result.GradientL2}
}

func (m *hostMatrix) Program() trainingprogram.TrainingProgram { return m.program }

func (m *hostMatrix) OptimizerPlan() optimizer.Plan { return m.plan }

func (m *hostMatrix) Snapshot() optimizer.State { return m.muon.Snapshot() }

func (m *hostMatrix) SaveWeights(path string) error {
	return optimizer.SavePlanWeights(path, m.plan, m.weights)
}

// publishUnderObjective steps a host matrix under an objective kind's
// program and publishes it through the recipe a held-out run's session
// trains under, as the forecast and table routes do, then holds the
// checkpoint to loading with that program and being recorded in the store.
func publishUnderObjective(t *testing.T, ctx context.Context, store *overgodb.Store, recipeID artifact.ID, kind trainingprogram.ObjectiveKind, stream artifact.ID) (trainingprogram.Checkpoint, error) {
	t.Helper()
	config, err := trainingprogram.BuiltinOptimizerPolicy().Config(testMatrix.End)
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	matrix := newHostMatrix(t, kind, config)
	result := matrix.step()
	target := filepath.Join(t.TempDir(), "checkpoint")
	published, err := publishTrained(ctx, store, recipeID, target, matrix, trainingdata.StreamState{Identity: stream, Position: uint64(result.Step)}, false)
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	loaded, err := trainingprogram.LoadCheckpoint(target)
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	if _, recorded, err := store.Artifact(ctx, loaded.ID()); err != nil || !recorded || loaded.Program != matrix.program.ID() || loaded.ID() != published.ID() {
		t.Fatalf("checkpoint %s of program %s recorded %t (%v), want %s's checkpoint %s recorded", loaded.ID(), loaded.Program, recorded, err, matrix.program.ID(), published.ID())
	}
	return published, nil
}
