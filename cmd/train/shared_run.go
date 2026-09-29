package main

import (
	"cmp"
	"context"
	"fmt"
	"path/filepath"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/hostoptimizer"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// boundedRun is the one bounded training loop the cmd/train routes share. It
// takes the plan's updates through step under the shared guards: every
// step's loss is finite and carries gradient, the first step's wall is
// admitted against the plan's projected-wall bound, and the objective loss
// re-evaluates after training below its value before. A route that opened a
// supervised session samples it every step and finishes it with the outcome.
type boundedRun struct {
	name    string
	plan    trainingprogram.TrainingSessionPlan
	session routeSession
	step    func(step int) (optimizer.ObservedStepResult, error)
	loss    func() (float64, error)
}

// train runs the loop and records its outcome as the session observation.
func (run boundedRun) train(ctx context.Context) error {
	started := time.Now()
	err := run.descend()
	run.session.observer.Phase(runrecord.PhaseForwardBackward, time.Since(started))
	return run.session.finish(ctx, run.plan, err)
}

func (run boundedRun) descend() error {
	before, err := run.loss()
	if err != nil {
		return err
	}
	started := time.Now()
	for step := range run.plan.Updates() {
		stepStarted := time.Now()
		result, err := run.step(step)
		if err != nil {
			return err
		}
		run.session.observer.SampleStep()
		wall := time.Since(stepStarted)
		fmt.Printf("step %d/%d: loss=%.6f grad_l2=%.4g lr=%.4g wall=%s\n",
			step+1, run.plan.Updates(), result.Loss, result.GradientL2, result.LearningRate, wall.Round(time.Millisecond))
		if !checked.Finite64(result.Loss) {
			return fmt.Errorf("%s: step %d loss is non-finite", run.name, step+1)
		}
		if !(result.GradientL2 > 0) {
			return fmt.Errorf("%s: step %d carried no gradient", run.name, step+1)
		}
		if err := run.plan.AdmitStepWall(wall); step == 0 && err != nil {
			return fmt.Errorf("%s: %w", run.name, err)
		}
	}
	after, err := run.loss()
	if err != nil {
		return err
	}
	if !checked.Finite64(after) || !(after < before) {
		return fmt.Errorf("%s: loss did not descend: before=%.6f after=%.6f", run.name, before, after)
	}
	fmt.Printf("descent: steps=%d loss %.6f -> %.6f total_wall=%s\n",
		run.plan.Updates(), before, after, time.Since(started).Round(time.Millisecond))
	return nil
}

// routeSession is a route's supervised model-session: the recipe the route
// bootstrapped for its model, that recipe's resolved authority, and the
// observer recording the run. The zero value supervises nothing.
type routeSession struct {
	store         *overgodb.Store
	model, recipe artifact.ID
	authority     trainingworkflow.SessionAuthority
	observer      *trainingworkflow.Observer
}

// openSession activates the token-training recipe bootstrapped over dataset
// for the model weights and admits the session before any weight touches
// memory.
func openSession(ctx context.Context, store *overgodb.Store, weights, dataset string) (routeSession, error) {
	recipeID, err := trainingworkflow.BootstrapTokenRecipe(ctx, store, weights, dataset)
	if err != nil {
		return routeSession{}, err
	}
	definition, err := recipe.RequireDefinition(ctx, store, recipeID)
	if err != nil {
		return routeSession{}, err
	}
	observer, err := trainingworkflow.NewObserver(store, false)
	if err != nil {
		return routeSession{}, err
	}
	if err := observer.Admit(ctx, definition.Model, recipeID); err != nil {
		return routeSession{}, err
	}
	authority, err := trainingworkflow.ResolveSessionAuthority(ctx, store, definition.Model, recipeID)
	if err != nil {
		return routeSession{}, err
	}
	fmt.Printf("session admitted: model=%s recipe=%s\n", definition.Model, recipeID)
	return routeSession{store: store, model: definition.Model, recipe: recipeID, authority: authority, observer: observer}, nil
}

// finish records the run's outcome as the session observation; the run's own
// error outranks the observation's.
func (session routeSession) finish(ctx context.Context, plan trainingprogram.TrainingSessionPlan, runErr error) error {
	streamed := uint64(plan.Updates()) * uint64(plan.MaximumSequence())
	observation, err := session.observer.Finish(ctx, session.model, session.recipe, runErr, streamed)
	if err := cmp.Or(runErr, err); err != nil {
		return err
	}
	if observation.Valid() {
		fmt.Printf("session observation: %s\n", observation)
	}
	return nil
}

// publish publishes what the session trained as a checkpoint at target and
// records it in the store, so the loop can judge it. write stages the
// trained weights, program and state are the trainer's own, stream is where
// the run stopped in the recipe objective's dataset, and the objective
// supplies the dataset authority the run plan binds.
func (session routeSession) publish(ctx context.Context, target string, program trainingprogram.TrainingProgram, state hostoptimizer.State, stream trainingdata.StreamState, write func(stage string) error) (trainingprogram.Checkpoint, error) {
	policies, err := trainingprogram.PoliciesFromRecipe(session.authority.Program.Definition())
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	objective, err := trainingprogram.LoadObjective(ctx, session.store, policies.Objective)
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	plan, err := trainingprogram.CompileTrainingRunPlanFromRepository(ctx, session.store, trainingprogram.RunSpec{
		Recipe: session.recipe, Initial: trainingprogram.InitialStateSpec{Model: session.model},
		Dataset: objective.Dataset, Split: objective.Split, Signature: objective.Signature,
		Processors: objective.Processors, Projectors: objective.Projectors, Codecs: objective.Codecs,
		Policies: policies, Program: program,
	})
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	// The counter RNG the dense workflow's checkpoints record.
	rng, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/counter-rng/v1"))
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	checkpoint, err := trainingprogram.PublishCheckpoint(target, trainingprogram.CheckpointSpec{
		RunPlan: plan.ID(), Program: program.ID(), Model: session.model, Dataset: objective.Dataset, Split: objective.Split,
		Stream: trainingprogram.DatasetState(stream), Optimizer: state, ParameterCount: len(state.Momentum),
		RNG: []trainingprogram.RNGState{
			{Name: "augmentation", Algorithm: rng, Counter: uint64(state.Step)}, {Name: "data", Algorithm: rng, Counter: stream.Position},
		},
		Processors: objective.Processors, Projectors: objective.Projectors, Codecs: objective.Codecs,
		Lineage: []trainingprogram.LineageParent{
			{Artifact: session.model, Relation: artifact.RelationTrainedFrom},
			{Artifact: objective.Dataset, Relation: artifact.RelationDependsOn},
			{Artifact: objective.Split, Relation: artifact.RelationDependsOn},
			{Artifact: program.ID(), Relation: artifact.RelationProducedBy},
			{Artifact: plan.ID(), Relation: artifact.RelationDependsOn},
		},
	}, write)
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	// A deterministic rerun publishes the same checkpoint into another
	// directory, which its record adds; the record's key names the directory.
	directory, err := filepath.Abs(target)
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	batch, err := checkpoint.Batch("training/evidence/checkpoint/"+checkpoint.ID().String()+"/"+directory, directory)
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	// The compiled program and run plan are exact external publications the
	// checkpoint's lineage names; registering them lets the record resolve.
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: program.ID()}, artifact.Descriptor{ID: plan.ID()})
	if err := trainingworkflow.PublishTrainingEvidence(ctx, session.store, trainingworkflow.TrainingEvidencePublication{Batch: batch}); err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	return checkpoint, nil
}
