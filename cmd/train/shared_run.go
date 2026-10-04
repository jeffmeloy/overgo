package main

import (
	"cmp"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/hostoptimizer"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
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
// supervised session samples it every step, publishes what descended, and
// finishes the session with the outcome.
type boundedRun struct {
	name    string
	plan    trainingprogram.TrainingSessionPlan
	session routeSession
	step    func(step int) (optimizer.ObservedStepResult, error)
	loss    func() (float64, error)
	publish func(ctx context.Context) error
}

// train runs the loop, publishes what descended, and records the outcome
// as the session observation.
func (run boundedRun) train(ctx context.Context) error {
	return run.session.finish(ctx, run.plan, run.trained(ctx))
}

// trained runs the loop until ctx ends and publishes what descended. A plan
// whose objective the session's recipe does not declare is refused before
// any step: the checkpoint would name a program the run did not execute. A
// run whose context ended publishes nothing.
func (run boundedRun) trained(ctx context.Context) error {
	if trained, declared := run.plan.Objective(), run.session.authority.Objective; trained != declared {
		return fmt.Errorf("%s: the trainer steps %s, but the session's recipe declares %s; its checkpoint would name a program the run did not execute", run.name, trained, declared)
	}
	started := time.Now()
	err := run.descend(ctx)
	run.session.observer.Phase(runrecord.PhaseForwardBackward, time.Since(started))
	if err == nil {
		err = ctx.Err()
	}
	if err != nil || run.publish == nil {
		return err
	}
	return run.publish(ctx)
}

// descend takes the plan's updates, stopping between updates once ctx ends.
func (run boundedRun) descend(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	before, err := run.loss()
	if err != nil {
		return err
	}
	started := time.Now()
	for step := range run.plan.Updates() {
		if err := ctx.Err(); err != nil {
			return err
		}
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
// bootstrapped for its model, that recipe's resolved authority, the
// model's weights, the dataset bytes the recipe grounds as the session read
// them, the scratch construction a model built from nothing starts from,
// the observer recording the run, and the checkpoint it published. The
// zero value supervises nothing.
type routeSession struct {
	store         *overgodb.Store
	model, recipe artifact.ID
	authority     trainingworkflow.SessionAuthority
	weights       string
	// data is the dataset as the session read it, once: the bytes its
	// recipe grounds, the run trains on, and its checkpoint records a
	// position in.
	data     []byte
	scratch  *trainingprogram.ScratchConstruction
	observer *trainingworkflow.Observer
	// published is the checkpoint the session published, which its finish
	// verifies the recipe on; every copy of the session shares it.
	published *artifact.ID
}

// generationContract is the contract of a trainer that steps kind from
// input to output, which metric would judge.
func generationContract(kind trainingprogram.ObjectiveKind, input, output recipecontract.Modality, metric trainingprogram.EvaluationMetric) trainingworkflow.RecipeContract {
	return trainingworkflow.RecipeContract{
		Kind: kind, Metric: metric,
		Signature: recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{input}, Outputs: []recipecontract.Modality{output}},
	}
}

// openSession opens the session of a trainer that steps token prediction
// over a text dataset, as openSessionAs does.
func openSession(ctx context.Context, store *overgodb.Store, weights, dataset string) (routeSession, error) {
	return openSessionAs(ctx, store, weights, dataset, trainingworkflow.TokenContract)
}

// openSessionAs publishes the training recipe whose bootstrap objective
// declares contract over the dataset for the model weights, and admits the
// session before any weight touches memory. A route opens under the
// contract its trainer steps.
func openSessionAs(ctx context.Context, store *overgodb.Store, weights, dataset string, contract trainingworkflow.RecipeContract) (routeSession, error) {
	session, err := recipeSession(ctx, store, weights, dataset, contract)
	if err != nil {
		return routeSession{}, err
	}
	return session.admitted(ctx)
}

// recipeSession binds the training recipe bootstrapped under contract over
// the dataset for the model weights; it supervises nothing until admitted.
func recipeSession(ctx context.Context, store *overgodb.Store, weights, dataset string, contract trainingworkflow.RecipeContract) (routeSession, error) {
	data, err := os.ReadFile(dataset)
	if err != nil {
		return routeSession{}, err
	}
	recipeID, err := trainingworkflow.BootstrapRecipe(ctx, store, weights, data, contract)
	if err != nil {
		return routeSession{}, err
	}
	session, err := bindSession(ctx, store, recipeID)
	if err != nil {
		return routeSession{}, err
	}
	session.weights, session.data = weights, data
	return session, nil
}

// scratchSession publishes the training recipe bootstrapped under contract
// over a scratch construction's dataset for the model it initializes, and
// admits the session; what the session publishes records the construction
// as its initial state. processor reads the construction's dataset.
func scratchSession(ctx context.Context, store *overgodb.Store, construction trainingprogram.ScratchConstruction, processor artifact.ID, contract trainingworkflow.RecipeContract) (routeSession, error) {
	recipeID, err := trainingworkflow.BootstrapScratchRecipe(ctx, store, construction, processor, contract)
	if err != nil {
		return routeSession{}, err
	}
	session, err := bindSession(ctx, store, recipeID)
	if err != nil {
		return routeSession{}, err
	}
	session.scratch = &construction
	return session.admitted(ctx)
}

// admitted opens the session's observer and admits the session before any
// weight touches memory.
func (session routeSession) admitted(ctx context.Context) (routeSession, error) {
	var err error
	if session.observer, err = trainingworkflow.NewObserver(session.store, false); err != nil {
		return routeSession{}, err
	}
	if err := session.observer.Admit(ctx, session.model, session.recipe); err != nil {
		return routeSession{}, err
	}
	fmt.Printf("session admitted: model=%s recipe=%s\n", session.model, session.recipe)
	return session, nil
}

// bindSession resolves the authority of a published training recipe, so
// what a run trained under it publishes; the session supervises nothing,
// since the run that bootstrapped the recipe observes itself.
func bindSession(ctx context.Context, store *overgodb.Store, recipeID artifact.ID) (routeSession, error) {
	definition, err := recipe.RequireDefinition(ctx, store, recipeID)
	if err != nil {
		return routeSession{}, err
	}
	authority, err := trainingworkflow.ResolveSessionAuthority(ctx, store, definition.Model, recipeID)
	if err != nil {
		return routeSession{}, err
	}
	return routeSession{store: store, model: definition.Model, recipe: recipeID, authority: authority, published: new(artifact.ID)}, nil
}

// weightWriter is what a trainer hands the checkpoint it publishes: its
// optimizer state and the weights that state pairs with.
type weightWriter interface {
	Snapshot() optimizer.State
	SaveWeights(path string) error
}

// checkpointed is a trainer that compiled the program it trains under.
type checkpointed interface {
	weightWriter
	Program() trainingprogram.TrainingProgram
}

// planned is a trainer whose checkpoint names the program its session's
// objective compiles over the trainer's optimizer plan.
type planned interface {
	weightWriter
	OptimizerPlan() optimizer.Plan
}

// publishTrained publishes, as a checkpoint at target, what trainer trained
// under the recipe a held-out run's session trained under; stream is where
// the training stopped in the objective's training membership.
func publishTrained(ctx context.Context, store *overgodb.Store, recipeID artifact.ID, target string, trainer checkpointed, stream trainingdata.StreamState) (trainingprogram.Checkpoint, error) {
	session, err := bindSession(ctx, store, recipeID)
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	return session.publishWeights(ctx, target, trainer.Program(), trainer, stream)
}

// publishPlanned publishes, as a checkpoint at target, what trainer trained
// over records of the dataset the session read, under the program the
// session's objective compiles over the trainer's optimizer plan.
func (session routeSession) publishPlanned(ctx context.Context, target string, trainer planned, records int) error {
	stream, err := trainingworkflow.TextStream(session.data, uint64(records))
	if err != nil {
		return err
	}
	program, err := trainingprogram.CompileObjectiveProgram(session.authority.Objective, nil, trainer.OptimizerPlan())
	if err != nil {
		return err
	}
	_, err = session.publishWeights(ctx, target, program, trainer, stream)
	return err
}

// servable is a trainer that names each tensor it trains as its GGUF base
// names it, so its run publishes the trained model in the base's layout.
type servable interface {
	planned
	TrainedTensor(name string) ([]float32, bool)
}

// publishWeights publishes, as a checkpoint at target, what trainer trained
// under program; stream is where the training stopped in the recipe
// objective's dataset. A servable trainer over a GGUF base publishes the
// trained model there, sharing the tensors it never trained with every
// other run over that base the session's store records, and the
// checkpoint's record registers it as the candidate the server opens.
func (session routeSession) publishWeights(ctx context.Context, target string, program trainingprogram.TrainingProgram, trainer weightWriter, stream trainingdata.StreamState) (trainingprogram.Checkpoint, error) {
	model, served := trainer.(servable)
	served = served && strings.EqualFold(filepath.Ext(session.weights), ".gguf")
	return session.publish(ctx, target, program, trainer.Snapshot(), stream, func(stage string) error {
		if served {
			return trainingworkflow.WriteTrainedModel(ctx, session.store, session.weights, stage, model.OptimizerPlan().GroupCount(), model.TrainedTensor)
		}
		return trainer.SaveWeights(filepath.Join(stage, trainingprogram.CheckpointWeights))
	})
}

// finish records the run's outcome as the session observation, a cancelled
// run's included, and, when the run succeeded, verifies the session's
// recipe on it and the checkpoint the session published; the run's own
// error outranks the observation's.
func (session routeSession) finish(ctx context.Context, plan trainingprogram.TrainingSessionPlan, runErr error) error {
	ctx = context.WithoutCancel(ctx)
	streamed := uint64(plan.Updates()) * uint64(plan.MaximumSequence())
	observation, err := session.observer.Finish(ctx, session.model, session.recipe, runErr, streamed)
	if err := cmp.Or(runErr, err); err != nil {
		return err
	}
	if !observation.Valid() {
		return nil
	}
	fmt.Printf("session observation: %s\n", observation)
	evidence := []artifact.ID{observation}
	if session.published != nil && session.published.Valid() {
		evidence = append(evidence, *session.published)
	}
	return session.observer.Verify(ctx, session.recipe, evidence...)
}

// settle finishes the session with *err when no run finished it, so a setup
// that fails after admission still releases what admission acquired and
// records its outcome; a route defers it as soon as its session opens. A
// finish that fails reports only when nothing failed before it.
func (session routeSession) settle(ctx context.Context, err *error) {
	if _, finishErr := session.observer.Finish(context.WithoutCancel(ctx), session.model, session.recipe, *err, 0); *err == nil {
		*err = finishErr
	}
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
	// A model built from nothing starts from its scratch construction; any
	// other starts from the model the session trains.
	initial := trainingprogram.InitialStateSpec{Model: session.model}
	if session.scratch != nil {
		initial = trainingprogram.InitialStateSpec{Scratch: session.scratch}
	}
	plan, err := trainingprogram.CompileTrainingRunPlanFromRepository(ctx, session.store, trainingprogram.RunSpec{
		Recipe: session.recipe, Initial: initial,
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
	lineage := []trainingprogram.LineageParent{
		{Artifact: session.model, Relation: artifact.RelationTrainedFrom},
		{Artifact: objective.Dataset, Relation: artifact.RelationDependsOn},
		{Artifact: objective.Split, Relation: artifact.RelationDependsOn},
		{Artifact: program.ID(), Relation: artifact.RelationProducedBy},
		{Artifact: plan.ID(), Relation: artifact.RelationDependsOn},
	}
	if session.scratch != nil {
		lineage = append(lineage, trainingprogram.LineageParent{Artifact: session.scratch.ID(), Relation: artifact.RelationDependsOn})
	}
	checkpoint, err := trainingprogram.PublishCheckpoint(target, trainingprogram.CheckpointSpec{
		RunPlan: plan.ID(), Program: program.ID(), Model: session.model, Dataset: objective.Dataset, Split: objective.Split,
		Stream: trainingprogram.DatasetState(stream), Optimizer: state, ParameterCount: state.Parameters(),
		RNG: []trainingprogram.RNGState{
			{Name: "augmentation", Algorithm: rng, Counter: uint64(state.Step)}, {Name: "data", Algorithm: rng, Counter: stream.Position},
		},
		Processors: objective.Processors, Projectors: objective.Projectors, Codecs: objective.Codecs,
		Lineage: lineage,
	}, write)
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	fmt.Printf("checkpoint=%s written to %s\n", checkpoint.ID(), target)
	// The record names the program, run plan and scratch construction the
	// checkpoint's lineage cites, and registers a trained GGUF it holds.
	candidate, err := trainingworkflow.RecordCheckpoint(ctx, session.store, target, checkpoint)
	if err != nil {
		return trainingprogram.Checkpoint{}, err
	}
	if candidate.Definition.ID.Valid() {
		fmt.Printf("model=%s candidate=%s (not activated)\n", candidate.Inventory.Manifest.ID, candidate.Definition.ID)
	}
	if session.published != nil {
		*session.published = checkpoint.ID()
	}
	return checkpoint, nil
}
