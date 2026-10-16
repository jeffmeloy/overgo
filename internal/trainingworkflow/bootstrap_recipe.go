package trainingworkflow

import (
	"context"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/trainingprogram"
	"overgo/internal/workflowrecipe"
)

// BootstrapTokenRecipe publishes a token-training recipe for one model as a
// candidate — the recipe authority a session-supervised training run
// resolves before touching weights. Every dependency is a real committed
// artifact: the dataset grounds by content, and the split and policy
// profiles identify by declared names. The recipe definition identity
// returns for the caller's run, whose own success verifies and activates
// it.
func BootstrapTokenRecipe(ctx context.Context, store *overgodb.Store, modelPath, datasetPath string) (artifact.ID, error) {
	datasetBytes, err := os.ReadFile(datasetPath)
	if err != nil {
		return artifact.ID{}, err
	}
	return BootstrapRecipe(ctx, store, modelPath, datasetBytes, TokenContract)
}

// RecipeContract is the objective a bootstrapped training recipe declares:
// the loss kind the run's trainer steps, the modalities it reads and
// produces, and the metric that would judge it.
type RecipeContract struct {
	Kind      trainingprogram.ObjectiveKind
	Signature recipecontract.ModalitySignature
	Metric    trainingprogram.EvaluationMetric
}

// TokenContract is next-token prediction over text, judged by token
// accuracy.
var TokenContract = RecipeContract{
	Kind: trainingprogram.ObjectiveTokenPrediction, Signature: textSignature, Metric: trainingprogram.MetricTokenAccuracy,
}

// BootstrapRecipe publishes, as BootstrapTokenRecipe does, the training
// recipe for the model weights at modelPath whose bootstrap objective
// declares contract over the dataset bytes, the bytes the run reads. A
// contract no objective admits, such as a loss kind over an output modality
// its contract excludes, is refused.
func BootstrapRecipe(ctx context.Context, store *overgodb.Store, modelPath string, datasetBytes []byte, contract RecipeContract) (artifact.ID, error) {
	modelID, err := identifyAndRecordWeights(ctx, store, modelPath)
	if err != nil {
		return artifact.ID{}, err
	}
	// Dataset, split and processor identify exactly as the training run
	// derives them, so the run authority equals the stored objective.
	datasetID, splitID, processorID, err := TextAuthority(datasetBytes)
	if err != nil {
		return artifact.ID{}, err
	}
	return bootstrapObjective(ctx, store, modelID, datasetID, splitID, processorID, contract)
}

// TextAuthority names a dataset a run reads whole as UTF-8 text: the
// dataset by its bytes, its one split of every record, and the processor
// that reads it.
func TextAuthority(raw []byte) (dataset, split, processor artifact.ID, err error) {
	authority, _, err := datasetAuthority(raw, textProcessor)
	return authority.Dataset, authority.Split, authority.Processor, err
}

// BootstrapScratchRecipe publishes the training recipe for a scratch
// construction, as BootstrapRecipe does for a weights file: the recipe's
// model is the model the construction initializes, and its bootstrap
// objective declares contract over the construction's dataset and split
// through the processor that reads them.
func BootstrapScratchRecipe(ctx context.Context, store *overgodb.Store, construction trainingprogram.ScratchConstruction, processor artifact.ID, contract RecipeContract) (artifact.ID, error) {
	return bootstrapObjective(ctx, store, construction.InitializedModel(), construction.Dataset(), construction.Split(), processor, contract)
}

// bootstrapObjective publishes the training recipe for a model under the
// bootstrap objective contract declares over a dataset, its split and the
// processor that reads it.
func bootstrapObjective(ctx context.Context, store *overgodb.Store, modelID, datasetID, splitID, processorID artifact.ID, contract RecipeContract, evaluators ...artifact.ID) (artifact.ID, error) {
	profile := func(name string) (artifact.ID, error) {
		return artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/training-bootstrap/"+name))
	}
	names := []string{"loss", "evaluation", "precision", "placement", "memory", "checkpoint", "promotion"}
	profiles := make(map[string]artifact.ID, len(names))
	for _, name := range names {
		id, err := profile(name)
		if err != nil {
			return artifact.ID{}, err
		}
		profiles[name] = id
	}
	evidenceID, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("overgo/training-bootstrap/objective-evidence"))
	if err != nil {
		return artifact.ID{}, err
	}
	// The token contract's objective keeps the name it always had, so the
	// recipes it already bootstrapped keep their identities.
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: strings.ReplaceAll(string(contract.Kind), "-", " ") + " bootstrap", Kind: contract.Kind,
		Signature: contract.Signature.Clone(),
		Dataset:   datasetID, Split: splitID, Processors: []artifact.ID{processorID},
		Loss: profiles["loss"], Evaluation: profiles["evaluation"],
		Metric: contract.Metric, Evidence: []artifact.ID{evidenceID},
		Authority: trainingprogram.ObjectiveApproved,
	})
	if err != nil {
		return artifact.ID{}, fmt.Errorf("training workflow: no bootstrap objective declares %s %v judged by %s: %w", contract.Kind, contract.Signature, contract.Metric, err)
	}
	return bootstrapRecipeForObjective(ctx, store, modelID, objective, profiles, evidenceID, evaluators...)
}

// bootstrapRewardRecipe publishes, as BootstrapRecipe does, the training
// recipe for a reward session over the model weights at modelPath and a suite
// of instruction prompts, the bytes the session reads: its objective is the
// reward objective over those prompts, read by the processor a reward session
// reads them with, and its recipe scores a policy's answers by the prompts'
// scorer, which it names as its evaluator.
func bootstrapRewardRecipe(ctx context.Context, store *overgodb.Store, modelPath string, prompts []byte) (artifact.ID, error) {
	suite, err := evaluation.ParseInstructionPrompts(prompts)
	if err != nil {
		return artifact.ID{}, err
	}
	evaluator, err := suite.Evaluator()
	if err != nil {
		return artifact.ID{}, err
	}
	modelID, err := identifyAndRecordWeights(ctx, store, modelPath)
	if err != nil {
		return artifact.ID{}, err
	}
	authority, _, err := datasetAuthority(prompts, rewardProcessor)
	if err != nil {
		return artifact.ID{}, err
	}
	return bootstrapObjective(ctx, store, modelID, authority.Dataset, authority.Split, authority.Processor,
		RecipeContract{Kind: trainingprogram.ObjectiveGRPO, Signature: textSignature, Metric: trainingprogram.MetricTokenAccuracy}, evaluator)
}

// BootstrapObjectiveRecipe publishes the training recipe for a model
// under a REGISTERED objective: the objective alias resolves from the
// store, the objective's own corpus and contracts become the recipe's
// authority, and the recipe carries the same session-supervised chain
// as the bootstrap -- one objective schema, whatever the loss.
func BootstrapObjectiveRecipe(ctx context.Context, store *overgodb.Store, modelPath, objectiveAlias string) (artifact.ID, error) {
	modelID, err := identifyAndRecordWeights(ctx, store, modelPath)
	if err != nil {
		return artifact.ID{}, err
	}
	objective, err := LoadObjectiveAlias(ctx, store, objectiveAlias)
	if err != nil {
		return artifact.ID{}, err
	}
	return BootstrapObjectiveForModel(ctx, store, modelID, objective)
}

// BootstrapObjectiveForModel publishes a training candidate under the caller's exact model and registered objective.
//   - Callers retain model inventory ownership so non-Hugging-Face layouts keep their serving identity.
func BootstrapObjectiveForModel(ctx context.Context, store *overgodb.Store, modelID artifact.ID, objective trainingprogram.ObjectiveDocument) (artifact.ID, error) {
	if modelID.Kind() != artifact.KindModel {
		return artifact.ID{}, errors.New("training workflow: objective bootstrap requires a model artifact")
	}
	profile := func(name string) (artifact.ID, error) {
		return artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/training-bootstrap/"+name))
	}
	profiles := map[string]artifact.ID{}
	for _, name := range []string{"precision", "placement", "memory", "checkpoint", "promotion"} {
		id, err := profile(name)
		if err != nil {
			return artifact.ID{}, err
		}
		profiles[name] = id
	}
	if len(objective.Evidence) == 0 {
		return artifact.ID{}, errors.New("training workflow: registered objective carries no evidence")
	}
	return bootstrapRecipeForObjective(ctx, store, modelID, objective, profiles, objective.Evidence[0])
}

// identifyAndRecordWeights binds the consumed weights through their inventory
// and commits their locations for the identified model. Bootstrap is the
// one moment the identity and the path are both in hand, so every
// bootstrapped activation stays discoverable by the training lanes; a
// location the store already holds for this path is left as the recorded
// fact. Weights that are several files, given by one of them, leave no
// training activation standing under that one file's own identity
// (retireLoneFile).
func identifyAndRecordWeights(ctx context.Context, store *overgodb.Store, modelPath string) (artifact.ID, error) {
	modelID, batch, err := identifyWeights(modelPath)
	if err != nil {
		return artifact.ID{}, err
	}
	location := batch.Locations[len(batch.Locations)-1].Location
	batch.Key = "training/bootstrap/location/" + modelID.String() + "/" + location.Value
	if len(batch.Manifests) != 0 {
		// Registration already recorded these manifests and locations when it
		// holds the model: nothing to add, and the run goes on.
		if _, err := artifact.Publish(ctx, store, batch); err != nil {
			return artifact.ID{}, err
		}
		return modelID, retireLoneFile(ctx, store, modelPath, modelID)
	}
	existing, err := store.Locations(ctx, modelID)
	if err != nil {
		return artifact.ID{}, err
	}
	for _, have := range existing {
		if have.Kind == artifact.LocationFile && have.Value == location.Value {
			return modelID, nil
		}
	}
	if _, found, err := store.Artifact(ctx, modelID); err != nil {
		return artifact.ID{}, err
	} else if found {
		batch.Artifacts = nil
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		return artifact.ID{}, err
	}
	return modelID, nil
}

// loneFileReason is the reason recorded with the retirement of an activation
// that a file of a larger model held as a model of its own.
const loneFileReason = "a run over these weights binds the model they are a part of, so nothing runs under this activation"

// retireLoneFile retires the training activation that the file at modelPath
// holds as a model of its own, when a run over that file binds bound, the
// model the file is a part of. Such an activation was made before the model
// was named by every file it runs through. No run binds its model any more,
// so its proof can never run. It is retired as orphaned, and the evidence is
// what a run binds from each file the store locates that model at
// (modelrecipe.RetireOrphanedActivation).
//
// Nothing is retired for a path that is no file, for a file that holds no
// training activation of its own, or while one of that model's files is still
// bound as that model. Nothing is retired from a tree whose code revision
// cannot be named either, where no claim is recorded: the next run from a
// clean one retires it.
func retireLoneFile(ctx context.Context, store *overgodb.Store, modelPath string, bound artifact.ID) error {
	info, err := os.Stat(modelPath)
	if err != nil || info.IsDir() {
		return err
	}
	file, err := os.Open(modelPath)
	if err != nil {
		return err
	}
	lone, _, identifyErr := artifact.Identify(artifact.KindModel, file)
	if err := errors.Join(identifyErr, file.Close()); err != nil || lone == bound {
		return err
	}
	if _, active, err := artifact.ResolveAlias(ctx, store, modelrecipe.ActiveAliasPrefix(recipe.TaskTraining)+lone.String()); err != nil || !active {
		return err
	}
	revision, err := executableCodeCommit(".")
	if err != nil {
		fmt.Printf("the training activation of %s, which a run over %s no longer binds, is not retired: %v\n", lone, modelPath, err)
		return nil
	}
	switch err := modelrecipe.RetireOrphanedActivation(ctx, store, lone, recipe.TaskTraining, revision, loneFileReason, WeightsIdentity); {
	case errors.Is(err, modelrecipe.ErrTrustedActivation):
		return nil
	case err != nil:
		return err
	}
	fmt.Printf("retired the training activation of %s as orphaned: a run over %s binds %s\n", lone, modelPath, bound)
	return nil
}

// bootstrapRecipeForObjective assembles, grounds, and publishes as a
// candidate the four-node training recipe for one model and one approved
// objective. No verification is fabricated here: the recipe activates on the
// verification a run under it records (Observer.Verify).
func bootstrapRecipeForObjective(
	ctx context.Context,
	store *overgodb.Store,
	modelID artifact.ID,
	objective trainingprogram.ObjectiveDocument,
	profiles map[string]artifact.ID,
	evidenceID artifact.ID,
	evaluators ...artifact.ID,
) (artifact.ID, error) {
	content, err := objective.Content()
	if err != nil {
		return artifact.ID{}, err
	}
	optimizerPolicy := trainingprogram.BuiltinOptimizerPolicy()
	optimizerContent, err := optimizerPolicy.Content()
	if err != nil {
		return artifact.ID{}, err
	}
	dependencies := []recipe.Dependency{
		{Role: recipe.DependencyModel, Artifact: modelID},
		{Role: recipe.DependencyObjective, Artifact: objective.ID},
		{Role: recipe.DependencyPrecision, Artifact: profiles["precision"]},
		{Role: recipe.DependencyPlacement, Artifact: profiles["placement"]},
		{Role: recipe.DependencyMemory, Artifact: profiles["memory"]},
		{Role: recipe.DependencyOptimizer, Artifact: optimizerPolicy.ID},
		{Role: recipe.DependencyCheckpointPolicy, Artifact: profiles["checkpoint"]},
		// Evaluation binds the objective's own evaluation profile: for the
		// token bootstrap this is the same derived identity as before, and a
		// registered objective brings its own.
		{Role: recipe.DependencyEvaluation, Artifact: objective.Evaluation},
		{Role: recipe.DependencyPromotion, Artifact: profiles["promotion"]},
	}
	backward := recipe.Node{ID: "backward", Module: workflowrecipe.ModuleBackward, Placement: recipe.PlacementHost}
	optimize := recipe.Node{ID: "optimize", Module: workflowrecipe.ModuleOptimize, Placement: recipe.PlacementHost}
	nodes := []recipe.Node{
		{ID: "batch", Module: workflowrecipe.ModuleBatchDataset, Placement: recipe.PlacementHost},
		{ID: "forward", Module: workflowrecipe.ModuleTrainingForward, Placement: recipe.PlacementHost},
		backward, optimize,
	}
	edge := func(fromNode recipe.NodeID, fromPort recipe.PortName, toNode recipe.NodeID, toPort recipe.PortName) recipe.Edge {
		return recipe.Edge{From: recipe.Endpoint{Node: fromNode, Port: fromPort}, To: recipe.Endpoint{Node: toNode, Port: toPort}}
	}
	edges := []recipe.Edge{
		edge("batch", "batch", "forward", "batch"),
		edge("forward", "loss", "backward", "loss"),
		edge("backward", "gradients", "optimize", "gradients"),
	}
	// A reward objective batches rollouts, scores them under the policy and
	// weighs them by its evaluators' rewards, where a token objective takes
	// one forward to its loss.
	if objective.Kind == trainingprogram.ObjectiveGRPO {
		nodes = []recipe.Node{
			{ID: "batch", Module: workflowrecipe.ModuleBatchRollout, Placement: recipe.PlacementHost},
			{ID: "policy", Module: workflowrecipe.ModuleScorePolicy, Placement: recipe.PlacementHost},
			{ID: "objective", Module: workflowrecipe.ModuleGRPOObjective, Placement: recipe.PlacementHost},
			backward, optimize,
		}
		edges = []recipe.Edge{
			edge("batch", "batch", "policy", "batch"), edge("policy", "scores", "objective", "policy"),
			edge("objective", "loss", "backward", "loss"), edge("backward", "gradients", "optimize", "gradients"),
		}
		for slot, evaluator := range evaluators {
			dependencies = append(dependencies, recipe.Dependency{Role: recipe.DependencyEvaluator, Slot: uint32(slot), Artifact: evaluator})
		}
	}
	definition, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskTraining, dependencies, nodes, edges, nil,
		[]recipe.Output{{Name: "checkpoint", Data: recipe.DataCheckpoint, Source: recipe.Endpoint{Node: "optimize", Port: "checkpoint"}}},
	)
	if err != nil {
		return artifact.ID{}, err
	}
	// Every bootstrap grounds the objective and the identities it and the
	// recipe name, even when the objective or the recipe is already stored:
	// a store rebuilt from one that never declared an objective's dataset
	// and split holds both without them, and a run under the recipe would
	// train to the end and fail only when its checkpoint resolves the
	// objective. Identities already grounded keep their stored facts; only
	// absent ones are declared, so a rerun over a grounded authority writes
	// nothing.
	authority := artifact.Batch{Key: "training/bootstrap/authority/" + objective.ID.String()}
	if _, ok, err := store.Artifact(ctx, objective.ID); err != nil {
		return artifact.ID{}, err
	} else if !ok {
		authority.Contents = []artifact.Content{content}
	}
	candidates := []artifact.ID{modelID, evidenceID, objective.Dataset, objective.Split, objective.Loss, objective.Evaluation}
	candidates = append(slices.Concat(candidates, objective.Processors), evaluators...)
	for _, id := range profiles {
		candidates = append(candidates, id)
	}
	for _, id := range candidates {
		if _, ok, err := store.Artifact(ctx, id); err != nil {
			return artifact.ID{}, err
		} else if !ok {
			authority.Artifacts = append(authority.Artifacts, artifact.Descriptor{ID: id})
		}
	}
	if len(authority.Artifacts) > 0 || len(authority.Contents) > 0 {
		if _, err := store.Commit(ctx, authority); err != nil {
			return artifact.ID{}, err
		}
	}
	// Reruns with the same model and dataset re-derive the SAME authority: an
	// already-published definition returns as-is, and each write below is guarded
	// so the append-only store never sees a same-key batch whose content
	// drifted with grounding order (ErrBatchKeyConflict on rerun otherwise).
	state, published, err := modelrecipe.Status(ctx, store, definition.ID)
	if err != nil {
		return artifact.ID{}, err
	}
	if published {
		fmt.Printf("training recipe already published: %s (model %s, %s)\n", definition.ID, modelID, state)
		return definition.ID, nil
	}
	if ok, err := store.HasContent(ctx, optimizerPolicy.ID); err != nil {
		return artifact.ID{}, err
	} else if !ok {
		if _, err := store.Commit(ctx, artifact.Batch{
			Key: "training/bootstrap/optimizer/" + optimizerPolicy.ID.String(), Contents: []artifact.Content{optimizerContent},
		}); err != nil {
			return artifact.ID{}, err
		}
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "training/bootstrap/candidate/"+definition.ID.String(), definition); err != nil {
		return artifact.ID{}, err
	}
	fmt.Printf("training recipe published: %s (model %s); the first run under it that succeeds verifies and activates it\n", definition.ID, modelID)
	return definition.ID, nil
}
