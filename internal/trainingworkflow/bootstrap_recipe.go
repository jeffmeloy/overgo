package trainingworkflow

import (
	"context"
	"errors"
	"fmt"
	"os"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/trainingprogram"
	"overgo/internal/workflowrecipe"
)

// BootstrapTokenRecipe publishes, verifies, and activates a token-training
// recipe for one model at the experimental evidence tier — the recipe
// authority a session-supervised training run resolves before touching
// weights. Every dependency is a real committed artifact: the dataset
// grounds by content, the split and policy profiles identify by declared
// names, and the activation reason records that this is the bootstrap.
// The activated recipe definition identity returns for the caller's run.
func BootstrapTokenRecipe(ctx context.Context, store *overgodb.Store, modelPath, datasetPath string) (artifact.ID, error) {
	modelFile, err := os.Open(modelPath)
	if err != nil {
		return artifact.ID{}, err
	}
	modelID, _, err := artifact.Identify(artifact.KindModel, modelFile)
	modelFile.Close()
	if err != nil {
		return artifact.ID{}, err
	}
	datasetBytes, err := os.ReadFile(datasetPath)
	if err != nil {
		return artifact.ID{}, err
	}
	datasetID, err := artifact.IdentifyBytes(artifact.KindDataset, datasetBytes)
	if err != nil {
		return artifact.ID{}, err
	}
	// Split and processor identify exactly as the training run derives them
	// (datasetAuthority in trainingworkflow), so the run authority equals the
	// stored objective.
	splitID, err := artifact.IdentifyBytes(artifact.KindDatasetShard, append([]byte("all\x00"), datasetBytes...))
	if err != nil {
		return artifact.ID{}, err
	}
	processorID, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/training/text-utf8/v1"))
	if err != nil {
		return artifact.ID{}, err
	}
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
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: "token prediction bootstrap", Kind: trainingprogram.ObjectiveTokenPrediction,
		Signature: recipecontract.ModalitySignature{
			Inputs:  []recipecontract.Modality{recipecontract.ModalityText},
			Outputs: []recipecontract.Modality{recipecontract.ModalityText},
		},
		Dataset: datasetID, Split: splitID, Processors: []artifact.ID{processorID},
		Loss: profiles["loss"], Evaluation: profiles["evaluation"],
		Metric: trainingprogram.MetricTokenAccuracy, Evidence: []artifact.ID{evidenceID},
		Authority: trainingprogram.ObjectiveApproved,
	})
	if err != nil {
		return artifact.ID{}, err
	}
	return bootstrapRecipeForObjective(ctx, store, modelID, objective, profiles, evidenceID)
}

// BootstrapObjectiveRecipe activates the training recipe for a model
// under a REGISTERED objective: the objective alias resolves from the
// store, the objective's own corpus and contracts become the recipe's
// authority, and the recipe carries the same session-supervised chain
// as the bootstrap -- one objective schema, whatever the loss.
func BootstrapObjectiveRecipe(ctx context.Context, store *overgodb.Store, modelPath, objectiveAlias string) (artifact.ID, error) {
	modelFile, err := os.Open(modelPath)
	if err != nil {
		return artifact.ID{}, err
	}
	modelID, _, err := artifact.Identify(artifact.KindModel, modelFile)
	modelFile.Close()
	if err != nil {
		return artifact.ID{}, err
	}
	target, found, err := store.ResolveAlias(ctx, objectiveAlias)
	if err != nil {
		return artifact.ID{}, err
	}
	if !found {
		return artifact.ID{}, fmt.Errorf("training workflow: objective alias %q is not registered", objectiveAlias)
	}
	objective, err := trainingprogram.LoadObjective(ctx, store, target)
	if err != nil {
		return artifact.ID{}, err
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

// bootstrapRecipeForObjective assembles, grounds, and activates the
// four-node training recipe for one model and one approved objective.
func bootstrapRecipeForObjective(
	ctx context.Context,
	store *overgodb.Store,
	modelID artifact.ID,
	objective trainingprogram.ObjectiveDocument,
	profiles map[string]artifact.ID,
	evidenceID artifact.ID,
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
	nodes := []recipe.Node{
		{ID: "batch", Module: workflowrecipe.ModuleBatchDataset, Placement: recipe.PlacementHost},
		{ID: "forward", Module: workflowrecipe.ModuleTrainingForward, Placement: recipe.PlacementHost},
		{ID: "backward", Module: workflowrecipe.ModuleBackward, Placement: recipe.PlacementHost},
		{ID: "optimize", Module: workflowrecipe.ModuleOptimize, Placement: recipe.PlacementHost},
	}
	edge := func(fromNode recipe.NodeID, fromPort recipe.PortName, toNode recipe.NodeID, toPort recipe.PortName) recipe.Edge {
		return recipe.Edge{From: recipe.Endpoint{Node: fromNode, Port: fromPort}, To: recipe.Endpoint{Node: toNode, Port: toPort}}
	}
	definition, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskTraining, dependencies, nodes,
		[]recipe.Edge{
			edge("batch", "batch", "forward", "batch"),
			edge("forward", "loss", "backward", "loss"),
			edge("backward", "gradients", "optimize", "gradients"),
		}, nil,
		[]recipe.Output{{Name: "checkpoint", Data: recipe.DataCheckpoint, Source: recipe.Endpoint{Node: "optimize", Port: "checkpoint"}}},
	)
	if err != nil {
		return artifact.ID{}, err
	}
	// Reruns with the same model and dataset re-derive the SAME authority: an
	// already-active definition returns as-is, and each write below is guarded
	// so the append-only store never sees a same-key batch whose content
	// drifted with grounding order (ErrBatchKeyConflict on rerun otherwise).
	state, published, err := modelrecipe.Status(ctx, store, definition.ID)
	if err != nil {
		return artifact.ID{}, err
	}
	if state == recipe.StatusActive {
		fmt.Printf("training recipe already active: %s (model %s)\n", definition.ID, modelID)
		return definition.ID, nil
	}
	if _, ok, err := store.Artifact(ctx, objective.ID); err != nil {
		return artifact.ID{}, err
	} else if !ok {
		// Identities already grounded by prior claims keep their stored facts;
		// only absent identities are declared.
		candidates := []artifact.ID{modelID, evidenceID, objective.Dataset, objective.Split, objective.Loss, objective.Evaluation}
		candidates = append(candidates, objective.Processors...)
		for _, id := range profiles {
			candidates = append(candidates, id)
		}
		var descriptors []artifact.Descriptor
		for _, id := range candidates {
			if _, ok, err := store.Artifact(ctx, id); err != nil {
				return artifact.ID{}, err
			} else if !ok {
				descriptors = append(descriptors, artifact.Descriptor{ID: id})
			}
		}
		if _, err := store.Commit(ctx, artifact.Batch{
			Key: "training/bootstrap/authority/" + objective.ID.String(), Artifacts: descriptors, Contents: []artifact.Content{content},
		}); err != nil {
			return artifact.ID{}, err
		}
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
	if !published {
		if _, _, err := modelrecipe.PublishCandidate(ctx, store, "training/bootstrap/candidate/"+definition.ID.String(), definition); err != nil {
			return artifact.ID{}, err
		}
	}
	verification, err := modelrecipetest.PublishVerification(ctx, store, "training/bootstrap/verification/"+definition.ID.String(), definition.ID)
	if err != nil {
		return artifact.ID{}, err
	}
	if err := modelrecipe.ActivateCapability(ctx, store, definition, verification,
		recipe.EvidenceExperimental, "session-supervised training substrate bootstrap"); err != nil {
		return artifact.ID{}, err
	}
	fmt.Printf("training recipe activated: %s (model %s, experimental evidence tier)\n", definition.ID, modelID)
	return definition.ID, nil
}
