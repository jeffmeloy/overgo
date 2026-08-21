package main

import (
	"context"
	"fmt"
	"os"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/repodb"
	"overgo/internal/trainingprogram"
	"overgo/internal/workflowrecipe"
)

// bootstrapTrainingRecipe publishes, verifies, and activates a token-training
// recipe for one model at the experimental evidence tier — the recipe
// authority a session-supervised training run resolves before touching
// weights. Every dependency is a real committed artifact: the dataset
// grounds by content, the split and policy profiles identify by declared
// names, and the activation reason records that this is the bootstrap.
func bootstrapTrainingRecipe(store *repodb.Store, modelPath, datasetPath string) error {
	ctx := context.Background()
	modelFile, err := os.Open(modelPath)
	if err != nil {
		return err
	}
	modelID, _, err := artifact.Identify(artifact.KindModel, modelFile)
	modelFile.Close()
	if err != nil {
		return err
	}
	datasetBytes, err := os.ReadFile(datasetPath)
	if err != nil {
		return err
	}
	datasetID, err := artifact.IdentifyBytes(artifact.KindDataset, datasetBytes)
	if err != nil {
		return err
	}
	// Split and processor identify exactly as the training run derives them
	// (datasetAuthority in trainingworkflow), so the run authority equals the
	// stored objective.
	splitID, err := artifact.IdentifyBytes(artifact.KindDatasetShard, append([]byte("all\x00"), datasetBytes...))
	if err != nil {
		return err
	}
	processorID, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/training/text-utf8/v1"))
	if err != nil {
		return err
	}
	profile := func(name string) (artifact.ID, error) {
		return artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/training-bootstrap/"+name))
	}
	names := []string{"loss", "evaluation", "precision", "placement", "memory", "checkpoint", "promotion"}
	profiles := make(map[string]artifact.ID, len(names))
	for _, name := range names {
		id, err := profile(name)
		if err != nil {
			return err
		}
		profiles[name] = id
	}
	evidenceID, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("overgo/training-bootstrap/objective-evidence"))
	if err != nil {
		return err
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
		return err
	}
	content, err := objective.Content()
	if err != nil {
		return err
	}
	optimizerPolicy := trainingprogram.BuiltinOptimizerPolicy()
	optimizerContent, err := optimizerPolicy.Content()
	if err != nil {
		return err
	}
	// Identities already grounded by prior claims keep their stored facts;
	// only absent identities are declared.
	candidates := []artifact.ID{modelID, evidenceID, datasetID, splitID, processorID}
	for _, name := range names {
		candidates = append(candidates, profiles[name])
	}
	var descriptors []artifact.Descriptor
	for _, id := range candidates {
		if _, ok, err := store.Artifact(ctx, id); err != nil {
			return err
		} else if !ok {
			descriptors = append(descriptors, artifact.Descriptor{ID: id})
		}
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "training/bootstrap/authority/" + objective.ID.String(), Artifacts: descriptors,
		Contents: []artifact.Content{content, optimizerContent},
	}); err != nil {
		return err
	}
	dependencies := []recipe.Dependency{
		{Role: recipe.DependencyModel, Artifact: modelID},
		{Role: recipe.DependencyObjective, Artifact: objective.ID},
		{Role: recipe.DependencyPrecision, Artifact: profiles["precision"]},
		{Role: recipe.DependencyPlacement, Artifact: profiles["placement"]},
		{Role: recipe.DependencyMemory, Artifact: profiles["memory"]},
		{Role: recipe.DependencyOptimizer, Artifact: optimizerPolicy.ID},
		{Role: recipe.DependencyCheckpointPolicy, Artifact: profiles["checkpoint"]},
		{Role: recipe.DependencyEvaluation, Artifact: profiles["evaluation"]},
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
		return err
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "training/bootstrap/candidate/"+definition.ID.String(), definition); err != nil {
		return err
	}
	verification, err := modelrecipetest.PublishVerification(ctx, store, "training/bootstrap/verification/"+definition.ID.String(), definition.ID)
	if err != nil {
		return err
	}
	if err := modelrecipe.ActivateCapability(ctx, store, definition, verification,
		recipe.EvidenceExperimental, "session-supervised training substrate bootstrap"); err != nil {
		return err
	}
	fmt.Printf("training recipe activated: %s (model %s, experimental evidence tier)\n", definition.ID, modelID)
	return nil
}
