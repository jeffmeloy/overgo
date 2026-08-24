package workflowrecipe

import (
	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

const (
	ModuleTokenize        recipe.ModuleID = "text.tokenize"
	ModuleDetokenize      recipe.ModuleID = "text.detokenize"
	ModuleGenerate        recipe.ModuleID = "model.generate"
	ModuleEmbed           recipe.ModuleID = "model.embed"
	ModulePool            recipe.ModuleID = "embedding.pool"
	ModuleNormalize       recipe.ModuleID = "embedding.normalize"
	ModuleRerankPrepare   recipe.ModuleID = "rerank.prepare"
	ModuleScore           recipe.ModuleID = "model.score"
	ModuleOrder           recipe.ModuleID = "rerank.order"
	ModuleDecodeImage     recipe.ModuleID = "media.decode-image"
	ModuleDecodeAudio     recipe.ModuleID = "media.decode-audio"
	ModuleDecodeVideo     recipe.ModuleID = "media.decode-video"
	ModuleProjectImage    recipe.ModuleID = "projector.image"
	ModuleProjectAudio    recipe.ModuleID = "projector.audio"
	ModuleProjectVideo    recipe.ModuleID = "projector.video"
	ModuleBatchDataset    recipe.ModuleID = "training.batch-dataset"
	ModuleTrainingForward recipe.ModuleID = "training.forward"
	ModuleBackward        recipe.ModuleID = "training.backward"
	ModuleOptimize        recipe.ModuleID = "training.optimize"
	ModuleBatchPreference recipe.ModuleID = "training.batch-preference"
	ModuleScorePolicy     recipe.ModuleID = "training.score-policy"
	ModuleScoreReference  recipe.ModuleID = "training.score-reference"
	ModuleDPOObjective    recipe.ModuleID = "training.dpo"
	ModuleBatchRollout    recipe.ModuleID = "training.batch-rollout"
	ModuleGRPOObjective   recipe.ModuleID = "training.grpo"
	ModuleSelectComponent recipe.ModuleID = "composition.select-component"
	ModuleTrainBridge     recipe.ModuleID = "composition.train-bridge"
	ModuleEvaluateBridge  recipe.ModuleID = "composition.evaluate-bridge"
	// ModuleFitModel owns model fitting.
	ModuleFitModel recipe.ModuleID = "model-build.fit"
	// ModuleEvaluateModel owns fitted-model evaluation.
	ModuleEvaluateModel recipe.ModuleID = "model-build.evaluate"
	// ModuleRecordModel owns model evidence publication.
	ModuleRecordModel recipe.ModuleID = "model-build.record"
	// ModulePromoteModel owns model promotion decisions.
	ModulePromoteModel recipe.ModuleID = "model-build.promote"

	// ModelBuildStatePort carries durable construction state.
	ModelBuildStatePort recipe.PortName = "state"
	// BuildRecipeFact identifies recipe authority.
	BuildRecipeFact recipe.PortName = "recipe"
	// BuildDatasetFact identifies dataset authority.
	BuildDatasetFact recipe.PortName = "dataset"
	// BuildConstructionFact identifies construction authority.
	BuildConstructionFact recipe.PortName = "construction"
	// BuildModelFact identifies the current model.
	BuildModelFact recipe.PortName = "model"
	// BuildCheckpointFact identifies the fitted checkpoint.
	BuildCheckpointFact recipe.PortName = "checkpoint"
	// BuildRunFact identifies the evaluation run.
	BuildRunFact recipe.PortName = "run"
	// BuildEvaluationFact identifies evaluation output.
	BuildEvaluationFact recipe.PortName = "evaluation"
	// BuildEvidenceFact identifies published evidence.
	BuildEvidenceFact recipe.PortName = "evidence"
	// BuildDecisionFact identifies the promotion decision.
	BuildDecisionFact recipe.PortName = "decision"
)

var catalog = mustCatalog()

func Catalog() *recipe.Catalog { return catalog }

func Module(id recipe.ModuleID) (recipe.Module, bool) { return catalog.Module(id) }

var placements = []recipe.Placement{
	recipe.PlacementHost, recipe.PlacementDevice, recipe.PlacementHybrid,
}

func mustCatalog() *recipe.Catalog {
	modules := []recipe.Module{
		module(ModuleTokenize, tasks(recipe.TaskGeneration, recipe.TaskEmbedding),
			ports(port("text", recipe.DataText, recipe.CardinalityOne)),
			ports(port("tokens", recipe.DataTokens, recipe.CardinalityOne))),
		module(ModuleDetokenize, tasks(recipe.TaskGeneration),
			ports(port("tokens", recipe.DataTokens, recipe.CardinalityOne)),
			ports(port("text", recipe.DataText, recipe.CardinalityOne))),
		module(ModuleGenerate, tasks(recipe.TaskGeneration),
			ports(port("tokens", recipe.DataTokens, recipe.CardinalityOne)),
			ports(port("tokens", recipe.DataTokens, recipe.CardinalityOne))),
		module(ModuleEmbed, tasks(recipe.TaskEmbedding),
			ports(port("tokens", recipe.DataTokens, recipe.CardinalityOne)),
			ports(port("embeddings", recipe.DataEmbeddings, recipe.CardinalityOne))),
		module(ModulePool, tasks(recipe.TaskEmbedding),
			ports(port("embeddings", recipe.DataEmbeddings, recipe.CardinalityOne)),
			ports(port("embeddings", recipe.DataEmbeddings, recipe.CardinalityOne))),
		module(ModuleNormalize, tasks(recipe.TaskEmbedding),
			ports(port("embeddings", recipe.DataEmbeddings, recipe.CardinalityOne)),
			ports(port("embeddings", recipe.DataEmbeddings, recipe.CardinalityOne))),
		module(ModuleRerankPrepare, tasks(recipe.TaskRerank),
			ports(
				port("documents", recipe.DataText, recipe.CardinalityOneOrMany),
				port("query", recipe.DataText, recipe.CardinalityOne),
			), ports(port("tokens", recipe.DataTokens, recipe.CardinalityOneOrMany))),
		module(ModuleScore, tasks(recipe.TaskRerank),
			ports(port("tokens", recipe.DataTokens, recipe.CardinalityOneOrMany)),
			ports(port("scores", recipe.DataScores, recipe.CardinalityOneOrMany))),
		module(ModuleOrder, tasks(recipe.TaskRerank),
			ports(port("scores", recipe.DataScores, recipe.CardinalityOneOrMany)),
			ports(port("ranking", recipe.DataRanking, recipe.CardinalityOne))),
		decodeModule(ModuleDecodeImage, recipe.DataImage, recipe.DataImageTensor),
		decodeModule(ModuleDecodeAudio, recipe.DataAudio, recipe.DataAudioTensor),
		decodeModule(ModuleDecodeVideo, recipe.DataVideo, recipe.DataVideoTensor),
		projectModule(ModuleProjectImage, recipe.DataImageTensor),
		projectModule(ModuleProjectAudio, recipe.DataAudioTensor),
		projectModule(ModuleProjectVideo, recipe.DataVideoTensor),
		module(ModuleBatchDataset, tasks(recipe.TaskTraining), nil,
			ports(port("batch", recipe.DataBatch, recipe.CardinalityOne))),
		module(ModuleTrainingForward, tasks(recipe.TaskTraining),
			ports(port("batch", recipe.DataBatch, recipe.CardinalityOne)),
			ports(port("loss", recipe.DataLoss, recipe.CardinalityOne))),
		module(ModuleBackward, tasks(recipe.TaskTraining),
			ports(port("loss", recipe.DataLoss, recipe.CardinalityOne)),
			ports(port("gradients", recipe.DataGradients, recipe.CardinalityOne))),
		module(ModuleOptimize, tasks(recipe.TaskTraining),
			ports(port("gradients", recipe.DataGradients, recipe.CardinalityOne)),
			ports(port("checkpoint", recipe.DataCheckpoint, recipe.CardinalityOne))),
	}
	modules = append(modules, preferenceModules()...)
	modules = append(modules, rolloutModules()...)
	modules = append(modules, compositionModules()...)
	modules = append(modules, modelBuildModules()...)
	catalog, err := recipe.NewCatalog(modules...)
	if err != nil {
		panic(err)
	}
	return catalog
}

func modelBuildModules() []recipe.Module {
	state := ports(port(ModelBuildStatePort, recipe.DataArtifact, recipe.CardinalityOne))
	required := []recipe.ArtifactRequirement{
		{Name: BuildRecipeFact, Kind: artifact.KindRecipe, Preserve: true},
		{Name: BuildDatasetFact, Kind: artifact.KindDataset, Preserve: true},
		{Name: BuildConstructionFact, Kind: artifact.KindRecipe, Preserve: true},
	}
	stage := func(node recipe.NodeID, id, next recipe.ModuleID, produced ...recipe.ArtifactRequirement) recipe.Module {
		required = append(required, produced...)
		contract := module(id, tasks(recipe.TaskTraining), state, state)
		contract.Postconditions = append([]recipe.ArtifactRequirement(nil), required...)
		contract.StageNode, contract.Next = node, next
		return contract
	}
	return []recipe.Module{
		stage("fit", ModuleFitModel, ModuleEvaluateModel,
			recipe.ArtifactRequirement{Name: BuildModelFact, Kind: artifact.KindModel},
			recipe.ArtifactRequirement{Name: BuildCheckpointFact, Kind: artifact.KindCheckpoint}),
		stage("evaluate", ModuleEvaluateModel, ModuleRecordModel,
			recipe.ArtifactRequirement{Name: BuildRunFact, Kind: artifact.KindRun},
			recipe.ArtifactRequirement{Name: BuildEvaluationFact, Kind: artifact.KindEvaluation}),
		stage("record", ModuleRecordModel, ModulePromoteModel,
			recipe.ArtifactRequirement{Name: BuildEvidenceFact, Kind: artifact.KindEvidence}),
		stage("promote", ModulePromoteModel, "",
			recipe.ArtifactRequirement{Name: BuildDecisionFact, Kind: artifact.KindEvidence}),
	}
}

func compositionModules() []recipe.Module {
	return []recipe.Module{
		module(ModuleSelectComponent, tasks(recipe.TaskTraining), nil,
			ports(port("component", recipe.DataTensor, recipe.CardinalityOne))),
		module(ModuleTrainBridge, tasks(recipe.TaskTraining),
			ports(port("component", recipe.DataTensor, recipe.CardinalityOne)),
			ports(port("bridge", recipe.DataArtifact, recipe.CardinalityOne))),
		module(ModuleEvaluateBridge, tasks(recipe.TaskTraining),
			ports(port("bridge", recipe.DataArtifact, recipe.CardinalityOne)),
			ports(port("metrics", recipe.DataMetrics, recipe.CardinalityOne))),
	}
}

func rolloutModules() []recipe.Module {
	return []recipe.Module{
		module(ModuleBatchRollout, tasks(recipe.TaskTraining), nil,
			ports(port("batch", recipe.DataPreferenceBatch, recipe.CardinalityOne))),
		module(ModuleGRPOObjective, tasks(recipe.TaskTraining),
			ports(port("policy", recipe.DataSequenceScores, recipe.CardinalityOne)),
			ports(port("loss", recipe.DataLoss, recipe.CardinalityOne))),
	}
}

func preferenceModules() []recipe.Module {
	return []recipe.Module{
		module(ModuleBatchPreference, tasks(recipe.TaskTraining), nil,
			ports(port("batch", recipe.DataPreferenceBatch, recipe.CardinalityOne))),
		module(ModuleScorePolicy, tasks(recipe.TaskTraining),
			ports(port("batch", recipe.DataPreferenceBatch, recipe.CardinalityOne)),
			ports(port("scores", recipe.DataSequenceScores, recipe.CardinalityOne))),
		module(ModuleScoreReference, tasks(recipe.TaskTraining),
			ports(port("batch", recipe.DataPreferenceBatch, recipe.CardinalityOne)),
			ports(port("scores", recipe.DataSequenceScores, recipe.CardinalityOne))),
		module(ModuleDPOObjective, tasks(recipe.TaskTraining),
			ports(
				port("policy", recipe.DataSequenceScores, recipe.CardinalityOne),
				port("reference", recipe.DataSequenceScores, recipe.CardinalityOne),
			), ports(port("loss", recipe.DataLoss, recipe.CardinalityOne))),
	}
}

func module(
	id recipe.ModuleID,
	tasks []recipe.Task,
	inputs, outputs []recipe.Port,
) recipe.Module {
	return recipe.Module{
		ID: id, Tasks: tasks, Placements: placements, Inputs: inputs, Outputs: outputs,
	}
}

func decodeModule(id recipe.ModuleID, mediaData, tensorData recipe.DataKind) recipe.Module {
	return module(id, tasks(recipe.TaskProjection),
		ports(port("media", mediaData, recipe.CardinalityOne)),
		ports(port("tensor", tensorData, recipe.CardinalityOne)))
}

func projectModule(id recipe.ModuleID, tensorData recipe.DataKind) recipe.Module {
	return module(id, tasks(recipe.TaskProjection),
		ports(port("tensor", tensorData, recipe.CardinalityOne)),
		ports(port("embeddings", recipe.DataEmbeddings, recipe.CardinalityOne)))
}

func tasks(values ...recipe.Task) []recipe.Task { return values }
func ports(values ...recipe.Port) []recipe.Port { return values }

func port(name recipe.PortName, data recipe.DataKind, cardinality recipe.Cardinality) recipe.Port {
	return recipe.Port{Name: name, Data: data, Cardinality: cardinality}
}
