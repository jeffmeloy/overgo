package workflowrecipe

import "llamacpp2go/internal/recipe"

var placements = []recipe.Placement{
	recipe.PlacementHost, recipe.PlacementDevice, recipe.PlacementHybrid,
}

func mustCatalog() *recipe.Catalog {
	catalog, err := recipe.NewCatalog(
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
		decodeModule(ModuleDecodeImage, recipe.DataImage),
		decodeModule(ModuleDecodeAudio, recipe.DataAudio),
		decodeModule(ModuleDecodeVideo, recipe.DataVideo),
		module(ModuleProject, tasks(recipe.TaskProjection),
			ports(port("tensor", recipe.DataTensor, recipe.CardinalityOne)),
			ports(port("embeddings", recipe.DataEmbeddings, recipe.CardinalityOne))),
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
	)
	if err != nil {
		panic(err)
	}
	return catalog
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

func decodeModule(id recipe.ModuleID, data recipe.DataKind) recipe.Module {
	return module(id, tasks(recipe.TaskProjection),
		ports(port("media", data, recipe.CardinalityOne)),
		ports(port("tensor", recipe.DataTensor, recipe.CardinalityOne)))
}

func tasks(values ...recipe.Task) []recipe.Task { return values }
func ports(values ...recipe.Port) []recipe.Port { return values }

func port(name recipe.PortName, data recipe.DataKind, cardinality recipe.Cardinality) recipe.Port {
	return recipe.Port{Name: name, Data: data, Cardinality: cardinality}
}
