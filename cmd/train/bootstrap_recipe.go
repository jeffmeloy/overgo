package main

import (
	"context"

	"overgo/internal/repodb"
	"overgo/internal/trainingworkflow"
)

func bootstrapTrainingRecipe(store *repodb.Store, modelPath, datasetPath string) error {
	_, err := trainingworkflow.BootstrapTokenRecipe(context.Background(), store, modelPath, datasetPath)
	return err
}
