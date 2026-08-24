package main

import (
	"context"

	"overgo/internal/overgodb"
	"overgo/internal/trainingworkflow"
)

func bootstrapTrainingRecipe(store *overgodb.Store, modelPath, datasetPath string) error {
	_, err := trainingworkflow.BootstrapTokenRecipe(context.Background(), store, modelPath, datasetPath)
	return err
}
