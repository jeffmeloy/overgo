package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"

	"overgo/internal/artifact"
	"overgo/internal/longform"
	"overgo/internal/overgodb"
)

func readCorpus(options options, limit int) (string, error) {
	if options.corpusText != "" {
		return options.corpusText, nil
	}
	if options.Corpus == "" {
		return longform.Corpus(options.Root, limit)
	}
	data, err := os.ReadFile(options.Corpus)
	if err != nil {
		return "", err
	}
	if len(data) == 0 {
		return "", errors.New("longform: fixed corpus is empty")
	}
	return string(data), nil
}

func bindBaselines(ctx context.Context, options options, targets []target) error {
	corpus, err := readCorpus(options, 0)
	if err != nil {
		return err
	}
	digest := sha256.Sum256([]byte(corpus))
	store, err := overgodb.OpenReadOnly(options.Repository)
	if err != nil {
		return err
	}
	defer store.Close()
	selected := make(map[artifact.ID]longform.Summary)
	for _, text := range options.Baselines {
		id, err := artifact.ParseID(text)
		if err != nil {
			return err
		}
		baseline, err := longform.ReadBaseline(ctx, store, id)
		if err != nil {
			return err
		}
		if baseline.Result.Inputs.CorpusDigest != hex.EncodeToString(digest[:]) || baseline.Result.Floors != longform.DeclaredFloors() {
			return errors.New("longform: baseline corpus or floors differ from this experiment")
		}
		model := baseline.Result.Inputs.Model
		if _, duplicate := selected[model]; duplicate {
			return fmt.Errorf("longform: multiple baselines for %s", model)
		}
		selected[model] = baseline
	}
	for index := range targets {
		baseline, found := selected[targets[index].weights]
		if !found {
			return fmt.Errorf("longform: no explicit baseline for %s", targets[index].entry.Location)
		}
		targets[index].record = baseline
		delete(selected, targets[index].weights)
	}
	if len(selected) != 0 {
		return errors.New("longform: baseline supplied for an unselected model")
	}
	return nil
}
