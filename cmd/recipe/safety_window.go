package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"os"
	"strconv"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// deriveSafetyWindow derives one live comparison window from a metric
// history and publishes the derivation as evidence citing the history it
// was read from. Boundaries are order statistics; nothing is hand-picked.
func deriveSafetyWindow(arguments []string) error {
	flags := flag.NewFlagSet("recipe safety-window", flag.ContinueOnError)
	repoFlag := flags.String("repo", "", "OvergoDB store; empty resolves via the data-root contract")
	metricFlag := flags.String("metric", "", "metric the window bounds")
	directionFlag := flags.String("direction", string(runrecord.DirectionMaximize), "metric direction (maximize|minimize)")
	historyFlag := flags.String("history", "", "committed history evidence artifact ID the values were read from")
	valuesFlag := flags.String("values", "", "comma-separated observed metric values")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if strings.TrimSpace(*metricFlag) == "" || strings.TrimSpace(*historyFlag) == "" ||
		strings.TrimSpace(*valuesFlag) == "" || flags.NArg() != 0 {
		return errors.New(
			"usage: recipe safety-window -metric <name> -history <evidence-id> -values <v,...> [-direction maximize|minimize] [-repo <store>]",
		)
	}
	historyEvidence, err := artifact.ParseID(*historyFlag)
	if err != nil {
		return err
	}
	var history []float64
	for _, text := range strings.Split(*valuesFlag, ",") {
		value, err := strconv.ParseFloat(strings.TrimSpace(text), 64)
		if err != nil {
			return err
		}
		history = append(history, value)
	}
	window, err := evaluation.DeriveLiveSafetyWindow(
		*metricFlag, runrecord.Direction(*directionFlag), history, historyEvidence,
	)
	if err != nil {
		return err
	}
	repository := strings.TrimSpace(*repoFlag)
	if repository == "" {
		roots, err := dataroot.ResolveCurrent()
		if err != nil {
			return err
		}
		repository = roots.Store
	}
	ctx := context.Background()
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	published, err := evaluation.PublishLiveSafetyWindow(ctx, store, window)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(window); err != nil {
		return err
	}
	_, err = os.Stdout.WriteString(published.String() + "\n")
	return err
}
