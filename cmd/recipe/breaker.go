package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/evaluation"
	"overgo/internal/loop"
	"overgo/internal/overgodb"
)

// judgeBreaker judges one live observation window against a published
// safety-window derivation and publishes the transition. Observations are
// metric values; "skip" marks skipped or inapplicable work, which never
// enters the ratios.
func judgeBreaker(arguments []string) error {
	flags := flag.NewFlagSet("recipe breaker", flag.ContinueOnError)
	repoFlag := flags.String("repo", "", "OvergoDB store; empty resolves via the data-root contract")
	windowFlag := flags.String("window", "", "published safety-window derivation artifact ID")
	valuesFlag := flags.String("values", "", "comma-separated observed values; skip marks inapplicable work")
	priorFlag := flags.Uint64("prior", 0, "consecutive breached windows before this one")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if strings.TrimSpace(*windowFlag) == "" || strings.TrimSpace(*valuesFlag) == "" || flags.NArg() != 0 {
		return errors.New(
			"usage: recipe breaker -window <derivation-id> -values <v|skip,...> [-prior <n>] [-repo <store>]",
		)
	}
	windowEvidence, err := artifact.ParseID(*windowFlag)
	if err != nil {
		return err
	}
	var observations []loop.LiveObservation
	for _, text := range strings.Split(*valuesFlag, ",") {
		text = strings.TrimSpace(text)
		if text == "skip" {
			observations = append(observations, loop.LiveObservation{})
			continue
		}
		value, err := strconv.ParseFloat(text, 64)
		if err != nil {
			return err
		}
		observations = append(observations, loop.LiveObservation{Value: value, Applicable: true})
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
	content, found, err := artifact.ReadContent(ctx, store, windowEvidence)
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("recipe: safety-window derivation %s is not committed", windowEvidence)
	}
	var window evaluation.LiveSafetyWindow
	if err := json.Unmarshal(content.Data, &window); err != nil {
		return err
	}
	transition, err := loop.JudgeCircuitBreaker(window, windowEvidence, observations, *priorFlag)
	if err != nil {
		return err
	}
	published, err := loop.PublishCircuitBreakerTransition(ctx, store, transition)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(transition); err != nil {
		return err
	}
	_, err = os.Stdout.WriteString(published.String() + "\n")
	return err
}
