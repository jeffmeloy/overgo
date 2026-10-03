package main

import (
	"slices"
	"strings"

	"overgo/internal/artifact"
)

// A DNA model's inference evidence has no long-form guard: the guard reads a
// text corpus for continuation quality, which says nothing of a genome. Its
// owner is validate -smoke's native-likelihood oracle, which runs the model
// against its exact reference cases and records a bound run at the commit it
// ran on. It takes seconds, so every cycle runs it.

// dnaModality is the declared evaluation domain of a DNA model.
const dnaModality = "dna"

// smokeBudget is validate -smoke's execution budget for one DNA model's run.
const smokeBudget = "15m"

// smokeCommand re-acquires a DNA model's smoke run into the store validate
// selected and read, so the lane never resolves a default store of its own.
func smokeCommand(model artifact.ID, store string) []string {
	return []string{"run", "./cmd/validate", "-smoke", "-budget", smokeBudget, "-model", model.String(), "-repo", store}
}

// dnaGuard reports a DNA model's inference cell, whose evidence is its smoke
// run rather than a long-form record.
func dnaGuard(cell ModelValidation) bool {
	return cell.Validation == guardValidation && slices.Contains(strings.Split(cell.Modality, ","), dnaModality)
}
