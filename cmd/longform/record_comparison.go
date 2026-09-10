package main

import (
	"context"
	"errors"
	"fmt"
	"io"

	"overgo/internal/artifact"
	"overgo/internal/longform"
	"overgo/internal/overgodb"
)

func compareStoredRecords(ctx context.Context, options options, output io.Writer) error {
	store, err := overgodb.OpenReadOnly(options.Repository)
	if err != nil {
		return err
	}
	defer store.Close()
	seen := make(map[artifact.ID]bool)
	read := func(texts []string) ([]longform.Summary, error) {
		var records []longform.Summary
		for _, text := range texts {
			id, err := artifact.ParseID(text)
			if err != nil {
				return nil, err
			}
			if seen[id] {
				return nil, fmt.Errorf("longform: duplicate or self-comparison record %s", id)
			}
			seen[id] = true
			record, err := longform.ReadBaseline(ctx, store, id)
			if err != nil {
				return nil, err
			}
			records = append(records, record)
		}
		return records, nil
	}
	references, err := read(options.Baselines)
	if err != nil {
		return err
	}
	candidates, err := read(options.CompareRecords)
	if err != nil {
		return err
	}
	used := make(map[artifact.ID]bool)
	var failures []error
	comparisons := 0
	for _, candidate := range candidates {
		fresh := candidate.Result
		fmt.Fprintf(output, "candidate=%s short_prompt_ms=%.4f short_prompt_tok_s=%.1f short_nll=%.4f retained_bytes=%d peak_bytes=%d\n",
			candidate.Record, fresh.Shape.Measure.PromptMilliseconds, fresh.Shape.Measure.PromptTokensPerSecond,
			fresh.Shape.NLL, fresh.Shape.Measure.Memory.CurrentBytes, fresh.Shape.Measure.Memory.PeakBytes)
		matched := false
		for _, reference := range references {
			if reference.Result.Inputs.Model != candidate.Result.Inputs.Model {
				continue
			}
			matched = true
			used[reference.Record] = true
			comparisons++
			verdict := longform.Compare(reference.Result, fresh, reference.Result.Floors, reference.Result.Floors.CheckRungCeiling)
			fmt.Fprintf(output, "reference=%s candidate=%s verdict=%s\n", reference.Record, candidate.Record, verdict)
			if !verdict.Passed {
				failures = append(failures, fmt.Errorf("longform: %s against %s: %s", candidate.Record, reference.Record, verdict))
			}
		}
		if !matched {
			failures = append(failures, fmt.Errorf("longform: candidate %s has no matching reference", candidate.Record))
		}
	}
	for _, reference := range references {
		if !used[reference.Record] {
			failures = append(failures, fmt.Errorf("longform: unused reference %s", reference.Record))
		}
	}
	fmt.Fprintf(output, "stored comparison: references=%d candidates=%d comparisons=%d failures=%d; no model executions or publications\n", len(references), len(candidates), comparisons, len(failures))
	return errors.Join(failures...)
}
