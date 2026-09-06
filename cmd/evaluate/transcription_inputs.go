package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
)

// Read each immutable container once per selected column, verifying its actual
// identity before extracting values. A manifest cannot attach arbitrary bytes
// to a declared Parquet origin, and a batch does not rescan a file per case.
func readTranscriptionInputs(ctx context.Context, base string, specifications []transcriptionResourceManifestInput) ([]evaluation.TranscriptionResourceInput, error) {
	type groupKey struct{ path, column string }
	groups := make(map[groupKey][]int)
	var order []groupKey
	inputs := make([]evaluation.TranscriptionResourceInput, len(specifications))
	for index, spec := range specifications {
		if err := spec.Policy.Validate(); err != nil {
			return nil, err
		}
		if spec.Path == "" || spec.Origin.Container.Kind() != artifact.KindFile || spec.Origin.ValueIndex >= uint64(math.MaxInt) || spec.Origin.Column == "" && spec.Origin.ValueIndex != 0 {
			return nil, errors.New("evaluate: invalid transcription container selector")
		}
		key := groupKey{filepath.Clean(resolveEvaluationPath(base, spec.Path)), spec.Origin.Column}
		if _, found := groups[key]; !found {
			order = append(order, key)
		}
		groups[key] = append(groups[key], index)
		inputs[index] = evaluation.TranscriptionResourceInput{Name: spec.Name, Origin: spec.Origin, Policy: spec.Policy}
	}
	identities := make(map[string]artifact.ID)
	for _, key := range order {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		members := groups[key]
		id, found := identities[key.path]
		if !found {
			file, err := os.Open(key.path)
			if err != nil {
				return nil, err
			}
			identified, _, identifyErr := artifact.Identify(artifact.KindFile, file)
			if err := errors.Join(identifyErr, file.Close()); err != nil {
				return nil, err
			}
			id = identified
			identities[key.path] = id
		}
		selected := make(map[uint64][]int)
		var limit int
		var maximum uint64
		for _, index := range members {
			spec := specifications[index]
			if id != spec.Origin.Container {
				return nil, fmt.Errorf("evaluate: container identity differs for %s", spec.Name)
			}
			selected[spec.Origin.ValueIndex] = append(selected[spec.Origin.ValueIndex], index)
			limit = max(limit, int(spec.Origin.ValueIndex)+1)
			maximum = max(maximum, spec.Policy.MaximumEncodedBytes)
		}
		if key.column == "" {
			file, err := os.Open(key.path)
			if err != nil {
				return nil, err
			}
			data, readErr := io.ReadAll(io.LimitReader(file, int64(maximum)+1))
			if err := errors.Join(readErr, file.Close()); err != nil {
				return nil, err
			}
			if uint64(len(data)) > maximum {
				return nil, errors.New("evaluate: audio file exceeds encoded payload budget")
			}
			for _, index := range members {
				inputs[index].Data = slices.Clone(data)
			}
			continue
		}
		seen := make(map[uint64]bool)
		err := dataset.ReadParquetTextRows(key.path, key.column, limit, func(ordinal uint64, value string) error {
			if err := ctx.Err(); err != nil {
				return err
			}
			for _, index := range selected[ordinal] {
				inputs[index].Data = []byte(value)
				seen[ordinal] = true
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
		if len(seen) != len(selected) {
			return nil, errors.New("evaluate: selected transcription value is absent")
		}
	}
	return inputs, nil
}
