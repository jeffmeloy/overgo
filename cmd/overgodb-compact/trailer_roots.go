package main

import (
	"context"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/gitauthority"
)

// trailerRoots collects every store identity the checkout's commit messages
// name -- the Overgo-Gate-Preparation, Overgo-Manifest-Plan and
// Overgo-Code-Manifest trailers and any identity a body cites -- once each,
// sorted. The plan completion authority proves every landed commit from
// these, so they root the live set exactly as committed documents do.
func trailerRoots(checkout string) ([]artifact.ID, error) {
	messages, err := gitauthority.Query(context.Background(), checkout, "log", "--format=%B")
	if err != nil {
		return nil, fmt.Errorf("git log for trailer roots: %w", err)
	}
	seen := map[artifact.ID]bool{}
	var roots []artifact.ID
	for _, match := range identityPattern.FindAll(messages, allMatches) {
		id, err := artifact.ParseID(string(match))
		if err != nil || seen[id] {
			continue
		}
		seen[id] = true
		roots = append(roots, id)
	}
	slices.SortFunc(roots, artifact.CompareID)
	return roots, nil
}

// mergeRoots unions two sorted root lists.
func mergeRoots(left, right []artifact.ID) []artifact.ID {
	merged := slices.Concat(left, right)
	slices.SortFunc(merged, artifact.CompareID)
	return slices.Compact(merged)
}
