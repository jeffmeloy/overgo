package main

import (
	"cmp"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

type guardRequest struct {
	Model          artifact.ID `json:"model"`
	Baseline       string      `json:"baseline"`
	Location       string      `json:"location"`
	MeasuredWallNS int64       `json:"measured_wall_ns"`
}

type guardSelection struct {
	Surface    string              `json:"surface"`
	Status     string              `json:"status"`
	Reason     string              `json:"reason"`
	Requests   []guardRequest      `json:"requests"`
	Coverage   guardCoverageReport `json:"coverage,omitzero"`
	Exclusions string              `json:"exclusions"`
}

func readGuardPaths(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var paths []string
	for line := range strings.SplitSeq(string(data), "\n") {
		if line = strings.TrimSpace(line); line != "" {
			paths = append(paths, line)
		}
	}
	return paths, nil
}

func selectGuardChecks(coverage guardCoverageReport, reason string, coverageErr error) (guardSelection, error) {
	selection := guardSelection{Surface: coverage.Surface, Status: "incomplete", Reason: reason, Coverage: coverage,
		Exclusions: "exact-model coverage; no operator-specific cross-model equivalence is established; no measurements, GPU tests, benchmark suites or modality validation executed"}
	if coverageErr != nil {
		return selection, coverageErr
	}
	if coverage.Selected == 0 || coverage.Surface == "" || coverage.Covered != coverage.Selected ||
		coverage.Explicit != coverage.Selected || len(coverage.Entries) != coverage.Selected || coverage.Uncovered != 0 || len(coverage.Unused) != 0 || reason == "" {
		return selection, errors.New("guard selection: incomplete coverage denominator or source identity")
	}
	seen := map[artifact.ID]bool{}
	var requests []guardRequest
	for _, entry := range coverage.Entries {
		record, err := artifact.ParseID(entry.Baseline)
		if err != nil || record.Kind() != artifact.KindEvidence || entry.Gap != "" || !entry.Model.Valid() || entry.Model.Kind() != artifact.KindModel ||
			entry.Location == "" || entry.MeasuredWallNS <= 0 || seen[entry.Model] {
			return selection, fmt.Errorf("guard selection: invalid or duplicate model coverage for %s", entry.Location)
		}
		seen[entry.Model] = true
		requests = append(requests, guardRequest{Model: entry.Model, Baseline: entry.Baseline,
			Location: entry.Location, MeasuredWallNS: entry.MeasuredWallNS})
	}
	// Order work by the explicit record's observed cost, never replace a
	// baseline with the fastest historical sample or invent equivalence.
	slices.SortFunc(requests, func(a, b guardRequest) int {
		if order := cmp.Compare(a.MeasuredWallNS, b.MeasuredWallNS); order != 0 {
			return order
		}
		return strings.Compare(a.Model.String(), b.Model.String())
	})
	selection.Requests = requests
	selection.Status = "ready"
	return selection, nil
}
