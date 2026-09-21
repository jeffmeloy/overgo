package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"

	"overgo/internal/processcontrol"
)

// guardRecord is the newest long-form record of one weights location, as
// cmd/longform latest-records prints it. The planner asks that command and
// does not read the records itself: reading them links the device stack the
// records were measured with, and a planner that loads no model has no
// business in the device group, which is what the device-group ratchet
// refused when this first imported the long-form package.
type guardRecord struct {
	Location string   `json:"location"`
	Surface  string   `json:"surface"`
	Commit   string   `json:"commit"`
	Passed   bool     `json:"passed"`
	Reasons  []string `json:"reasons"`
}

// surfaceShown is how much of a surface digest a refusal shows.
const surfaceShown = 12

// latestGuardRecords asks the long-form command for every location's newest
// record.
func latestGuardRecords(ctx context.Context, root, store string) (map[string]guardRecord, error) {
	var stdout, stderr bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "go", Args: []string{"run", "./cmd/longform", "latest-records", store}, Dir: root,
		Stdout: &stdout, Stderr: &stderr,
	})
	if err != nil {
		return nil, fmt.Errorf("validate: long-form records: %w", err)
	}
	if receipt.ExitCode != 0 {
		return nil, fmt.Errorf("validate: long-form records: exit=%d: %s", receipt.ExitCode, strings.TrimSpace(stderr.String()))
	}
	var listed []guardRecord
	if err := json.Unmarshal(stdout.Bytes(), &listed); err != nil {
		return nil, fmt.Errorf("validate: long-form records: %w", err)
	}
	records := make(map[string]guardRecord, len(listed))
	for _, record := range listed {
		records[record.Location] = record
	}
	return records, nil
}

// locationKey normalizes a weights path as the long-form records key it, so a
// record registered with one spelling of a path meets a cell with another.
func locationKey(path string) string {
	return strings.ToLower(filepath.ToSlash(filepath.Clean(path)))
}

// guardCurrent says why a location's guard evidence is not current at the
// surface the code is at now, with the run that lifts it; nil when its newest
// record ran on this surface and met its floors.
func guardCurrent(records map[string]guardRecord, location, surface string) error {
	rerun := fmt.Sprintf("run: go run ./cmd/longform -guard -publish -corpus %s -budget %s -model-budget %s %q", guardCorpus, guardBudget, guardModelBudget, location)
	record, found := records[locationKey(location)]
	switch {
	case !found:
		return fmt.Errorf("no long-form record for this location; %s", rerun)
	case record.Surface != surface:
		return fmt.Errorf("its long-form record measured inference surface %.*s at commit %.*s, the code is at surface %.*s; %s",
			surfaceShown, record.Surface, surfaceShown, record.Commit, surfaceShown, surface, rerun)
	case !record.Passed:
		return fmt.Errorf("its long-form record failed its floors on this surface (%s); fix the runtime, then %s", strings.Join(record.Reasons, "; "), rerun)
	}
	return nil
}
