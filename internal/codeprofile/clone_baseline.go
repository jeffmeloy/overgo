package codeprofile

import (
	"fmt"
	"os"

	"overgo/internal/jsonfile"
)

// CloneBaselineFile is the repository-relative ratchet document: the
// reviewed ceiling on duplicate-excess AST nodes. The gate refuses a
// commit whose census exceeds it, so duplication can only grow through
// a deliberate, versioned edit of this file.
const CloneBaselineFile = "docs/clone_baseline.json"

// CloneBaseline is the versioned duplicate-excess ceiling.
type CloneBaseline struct {
	Version uint16 `json:"version"`
	Doc     string `json:"doc,omitzero"`
	// DuplicateExcessNodes is the ceiling; clone-census -update-baseline
	// lowers it to the measured value after a tightening campaign.
	DuplicateExcessNodes int `json:"duplicate_excess_nodes"`
}

// LoadCloneBaseline reads the ratchet; an absent file declares no
// ceiling. Decoding is strict because the file is gate authority.
func LoadCloneBaseline(path string) (CloneBaseline, bool, error) {
	var baseline CloneBaseline
	if _, err := os.Stat(path); err != nil {
		return CloneBaseline{}, false, nil
	}
	if err := jsonfile.DecodeStrict(path, &baseline); err != nil {
		return CloneBaseline{}, false, err
	}
	if baseline.Version != 1 || baseline.DuplicateExcessNodes < 0 {
		return CloneBaseline{}, false, fmt.Errorf("codeprofile: clone baseline needs version 1 and a nonnegative ceiling")
	}
	return baseline, true, nil
}

// AdmitCloneBaseline compares a measured census against the ratchet.
func AdmitCloneBaseline(baseline CloneBaseline, measured int) error {
	if measured > baseline.DuplicateExcessNodes {
		return fmt.Errorf(
			"duplicate-excess %d exceeds the %s ceiling %d; consolidate the clone or raise the ceiling in a reviewed edit",
			measured, CloneBaselineFile, baseline.DuplicateExcessNodes,
		)
	}
	return nil
}
