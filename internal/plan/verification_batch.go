package plan

import (
	"errors"
	"fmt"
	"io/fs"
	"strings"

	"overgo/internal/testevidence"
	"overgo/internal/textcheck"
)

// VerificationBatch bounds implementation subdivisions of one plan step.
// Checkpoints remain obligations, not completed rows or executable policy.
// The parent step's verifier is the final integration acceptance.
type VerificationBatch struct {
	Scope       []string                 `json:"scope"`
	Rationale   string                   `json:"rationale"`
	ReopenWhen  string                   `json:"reopen_when"`
	Checkpoints []VerificationCheckpoint `json:"checkpoints"`
	// Flush: optional keyed accumulation bounds; absent -> checkpoints run
	// as declared without accumulation.
	Flush *BatchFlush `json:"flush,omitempty"`
}

// VerificationCheckpoint declares one acceptance within a step-local batch.
// Dependencies name earlier checkpoints in this batch, never other plan rows.
type VerificationCheckpoint struct {
	ID        string   `json:"id"`
	Title     string   `json:"title"`
	Verify    string   `json:"verify"`
	DependsOn []string `json:"depends_on,omitempty"`
}

// GateCheckName is the checkpoint's subordinate full-gate acceptance name.
func (checkpoint VerificationCheckpoint) GateCheckName() string {
	return "acceptance-" + checkpoint.ID
}

func validateVerificationBatch(batch *VerificationBatch) error {
	if batch == nil {
		return nil
	}
	if len(batch.Scope) == 0 || len(batch.Checkpoints) == 0 ||
		!validAutomationDetail(batch.Rationale) || !validAutomationDetail(batch.ReopenWhen) {
		return errors.New("verification batch requires scope, checkpoints, rationale and reopen condition")
	}
	if err := validateBatchFlush(batch.Flush); err != nil {
		return err
	}
	paths := map[string]bool{}
	for _, candidate := range batch.Scope {
		if candidate == "." || !fs.ValidPath(candidate) ||
			strings.TrimSpace(candidate) != candidate || strings.ContainsAny(candidate, "\\:*?[\x00\r\n") {
			return fmt.Errorf("verification batch scope is not a canonical literal repository path: %q", candidate)
		}
		key := strings.ToLower(candidate)
		if paths[key] {
			return fmt.Errorf("verification batch scope is duplicated: %q", candidate)
		}
		paths[key] = true
	}
	seen := map[string]bool{}
	for _, checkpoint := range batch.Checkpoints {
		if !validPlanID(checkpoint.ID) || !textcheck.LowerIdentifier(checkpoint.ID, len(checkpoint.ID)) || seen[checkpoint.ID] ||
			!validAutomationText(checkpoint.Title) || !validAutomationDetail(checkpoint.Verify) {
			return fmt.Errorf("verification batch checkpoint %q is invalid or duplicated", checkpoint.ID)
		}
		if err := testevidence.ValidateGoTestCommand(checkpoint.Verify); err != nil {
			return fmt.Errorf("verification batch checkpoint %s: %w", checkpoint.ID, err)
		}
		dependencies := map[string]bool{}
		for _, dependency := range checkpoint.DependsOn {
			if !seen[dependency] || dependencies[dependency] {
				return fmt.Errorf("verification batch checkpoint %s: dependency %q must name a distinct earlier checkpoint", checkpoint.ID, dependency)
			}
			dependencies[dependency] = true
		}
		seen[checkpoint.ID] = true
	}
	return nil
}
