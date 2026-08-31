package gate

import (
	"context"
	"fmt"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

func admitStoredReview(repo, storePath, verdictText, targetHead string) error {
	verdictID, err := artifact.ParseID(verdictText)
	if err != nil || verdictID.Kind() != artifact.KindEvidence {
		return fmt.Errorf("gate: invalid review verdict ID %q", verdictText)
	}
	store, err := overgodb.OpenReadOnly(filepath.Join(repo, storePath))
	if err != nil {
		return fmt.Errorf("gate: open review store: %w", err)
	}
	defer store.Close()
	admission, err := runrecord.LoadReviewAdmission(context.Background(), store, verdictID)
	if err != nil {
		return fmt.Errorf("gate: load review admission: %w", err)
	}
	return runrecord.AdmitReview(targetHead, admission)
}
