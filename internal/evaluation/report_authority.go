package evaluation

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"overgo/internal/artifact"
)

// CommittedReportPassed reads one committed evaluation report by exact
// identity and reports whether it passed. Only the typed evaluation
// report contracts qualify -- an arbitrary evidence artifact, an
// absent identity, or an unrecognized schema is refused, so a caller
// gating authority on "a committed evaluation exists and passed"
// cannot be satisfied by assertion.
func CommittedReportPassed(ctx context.Context, reader artifact.Reader, id artifact.ID) (bool, error) {
	if ctx == nil || reader == nil {
		return false, errors.New("evaluation: nil report context or reader")
	}
	content, found, err := artifact.ReadContent(ctx, reader, id)
	if err != nil {
		return false, err
	}
	if !found {
		return false, fmt.Errorf("evaluation: report %s is not committed", id)
	}
	mediaType, schema := content.Descriptor.MediaType, content.Descriptor.Schema
	switch {
	case mediaType == campaignReportMediaType && schema == campaignReportSchema,
		mediaType == shardReportMediaType && schema == shardReportSchema:
		var value struct {
			Passed bool `json:"passed"`
		}
		if err := json.Unmarshal(content.Data, &value); err != nil {
			return false, err
		}
		return value.Passed, nil
	default:
		return false, fmt.Errorf("evaluation: %s is not an evaluation report", id)
	}
}
