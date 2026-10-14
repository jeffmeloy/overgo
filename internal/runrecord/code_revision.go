package runrecord

import (
	"context"
	"errors"

	"overgo/internal/artifact"
)

const (
	// codeRevisionMediaType identifies an exact source revision used by a run.
	codeRevisionMediaType = "application/vnd.overgo.code-revision+json"
	// codeRevisionSchema identifies the code revision evidence contract.
	codeRevisionSchema = "overgo/code-revision/v1"
)

// CodeRevision gives a content-addressed evidence identity to the full Git
// revision already carried by bound run records.
type CodeRevision struct {
	Version uint16      `json:"version"`
	Commit  string      `json:"commit"`
	ID      artifact.ID `json:"-"`
}

var codeRevisionCodec = artifact.JSONDocumentCodec(
	"code revision", artifact.KindEvidence, codeRevisionMediaType, codeRevisionSchema,
	func(value *CodeRevision) error {
		if value == nil || value.Version != artifact.InitialDocumentVersion || !validCodeCommit(value.Commit) {
			return errors.New("run record: invalid code revision")
		}
		return nil
	},
	func(value CodeRevision) artifact.ID { return value.ID },
	func(value *CodeRevision, id artifact.ID) { value.ID = id },
)

// NewCodeRevision validates and identifies an exact source revision.
func NewCodeRevision(commit string) (CodeRevision, error) {
	if commit == "" {
		return CodeRevision{}, errors.New("run record: code revision is absent")
	}
	return codeRevisionCodec.New(CodeRevision{Version: artifact.InitialDocumentVersion, Commit: commit})
}

// RequireCodeRevision loads one exact source revision through its owning contract.
func RequireCodeRevision(ctx context.Context, reader artifact.Reader, id artifact.ID) (CodeRevision, error) {
	return codeRevisionCodec.Require(ctx, reader, id)
}

// Publication returns the canonical content-addressed revision batch.
func (value CodeRevision) Publication() (artifact.Batch, error) {
	content, err := codeRevisionCodec.Content(value)
	if err != nil {
		return artifact.Batch{}, err
	}
	return artifact.NewDocumentBatch(
		"training/code-revision/"+value.ID.String(), []artifact.Content{content}, nil, nil,
	)
}
