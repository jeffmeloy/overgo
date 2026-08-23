package operatoraction

import (
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

const (
	approvalMediaType = "application/vnd.overgo.approval-request+json"
	approvalSchema    = "overgo/approval-request/v1"
)

// ApprovalRequest binds exact blocked work to one operator action.
type ApprovalRequest struct {
	Version   uint16      `json:"version"`
	Operation artifact.ID `json:"operation"`
	Recipe    artifact.ID `json:"recipe"`
	Tool      string      `json:"tool"`
	Arguments []string    `json:"arguments"`
	Prior     artifact.ID `json:"prior,omitzero"`
	ID        artifact.ID `json:"-"`
}

var approvalCodec = artifact.JSONDocumentCodec(
	"approval request", artifact.KindEvidence, approvalMediaType, approvalSchema,
	canonicalizeApproval,
	func(value ApprovalRequest) artifact.ID { return value.ID },
	func(value *ApprovalRequest, id artifact.ID) { value.ID = id },
	func(value ApprovalRequest) ApprovalRequest {
		value.Arguments = slices.Clone(value.Arguments)
		return value
	},
)

// NewApprovalRequest identifies exact operation recovery work.
func NewApprovalRequest(operation, recipe artifact.ID, action Action, prior artifact.ID) (ApprovalRequest, error) {
	return approvalCodec.New(ApprovalRequest{
		Version: artifact.InitialDocumentVersion, Operation: operation, Recipe: recipe,
		Tool: action.Code, Arguments: slices.Clone(action.Argv), Prior: prior,
	})
}

// Content returns the durable approval request.
func (request ApprovalRequest) Content() (artifact.Content, error) {
	return approvalCodec.Content(request)
}

// Binds reports exact action identity and arguments.
func (request ApprovalRequest) Binds(action Action) bool {
	return action.Validate() == nil && request.Tool == action.Code && slices.Equal(request.Arguments, action.Argv)
}

func canonicalizeApproval(value *ApprovalRequest) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Operation.Kind() != artifact.KindEvidence || value.Recipe.Kind() != artifact.KindRecipe ||
		!bounded(value.Tool) || strings.ContainsAny(value.Tool, " \t") ||
		len(value.Arguments) == 0 || len(value.Arguments) > maxArguments ||
		(value.Prior.Valid() && value.Prior.Kind() != artifact.KindEvidence) {
		return errors.New("operator action: invalid approval request")
	}
	for _, argument := range value.Arguments {
		if argument == "" || len(argument) > maxTextBytes || strings.ContainsAny(argument, "\x00\r\n") {
			return errors.New("operator action: invalid approval arguments")
		}
	}
	value.Arguments = slices.Clone(value.Arguments)
	return nil
}
