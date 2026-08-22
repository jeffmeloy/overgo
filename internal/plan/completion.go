package plan

import (
	"errors"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

const (
	completionReceiptMediaType = "application/vnd.overgo.plan-completion+json"
	completionReceiptSchema    = "overgo/plan-completion/v1"
)

// CompletionReceipt binds a removed open-work row to the exact successful
// gate and Git commit that completed it.
type CompletionReceipt struct {
	Version    uint16      `json:"version"`
	Item       string      `json:"item"`
	Step       string      `json:"step"`
	Owner      string      `json:"owner,omitempty"`
	Authority  artifact.ID `json:"authority"`
	CodeCommit string      `json:"code_commit"`
	GateResult artifact.ID `json:"gate_result"`
	ID         artifact.ID `json:"-"`
}

var completionReceiptCodec = artifact.JSONDocumentCodec(
	"plan completion receipt", artifact.KindEvidence, completionReceiptMediaType, completionReceiptSchema,
	canonicalizeCompletionReceipt, func(value CompletionReceipt) artifact.ID { return value.ID },
	func(value *CompletionReceipt, id artifact.ID) { value.ID = id }, func(value CompletionReceipt) CompletionReceipt { return value },
)

// NewCompletionReceipt constructs exact post-commit completion evidence.
func NewCompletionReceipt(item, step, owner string, authority artifact.ID, codeCommit string, gateResult artifact.ID) (CompletionReceipt, error) {
	return completionReceiptCodec.New(CompletionReceipt{
		Version: artifact.InitialDocumentVersion, Item: item, Step: step, Owner: owner,
		Authority: authority, CodeCommit: codeCommit, GateResult: gateResult,
	})
}

func canonicalizeCompletionReceipt(receipt *CompletionReceipt) error {
	if receipt == nil || receipt.Version != artifact.InitialDocumentVersion ||
		!textcheck.Bounded(receipt.Item, automationRoleMaxBytes, "\x00\r\n/") ||
		!textcheck.Bounded(receipt.Step, automationRoleMaxBytes, "\x00\r\n/") ||
		receipt.Owner != "" && !textcheck.Bounded(receipt.Owner, automationRoleMaxBytes, "\x00\r\n") ||
		receipt.Authority.Kind() != artifact.KindEvidence || receipt.GateResult.Kind() != artifact.KindEvidence ||
		!validCommit(strings.TrimSpace(receipt.CodeCommit)) {
		return errors.New("plan: invalid completion receipt")
	}
	receipt.CodeCommit = strings.TrimSpace(receipt.CodeCommit)
	return nil
}

// Content encodes the immutable completion receipt.
func (receipt CompletionReceipt) Content() (artifact.Content, error) {
	return completionReceiptCodec.Content(receipt)
}

// Lineage binds the receipt to its preparation authority and gate result.
func (receipt CompletionReceipt) Lineage() []artifact.Lineage {
	return []artifact.Lineage{
		{Child: receipt.ID, Parent: receipt.Authority, Relation: artifact.RelationDependsOn},
		{Child: receipt.ID, Parent: receipt.GateResult, Relation: artifact.RelationDependsOn},
	}
}
