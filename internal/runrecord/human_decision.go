package runrecord

import (
	"context"
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/operatoraction"
)

const (
	// HumanDecisionMediaType identifies human decision documents.
	HumanDecisionMediaType = "application/vnd.overgo.human-decision+json"
	// HumanDecisionSchema identifies the human decision contract.
	HumanDecisionSchema    = "overgo/human-decision/v1"
	humanDecisionAliasRoot = "operations/decision/"
)

// HumanDecision records one exact approval outcome.
type HumanDecision struct {
	Version   uint16                `json:"version"`
	Request   artifact.ID           `json:"request"`
	Operation artifact.ID           `json:"operation"`
	Recipe    artifact.ID           `json:"recipe"`
	Tool      string                `json:"tool"`
	Arguments []string              `json:"arguments"`
	Answer    operatoraction.Answer `json:"answer"`
	Prior     artifact.ID           `json:"prior,omitzero"`
	ID        artifact.ID           `json:"-"`
}

var humanDecisionCodec = artifact.JSONDocumentCodec(
	"human decision", artifact.KindEvidence, HumanDecisionMediaType, HumanDecisionSchema,
	canonicalizeHumanDecision,
	func(value HumanDecision) artifact.ID { return value.ID },
	func(value *HumanDecision, id artifact.ID) { value.ID = id },
	func(value HumanDecision) HumanDecision {
		value.Arguments = slices.Clone(value.Arguments)
		return value
	},
)

// NewHumanDecision binds an answer to the exact approval request.
func NewHumanDecision(request operatoraction.ApprovalRequest, answer operatoraction.Answer) (HumanDecision, error) {
	return humanDecisionCodec.New(HumanDecision{
		Version: artifact.InitialDocumentVersion, Request: request.ID,
		Operation: request.Operation, Recipe: request.Recipe, Tool: request.Tool,
		Arguments: slices.Clone(request.Arguments), Answer: answer, Prior: request.Prior,
	})
}

// ResolveHumanDecision returns the latest decision for one operation.
func ResolveHumanDecision(ctx context.Context, reader artifact.Reader, operation artifact.ID) (HumanDecision, bool, error) {
	id, found, err := artifact.ResolveAlias(ctx, reader, humanDecisionAliasRoot+operation.String())
	if err != nil || !found {
		return HumanDecision{}, found, err
	}
	value, err := humanDecisionCodec.Require(ctx, reader, id)
	return value, err == nil, err
}

// PublishHumanDecision atomically advances one operation decision chain.
func PublishHumanDecision(
	ctx context.Context,
	repository artifact.Repository,
	request operatoraction.ApprovalRequest,
	decision HumanDecision,
) error {
	if ctx == nil || repository == nil || !decision.Binds(request) {
		return errors.New("run record: invalid human decision publication")
	}
	prior, found, err := ResolveHumanDecision(ctx, repository, request.Operation)
	if err != nil || found != request.Prior.Valid() || found && prior.ID != request.Prior {
		if err == nil {
			err = errors.New("run record: human decision chain differs")
		}
		return err
	}
	requestContent, err := request.Content()
	if err != nil {
		return err
	}
	decisionContent, err := humanDecisionCodec.Content(decision)
	if err != nil {
		return err
	}
	alias := artifact.AliasBinding{Name: humanDecisionAliasRoot + request.Operation.String(), Target: decision.ID}
	if found {
		alias.Previous = artifact.IDPointer(prior.ID)
	}
	parents := []artifact.ID{request.ID}
	if request.Prior.Valid() {
		parents = append(parents, request.Prior)
	}
	batch, err := artifact.NewDocumentBatch(
		"operation/decision/"+decision.ID.String(), []artifact.Content{requestContent, decisionContent},
		artifact.DependencyLineage(decision.ID, parents...),
		[]artifact.AliasBinding{alias},
	)
	if err != nil {
		return err
	}
	_, err = artifact.CommitBatch(ctx, repository, batch)
	return err
}

// Binds reports exact request, operation, recipe, tool, arguments, and prior decision.
func (decision HumanDecision) Binds(request operatoraction.ApprovalRequest) bool {
	return decision.Request == request.ID && decision.Operation == request.Operation &&
		decision.Recipe == request.Recipe && decision.Tool == request.Tool &&
		slices.Equal(decision.Arguments, request.Arguments) && decision.Prior == request.Prior
}

func canonicalizeHumanDecision(value *HumanDecision) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Request.Kind() != artifact.KindEvidence || value.Operation.Kind() != artifact.KindEvidence ||
		value.Recipe.Kind() != artifact.KindRecipe || value.Tool == "" ||
		strings.TrimSpace(value.Tool) != value.Tool || strings.ContainsAny(value.Tool, " \t\x00\r\n") || len(value.Arguments) == 0 ||
		(value.Answer != operatoraction.AnswerGrant && value.Answer != operatoraction.AnswerDecline) ||
		(value.Prior.Valid() && value.Prior.Kind() != artifact.KindEvidence) {
		return errors.New("run record: invalid human decision")
	}
	for _, argument := range value.Arguments {
		if argument == "" || strings.ContainsAny(argument, "\x00\r\n") {
			return errors.New("run record: invalid human decision arguments")
		}
	}
	value.Arguments = slices.Clone(value.Arguments)
	return nil
}
