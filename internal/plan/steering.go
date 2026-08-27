package plan

import (
	"context"
	"errors"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

// AdmitSteeringProposal validates every referenced artifact and compiles rows
// only through operator-owned verifier bindings. Proposal text is inert.
func AdmitSteeringProposal(ctx context.Context, reader artifact.Reader, current Plan, proposal recipe.SteeringProposal, verifierBindings map[artifact.ID]string) (Plan, error) {
	if ctx == nil || reader == nil {
		return Plan{}, errors.New("plan: steering admission authority is absent")
	}
	if err := Validate(current); err != nil {
		return Plan{}, err
	}
	if err := proposal.ValidateIdentity(); err != nil {
		return Plan{}, err
	}
	for _, id := range append(append([]artifact.ID{proposal.Evaluation}, proposal.AffectedAuthorities...), proposal.Measurements...) {
		if _, found, err := reader.Artifact(ctx, id); err != nil || !found {
			return Plan{}, errors.Join(errors.New("plan: steering reference is absent"), err)
		}
	}
	itemIndex := map[string]int{}
	for index, item := range current.Items {
		itemIndex[item.ID] = index
	}
	for _, row := range proposal.Rows {
		verify, bound := verifierBindings[row.Verifier]
		if !bound || strings.TrimSpace(verify) == "" {
			return Plan{}, errors.New("plan: steering verifier has no deterministic binding")
		}
		if _, found, err := reader.Artifact(ctx, row.Verifier); err != nil || !found {
			return Plan{}, errors.Join(errors.New("plan: steering verifier authority is absent"), err)
		}
		index, found := itemIndex[row.Item]
		if !found {
			current.Items = append(current.Items, Item{ID: row.Item, Title: proposal.Goal, Status: StatusOpen})
			index = len(current.Items) - 1
			itemIndex[row.Item] = index
		}
		for _, existing := range current.Items[index].Steps {
			if existing.ID == row.Step {
				return Plan{}, errors.New("plan: steering row duplicates existing work")
			}
		}
		current.Items[index].Steps = append(current.Items[index].Steps, Step{ID: row.Step, Title: row.Title, Status: StatusOpen, Verify: verify, Rationale: "admitted steering proposal " + proposal.ID.String(), Capabilities: append([]string(nil), row.Capabilities...)})
	}
	return current, Validate(current)
}
