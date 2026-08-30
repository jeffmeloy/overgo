package runrecord

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"

	"overgo/internal/artifact"
)

const (
	driverDecisionMediaType = "application/vnd.overgo.driver-decision+json"
	driverDecisionSchema    = "overgo/driver-decision/v2"
	driverDecisionVersion   = artifact.SecondDocumentVersion
)

// DriverOptionState names the exact lifecycle boundary at which an incumbent
// or admitted candidate was considered. Eligibility is deliberately separate
// from a recipe lifecycle: an admission does not manufacture a recipe event.
type DriverOptionState string

const (
	// DriverIncumbentActive identifies the currently serving recipe.
	DriverIncumbentActive DriverOptionState = "incumbent-active"
	// DriverCandidateEligible identifies an admitted, unrealized candidate.
	DriverCandidateEligible DriverOptionState = "candidate-eligible"
	// DriverCandidateCreated identifies a realized candidate awaiting validation.
	DriverCandidateCreated DriverOptionState = "candidate-created"
	// DriverCandidateValidated identifies a candidate awaiting evaluation evidence.
	DriverCandidateValidated DriverOptionState = "candidate-validated"
	// DriverCandidateVerified identifies a candidate awaiting promotion.
	DriverCandidateVerified DriverOptionState = "candidate-verified"
	// DriverCandidateActive identifies a promoted candidate with placement.
	DriverCandidateActive DriverOptionState = "candidate-active"
	// DriverCandidateRefused identifies a terminal rejected candidate.
	DriverCandidateRefused DriverOptionState = "candidate-refused"
	// DriverCandidateSuperseded identifies a terminal replaced candidate.
	DriverCandidateSuperseded DriverOptionState = "candidate-superseded"
)

// DriverEvidenceNeed is the closed vocabulary for work that prevents an
// option from being selected. Target names the exact owner input for that work.
type DriverEvidenceNeed string

const (
	// DriverNeedRealization requests compilation of an admitted candidate.
	DriverNeedRealization DriverEvidenceNeed = "realization"
	// DriverNeedValidation requests structural recipe validation.
	DriverNeedValidation DriverEvidenceNeed = "validation"
	// DriverNeedEvaluation requests isolated performance evidence.
	DriverNeedEvaluation DriverEvidenceNeed = "evaluation"
	// DriverNeedAblation requests causal contribution evidence.
	DriverNeedAblation DriverEvidenceNeed = "ablation"
	// DriverNeedPlacement requests a non-serving or serving placement.
	DriverNeedPlacement DriverEvidenceNeed = "placement"
	// DriverNeedPromotion requests an evidence-gated activation decision.
	DriverNeedPromotion DriverEvidenceNeed = "promotion"
)

// DriverEvidenceGap records explicit missing evidence and its exact measured
// resource cost. CostAuthority names the immutable owner of that estimate.
type DriverEvidenceGap struct {
	Need          DriverEvidenceNeed `json:"need"`
	Target        artifact.ID        `json:"target"`
	CostUnits     uint64             `json:"cost_units"`
	CostUnit      string             `json:"cost_unit"`
	CostAuthority artifact.ID        `json:"cost_authority"`
}

// DriverOption is the package-neutral projection of one considered incumbent
// or admitted cross-domain candidate. The owning adapters must reverify the
// named admission, lifecycle event, and placement before acting.
type DriverOption struct {
	Candidate artifact.ID         `json:"candidate,omitzero"`
	Admission artifact.ID         `json:"admission,omitzero"`
	Recipe    artifact.ID         `json:"recipe,omitzero"`
	Lifecycle artifact.ID         `json:"lifecycle,omitzero"`
	State     DriverOptionState   `json:"state"`
	Placement artifact.ID         `json:"placement,omitzero"`
	Evidence  []artifact.ID       `json:"evidence,omitempty"`
	Missing   []DriverEvidenceGap `json:"missing,omitempty"`
}

// DriverBudgetState is a materialized balance over one exact immutable grant
// and its exact charge set. Remaining is always recomputed from Issued and
// Consumed; the constructor derives all three values through BudgetBalance.
type DriverBudgetState struct {
	Grant     artifact.ID   `json:"grant"`
	Charges   []artifact.ID `json:"charges,omitempty"`
	Unit      string        `json:"unit"`
	Issued    uint64        `json:"issued"`
	Consumed  uint64        `json:"consumed,omitzero"`
	Remaining uint64        `json:"remaining,omitzero"`
}

// DriverBudgetReference is the wire-safe input to budget derivation. The grant
// is reloaded by exact identity and its complete committed BudgetCharge child
// set is derived from lineage; callers cannot omit charges from the balance.
type DriverBudgetReference struct {
	Grant artifact.ID `json:"grant"`
}

// DriverStopState is the derived terminal condition for one decision.
type DriverStopState string

const (
	// DriverStopContinue keeps the supervised decision loop open.
	DriverStopContinue DriverStopState = "continue"
	// DriverStopOperator records an explicit operator stop.
	DriverStopOperator DriverStopState = "operator-stop"
	// DriverStopBudgetExhausted records a derived exhausted budget.
	DriverStopBudgetExhausted DriverStopState = "budget-exhausted"
	// DriverStopSaturated records evidence that further work has saturated.
	DriverStopSaturated DriverStopState = "saturated"
	// DriverStopRefused records an evidence-bound refusal.
	DriverStopRefused DriverStopState = "refused"
)

// DriverAction is the closed, evidence-only description of what the Go driver
// may ask an existing owner to do next. It grants no execution authority.
type DriverAction string

const (
	// DriverActionSelect selects an already active placement.
	DriverActionSelect DriverAction = "select"
	// DriverActionRealize requests candidate realization.
	DriverActionRealize DriverAction = "realize"
	// DriverActionEvaluate requests isolated evaluation.
	DriverActionEvaluate DriverAction = "evaluate"
	// DriverActionAblate requests contribution isolation.
	DriverActionAblate DriverAction = "ablate"
	// DriverActionRequestEvidence requests other typed missing evidence.
	DriverActionRequestEvidence DriverAction = "request-evidence"
	// DriverActionStop stops without granting mutation authority.
	DriverActionStop DriverAction = "stop"
	// DriverActionRefuse refuses further work.
	DriverActionRefuse DriverAction = "refuse"
)

// DriverTieBreak names the one deterministic ordering used when equal-cost
// work is present. Identity remains the final non-scalar tie-break.
type DriverTieBreak string

// DriverTieCostThenActionThenIdentity orders work by measured cost, then action
// class and artifact identity.
const DriverTieCostThenActionThenIdentity DriverTieBreak = "cost-then-action-priority-then-artifact-identity"

// DriverDecisionFacts are exact inputs to the sole decision constructor.
// Derived fields are intentionally absent, so callers cannot assert a winner,
// stop condition, budget balance, or action.
type DriverDecisionFacts struct {
	Goal               artifact.ID           `json:"goal"`
	Causal             CausalContext         `json:"causal"`
	Head               artifact.CommitID     `json:"head"`
	Options            []DriverOption        `json:"options"`
	InteractionBudget  DriverBudgetReference `json:"interaction_budget"`
	ResourceBudget     DriverBudgetReference `json:"resource_budget"`
	SaturationEvidence []artifact.ID         `json:"saturation_evidence,omitempty"`
	OperatorStop       artifact.ID           `json:"operator_stop,omitzero"`
	Refusal            artifact.ID           `json:"refusal,omitzero"`
}

// DriverDecision is the canonical immutable fact for what the RSI driver does
// next. It records evidence and a derived request; lifecycle, placement,
// promotion, and execution remain with their existing owners.
type DriverDecision struct {
	Version            uint16             `json:"version"`
	Goal               artifact.ID        `json:"goal"`
	Causal             CausalContext      `json:"causal"`
	Head               artifact.CommitID  `json:"head"`
	Incumbent          artifact.ID        `json:"incumbent"`
	Options            []DriverOption     `json:"options"`
	Interaction        DriverBudgetState  `json:"interaction"`
	Resources          DriverBudgetState  `json:"resources"`
	SaturationEvidence []artifact.ID      `json:"saturation_evidence,omitempty"`
	OperatorStop       artifact.ID        `json:"operator_stop,omitzero"`
	Refusal            artifact.ID        `json:"refusal,omitzero"`
	Stop               DriverStopState    `json:"stop"`
	Action             DriverAction       `json:"action"`
	Subject            artifact.ID        `json:"subject,omitzero"`
	Target             artifact.ID        `json:"target,omitzero"`
	Need               DriverEvidenceNeed `json:"need,omitzero"`
	TieBreak           DriverTieBreak     `json:"tie_break"`
	ID                 artifact.ID        `json:"-"`
}

var driverDecisionCodec = artifact.JSONDocumentCodec(
	"driver decision", artifact.KindEvidence, driverDecisionMediaType, driverDecisionSchema,
	canonicalizeDriverDecision,
	func(value DriverDecision) artifact.ID { return value.ID },
	func(value *DriverDecision, id artifact.ID) { value.ID = id },
	cloneDriverDecision,
)

// NewDriverDecision reloads exact budget authorities, then derives,
// canonicalizes, and identifies one driver fact.
func NewDriverDecision(ctx context.Context, reader artifact.Reader, facts DriverDecisionFacts) (DriverDecision, error) {
	return newDriverDecision(ctx, reader, facts)
}

func newDriverDecision(ctx context.Context, reader artifact.Reader, facts DriverDecisionFacts) (DriverDecision, error) {
	interaction, err := deriveDriverBudgetState(ctx, reader, facts.InteractionBudget)
	if err != nil {
		return DriverDecision{}, err
	}
	resources, err := deriveDriverBudgetState(ctx, reader, facts.ResourceBudget)
	if err != nil {
		return DriverDecision{}, err
	}
	causal := cloneCausal(&facts.Causal)
	return driverDecisionCodec.New(DriverDecision{
		Version: driverDecisionVersion, Goal: facts.Goal, Causal: *causal, Head: facts.Head,
		Options: cloneDriverOptions(facts.Options), Interaction: interaction, Resources: resources,
		SaturationEvidence: slices.Clone(facts.SaturationEvidence), OperatorStop: facts.OperatorStop,
		Refusal: facts.Refusal, TieBreak: DriverTieCostThenActionThenIdentity,
	})
}

// RequireDriverDecision loads one persisted decision and re-derives it from
// the complete current charge sets of its exact budget grants. A decision
// ceases to be executable as soon as either budget ledger changes.
func RequireDriverDecision(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (DriverDecision, error) {
	decision, err := RequireRecordedDriverDecision(ctx, reader, id)
	if err != nil {
		return DriverDecision{}, err
	}
	replayed, err := newDriverDecision(ctx, reader, DriverDecisionFacts{
		Goal: decision.Goal, Causal: decision.Causal, Head: decision.Head,
		Options:            cloneDriverOptions(decision.Options),
		InteractionBudget:  DriverBudgetReference{Grant: decision.Interaction.Grant},
		ResourceBudget:     DriverBudgetReference{Grant: decision.Resources.Grant},
		SaturationEvidence: slices.Clone(decision.SaturationEvidence),
		OperatorStop:       decision.OperatorStop, Refusal: decision.Refusal,
	})
	if err != nil {
		return DriverDecision{}, err
	}
	if replayed.ID != decision.ID {
		return DriverDecision{}, fmt.Errorf("run record: driver decision is stale: %s", id)
	}
	return replayed, nil
}

// RequireRecordedDriverDecision loads the immutable decision with its exact
// stored authority lineage without asserting that its budget ledgers are still
// executable. Consumers replaying historical closure use this after the
// decision's own charge has necessarily made RequireDriverDecision stale.
func RequireRecordedDriverDecision(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (DriverDecision, error) {
	return driverDecisionCodec.RequireExactLineage(ctx, reader, id, DriverDecision.Lineage)
}

// Content returns the canonical driver-decision document.
func (value DriverDecision) Content() (artifact.Content, error) {
	return driverDecisionCodec.Content(value)
}

// Lineage binds every exact authority considered by the decision. Causal
// projection remains separate and is added by the common controller batch.
func (value DriverDecision) Lineage() []artifact.Lineage {
	parents := []artifact.ID{
		value.Goal, value.Incumbent, value.Interaction.Grant, value.Resources.Grant,
		value.OperatorStop, value.Refusal,
	}
	parents = append(parents, value.Interaction.Charges...)
	parents = append(parents, value.Resources.Charges...)
	parents = append(parents, value.SaturationEvidence...)
	for _, option := range value.Options {
		parents = append(parents, option.Candidate, option.Admission, option.Recipe, option.Lifecycle)
		parents = append(parents, option.Evidence...)
		for _, gap := range option.Missing {
			parents = append(parents, gap.Target, gap.CostAuthority)
		}
	}
	parents = slices.DeleteFunc(parents, func(id artifact.ID) bool { return !id.Valid() })
	slices.SortFunc(parents, artifact.CompareID)
	return artifact.DependencyLineage(value.ID, slices.Compact(parents)...)
}

func deriveDriverBudgetState(
	ctx context.Context,
	reader artifact.Reader,
	reference DriverBudgetReference,
) (DriverBudgetState, error) {
	if ctx == nil || reader == nil || reference.Grant.Kind() != artifact.KindEvidence {
		return DriverBudgetState{}, errors.New("run record: driver budget authority is absent")
	}
	budget, err := RequireBudget(ctx, reader, reference.Grant)
	if err != nil {
		return DriverBudgetState{}, err
	}
	edges, err := reader.Children(ctx, reference.Grant)
	if err != nil {
		return DriverBudgetState{}, err
	}
	chargeSet := make(map[artifact.ID]struct{})
	for _, edge := range edges {
		if edge.Parent != reference.Grant || edge.Relation != artifact.RelationDependsOn {
			continue
		}
		descriptor, found, err := reader.Artifact(ctx, edge.Child)
		if err != nil {
			return DriverBudgetState{}, err
		}
		if !found {
			return DriverBudgetState{}, errors.New("run record: driver budget child is absent")
		}
		if descriptor.MediaType == BudgetChargeMediaType && descriptor.Schema == BudgetChargeSchema {
			chargeSet[edge.Child] = struct{}{}
		}
	}
	chargeIDs := slices.Collect(maps.Keys(chargeSet))
	if err := canonicalIDs(chargeIDs); err != nil {
		return DriverBudgetState{}, errors.Join(errors.New("run record: invalid driver budget charges"), err)
	}
	charges := make([]BudgetCharge, len(chargeIDs))
	for index, id := range chargeIDs {
		charges[index], err = budgetChargeCodec.Require(ctx, reader, id)
		if err != nil {
			return DriverBudgetState{}, err
		}
	}
	remaining, err := BudgetBalance(budget, charges)
	if err != nil {
		return DriverBudgetState{}, err
	}
	return DriverBudgetState{
		Grant: budget.ID, Charges: chargeIDs, Unit: budget.Unit, Issued: budget.Issued,
		Consumed: budget.Issued - remaining, Remaining: remaining,
	}, nil
}

func canonicalizeDriverDecision(value *DriverDecision) error {
	if value == nil || value.Version != driverDecisionVersion || !value.Goal.Valid() || !value.Head.Valid() ||
		value.Causal.Validate() != nil || value.Interaction.Grant == value.Resources.Grant ||
		len(value.Options) == 0 || len(value.Options) > MaximumAttemptPopulation ||
		value.TieBreak != DriverTieCostThenActionThenIdentity {
		return errors.New("run record: invalid driver decision")
	}
	if err := canonicalizeDriverBudgetState(&value.Interaction); err != nil {
		return err
	}
	if err := canonicalizeDriverBudgetState(&value.Resources); err != nil {
		return err
	}
	if value.OperatorStop.Valid() && value.OperatorStop.Kind() != artifact.KindEvidence ||
		value.Refusal.Valid() && value.Refusal.Kind() != artifact.KindEvidence ||
		value.OperatorStop.Valid() && value.Refusal.Valid() {
		return errors.New("run record: invalid driver stop evidence")
	}
	value.SaturationEvidence = slices.Clone(value.SaturationEvidence)
	if err := canonicalIDs(value.SaturationEvidence); err != nil && len(value.SaturationEvidence) != 0 {
		return errors.Join(errors.New("run record: invalid driver saturation evidence"), err)
	}
	value.Options = cloneDriverOptions(value.Options)
	slices.SortFunc(value.Options, compareDriverOptions)
	seen := make(map[artifact.ID]bool, len(value.Options))
	incumbents := 0
	value.Incumbent = artifact.ID{}
	for index := range value.Options {
		if err := canonicalizeDriverOption(&value.Options[index], value.Resources.Unit); err != nil {
			return err
		}
		subject := driverOptionSubject(value.Options[index])
		if seen[subject] {
			return errors.New("run record: duplicate driver option")
		}
		seen[subject] = true
		if value.Options[index].State == DriverIncumbentActive {
			incumbents++
			value.Incumbent = value.Options[index].Recipe
		}
	}
	if incumbents != 1 {
		return errors.New("run record: driver decision requires one active incumbent")
	}
	value.Stop, value.Action, value.Subject, value.Target, value.Need = deriveDriverSelection(*value)
	return nil
}

func canonicalizeDriverBudgetState(value *DriverBudgetState) error {
	if value == nil || value.Grant.Kind() != artifact.KindEvidence || value.Unit == "" || !validUnit(value.Unit) ||
		value.Issued == 0 || value.Consumed > value.Issued {
		return errors.New("run record: invalid driver budget state")
	}
	value.Charges = slices.Clone(value.Charges)
	if err := canonicalIDs(value.Charges); err != nil && len(value.Charges) != 0 {
		return errors.Join(errors.New("run record: invalid driver budget charge identities"), err)
	}
	if len(value.Charges) == 0 && value.Consumed != 0 || len(value.Charges) != 0 && value.Consumed == 0 {
		return errors.New("run record: driver budget charge total differs")
	}
	value.Remaining = value.Issued - value.Consumed
	return nil
}

func canonicalizeDriverOption(value *DriverOption, resourceUnit string) error {
	if value == nil {
		return errors.New("run record: invalid driver option")
	}
	validState := false
	switch value.State {
	case DriverIncumbentActive, DriverCandidateEligible, DriverCandidateCreated, DriverCandidateValidated,
		DriverCandidateVerified, DriverCandidateActive, DriverCandidateRefused, DriverCandidateSuperseded:
		validState = true
	}
	if !validState || value.Placement.Valid() && value.Placement.Kind() != artifact.KindProfile {
		return errors.New("run record: invalid driver option")
	}
	incumbent := value.State == DriverIncumbentActive
	if incumbent {
		if value.Candidate.Valid() || value.Admission.Valid() || value.Recipe.Kind() != artifact.KindRecipe ||
			value.Lifecycle.Kind() != artifact.KindEvidence || !value.Placement.Valid() || len(value.Missing) != 0 {
			return errors.New("run record: invalid incumbent driver option")
		}
	} else if value.State == DriverCandidateEligible {
		if value.Candidate.Kind() != artifact.KindRecipe || value.Admission.Kind() != artifact.KindEvidence ||
			value.Recipe.Valid() || value.Lifecycle.Valid() || value.Placement.Valid() ||
			len(value.Missing) != 1 || value.Missing[0].Need != DriverNeedRealization || value.Missing[0].Target != value.Candidate {
			return errors.New("run record: eligible candidate claimed realized state")
		}
	} else if value.Candidate.Kind() != artifact.KindRecipe || value.Admission.Kind() != artifact.KindEvidence ||
		value.Recipe.Kind() != artifact.KindRecipe || value.Lifecycle.Kind() != artifact.KindEvidence ||
		value.Candidate == value.Recipe {
		return errors.New("run record: candidate option lacks exact admission or lifecycle subject")
	} else if value.State == DriverCandidateActive && (!value.Placement.Valid() || len(value.Missing) != 0) {
		return errors.New("run record: active candidate lacks exact placement")
	} else if (value.State == DriverCandidateCreated || value.State == DriverCandidateValidated ||
		value.State == DriverCandidateVerified) && len(value.Missing) == 0 {
		return errors.New("run record: non-active candidate omits its missing work")
	} else if (value.State == DriverCandidateRefused || value.State == DriverCandidateSuperseded) && len(value.Missing) != 0 {
		return errors.New("run record: terminal candidate carries executable work")
	}
	value.Evidence = slices.Clone(value.Evidence)
	if err := canonicalIDs(value.Evidence); err != nil && len(value.Evidence) != 0 {
		return errors.Join(errors.New("run record: invalid driver option evidence"), err)
	}
	value.Missing = slices.Clone(value.Missing)
	slices.SortFunc(value.Missing, compareDriverEvidenceGaps)
	for index, gap := range value.Missing {
		if !driverEvidenceNeedValid(gap.Need) || !driverOptionNeedAllowed(value.State, gap.Need) ||
			!gap.Target.Valid() || gap.CostUnits == 0 || gap.CostUnit != resourceUnit ||
			!driverGapAuthorityAllowed(*value, gap.CostAuthority) || slices.Contains(value.Evidence, gap.Target) ||
			index > 0 && gap.Need == value.Missing[index-1].Need && gap.Target == value.Missing[index-1].Target {
			return errors.New("run record: invalid or duplicate driver evidence gap")
		}
	}
	return nil
}

func deriveDriverSelection(value DriverDecision) (DriverStopState, DriverAction, artifact.ID, artifact.ID, DriverEvidenceNeed) {
	switch {
	case value.OperatorStop.Valid():
		return DriverStopOperator, DriverActionStop, artifact.ID{}, value.OperatorStop, ""
	case value.Refusal.Valid():
		return DriverStopRefused, DriverActionRefuse, artifact.ID{}, value.Refusal, ""
	case value.Interaction.Remaining == 0 || value.Resources.Remaining == 0:
		return DriverStopBudgetExhausted, DriverActionRefuse, artifact.ID{}, artifact.ID{}, ""
	case len(value.SaturationEvidence) != 0:
		return DriverStopSaturated, DriverActionStop, artifact.ID{}, value.SaturationEvidence[0], ""
	}
	var selected DriverOption
	var gap DriverEvidenceGap
	pending, found := false, false
	for _, option := range value.Options {
		if option.State == DriverCandidateRefused || option.State == DriverCandidateSuperseded {
			continue
		}
		for _, candidate := range option.Missing {
			pending = true
			if candidate.CostUnits > value.Resources.Remaining {
				continue
			}
			if !found || compareDriverWork(candidate, driverOptionSubject(option), gap, driverOptionSubject(selected)) < 0 {
				selected, gap, found = option, candidate, true
			}
		}
	}
	if found {
		action := DriverActionRequestEvidence
		switch gap.Need {
		case DriverNeedRealization:
			action = DriverActionRealize
		case DriverNeedEvaluation:
			action = DriverActionEvaluate
		case DriverNeedAblation:
			action = DriverActionAblate
		}
		return DriverStopContinue, action, driverOptionSubject(selected), gap.Target, gap.Need
	}
	if pending {
		return DriverStopBudgetExhausted, DriverActionRefuse, artifact.ID{}, artifact.ID{}, ""
	}
	for _, option := range value.Options {
		if option.State == DriverCandidateActive {
			return DriverStopContinue, DriverActionSelect, driverOptionSubject(option), option.Placement, ""
		}
	}
	return DriverStopContinue, DriverActionSelect, value.Incumbent, driverIncumbentPlacement(value.Options), ""
}

func compareDriverWork(left DriverEvidenceGap, leftSubject artifact.ID, right DriverEvidenceGap, rightSubject artifact.ID) int {
	return cmp.Or(
		cmp.Compare(left.CostUnits, right.CostUnits),
		cmp.Compare(driverEvidencePriority(left.Need), driverEvidencePriority(right.Need)),
		artifact.CompareID(leftSubject, rightSubject), artifact.CompareID(left.Target, right.Target),
		artifact.CompareID(left.CostAuthority, right.CostAuthority),
	)
}

func compareDriverOptions(left, right DriverOption) int {
	return cmp.Or(artifact.CompareID(driverOptionSubject(left), driverOptionSubject(right)), cmp.Compare(left.State, right.State))
}

func compareDriverEvidenceGaps(left, right DriverEvidenceGap) int {
	return cmp.Or(
		cmp.Compare(driverEvidencePriority(left.Need), driverEvidencePriority(right.Need)),
		artifact.CompareID(left.Target, right.Target), cmp.Compare(left.CostUnits, right.CostUnits),
		artifact.CompareID(left.CostAuthority, right.CostAuthority),
	)
}

func driverGapAuthorityAllowed(option DriverOption, authority artifact.ID) bool {
	if !authority.Valid() {
		return false
	}
	return authority == option.Candidate || authority == option.Admission || authority == option.Recipe ||
		authority == option.Lifecycle || slices.Contains(option.Evidence, authority) ||
		slices.ContainsFunc(option.Missing, func(gap DriverEvidenceGap) bool { return gap.Target == authority })
}

func driverEvidencePriority(value DriverEvidenceNeed) int {
	return slices.Index([]DriverEvidenceNeed{
		DriverNeedRealization, DriverNeedValidation, DriverNeedEvaluation,
		DriverNeedAblation, DriverNeedPlacement, DriverNeedPromotion,
	}, value)
}

func driverEvidenceNeedValid(value DriverEvidenceNeed) bool {
	return driverEvidencePriority(value) >= 0
}

func driverOptionNeedAllowed(state DriverOptionState, need DriverEvidenceNeed) bool {
	switch state {
	case DriverCandidateEligible:
		return need == DriverNeedRealization
	case DriverCandidateCreated:
		return need == DriverNeedValidation || need == DriverNeedPlacement
	case DriverCandidateValidated:
		return need == DriverNeedEvaluation || need == DriverNeedAblation || need == DriverNeedPlacement
	case DriverCandidateVerified:
		return need == DriverNeedPromotion || need == DriverNeedPlacement
	default:
		return false
	}
}

func driverOptionSubject(value DriverOption) artifact.ID {
	if value.Candidate.Valid() {
		return value.Candidate
	}
	return value.Recipe
}

func driverIncumbentPlacement(options []DriverOption) artifact.ID {
	for _, option := range options {
		if option.State == DriverIncumbentActive {
			return option.Placement
		}
	}
	return artifact.ID{}
}

func cloneDriverOptions(values []DriverOption) []DriverOption {
	cloned := slices.Clone(values)
	for index := range cloned {
		cloned[index].Evidence = slices.Clone(cloned[index].Evidence)
		cloned[index].Missing = slices.Clone(cloned[index].Missing)
	}
	return cloned
}

func cloneDriverDecision(value DriverDecision) DriverDecision {
	value.Causal = *cloneCausal(&value.Causal)
	value.Options = cloneDriverOptions(value.Options)
	value.Interaction.Charges = slices.Clone(value.Interaction.Charges)
	value.Resources.Charges = slices.Clone(value.Resources.Charges)
	value.SaturationEvidence = slices.Clone(value.SaturationEvidence)
	return value
}
