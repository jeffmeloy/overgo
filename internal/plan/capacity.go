package plan

import (
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/textcheck"
)

// SlotCost is a row's declared reservation on one lane, in device and host
// units; a zero cost reserves nothing.
type SlotCost struct {
	Device int `json:"device,omitzero"`
	Host   int `json:"host,omitzero"`
}

// LabelRequirement is a row's demand on a lane label: required, or weighted
// preference when not required.
type LabelRequirement struct {
	Label    string `json:"label"`
	Required bool   `json:"required,omitzero"`
	Weight   int    `json:"weight,omitzero"`
}

// LaneCapacity is one lane's declaration: its labels and the device and
// host units it can hold at once.
type LaneCapacity struct {
	Lane     string   `json:"lane"`
	Labels   []string `json:"labels,omitempty"`
	Capacity SlotCost `json:"capacity"`
}

// SlotDemand is one row's declared cost and labels for assignment.
type SlotDemand struct {
	Row    Ref
	Cost   SlotCost
	Labels []LabelRequirement
}

// SlotAssignment is the scheduler's verdict for one row: the lane charged,
// or the reason the row waits.
type SlotAssignment struct {
	Row   Ref    `json:"row"`
	Lane  string `json:"lane,omitzero"`
	Waits string `json:"waits,omitzero"`
}

func (cost SlotCost) fits(free SlotCost) bool {
	return cost.Device <= free.Device && cost.Host <= free.Host
}

func (cost SlotCost) minus(other SlotCost) SlotCost {
	return SlotCost{Device: cost.Device - other.Device, Host: cost.Host - other.Host}
}

func validateSlotCost(cost SlotCost) error {
	if cost.Device < 0 || cost.Host < 0 {
		return errors.New("slot cost units must be non-negative")
	}
	return nil
}

func validateLabelRequirements(labels []LabelRequirement) error {
	seen := map[string]bool{}
	for _, label := range labels {
		if !textcheck.LowerIdentifier(label.Label, automationRoleMaxBytes) || seen[label.Label] {
			return fmt.Errorf("label %q is not a unique lower identifier", label.Label)
		}
		if label.Weight < 0 {
			return fmt.Errorf("label %q has a negative weight", label.Label)
		}
		seen[label.Label] = true
	}
	return nil
}

// validateLanes: unique lane names, lower-identifier labels, non-negative capacity.
func validateLanes(lanes []LaneCapacity) error {
	seen := map[string]bool{}
	for _, lane := range lanes {
		if !validAutomationText(lane.Lane) || lane.Lane == "" || seen[lane.Lane] {
			return fmt.Errorf("plan lane %q is not a unique name", lane.Lane)
		}
		seen[lane.Lane] = true
		for _, label := range lane.Labels {
			if !textcheck.LowerIdentifier(label, automationRoleMaxBytes) {
				return fmt.Errorf("plan lane %s label %q is not a lower identifier", lane.Lane, label)
			}
		}
		if err := validateSlotCost(lane.Capacity); err != nil {
			return fmt.Errorf("plan lane %s: %w", lane.Lane, err)
		}
	}
	return nil
}

// AssignSlots charges each demand, in order, on the first lane that holds
// every required label and has free capacity for its cost; a row whose
// required label no lane holds, or whose cost exceeds every labelled lane's
// free capacity, waits with that reason. Reserved holds the units already
// charged per lane before this pass.
func AssignSlots(lanes []LaneCapacity, reserved map[string]SlotCost, demands []SlotDemand) ([]SlotAssignment, error) {
	if err := validateLanes(lanes); err != nil {
		return nil, err
	}
	free := make(map[string]SlotCost, len(lanes))
	for _, lane := range lanes {
		charged := reserved[lane.Lane]
		if err := validateSlotCost(charged); err != nil {
			return nil, fmt.Errorf("plan lane %s reservation: %w", lane.Lane, err)
		}
		free[lane.Lane] = lane.Capacity.minus(charged)
	}
	assignments := make([]SlotAssignment, 0, len(demands))
	for _, demand := range demands {
		if err := validateSlotCost(demand.Cost); err != nil {
			return nil, fmt.Errorf("plan row %s: %w", demand.Row, err)
		}
		if err := validateLabelRequirements(demand.Labels); err != nil {
			return nil, fmt.Errorf("plan row %s: %w", demand.Row, err)
		}
		assignment := SlotAssignment{Row: demand.Row}
		labelled := 0
		for _, lane := range lanes {
			if missing := missingRequiredLabel(lane, demand.Labels); missing != "" {
				continue
			}
			labelled++
			if demand.Cost.fits(free[lane.Lane]) {
				assignment.Lane = lane.Lane
				free[lane.Lane] = free[lane.Lane].minus(demand.Cost)
				break
			}
		}
		if assignment.Lane == "" {
			if labelled == 0 {
				assignment.Waits = "required label " + requiredLabels(demand.Labels) + " is held by no lane"
			} else {
				assignment.Waits = fmt.Sprintf("cost device=%d host=%d exceeds the free capacity of every labelled lane", demand.Cost.Device, demand.Cost.Host)
			}
		}
		assignments = append(assignments, assignment)
	}
	return assignments, nil
}

// missingRequiredLabel: the first required label the lane lacks, or "".
func missingRequiredLabel(lane LaneCapacity, labels []LabelRequirement) string {
	for _, label := range labels {
		if label.Required && !slices.Contains(lane.Labels, label.Label) {
			return label.Label
		}
	}
	return ""
}

func requiredLabels(labels []LabelRequirement) string {
	var required []string
	for _, label := range labels {
		if label.Required {
			required = append(required, label.Label)
		}
	}
	return strings.Join(required, ",")
}

// FormatSlotAssignments renders each assignment as one line.
func FormatSlotAssignments(assignments []SlotAssignment) string {
	lines := make([]string, 0, len(assignments))
	for _, assignment := range assignments {
		if assignment.Lane != "" {
			lines = append(lines, fmt.Sprintf("lane: %s -> %s", assignment.Row, assignment.Lane))
			continue
		}
		lines = append(lines, fmt.Sprintf("lane: %s waits: %s", assignment.Row, assignment.Waits))
	}
	return strings.Join(lines, "\n")
}

// validateStepSlot: a row's declared cost and labels.
func validateStepSlot(step Step) error {
	if step.SlotCost != nil {
		if err := validateSlotCost(*step.SlotCost); err != nil {
			return err
		}
	}
	return validateLabelRequirements(step.Labels)
}

// FrontierDemands collects the declared cost and labels of every frontier row, in
// frontier order; a row without a declaration demands nothing.
func FrontierDemands(d Plan, frontier []Ref) []SlotDemand {
	demands := make([]SlotDemand, 0, len(frontier))
	for _, ref := range frontier {
		demand := SlotDemand{Row: ref}
		if step, found := exactPlanStep(d, ref.Item, ref.Step); found {
			if step.SlotCost != nil {
				demand.Cost = *step.SlotCost
			}
			demand.Labels = step.Labels
		}
		demands = append(demands, demand)
	}
	return demands
}
