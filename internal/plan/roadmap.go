package plan

import (
	"encoding/json"
	"fmt"
	"slices"
	"sort"
	"strconv"
	"strings"

	"overgo/internal/strictjson"
	"overgo/internal/testevidence"
)

type roadmapDocument struct {
	Campaign                json.RawMessage  `json:"campaign"`
	Revision                json.RawMessage  `json:"revision"`
	Doctrine                json.RawMessage  `json:"doctrine"`
	Thesis                  json.RawMessage  `json:"thesis"`
	ControlPlaneStatus      json.RawMessage  `json:"control_plane_status"`
	GoImplementationQuality json.RawMessage  `json:"go_implementation_quality"`
	OperatorSet             json.RawMessage  `json:"operator_set"`
	Metric                  json.RawMessage  `json:"metric"`
	Graph                   roadmapGraph     `json:"graph_contract"`
	StateContract           json.RawMessage  `json:"roadmap_state_contract"`
	VerifierLegend          json.RawMessage  `json:"verifier_status_legend"`
	Items                   []roadmapItem    `json:"items"`
	Execution               roadmapExecution `json:"execution_order"`
	OpenRisks               json.RawMessage  `json:"open_risks"`
}

type roadmapGraph struct {
	Authority              string                             `json:"authority"`
	Validation             string                             `json:"validation"`
	MandatoryPreconditions roadmapMandatoryPreconditions      `json:"mandatory_preconditions"`
	Implementation         roadmapImplementationPreconditions `json:"implementation_preconditions"`
}

type roadmapMandatoryPreconditions struct {
	Rule                  string              `json:"rule"`
	ByCapability          map[string][]string `json:"by_capability"`
	CapabilityDefinitions map[string]string   `json:"capability_definitions"`
}

type roadmapImplementationPreconditions struct {
	Rule                string   `json:"rule"`
	Requires            []string `json:"requires"`
	WhyNotDependsOn     string   `json:"why_not_depends_on"`
	BootstrapExceptions []string `json:"bootstrap_exceptions"`
	LandedExemption     string   `json:"landed_exemption"`
}

type roadmapItem struct {
	ID                    string          `json:"id"`
	Title                 string          `json:"title"`
	Status                string          `json:"status"`
	Rationale             string          `json:"rationale"`
	Constraint            json.RawMessage `json:"constraint"`
	PhaseCapabilities     json.RawMessage `json:"phase_capabilities"`
	PhaseCapabilitiesNote json.RawMessage `json:"phase_capabilities_note"`
	Steps                 []roadmapStep   `json:"steps"`
}

type roadmapStep struct {
	ID                     string          `json:"id"`
	Title                  string          `json:"title"`
	Status                 string          `json:"status"`
	Verify                 string          `json:"verify"`
	VerifierStatus         string          `json:"verifier_status"`
	DependsOn              []string        `json:"depends_on"`
	Capabilities           []string        `json:"capabilities"`
	ImplementationLanguage string          `json:"implementation_language"`
	Acceptance             json.RawMessage `json:"acceptance"`
	Bootstrap              json.RawMessage `json:"bootstrap"`
	Caution                json.RawMessage `json:"caution"`
	ClosesOperator         json.RawMessage `json:"closes_operator"`
	Constraint             json.RawMessage `json:"constraint"`
	Cost                   json.RawMessage `json:"cost"`
	DecisionPoint          json.RawMessage `json:"decision_point"`
	Improves               json.RawMessage `json:"improves"`
	LivePlanConflict       json.RawMessage `json:"live_plan_conflict"`
	Note                   json.RawMessage `json:"note"`
	Origin                 json.RawMessage `json:"origin"`
	PortSource             json.RawMessage `json:"port_source"`
	Priority               json.RawMessage `json:"priority"`
	Rationale              json.RawMessage `json:"rationale"`
	SingleMachineCaveat    json.RawMessage `json:"single_machine_caveat"`
}

type roadmapExecution struct {
	Stages map[int][]string
	Landed []string
}

func (execution *roadmapExecution) UnmarshalJSON(data []byte) error {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	execution.Stages = map[int][]string{}
	for name, value := range fields {
		switch {
		case strings.HasPrefix(name, "stage_"):
			stage, err := strconv.Atoi(strings.TrimPrefix(name, "stage_"))
			if err != nil || stage < 1 {
				return fmt.Errorf("invalid roadmap stage field %q", name)
			}
			var ids []string
			if err := json.Unmarshal(value, &ids); err != nil {
				return err
			}
			execution.Stages[stage] = ids
		case name == "landed":
			if err := json.Unmarshal(value, &execution.Landed); err != nil {
				return err
			}
		case name == "derivation", name == "priority_within_stage", name == "bootstrap_exception":
		default:
			return fmt.Errorf("unknown roadmap execution field %q", name)
		}
	}
	return nil
}

// RoadmapEvidence is the authoritative state projection supplied by Git,
// RepoDB, the live plan, leases, and isolated verifier evidence.
type RoadmapEvidence struct {
	Live       map[string]bool
	InFlight   map[string]bool
	Landed     map[string]bool
	ProbeBound map[string]string
}

type proposedRoadmapItem struct {
	Disposition string `json:"disposition"`
	Item        Item   `json:"item"`
}

// RoadmapReport is advisory. Proposed rows still require cmd/plan authorization.
type RoadmapReport struct {
	States   map[string]string     `json:"states"`
	Stages   map[string]int        `json:"stages"`
	Proposed []proposedRoadmapItem `json:"proposed"`
}

// EvaluateRoadmap validates the design record and derives advisory state.
func EvaluateRoadmap(data []byte, evidence RoadmapEvidence) (RoadmapReport, error) {
	document, rows, err := parseRoadmap(data)
	if err != nil {
		return RoadmapReport{}, err
	}
	if err := validateRoadmapReferences(document, rows); err != nil {
		return RoadmapReport{}, err
	}
	stages, closures, err := deriveRoadmapGraph(rows)
	if err != nil {
		return RoadmapReport{}, err
	}
	if err := validateStoredStages(document.Execution, rows, stages); err != nil {
		return RoadmapReport{}, err
	}
	if err := validateRoadmapSafety(document, rows, closures, evidence); err != nil {
		return RoadmapReport{}, err
	}
	report := RoadmapReport{States: map[string]string{}, Stages: stages, Proposed: []proposedRoadmapItem{}}
	for id, row := range rows {
		state := "new"
		switch {
		case evidence.Landed[id]:
			state = "landed"
		case evidence.InFlight[id]:
			state = "in-flight"
		case evidence.Live[id]:
			state = "open"
		case evidence.ProbeBound[id] == row.Verify && dependenciesLanded(row.DependsOn, evidence.Landed) && goLandingReady(id, row, document, evidence.Landed):
			state = "ready"
			report.Proposed = append(report.Proposed, proposedRoadmapItem{Disposition: "PROPOSED", Item: Item{
				ID: id, Title: row.Title, Status: "open", Steps: []Step{{ID: "do", Title: row.Title, Status: "open", Verify: row.Verify}},
			}})
		}
		report.States[id] = state
	}
	sort.Slice(report.Proposed, func(i, j int) bool { return report.Proposed[i].Item.ID < report.Proposed[j].Item.ID })
	return report, nil
}

// RoadmapVerifier returns the canonical verifier for one validated row.
func RoadmapVerifier(data []byte, id string) (string, error) {
	_, rows, err := parseRoadmap(data)
	if err != nil {
		return "", err
	}
	row, ok := rows[id]
	if !ok {
		return "", fmt.Errorf("roadmap row %q is absent", id)
	}
	return row.Verify, nil
}

func parseRoadmap(data []byte) (roadmapDocument, map[string]roadmapStep, error) {
	var document roadmapDocument
	if err := strictjson.DecodeBytes(data, &document); err != nil {
		return roadmapDocument{}, nil, fmt.Errorf("parse roadmap: %w", err)
	}
	rows := map[string]roadmapStep{}
	for _, item := range document.Items {
		if item.ID == "" || item.Title == "" || len(item.Steps) == 0 {
			return roadmapDocument{}, nil, fmt.Errorf("roadmap group %q is incomplete", item.ID)
		}
		for _, row := range item.Steps {
			if row.ID == "" || row.Title == "" || rows[row.ID].ID != "" {
				return roadmapDocument{}, nil, fmt.Errorf("roadmap row id %q is empty or duplicated", row.ID)
			}
			if err := testevidence.ValidateGoTestCommand(row.Verify); err != nil {
				return roadmapDocument{}, nil, fmt.Errorf("roadmap row %s verifier: %w", row.ID, err)
			}
			rows[row.ID] = row
		}
	}
	return document, rows, nil
}

func validateRoadmapReferences(document roadmapDocument, rows map[string]roadmapStep) error {
	check := func(owner string, references []string) error {
		seen := map[string]bool{}
		for _, id := range references {
			if rows[id].ID == "" || seen[id] {
				return fmt.Errorf("roadmap %s references unknown or duplicate row %q", owner, id)
			}
			seen[id] = true
		}
		return nil
	}
	for id, row := range rows {
		if slices.Contains(row.DependsOn, id) {
			return fmt.Errorf("roadmap row %s depends on itself", id)
		}
		if err := check(id+" depends_on", row.DependsOn); err != nil {
			return err
		}
		for _, capability := range row.Capabilities {
			if _, ok := document.Graph.MandatoryPreconditions.CapabilityDefinitions[capability]; !ok {
				return fmt.Errorf("roadmap row %s declares unknown capability %q", id, capability)
			}
		}
	}
	for capability, required := range document.Graph.MandatoryPreconditions.ByCapability {
		if _, ok := document.Graph.MandatoryPreconditions.CapabilityDefinitions[capability]; !ok {
			return fmt.Errorf("roadmap preconditions name unknown capability %q", capability)
		}
		if err := check("capability "+capability, required); err != nil {
			return err
		}
	}
	for _, references := range [][]string{document.Graph.Implementation.Requires, document.Graph.Implementation.BootstrapExceptions, document.Execution.Landed} {
		if err := check("cross-reference", references); err != nil {
			return err
		}
	}
	return nil
}

func deriveRoadmapGraph(rows map[string]roadmapStep) (map[string]int, map[string]map[string]bool, error) {
	stages, closures, visiting := map[string]int{}, map[string]map[string]bool{}, map[string]bool{}
	var visit func(string) (int, error)
	visit = func(id string) (int, error) {
		if stages[id] != 0 {
			return stages[id], nil
		}
		if visiting[id] {
			return 0, fmt.Errorf("roadmap dependency cycle reaches %s", id)
		}
		visiting[id] = true
		stage := 1
		closure := map[string]bool{}
		for _, dependency := range rows[id].DependsOn {
			prior, err := visit(dependency)
			if err != nil {
				return 0, err
			}
			stage = max(stage, prior+1)
			closure[dependency] = true
			for inherited := range closures[dependency] {
				closure[inherited] = true
			}
		}
		delete(visiting, id)
		stages[id] = stage
		closures[id] = closure
		return stage, nil
	}
	for id := range rows {
		if _, err := visit(id); err != nil {
			return nil, nil, err
		}
	}
	return stages, closures, nil
}

func validateStoredStages(execution roadmapExecution, rows map[string]roadmapStep, stages map[string]int) error {
	landed := toSet(execution.Landed)
	seen := map[string]bool{}
	for stage, ids := range execution.Stages {
		if !sort.StringsAreSorted(ids) {
			return fmt.Errorf("roadmap stage_%d is not sorted", stage)
		}
		for _, id := range ids {
			if rows[id].ID == "" || seen[id] || landed[id] || stages[id] != stage {
				return fmt.Errorf("roadmap stage_%d contains misplaced row %q", stage, id)
			}
			seen[id] = true
		}
	}
	for id := range rows {
		if !landed[id] && !seen[id] {
			return fmt.Errorf("roadmap derived stages omit %s (stage %d)", id, stages[id])
		}
	}
	return nil
}

func validateRoadmapSafety(document roadmapDocument, rows map[string]roadmapStep, closures map[string]map[string]bool, evidence RoadmapEvidence) error {
	for id, row := range rows {
		for _, capability := range row.Capabilities {
			for _, required := range document.Graph.MandatoryPreconditions.ByCapability[capability] {
				if !closures[id][required] {
					return fmt.Errorf("roadmap row %s capability %s lacks mandatory dependency %s", id, capability, required)
				}
			}
		}
		if (evidence.Live[id] || evidence.InFlight[id]) && !dependenciesLanded(row.DependsOn, evidence.Landed) {
			return fmt.Errorf("roadmap live row %s has unsatisfied dependencies", id)
		}
		if evidence.Landed[id] && !goLandingReady(id, row, document, evidence.Landed) {
			return fmt.Errorf("roadmap Go row %s landed without implementation precondition", id)
		}
	}
	return nil
}

func goLandingReady(id string, row roadmapStep, document roadmapDocument, landed map[string]bool) bool {
	if row.ImplementationLanguage != "" && row.ImplementationLanguage != "go" || slices.Contains(document.Graph.Implementation.BootstrapExceptions, id) || slices.Contains(document.Execution.Landed, id) {
		return true
	}
	return dependenciesLanded(document.Graph.Implementation.Requires, landed)
}

func dependenciesLanded(ids []string, landed map[string]bool) bool {
	for _, id := range ids {
		if !landed[id] {
			return false
		}
	}
	return true
}

func toSet(ids []string) map[string]bool {
	result := make(map[string]bool, len(ids))
	for _, id := range ids {
		result[id] = true
	}
	return result
}
