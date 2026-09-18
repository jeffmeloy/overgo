package main

import (
	"bytes"
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"math/bits"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// Counts refer to committed acceptance phases, never examples or quality scores.
type modalityCoverageProjection struct {
	Registered, Activations, AcceptedActivations int
	Models                                       []modalityModelProjection
	Documents                                    map[string]string
	Proofs                                       []acceptedGateProof
}

type modalityModelProjection struct {
	Model            artifact.ID
	Name             string
	Present          bool
	Execution        string
	Domains          []string
	Cells            []modalityCellProjection
	Protocols        []modalityProtocolProjection
	Gap, RepairOwner string
}

// Recorded examples are not correct answers. Nil counts retain an unknown denominator.
type modalityProtocolProjection struct {
	Name, InputMode, Source, SourceKind, Unit string
	Recipe, Plan, CaseProfile, Report         artifact.ID `json:",omitzero"`
	Evidence                                  []artifact.ID
	Required, Recorded                        *uint64
	Gap, RepairOwner                          string
}

type modalityCellProjection struct {
	acceptedCell
	Implemented, Activated, Measured, Accepted bool
	RequiredPhases, AcceptedPhases             int
	HistoricalVerifier                         *mediaVerifierProjection `json:",omitzero"`
	Gap, RepairOwner                           string
}

func loadAcceptedGate(ctx context.Context, store *overgodb.Store, proof acceptedGateProof) (runrecord.GateResult, error) {
	finalized, found, err := runrecord.GateFinalizationForPreparation(ctx, store, proof.Preparation)
	if err != nil {
		return runrecord.GateResult{}, err
	}
	if !found || finalized.CodeCommit != proof.Commit || finalized.Outcome != runrecord.OutcomeSucceeded || finalized.Result == nil || *finalized.Result != proof.Result {
		return runrecord.GateResult{}, fmt.Errorf("coverage: %s lacks successful finalization", proof.Reference)
	}
	gate, err := runrecord.RequireGateResult(ctx, store, proof.Result)
	if err != nil {
		return gate, err
	}
	return gate, checkAcceptedGate(proof, gate)
}

func projectModalityCoverage(ctx context.Context, root string, store *overgodb.Store, names map[artifact.ID]string) (*modalityCoverageProjection, error) {
	entries, truncated, err := discovery.RegisteredCatalog(ctx, store, mediaCatalogLimit, discovery.LoadMemo(ctx, store))
	if err != nil {
		return nil, err
	}
	if truncated {
		return nil, errors.New("coverage: registered denominator truncated")
	}
	value := &modalityCoverageProjection{Registered: len(entries), Documents: map[string]string{}}
	var tasks []recipe.Task
	for _, entry := range entries {
		for _, capability := range entry.Capabilities {
			if !slices.Contains(tasks, capability.Task) {
				tasks = append(tasks, capability.Task)
			}
		}
	}
	rows, err := loadMediaReportRows(ctx, store, entries, mediaReportScope{Tasks: tasks}, nil, nil)
	if err != nil {
		return nil, err
	}
	type key struct {
		model artifact.ID
		task  recipe.Task
	}
	cells := map[key]acceptedCell{}
	measurements := map[key]*mediaVerifierProjection{}
	for _, row := range rows {
		if row.stale != "" {
			continue
		}
		measurements[key{row.modelID, recipe.Task(row.task)}], err = projectMediaVerifier(row.verification)
		if err != nil {
			return nil, err
		}
	}
	proofs := map[string]bool{}
	domains := map[artifact.ID][]string{}
	protocols := map[artifact.ID][]modalityProtocolProjection{}
	for _, path := range []string{"docs/verification/text-vision-coverage.json", "docs/verification/image-video-coverage.json", "docs/verification/specialized-coverage.json"} {
		var document imageVideoCoverage
		if err := jsonfile.DecodeStrict(filepath.Join(root, path), &document); err != nil {
			return nil, err
		}
		if document.Version != artifact.InitialDocumentVersion || document.Scope == "" {
			return nil, errors.New("coverage: missing scope")
		}
		for input, digest := range document.Inputs {
			if !filepath.IsLocal(input) {
				return nil, errors.New("coverage: nonlocal input")
			}
			data, err := os.ReadFile(filepath.Join(root, input))
			if err != nil {
				return nil, err
			}
			if fmt.Sprintf("%x", sha256.Sum256(data)) != digest {
				return nil, fmt.Errorf("coverage: input changed: %s", input)
			}
			if filepath.Ext(input) == ".json" {
				if err := collectModalityProtocols(ctx, store, data, document.Models, protocols); err != nil {
					return nil, fmt.Errorf("coverage: %s: %w", input, err)
				}
			}
		}
		data, err := os.ReadFile(filepath.Join(root, path))
		if err != nil {
			return nil, err
		}
		value.Documents[path] = fmt.Sprintf("%x", sha256.Sum256(data))
		for _, proof := range document.Proofs {
			id := proof.Reference + "#" + proof.Check
			if proofs[id] {
				return nil, errors.New("coverage: duplicate proof")
			}
			if _, err := loadAcceptedGate(ctx, store, proof); err != nil {
				return nil, err
			}
			proofs[id] = true
			value.Proofs = append(value.Proofs, proof)
		}
		for _, model := range document.Models {
			if prior, found := domains[model.Model]; found && !slices.Equal(prior, model.Domains) {
				return nil, errors.New("coverage: contradictory domains")
			}
			domains[model.Model] = model.Domains
			for _, cell := range model.Cells {
				k := key{model.Model, cell.Task}
				if _, found := cells[k]; found || len(cell.Proofs) == 0 || cell.Scope == "" {
					return nil, errors.New("coverage: duplicate or unbound cell")
				}
				for _, ref := range cell.Proofs {
					if !strings.Contains(ref, "#") {
						ref += "#"
					}
					if !proofs[ref] {
						return nil, fmt.Errorf("coverage: missing proof %s", ref)
					}
				}
				cells[k] = cell
			}
		}
		if len(document.Cases) != 0 {
			var protocol imageVideoProtocol
			if err := jsonfile.DecodeStrict(filepath.Join(root, imageVideoProtocolPath), &protocol); err != nil {
				return nil, err
			}
			if err := checkAcceptedMediaCases(protocol, document); err != nil {
				return nil, err
			}
			for _, binding := range document.Cases {
				index := slices.IndexFunc(protocol.Cases, func(c imageVideoCase) bool { return c.ID == binding.Case })
				declared := protocol.Cases[index]
				declared.Run, declared.Outputs = binding.Run, []artifact.ID{binding.Output}
				run, err := runrecord.RequireRun(ctx, store, binding.Run)
				if err != nil {
					return nil, err
				}
				if err := checkImageVideoCaseRun(declared, run); err != nil {
					return nil, err
				}
				// Each protocol entry declares one independent generation case.
				required, recorded := uint64(1), uint64(1)
				source, sourceKind := run.CodeCommit, "run"
				if source == "" {
					ref := cells[key{declared.Model, declared.Task}].Proofs[0]
					if !strings.Contains(ref, "#") {
						ref += "#"
					}
					proof := slices.IndexFunc(document.Proofs, func(p acceptedGateProof) bool { return p.Reference+"#"+p.Check == ref })
					if proof < 0 {
						return nil, errors.New("coverage: generation case has no acceptance source")
					}
					source, sourceKind = document.Proofs[proof].Commit, "acceptance gate; run source unrecorded"
				}
				protocols[declared.Model] = append(protocols[declared.Model], modalityProtocolProjection{Name: declared.ID, InputMode: "declared request", Source: source, SourceKind: sourceKind, Unit: "generation case", Recipe: declared.Recipe, Evidence: []artifact.ID{binding.Run, binding.Output, binding.Review}, Required: &required, Recorded: &recorded, Gap: declared.Scope + " Native and lifecycle acceptance only; broader quality remains unestablished.", RepairOwner: "final-model-validation/do"})
			}
		}
	}
	for _, entry := range entries {
		row := modalityModelProjection{Model: entry.Model, Name: cmp.Or(names[entry.Model], mediaModelName(entry.Location)), Present: entry.Present, Execution: "local"}
		row.Protocols = protocols[entry.Model]
		slices.SortFunc(row.Protocols, func(a, b modalityProtocolProjection) int { return strings.Compare(a.Name, b.Name) })
		if entry.KeyEnvironment != "" {
			row.Execution = "hosted"
		}
		row.Domains, _, err = modelartifact.EvalDomains(ctx, store, entry.Model)
		if err != nil {
			return nil, err
		}
		if len(entry.Capabilities) == 0 {
			row.Gap = "Registered without task activation or accepted protocol."
			row.RepairOwner = "final-model-validation/do"
		}
		for _, capability := range entry.Capabilities {
			cell := modalityCellProjection{acceptedCell: acceptedCell{Task: capability.Task, Recipe: capability.Recipe}, Activated: capability.Stale == "", Gap: "No accepted protocol bound to this activation.", RepairOwner: "final-model-validation/do"}
			cell.HistoricalVerifier = measurements[key{entry.Model, capability.Task}]
			cell.Measured = cell.HistoricalVerifier != nil
			definition, err := recipe.RequireDefinition(ctx, store, capability.Recipe)
			cell.Implemented = err == nil && definition.Task == capability.Task
			if err != nil {
				cell.Gap = err.Error()
			}
			if capability.Stale != "" {
				cell.Gap = capability.Stale
			}
			if accepted, found := cells[key{entry.Model, capability.Task}]; found {
				if !entry.Present || !cell.Activated || !cell.Implemented || accepted.Recipe != capability.Recipe || row.Execution != "local" || !slices.Equal(row.Domains, domains[entry.Model]) {
					return nil, errors.New("coverage: accepted activation changed")
				}
				cell.acceptedCell = accepted
				cell.Accepted = true
				cell.RequiredPhases, cell.AcceptedPhases = len(accepted.Proofs), len(accepted.Proofs)
				cell.Gap = "Acceptance retains producer source and protocol scope; current performance and broader quality are not established."
				value.AcceptedActivations++
				delete(cells, key{entry.Model, capability.Task})
			}
			row.Cells = append(row.Cells, cell)
			value.Activations++
		}
		value.Models = append(value.Models, row)
	}
	if len(cells) != 0 {
		return nil, errors.New("coverage: accepted activation omitted from catalog")
	}
	return value, checkModalityCoverageProjection(value, entries)
}

// Reconcile the view against its catalog, never trust aggregate counters.
func checkModalityCoverageProjection(value *modalityCoverageProjection, entries []discovery.CatalogEntry) error {
	if value.Registered != len(entries) || len(value.Models) != len(entries) {
		return errors.New("coverage: registered denominator differs")
	}
	proofs := map[string]bool{}
	for _, proof := range value.Proofs {
		proofs[proof.Reference+"#"+proof.Check] = true
	}
	activations, accepted := 0, 0
	for i, row := range value.Models {
		entry := entries[i]
		if row.Model != entry.Model || row.Present != entry.Present || len(row.Cells) != len(entry.Capabilities) {
			return errors.New("coverage: model or activation omitted")
		}
		if len(row.Cells) == 0 && (row.Gap == "" || row.RepairOwner == "") {
			return errors.New("coverage: inactive registration lacks disposition")
		}
		if err := checkModalityProtocols(row.Protocols); err != nil {
			return err
		}
		for j, cell := range row.Cells {
			capability := entry.Capabilities[j]
			if cell.Task != capability.Task || cell.Recipe != capability.Recipe || cell.Gap == "" || cell.RepairOwner == "" {
				return errors.New("coverage: changed cell or missing repair owner")
			}
			if cell.Accepted {
				if !cell.Implemented || !cell.Activated || !row.Present || row.Execution != "local" || cell.Scope == "" || len(cell.Proofs) == 0 || cell.RequiredPhases != len(cell.Proofs) || cell.AcceptedPhases != cell.RequiredPhases {
					return errors.New("coverage: unbound acceptance")
				}
				seen := map[string]bool{}
				for _, ref := range cell.Proofs {
					if !strings.Contains(ref, "#") {
						ref += "#"
					}
					if !proofs[ref] || seen[ref] {
						return errors.New("coverage: missing or duplicate phase")
					}
					seen[ref] = true
				}
				accepted++
			} else if cell.AcceptedPhases != 0 || cell.RequiredPhases != 0 || len(cell.Proofs) != 0 {
				return errors.New("coverage: unaccepted cell gained phase credit")
			}
			activations++
			if cell.Measured != (cell.HistoricalVerifier != nil) {
				return errors.New("coverage: unbound measurement")
			}
		}
	}
	if value.Activations != activations || value.AcceptedActivations != accepted {
		return errors.New("coverage: contradictory totals")
	}
	return nil
}

func checkModalityProtocols(values []modalityProtocolProjection) error {
	seen := map[string]bool{}
	for _, value := range values {
		if value.Name == "" || seen[value.Name] || value.InputMode == "" || value.Unit == "" || value.Gap == "" || value.RepairOwner == "" || !gitauthority.ValidObjectID(value.Source) || len(value.Evidence) == 0 {
			return fmt.Errorf("coverage: incomplete or duplicate protocol %q (source %q)", value.Name, value.Source)
		}
		seen[value.Name] = true
		if value.Recorded != nil && (value.Required == nil || *value.Recorded != *value.Required) || value.Required != nil && *value.Required == 0 {
			return errors.New("coverage: unbound or contradictory case count")
		}
	}
	return nil
}

func writeModalityCoverage(output *bytes.Buffer, value *modalityCoverageProjection) {
	output.WriteString("## Accepted protocol coverage\n\n")
	fmt.Fprintf(output, "%d registered models; %d task activations; %d activations carry accepted protocols. Counts below are acceptance phases, not examples, quality scores or current performance claims. Producer source revisions and exact verifier commands appear in the [typed report](media_report.json); input identities remain in its linked coverage documents. See also [benchmark results](BENCHMARK.md).\n\n", value.Registered, value.Activations, value.AcceptedActivations)
	output.WriteString("These are raw registrations, including fixture identities, not a count of validated models. Domains are catalog declarations; accepted input modalities appear in protocol scope even when a catalog domain is undeclared. Local results remain separate from hosted capabilities; mocked provider contracts confer no model-quality evidence. Historical resource evidence retains its original source. The [E4B resource refresh](verification/e4b-resource-refresh.json) does not establish current resource bounds after lifecycle changes; device-memory-retention/do owns that comparison.\n\n")
	output.WriteString("| Model | Execution / domains | Task | Implemented / activated / measured | Accepted / required phases | Protocol scope and remaining gap | Repair owner |\n| --- | --- | --- | --- | --- | --- | --- |\n")
	for _, row := range value.Models {
		for _, cell := range row.Cells {
			fmt.Fprintf(output, "| %s | %s / %s | %s | %t / %t / %t | %d / %d | %s %s | `%s` |\n", escapeMarkdown(row.Name), row.Execution, strings.Join(row.Domains, ", "), cell.Task, cell.Implemented, cell.Activated, cell.Measured, cell.AcceptedPhases, cell.RequiredPhases, escapeMarkdown(cell.Scope), escapeMarkdown(cell.Gap), cell.RepairOwner)
		}
	}
	output.WriteString("\n<details><summary>Registrations without task activation</summary>\n\n| Model identity | Name | Artifact present | Gap | Repair owner |\n| --- | --- | --- | --- | --- |\n")
	for _, row := range value.Models {
		if len(row.Cells) == 0 {
			fmt.Fprintf(output, "| `%s` | %s | %t | %s | `%s` |\n", row.Model, escapeMarkdown(row.Name), row.Present, row.Gap, row.RepairOwner)
		}
	}
	output.WriteString("\n</details>\n")
	output.WriteString("\n## Protocol case denominators\n\nRecorded examples are not correct answers. Units remain separate; there is no combined total across repeated protocols, generation cases and benchmark examples. Unknown counts receive no case credit. Exact evidence, recipe, plan and case-profile identities are in the typed report.\n\n| Model | Input mode | Protocol | Recorded / required | Unit | Producer source | Scope or gap |\n| --- | --- | --- | --- | --- | --- | --- |\n")
	for _, row := range value.Models {
		if len(row.Protocols) == 0 && len(row.Cells) > 0 {
			fmt.Fprintf(output, "| %s | Unspecified | See accepted phase scopes above | Unknown / unknown | Unrecorded | See phase proofs | Structured case receipt unavailable; modality-verification/experiment-resume owns extraction without reacquisition. |\n", escapeMarkdown(row.Name))
		}
		for _, protocol := range row.Protocols {
			count := func(v *uint64) string {
				if v == nil {
					return "Unknown"
				}
				return fmt.Sprint(*v)
			}
			fmt.Fprintf(output, "| %s | %s | %s | %s / %s | %s | `%s` (%s) | %s |\n", escapeMarkdown(row.Name), escapeMarkdown(protocol.InputMode), escapeMarkdown(protocol.Name), count(protocol.Recorded), count(protocol.Required), protocol.Unit, protocol.Source, protocol.SourceKind, escapeMarkdown(protocol.Gap))
		}
	}
	output.WriteString("\nZero required phases on an uncovered activation means no declared protocol, never a vacuous pass. Measurements below retain their original sources and unresolved historical attempts.\n\n")
}

// This join reuses committed acceptance, not model execution or a new evaluator.
// Original protocol scopes do not confer current-source or performance credit.
type acceptedCoverage struct {
	Version uint16
	Scope   string
	Models  []acceptedModel
	Proofs  []acceptedGateProof
	Inputs  map[string]string
}

type acceptedModel struct {
	Model   artifact.ID
	Domains []string
	Cells   []acceptedCell
}

type acceptedCell struct {
	Task   recipe.Task
	Recipe artifact.ID
	Proofs []string
	Scope  string
}

type acceptedGateProof struct {
	Reference, Commit, Verify string
	Preparation, Result       artifact.ID
	Check                     string `json:",omitzero"`
}

func checkAcceptedGate(proof acceptedGateProof, gate runrecord.GateResult) error {
	if gate.ID != proof.Result || gate.CodeCommit != proof.Commit || gate.Outcome != runrecord.OutcomeSucceeded {
		return errors.New("coverage: acceptance result failed or source changed")
	}
	for _, step := range gate.Steps {
		if step.Name == cmp.Or(proof.Check, "acceptance") && (step.Outcome == runrecord.StepSucceeded || step.Outcome == runrecord.StepReused) {
			return runrecord.VerifyCompletionAcceptanceEvidence(step.Evidence, proof.Reference, proof.Verify)
		}
	}
	return errors.New("coverage: required acceptance was not completed")
}

type imageVideoCoverage struct {
	acceptedCoverage
	Cases []acceptedMediaCase
}

type acceptedMediaCase struct {
	Case                string
	Run, Output, Review artifact.ID
}

// Only declarations already pinned by an accepted coverage document enter this view.
func collectModalityProtocols(ctx context.Context, reader artifact.Reader, data []byte, accepted []acceptedModel, models map[artifact.ID][]modalityProtocolProjection) error {
	var header struct {
		Model         json.RawMessage
		Cells, Claims json.RawMessage
	}
	if err := json.Unmarshal(data, &header); err != nil {
		return err
	}
	if len(header.Cells) == 0 && len(header.Claims) == 0 {
		return nil
	}
	var model artifact.ID
	if err := json.Unmarshal(header.Model, &model); err != nil || model.Kind() != artifact.KindModel {
		return nil
	}
	if !slices.ContainsFunc(accepted, func(value acceptedModel) bool { return value.Model == model }) {
		return nil
	}
	var values []modalityProtocolProjection
	if len(header.Cells) != 0 {
		var spec modelValidationSpecification
		if err := json.Unmarshal(data, &spec); err != nil {
			return err
		}
		for _, cell := range spec.Cells {
			parts := strings.Split(cell.Name, "/")
			if len(parts) != 3 {
				return errors.New("coverage: invalid modality protocol name")
			}
			value := modalityProtocolProjection{Name: cell.Name, InputMode: parts[1], Source: spec.CodeCommit, Recipe: cell.Recipe, Plan: cell.Plan, Evidence: []artifact.ID{cell.Evidence}}
			if strings.HasPrefix(cell.Name, "protocol/") {
				if err := checkProtocolValidationCell(ctx, reader, spec, cell); err != nil {
					return err
				}
				// The frozen native capture covers each declared modality/endpoint case.
				required, recorded := uint64(1), uint64(1)
				value.Required, value.Recorded, value.Unit = &required, &recorded, "protocol case"
			}
			if len(cell.Cases) != 0 {
				count := uint64(len(cell.Cases))
				value.Required = &count
			}
			values = append(values, value)
		}
	} else {
		var spec verificationSpecification
		if err := json.Unmarshal(data, &spec); err != nil {
			return err
		}
		for _, claim := range spec.Claims {
			value := modalityProtocolProjection{Name: claim.Capability, Source: claim.Commit, Evidence: slices.Clone(claim.Evidence), InputMode: "unspecified"}
			switch {
			case strings.HasPrefix(claim.Capability, "mmlu"), strings.HasPrefix(claim.Capability, "ifeval"), strings.HasPrefix(claim.Capability, "bbh"), claim.Capability == "native-text":
				value.InputMode = "text"
			case strings.HasPrefix(claim.Capability, "vqa-"), strings.HasPrefix(claim.Capability, "projection-grounded-"):
				value.InputMode = "image+text"
			}
			values = append(values, value)
		}
	}
	for _, value := range values {
		value.SourceKind = "frozen declaration"
		value.Unit = cmp.Or(value.Unit, "examples")
		value.Gap = "Retained protocol; case counts are unavailable in the supported structured report. No current performance claim."
		value.RepairOwner = "modality-verification/experiment-resume"
		if value.Recorded != nil {
			value.Gap = "Native protocol case accepted at the recorded source; no current performance claim."
			value.RepairOwner = "final-model-validation/do"
		}
		if err := projectProtocolCounts(ctx, reader, &value); err != nil {
			return err
		}
		prior := slices.IndexFunc(models[model], func(p modalityProtocolProjection) bool { return p.Name == value.Name })
		if prior >= 0 {
			left, _ := json.Marshal(models[model][prior])
			right, _ := json.Marshal(value)
			if !bytes.Equal(left, right) {
				return errors.New("coverage: contradictory protocol declarations")
			}
			continue
		}
		models[model] = append(models[model], value)
	}
	return nil
}

// Read schema-bound record fields without linking the evaluator's execution runtime.
// Acceptance remains with the frozen producer; these fields only project its counts.
func projectProtocolCounts(ctx context.Context, reader artifact.Reader, value *modalityProtocolProjection) error {
	for _, id := range value.Evidence {
		content, err := artifact.RequireTypedContent(ctx, reader, id)
		if err != nil {
			return err
		}
		if content.Descriptor.Schema == "overgo/gate-result/v1" && strings.HasPrefix(value.Name, "resources/") {
			gate, err := runrecord.RequireGateResult(ctx, reader, id)
			if err != nil {
				return err
			}
			return projectResourceCaseCounts(value, gate)
		}
		if content.Descriptor.Schema == "overgo/transcription-report/v1" && value.Required != nil {
			var report struct {
				Plan         artifact.ID
				Observations []json.RawMessage
			}
			if err := json.Unmarshal(content.Data, &report); err != nil {
				return err
			}
			if report.Plan != value.Plan || uint64(len(report.Observations)) != *value.Required {
				return errors.New("coverage: transcription case denominator differs")
			}
			recorded := uint64(len(report.Observations))
			value.Recorded = &recorded
			value.Report = id
			value.Gap = "Recorded transcription examples match the frozen selection; WER and CER remain separate quality metrics."
			value.RepairOwner = "final-model-validation/do"
			return nil
		}
		if content.Descriptor.Schema != "overgo/evaluation-evidence/v1" {
			continue
		}
		var evidence struct {
			Plan, Report, Recipe artifact.ID
			CodeCommit           string `json:"code_commit"`
		}
		if err := json.Unmarshal(content.Data, &evidence); err != nil {
			return err
		}
		if value.Plan.Valid() && value.Plan != evidence.Plan || value.Recipe.Valid() && value.Recipe != evidence.Recipe || value.Source != evidence.CodeCommit {
			return errors.New("coverage: evaluation source or binding differs")
		}
		value.Plan, value.Report, value.Recipe = evidence.Plan, evidence.Report, evidence.Recipe
		plan, err := artifact.RequireTypedContent(ctx, reader, evidence.Plan)
		if err != nil {
			return err
		}
		if plan.Descriptor.Schema != "overgo/evaluation-plan/v1" {
			return errors.New("coverage: unsupported evaluation plan")
		}
		var binding struct {
			CaseProfile artifact.ID `json:"case_profile"`
		}
		if err := json.Unmarshal(plan.Data, &binding); err != nil {
			return err
		}
		profile, err := artifact.RequireTypedContent(ctx, reader, binding.CaseProfile)
		if err != nil {
			return err
		}
		if profile.Descriptor.Schema != "overgo/evaluation-case-profile/v1" {
			return errors.New("coverage: unsupported case profile")
		}
		var declared struct{ Cases []json.RawMessage }
		if err := json.Unmarshal(profile.Data, &declared); err != nil {
			return err
		}
		if len(declared.Cases) == 0 {
			return errors.New("coverage: case profile denominator absent")
		}
		value.CaseProfile = binding.CaseProfile
		required := uint64(len(declared.Cases))
		value.Required = &required
		report, err := artifact.RequireTypedContent(ctx, reader, evidence.Report)
		if err != nil {
			return err
		}
		var observed struct {
			Plan         artifact.ID
			Cases        *uint64
			Observations []json.RawMessage
		}
		if err := json.Unmarshal(report.Data, &observed); err != nil {
			return err
		}
		if observed.Plan != evidence.Plan {
			return errors.New("coverage: report belongs to another plan")
		}
		switch report.Descriptor.Schema {
		case "overgo/evaluation-campaign/v1":
			value.Recorded = observed.Cases
		case "overgo/mmlu-pro-report/v1", "overgo/multiple-choice-report/v1", "overgo/grouped-choice-report/v1", "overgo/instruction-rules-report/v1", "overgo/generated-answer-report/v1":
			recorded := uint64(len(observed.Observations))
			value.Recorded = &recorded
			if err := checkProtocolCaseNames(declared.Cases, observed.Observations); err != nil {
				return err
			}
		}
		if value.Recorded != nil {
			if *value.Recorded != *value.Required {
				return errors.New("coverage: recorded case count differs from required profile")
			}
			value.Gap = "Recorded examples match the frozen case profile; correctness and quality remain the producer's metrics, not this count. No current performance claim."
			value.RepairOwner = "final-model-validation/do"
		}
		return nil
	}
	return nil
}

func projectResourceCaseCounts(value *modalityProtocolProjection, gate runrecord.GateResult) error {
	if gate.Outcome != runrecord.OutcomeSucceeded || gate.CodeCommit != value.Source || gate.Recipe != value.Recipe {
		return errors.New("coverage: resource producer differs or failed")
	}
	index := slices.IndexFunc(gate.Steps, func(step runrecord.GateStep) bool { return step.Name == "media-resource-contract" })
	if index < 0 || gate.Steps[index].Outcome != runrecord.StepSucceeded {
		return errors.New("coverage: resource contract absent or failed")
	}
	fields := map[string]string{}
	for part := range strings.SplitSeq(gate.Steps[index].Evidence, ";") {
		key, content, _ := strings.Cut(part, "=")
		if _, exists := fields[key]; exists {
			return errors.New("coverage: duplicate resource contract field")
		}
		fields[key] = content
	}
	// Reloads × cycles × actions is the producer-declared denominator per mode.
	required := uint64(1)
	for _, axis := range []string{"reloads", "cycles", "actions"} {
		n, err := strconv.ParseUint(fields[axis], 10, 64)
		if err != nil || n == 0 {
			return errors.New("coverage: invalid resource axis")
		}
		high, low := bits.Mul64(required, n)
		if high != 0 {
			return errors.New("coverage: resource denominator overflow")
		}
		required = low
	}
	var recorded uint64
	seen := map[artifact.ID]bool{}
	for _, step := range gate.Steps {
		if !strings.HasPrefix(step.Name, "media-r") || step.Name == "media-resource-contract" {
			continue
		}
		var observation struct {
			Mode        string
			Observation artifact.ID
		}
		if err := json.Unmarshal([]byte(step.Evidence), &observation); err != nil {
			return err
		}
		if observation.Mode != strings.TrimSuffix(value.InputMode, "-history") {
			continue
		}
		if !observation.Observation.Valid() || seen[observation.Observation] || step.Outcome != runrecord.StepSucceeded {
			return errors.New("coverage: missing, duplicate or failed resource case")
		}
		seen[observation.Observation] = true
		recorded++
	}
	if recorded != required {
		return errors.New("coverage: resource case denominator differs")
	}
	value.Required, value.Recorded, value.Unit = &required, &recorded, "recovery action"
	value.Gap = "Original resource and recovery observations retained. Later lifecycle changes require a targeted current resource comparison; these counts do not reaccept current peaks."
	value.RepairOwner = "device-memory-retention/do"
	return nil
}

func checkProtocolCaseNames(declared, observed []json.RawMessage) error {
	wanted := map[string]bool{}
	for _, raw := range declared {
		var row struct{ Name string }
		if err := json.Unmarshal(raw, &row); err != nil {
			return err
		}
		if row.Name == "" || wanted[row.Name] {
			return errors.New("coverage: missing or duplicate required case")
		}
		wanted[row.Name] = true
	}
	for _, raw := range observed {
		var row struct{ Name string }
		if err := json.Unmarshal(raw, &row); err != nil {
			return err
		}
		if !wanted[row.Name] {
			return errors.New("coverage: foreign or duplicate recorded case")
		}
		delete(wanted, row.Name)
	}
	if len(wanted) != 0 {
		return errors.New("coverage: required cases omitted")
	}
	return nil
}

// A validation specification selects exact existing evaluation evidence. It
// neither runs a model nor supplies replacement observations for missing data.
type modelValidationSpecification struct {
	Model      artifact.ID           `json:"model"`
	Projector  artifact.ID           `json:"projector"`
	CodeCommit string                `json:"code_commit"`
	MaskGate   artifact.ID           `json:"mask_gate"`
	MaskRun    artifact.ID           `json:"mask_run"`
	Cells      []modelValidationCell `json:"cells"`
}

type modelValidationCell struct {
	Name            string                 `json:"name"`
	Evidence        artifact.ID            `json:"evidence"`
	Run             artifact.ID            `json:"run,omitzero"`
	OracleSHA256    string                 `json:"oracle_sha256,omitzero"`
	Plan            artifact.ID            `json:"plan"`
	ModelDefinition artifact.ID            `json:"model_definition"`
	Recipe          artifact.ID            `json:"recipe"`
	Environment     artifact.ID            `json:"environment"`
	Dataset         artifact.ID            `json:"dataset"`
	Split           artifact.ID            `json:"split"`
	Shards          []artifact.ID          `json:"shards"`
	Cases           []string               `json:"cases,omitempty"`
	Bounds          []modelValidationBound `json:"bounds"`
}

type modelValidationBound struct {
	Metric    string              `json:"metric"`
	Unit      string              `json:"unit"`
	Direction runrecord.Direction `json:"direction"`
	Minimum   float64             `json:"minimum"`
	Maximum   float64             `json:"maximum"`
}

func checkAcceptedMediaCases(protocol imageVideoProtocol, coverage imageVideoCoverage) error {
	wanted := map[string]imageVideoCase{}
	for _, value := range protocol.Cases {
		if _, found := wanted[value.ID]; found {
			return errors.New("media acceptance: duplicate protocol case")
		}
		wanted[value.ID] = value
	}
	proofs := map[string]bool{}
	for _, proof := range coverage.Proofs {
		key := proof.Reference + "#" + proof.Check
		if proofs[key] {
			return errors.New("media acceptance: duplicate proof")
		}
		proofs[key] = true
	}
	for _, binding := range coverage.Cases {
		value, found := wanted[binding.Case]
		if !found || binding.Run.Kind() != artifact.KindRun || binding.Output.Kind() != artifact.KindOutput || binding.Review.Kind() != artifact.KindEvidence {
			return errors.New("media acceptance: absent, duplicate or invalid case")
		}
		bound := false
		for _, model := range coverage.Models {
			for _, cell := range model.Cells {
				if model.Model != value.Model || cell.Task != value.Task {
					continue
				}
				if bound || cell.Recipe != value.Recipe || len(cell.Proofs) == 0 {
					return errors.New("media acceptance: duplicate or changed activation")
				}
				for _, reference := range cell.Proofs {
					if !proofs[reference] {
						return fmt.Errorf("media acceptance: missing proof %s", reference)
					}
				}
				bound = true
			}
		}
		if !bound {
			return errors.New("media acceptance: case has no accepted activation")
		}
		delete(wanted, binding.Case)
	}
	if len(wanted) != 0 || len(coverage.Cases) == 0 {
		return errors.New("media acceptance: incomplete denominator")
	}
	return nil
}

func checkProtocolValidationCell(ctx context.Context, store artifact.Reader, spec modelValidationSpecification, cell modelValidationCell) error {
	if cell.Plan.Valid() || cell.Dataset.Valid() || cell.Split.Valid() || len(cell.Shards) != 0 || len(cell.Cases) != 0 || len(cell.Bounds) != 0 ||
		len(cell.OracleSHA256) != sha256.Size*2 || strings.Trim(cell.OracleSHA256, "0123456789abcdef") != "" {
		return errors.New("model validation: HTTP protocol proof requires its native oracle, not a renamed dataset evaluation")
	}
	proof, err := runrecord.VerifyGateRun(ctx, store, cell.Recipe, cell.Evidence, cell.Run)
	if err != nil {
		return err
	}
	if proof.Gate.CodeCommit != spec.CodeCommit || proof.Gate.Environment != cell.Environment || len(proof.Gate.Steps) != 1 {
		return errors.New("model validation: HTTP protocol producer authority differs")
	}
	step := proof.Gate.Steps[0]
	want := fmt.Sprintf("contract=e4b-http-modalities;oracle_sha256=%s;model_definition=%s;cases=18", cell.OracleSHA256, cell.ModelDefinition)
	if step.Name != "candidate-execution" || step.Phase != runrecord.PhaseTest || step.Outcome != runrecord.StepSucceeded || step.Evidence != want {
		return errors.New("model validation: complete executed HTTP protocol proof is absent")
	}
	return nil
}
