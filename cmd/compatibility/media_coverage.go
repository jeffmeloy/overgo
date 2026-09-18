package main

import (
	"bytes"
	"cmp"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
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
	Gap, RepairOwner string
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

func projectModalityCoverage(ctx context.Context, root string, store *overgodb.Store) (*modalityCoverageProjection, error) {
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
	}
	for _, entry := range entries {
		row := modalityModelProjection{Model: entry.Model, Name: mediaModelName(entry.Location), Present: entry.Present, Execution: "local"}
		if entry.KeyEnvironment != "" {
			row.Execution = "hosted"
		}
		row.Domains, _, err = modelrecipe.EvalDomains(ctx, store, entry.Model)
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
