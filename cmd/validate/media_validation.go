package main

import (
	"bytes"
	"cmp"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/testevidence"
)

const mediaValidationPath = "docs/image_video_validation.json"
const mediaAssertionsPath = "docs/image_video_assertions.json"

// mediaValidationIndex locates acquired test output. Verdicts come from the
// existing Go test evidence parser; scope descriptions are not acceptance.
type mediaValidationIndex struct {
	Version       uint16                 `json:"version"`
	Source        string                 `json:"source_base"`
	Environment   string                 `json:"environment"`
	Notes         []string               `json:"notes"`
	Checks        []mediaValidationCheck `json:"checks"`
	PriorCapacity artifact.ID            `json:"prior_capacity_checkpoint,omitzero"`
}

type mediaValidationCheck struct {
	Name          string                 `json:"name"`
	Source        string                 `json:"source_base,omitzero"`
	Evidence      artifact.ID            `json:"evidence"`
	Command       string                 `json:"command"`
	Scope         string                 `json:"scope"`
	SourceFiles   map[string]string      `json:"changed_source_sha256,omitempty"`
	Harness       map[string]artifact.ID `json:"overlay_sources,omitempty"`
	PriorEvidence artifact.ID            `json:"prior_evidence,omitzero"`
}

// Retained assertion outcomes remain distinct from current generation acceptance.
type mediaValidationProjection struct {
	Source        string                  `json:"source_base"`
	Environment   string                  `json:"environment"`
	Notes         []string                `json:"notes"`
	Checks        []mediaValidationResult `json:"checks"`
	Passed        int                     `json:"historical_passed"`
	Unresolved    int                     `json:"unresolved"`
	PriorCapacity artifact.ID             `json:"prior_capacity_checkpoint,omitzero"`
}

type mediaValidationResult struct {
	mediaValidationCheck
	Assertions  map[string][]string `json:"assertions"`
	Passed      bool                `json:"historical_assertions_passed"`
	Verdict     string              `json:"recorded_result"`
	RepairOwner string              `json:"repair_owner"`
}

// Declared overlay paths describe the producer workspace. Resolve their
// contents by identity, never by the current checkout's filesystem.
func checkMediaReceiptProvenance(ctx context.Context, reader artifact.Reader, check mediaValidationCheck) error {
	if check.Source != "" && !gitauthority.ValidObjectID(check.Source) {
		return errors.New("media validation: invalid acquisition source")
	}
	for path, digest := range check.SourceFiles {
		if _, err := artifact.ParseID("evidence:sha256:" + digest); err != nil || !filepath.IsLocal(path) {
			return fmt.Errorf("media validation: invalid changed-source binding for %q", path)
		}
	}
	var ids []artifact.ID
	if check.PriorEvidence.Valid() {
		if check.PriorEvidence.Kind() != artifact.KindEvidence {
			return errors.New("media validation: invalid prior evidence identity")
		}
		ids = append(ids, check.PriorEvidence)
	}
	for path, id := range check.Harness {
		if strings.TrimSpace(path) == "" || id.Kind() != artifact.KindEvidence || !id.Valid() {
			return fmt.Errorf("media validation: invalid overlay binding for %q", path)
		}
		ids = append(ids, id)
	}
	slices.SortFunc(ids, artifact.CompareID)
	if len(ids) != 0 {
		if err := artifact.ReadContents(ctx, reader, slices.Compact(ids), func(artifact.Content) error { return nil }); err != nil {
			return fmt.Errorf("media validation: retained overlay sources: %w", err)
		}
	}
	return nil
}

func checkMediaTestReceipt(name string, data []byte, required map[string][]string) error {
	if len(required) == 0 {
		return fmt.Errorf("%s: assertion contract absent", name)
	}
	report, err := testevidence.GoTestJSONReport(string(data))
	if err != nil {
		return err
	}
	if err := testevidence.RequireComplete(report); err != nil {
		return err
	}
	// PackagePassed supplies pass authority; observations supply membership.
	// Timing values never grant assertion credit.
	observed := report.TestCosts()
	for pkg, tests := range required {
		if len(tests) == 0 || !report.PackagePassed(pkg) {
			return fmt.Errorf("%s: required assertions incomplete in %s: %v", name, pkg, tests)
		}
		seen := map[string]bool{}
		for _, test := range tests {
			if test == "" || seen[test] || !slices.ContainsFunc(observed, func(value testevidence.TestExecution) bool {
				return value.Package == pkg && value.Name == test && value.Action == "pass"
			}) {
				return fmt.Errorf("%s: missing or duplicate assertion %s:%s", name, pkg, test)
			}
			seen[test] = true
		}
	}
	return nil
}

func mediaValidationVerdict(data []byte, required map[string][]string) (bool, string) {
	if err := checkMediaTestReceipt("media validation", data, required); err != nil {
		return false, "Not accepted: " + err.Error()
	}
	return true, "Passed declared assertions (historical; current scope unbound)"
}

func projectMediaValidation(ctx context.Context, root string, reader artifact.Reader) (*mediaValidationProjection, error) {
	var index mediaValidationIndex
	if err := jsonfile.DecodeStrict(filepath.Join(root, mediaValidationPath), &index); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	if index.Version != artifact.InitialDocumentVersion || !gitauthority.ValidObjectID(index.Source) || index.Environment == "" || len(index.Checks) == 0 {
		return nil, errors.New("media validation: incomplete checkpoint identity")
	}
	if index.PriorCapacity.Valid() && index.PriorCapacity.Kind() != artifact.KindEvidence {
		return nil, errors.New("media validation: invalid prior capacity identity")
	}
	var assertions map[string]map[string][]string
	if err := jsonfile.DecodeStrict(filepath.Join(root, mediaAssertionsPath), &assertions); err != nil {
		return nil, err
	}
	projection := &mediaValidationProjection{Source: index.Source, Environment: index.Environment, Notes: index.Notes, PriorCapacity: index.PriorCapacity, Checks: []mediaValidationResult{}}
	seen := map[string]bool{}
	for _, check := range index.Checks {
		if check.Name == "" || seen[check.Name] || check.Scope == "" || check.Command == "" || check.Evidence.Kind() != artifact.KindEvidence {
			return nil, errors.New("media validation: invalid or duplicate check")
		}
		seen[check.Name] = true
		check.Source = cmp.Or(check.Source, index.Source)
		result := mediaValidationResult{mediaValidationCheck: check, Assertions: assertions[check.Name], RepairOwner: "modality-verification/media-report"}
		content, found, err := artifact.ReadContent(ctx, reader, check.Evidence)
		result.Verdict = "Unavailable: evidence content absent"
		if err != nil {
			result.Verdict = "Invalid evidence: " + err.Error()
		} else if found {
			if err := checkMediaReceiptProvenance(ctx, reader, check); err != nil {
				result.Verdict = "Not accepted: " + err.Error()
			} else {
				result.Passed, result.Verdict = mediaValidationVerdict(content.Data, result.Assertions)
			}
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if result.Passed {
			projection.Passed++
		} else {
			projection.Unresolved++
		}
		projection.Checks = append(projection.Checks, result)
	}
	return projection, nil
}

func writeMediaValidation(output *bytes.Buffer, projection *mediaValidationProjection) {
	output.WriteString("## Current validation checkpoint\n\n")
	if projection == nil {
		output.WriteString("No current validation index is recorded.\n\n")
		return
	}
	fmt.Fprintf(output, "The [validation index](%s) records each acquisition's source base and changed component hashes. The default base is `%s`; individual checks can identify a later base. Historical failures retain their original outcomes. This checkpoint does not complete the campaign.\n\n%s\n\n", filepath.Base(mediaValidationPath), projection.Source, projection.Environment)
	fmt.Fprintf(output, "The [assertion contracts](%s) bind required test names to their packages. Passing these assertions alone does not establish current source, environment or model-quality acceptance.\n\n", filepath.Base(mediaAssertionsPath))
	fmt.Fprintf(output, "%d declared checks: %d historical passes; %d unresolved or unsuccessful. Current generation acceptance remains unbound.\n\n", len(projection.Checks), projection.Passed, projection.Unresolved)
	if len(projection.Notes) != 0 {
		output.WriteString("### Acquisition notes (historical)\n\n")
	}
	for _, note := range projection.Notes {
		fmt.Fprintf(output, "%s\n\n", strings.ReplaceAll(note, "](docs/", "]("))
	}
	output.WriteString("| Check | Recorded result | Original source | Scope and observations | Evidence |\n| --- | --- | --- | --- | --- |\n")
	for _, result := range projection.Checks {
		evidence := fmt.Sprintf("`%s`", result.Evidence)
		if result.PriorEvidence.Valid() {
			evidence += fmt.Sprintf("; prior `%s`", result.PriorEvidence)
		}
		fmt.Fprintf(output, "| %s | %s | `%s` | %s | %s |\n", escapeMarkdown(result.Name), escapeMarkdown(result.Verdict), shortCommit(result.Source), escapeMarkdown(result.Scope), evidence)
	}
	output.WriteString("\nResults describe the selected tests, including failures, rather than all supported requests. Commands and retained overlay sources in the index identify each acquisition. Test output is stored by content identity in OvergoDB.\n\n")
	output.WriteString("Samples below retain their original producing-run metadata. A newer validation can reproduce the same content, as noted above. Historical GIFs retain their original timing; they have not been regenerated by this checkpoint.\n\n")
}
