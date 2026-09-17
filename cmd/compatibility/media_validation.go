package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"

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
	Version     uint16                 `json:"version"`
	Source      string                 `json:"source_base"`
	Environment string                 `json:"environment"`
	Notes       []string               `json:"notes"`
	Checks      []mediaValidationCheck `json:"checks"`
}

type mediaValidationCheck struct {
	Name        string                 `json:"name"`
	Source      string                 `json:"source_base,omitzero"`
	Evidence    artifact.ID            `json:"evidence"`
	Command     string                 `json:"command"`
	Scope       string                 `json:"scope"`
	SourceFiles map[string]string      `json:"changed_source_sha256,omitempty"`
	Harness     map[string]artifact.ID `json:"overlay_sources,omitempty"`
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

func mediaValidationVerdict(data []byte, required map[string][]string) string {
	if err := checkMediaTestReceipt("media validation", data, required); err != nil {
		return "Not accepted: " + err.Error()
	}
	return "Passed declared assertions"
}

func writeMediaValidation(ctx context.Context, output *bytes.Buffer, root string, reader artifact.Reader) error {
	output.WriteString("## Current validation checkpoint\n\n")
	var index mediaValidationIndex
	if err := jsonfile.Decode(filepath.Join(root, mediaValidationPath), &index); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			output.WriteString("No current validation index is recorded.\n\n")
			return nil
		}
		return err
	}
	if index.Version != 1 || !gitauthority.ValidObjectID(index.Source) || index.Environment == "" || len(index.Checks) == 0 {
		return errors.New("media validation: incomplete checkpoint identity")
	}
	var assertions map[string]map[string][]string
	if err := jsonfile.DecodeStrict(filepath.Join(root, mediaAssertionsPath), &assertions); err != nil {
		return err
	}
	fmt.Fprintf(output, "The [validation index](%s) records each acquisition's source base and changed component hashes. The default base is `%s`; individual checks can identify a later base. Historical failures retain their original outcomes. This checkpoint does not complete the campaign.\n\n%s\n\n", mediaValidationPath, index.Source, index.Environment)
	fmt.Fprintf(output, "The [assertion contracts](%s) bind required test names to their packages. Passing these assertions alone does not establish current source, environment or model-quality acceptance.\n\n", mediaAssertionsPath)
	for _, note := range index.Notes {
		fmt.Fprintf(output, "%s\n\n", note)
	}
	output.WriteString("| Check | Recorded result | Scope and observations | Evidence |\n| --- | --- | --- | --- |\n")
	seen := map[string]bool{}
	for _, check := range index.Checks {
		if check.Source != "" && !gitauthority.ValidObjectID(check.Source) {
			return errors.New("media validation: invalid acquisition source")
		}
		if check.Name == "" || seen[check.Name] || check.Scope == "" || check.Command == "" || check.Evidence.Kind() != artifact.KindEvidence {
			return errors.New("media validation: invalid or duplicate check")
		}
		seen[check.Name] = true
		content, found, err := artifact.ReadContent(ctx, reader, check.Evidence)
		verdict := "Unavailable: evidence content absent"
		if err != nil {
			verdict = "Invalid evidence: " + err.Error()
		} else if found {
			verdict = mediaValidationVerdict(content.Data, assertions[check.Name])
		}
		fmt.Fprintf(output, "| %s | %s | %s | `%s` |\n", escapeMarkdown(check.Name), escapeMarkdown(verdict), escapeMarkdown(check.Scope), check.Evidence)
	}
	output.WriteString("\nResults describe the selected tests, including failures, rather than all supported requests. Commands and retained overlay sources in the index identify each acquisition. Test output is stored by content identity in OvergoDB.\n\n")
	output.WriteString("Samples below retain their original producing-run metadata. A newer validation can reproduce the same content, as noted above. Historical GIFs retain their original timing; they have not been regenerated by this checkpoint.\n\n")
	return nil
}
