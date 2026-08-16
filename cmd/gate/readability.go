package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"runtime"
	"strings"

	"overgo/internal/gostyle"
	"overgo/internal/repoanalysis"
)

func readabilityAdmissionActive() bool { return gostyle.EnforcementActive() }

func runReadabilityReport(repo, pathsCSV string, output io.Writer) error {
	gate := gateContext{repo: repo, stepEvidence: map[string]string{}}
	for _, path := range strings.Split(pathsCSV, ",") {
		if path = strings.TrimSpace(path); path != "" {
			gate.paths = append(gate.paths, filepath.ToSlash(path))
		}
	}
	if len(gate.paths) == 0 {
		return errors.New("readability report requires -paths")
	}
	if err := gate.expandDirectoryPaths(); err != nil {
		return err
	}
	_, err := gate.stepReadability()
	evidence := gate.stepEvidence["readability"]
	if evidence == "" {
		return errors.New(strings.Join(gate.honesty, "; "))
	}
	fmt.Fprintln(output, evidence)
	return err
}

type readabilityRuleEvidence struct {
	ID         string `json:"id"`
	Maturity   string `json:"m"`
	Selected   bool   `json:"selected"`
	SkipReason string `json:"skip,omitempty"`
	Introduced int    `json:"introduced,omitempty"`
	Resolved   int    `json:"resolved,omitempty"`
}

type readabilityEvidence struct {
	BaseIdentity      string                    `json:"base"`
	CandidateIdentity string                    `json:"candidate"`
	BuildContext      string                    `json:"context"`
	Rules             []readabilityRuleEvidence `json:"rules"`
	Analysis          gostyle.AnalysisStats     `json:"analysis"`
	AllocatedBytes    uint64                    `json:"allocated_bytes"`
	HeapBytesAfter    uint64                    `json:"heap_after"`
	PeakMemory        string                    `json:"peak_memory"`
	Blocking          []string                  `json:"blocking,omitempty"`
}

func (g *gateContext) stepReadability() (bool, error) {
	if len(g.changedGoFiles()) == 0 {
		return true, nil
	}
	if g.profileDirty {
		g.honesty = append(g.honesty, "go readability: unavailable because unplanned Go dirt contaminates candidate evidence")
		return true, nil
	}
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	selection, err := repoanalysis.HostBuildSelection(g.repo, "./internal/...", "./cmd/...")
	if err != nil {
		return false, err
	}
	candidate, err := g.sourceSnapshot()
	if err != nil {
		return false, err
	}
	base, err := sourceAtHEAD(g.repo, candidate)
	if err != nil {
		return false, err
	}
	report, err := gostyle.Compare(base, candidate, selection)
	if err != nil {
		return false, err
	}
	runtime.ReadMemStats(&after)
	evidence := readabilityEvidence{
		BaseIdentity: report.Base.Identity, CandidateIdentity: report.Candidate.Identity,
		BuildContext: report.Candidate.BuildContext, Analysis: report.Analysis,
		AllocatedBytes: after.TotalAlloc - before.TotalAlloc, HeapBytesAfter: after.Alloc,
		PeakMemory: "unavailable; allocation and ending heap recorded",
		Rules:      make([]readabilityRuleEvidence, 0, len(report.Rules)),
	}
	for _, delta := range report.Rules {
		rule := readabilityRuleEvidence{
			ID: delta.Rule.ID, Maturity: string(delta.Rule.Maturity),
			Selected: delta.Selected, SkipReason: delta.SkipReason,
			Introduced: len(delta.Introduced), Resolved: len(delta.Resolved),
		}
		if delta.Blocking {
			for _, diagnostic := range delta.Introduced {
				evidence.Blocking = append(evidence.Blocking, delta.Rule.ID+":"+diagnostic.File)
			}
		}
		evidence.Rules = append(evidence.Rules, rule)
	}
	structured, err := json.Marshal(evidence)
	if err != nil {
		return false, err
	}
	g.stepEvidence["readability"] = string(structured)
	g.honesty = append(g.honesty, fmt.Sprintf(
		"go readability: context=%s scanned=%d reused=%d passes=%d diagnostics=%d allocated_bytes=%d enforceable_regressions=%d; observe-only rules remain non-blocking",
		evidence.BuildContext, evidence.Analysis.FilesScanned, evidence.Analysis.FilesReused,
		evidence.Analysis.SyntaxPasses, evidence.Analysis.Diagnostics, evidence.AllocatedBytes, len(evidence.Blocking),
	))
	if len(evidence.Blocking) > 0 {
		return false, fmt.Errorf("new enforceable Go readability findings: %s", strings.Join(evidence.Blocking, ", "))
	}
	return false, nil
}
