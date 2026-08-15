package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"runtime"
	"strings"

	"overgo/internal/gostyle"
	"overgo/internal/repoanalysis"
)

type readabilityRuleEvidence struct {
	ID         string           `json:"id"`
	Maturity   gostyle.Maturity `json:"m"`
	Selected   bool             `json:"selected"`
	SkipReason string           `json:"skip,omitempty"`
	Introduced int              `json:"introduced,omitempty"`
	Resolved   int              `json:"resolved,omitempty"`
}

type readabilityEvidence struct {
	BaseIdentity       string                    `json:"base"`
	CandidateIdentity  string                    `json:"candidate"`
	BuildContext       string                    `json:"context"`
	Rules              []readabilityRuleEvidence `json:"rules"`
	Analysis           gostyle.AnalysisStats     `json:"analysis"`
	EncodedReportBytes int                       `json:"report_bytes"`
	ReportDigest       string                    `json:"report_sha256"`
	AllocatedBytes     uint64                    `json:"allocated_bytes"`
	HeapBytesAfter     uint64                    `json:"heap_after"`
	PeakMemory         string                    `json:"peak_memory"`
	Blocking           []string                  `json:"blocking,omitempty"`
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
	report, err := gostyle.Compare(base, candidate, gostyle.Options{Build: selection})
	if err != nil {
		return false, err
	}
	encoded, err := json.Marshal(report)
	if err != nil {
		return false, err
	}
	runtime.ReadMemStats(&after)
	evidence := readabilityEvidence{
		BaseIdentity: report.Base.Identity, CandidateIdentity: report.Candidate.Identity,
		BuildContext: report.Candidate.BuildContext, Analysis: report.Analysis,
		EncodedReportBytes: len(encoded), ReportDigest: fmt.Sprintf("%x", sha256.Sum256(encoded)),
		AllocatedBytes: after.TotalAlloc - before.TotalAlloc, HeapBytesAfter: after.Alloc,
		PeakMemory: "unavailable; allocation and ending heap recorded",
		Rules:      make([]readabilityRuleEvidence, 0, len(report.Rules)),
	}
	for _, delta := range report.Rules {
		rule := readabilityRuleEvidence{
			ID: delta.Rule.ID, Maturity: delta.Rule.Maturity,
			Selected: delta.Selected, SkipReason: compactSkipReason(delta.Rule.Mechanism, delta.Selected),
		}
		for _, change := range delta.Changes {
			switch change.State {
			case gostyle.Introduced:
				rule.Introduced++
			case gostyle.Resolved:
				rule.Resolved++
			}
			if delta.Rule.Maturity == gostyle.Enforceable && change.State == gostyle.Introduced {
				evidence.Blocking = append(evidence.Blocking, delta.Rule.ID+":"+change.Diagnostic.File)
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
		"go readability: context=%s scanned=%d reused=%d passes=%d diagnostics=%d report_bytes=%d allocated_bytes=%d enforceable_regressions=%d; observe-only rules remain non-blocking",
		evidence.BuildContext, evidence.Analysis.FilesScanned, evidence.Analysis.FilesReused,
		evidence.Analysis.SyntaxPasses, evidence.Analysis.Diagnostics, evidence.EncodedReportBytes,
		evidence.AllocatedBytes, len(evidence.Blocking),
	))
	if len(evidence.Blocking) > 0 {
		return false, fmt.Errorf("new enforceable Go readability findings: %s", strings.Join(evidence.Blocking, ", "))
	}
	return false, nil
}

func compactSkipReason(mechanism gostyle.Mechanism, selected bool) string {
	if selected {
		return ""
	}
	switch mechanism {
	case gostyle.Toolchain:
		return "toolchain-owner"
	case gostyle.TypeAware:
		return "type-pass-not-landed"
	case gostyle.Delegated:
		return "codeprofile-owner"
	default:
		return "unsupported"
	}
}
