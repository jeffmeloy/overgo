package main

import (
	"path/filepath"
	"testing"

	"overgo/internal/repoanalysis"
)

func TestArchitectureRatchetIncludesGoOnlyPolicy(t *testing.T) {
	architectureRSIFinding(t, "internal/loop/rogue_markdown.go", "package loop\nimport \"os\"\nfunc loadBehavior() { _, _ = os.ReadFile(\"skill.md\") }\n", "go-only", "non-go-behavior")
}

func TestArchitectureRatchetClassifiesCapabilityInvocationBoundaries(t *testing.T) {
	architectureRSIFinding(t, "internal/server/rogue_invocation.go", "package server\nimport tools \"overgo/internal/agenttool\"\nfunc invoke(e *tools.Executor) { _, _, _ = e.InvokeWithEffect(nil, tools.Manual{}, nil) }\n", "invocation-boundary", "unclassified")
}

func TestInternalOrchestrationDoesNotInvokeUTCPTransport(t *testing.T) {
	architectureRSIFinding(t, "internal/loop/rogue_transport.go", "package loop\nimport tools \"overgo/internal/agenttool\"\nfunc routeInternally(e *tools.Executor) { _, _, _ = e.InvokeWithEffect(nil, tools.Manual{}, nil) }\n", "direct-go", "transport-bypass")
}

func TestCrossDomainCandidateHasSingleAdmissionOwner(t *testing.T) {
	snapshot := architectureRSISnapshot(t)
	clean := architectureRSIAudit(t, snapshot)
	if err := clean.Error(); err != nil {
		t.Fatal(err)
	}
	candidate := architectureRSIOverlay(t, snapshot, map[string][]byte{
		"internal/controlleraction/duplicate_admission.go": []byte("package controlleraction\ntype AdmissionBinding struct{}\n"),
	})
	report := architectureRSIAudit(t, candidate)
	requireArchitectureFinding(t, report, "candidate-admission", "owner")
}

func TestCrossDomainPromotionLifecycleHasOneTransitionOwner(t *testing.T) {
	architectureRSIFinding(t, "internal/composition/duplicate_transition.go", "package composition\nfunc Transition() {}\n", "promotion-lifecycle", "owner")
}

func TestEvidenceDriverHasSingleDecisionOwner(t *testing.T) {
	snapshot := architectureRSISnapshot(t)
	owned := architectureRSIOverlay(t, snapshot, map[string][]byte{
		"internal/runrecord/driver_decision.go": []byte("package runrecord\ntype DriverDecision struct{}\nfunc NewDriverDecision() DriverDecision { return DriverDecision{} }\n"),
	})
	if report := architectureRSIAudit(t, owned); report.Error() != nil {
		t.Fatalf("reserved owner was refused: %v", report.Error())
	}
	duplicate := architectureRSIOverlay(t, owned, map[string][]byte{
		"internal/loop/driver_decision.go": []byte("package loop\ntype DriverDecision struct{}\n"),
	})
	report := architectureRSIAudit(t, duplicate)
	requireArchitectureFinding(t, report, "evidence-driver", "reserved-owner")
}

func TestRSIRuntimeIsGoOnly(t *testing.T) {
	snapshot := architectureRSISnapshot(t)
	tests := []struct {
		name string
		path string
		body string
		kind string
	}{
		{
			name: "runtime loader", path: "internal/loop/rogue_plugin.go", kind: "runtime-loader",
			body: "package loop\nimport \"plugin\"\nfunc load() { _, _ = plugin.Open(\"mechanism.so\") }\n",
		},
		{
			name: "interpreter", path: "internal/loop/rogue_python.go", kind: "interpreted-runtime",
			body: "package loop\nimport \"os/exec\"\nfunc run() { _ = exec.Command(\"python\", \"mechanism.py\").Run() }\n",
		},
		{
			name: "markdown behavior", path: "internal/loop/rogue_skill.go", kind: "non-go-behavior",
			body: "package loop\nimport \"os\"\nfunc load() { _, _ = os.ReadFile(\"SKILL.md\") }\n",
		},
		{
			name: "filesystem catalog", path: "internal/agenttool/rogue_catalog.go", kind: "filesystem-capability-discovery",
			body: "package agenttool\nimport \"os\"\nfunc discover() { _, _ = os.ReadDir(\"capabilities\") }\n",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			candidate := architectureRSIOverlay(t, snapshot, map[string][]byte{test.path: []byte(test.body)})
			report := architectureRSIAudit(t, candidate)
			requireArchitectureFinding(t, report, "go-only", test.kind)
		})
	}
}

func architectureRSIFinding(t *testing.T, path, body, family, kind string) {
	t.Helper()
	snapshot := architectureRSISnapshot(t)
	candidate := architectureRSIOverlay(t, snapshot, map[string][]byte{path: []byte(body)})
	report := architectureRSIAudit(t, candidate)
	requireArchitectureFinding(t, report, family, kind)
}

func architectureRSISnapshot(t *testing.T) repoanalysis.SourceSnapshot {
	t.Helper()
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := repoanalysis.DiscoverGo(repo, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

func architectureRSIOverlay(t *testing.T, snapshot repoanalysis.SourceSnapshot, overlay map[string][]byte) repoanalysis.SourceSnapshot {
	t.Helper()
	candidate, err := snapshot.Overlay(overlay)
	if err != nil {
		t.Fatal(err)
	}
	return candidate
}

func architectureRSIAudit(t *testing.T, snapshot repoanalysis.SourceSnapshot) repoanalysis.ProductionAuthorityReport {
	t.Helper()
	report, err := repoanalysis.AuditProductionAuthorityBoundaries(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	return report
}

func requireArchitectureFinding(t *testing.T, report repoanalysis.ProductionAuthorityReport, family, kind string) {
	t.Helper()
	for _, finding := range report.Findings {
		if finding.Family == family && finding.Kind == kind {
			return
		}
	}
	t.Fatalf("missing %s/%s finding: %+v", family, kind, report.Findings)
}
