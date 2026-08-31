package gate

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/closurescan"
	"overgo/internal/repoanalysis"
)

func TestArchitectureRatchetIncludesGoOnlyPolicy(t *testing.T) {
	rule, found := closurescan.EntryAuthorityRuleFor(closurescan.EntryAuthorityGoOnly)
	if !found || !strings.Contains(rule.Owner, "Go") {
		t.Fatalf("go-only entry authority rule = (found=%t, owner=%q)", found, rule.Owner)
	}
	gate := &gateContext{repo: filepath.Join("..", ".."), paths: []string{"README.md"}}
	if skipped, err := gate.stepArchitecture(); err != nil || skipped {
		t.Fatalf("architecture ratchet over the live tree = (skipped=%t, %v)", skipped, err)
	}
	governed := false
	for _, line := range gate.honesty {
		if strings.HasPrefix(line, "entry authority ratchet:") && strings.Contains(line, "go-only") {
			governed = true
		}
	}
	if !governed {
		t.Fatalf("gate ratchet does not report go-only governance: %q", gate.honesty)
	}
	root := t.TempDir()
	rogue := filepath.Join(root, "internal", "rogue")
	if err := os.MkdirAll(rogue, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(rogue, "rogue.go"),
		[]byte("package rogue\n\nconst deployHook = \"deploy.ps1\"\n"), 0o644,
	); err != nil {
		t.Fatal(err)
	}
	violation, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := closurescan.ValidateEntryAuthorities(
		violation, []closurescan.EntryAuthorityRule{rule},
	); err == nil || !strings.Contains(err.Error(), "internal/rogue/rogue.go bypasses the owner") {
		t.Fatalf("runtime script literal was not refused: %v", err)
	}
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
		"internal/controlleraction/duplicate_admission.go": []byte("package controlleraction\ntype CandidateAdmission struct{}\nfunc AdmitCandidate() {}\n"),
	})
	report := architectureRSIAudit(t, candidate)
	requireArchitectureFinding(t, report, "candidate-admission", "owner")
	bypass := architectureRSIOverlay(t, snapshot, map[string][]byte{
		"internal/loop/bypass_candidate_admission.go": []byte("package loop\nimport \"overgo/internal/runrecord\"\nfunc bypassCandidateAdmission() { _, _ = runrecord.AdmitCandidate() }\n"),
	})
	requireArchitectureFinding(t, architectureRSIAudit(t, bypass), "candidate-admission", "bypass")
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
	bypass := architectureRSIOverlay(t, snapshot, map[string][]byte{
		"internal/loop/bypass_driver_decision.go": []byte("package loop\nimport \"overgo/internal/runrecord\"\nfunc bypassDriverDecision() { _, _ = runrecord.NewDriverDecision(nil, nil, runrecord.DriverDecisionFacts{}) }\n"),
	})
	requireArchitectureFinding(t, architectureRSIAudit(t, bypass), "evidence-driver", "bypass")
}

func TestTrainingEvidenceSinglePublicationAuthority(t *testing.T) {
	requireReservedArchitectureOwner(t, "training-evidence-publication")
}

func TestSequentialControlHasSingleOwner(t *testing.T) {
	requireReservedArchitectureOwner(t, "sequential-control")
}

type reservedArchitectureOwnerCase struct {
	ownerPath, ownerBody         string
	duplicatePath, duplicateBody string
}

var reservedArchitectureOwnerCases = map[string]reservedArchitectureOwnerCase{
	"training-evidence-publication": {
		duplicatePath: "internal/runrecord/evidence_publication.go",
		duplicateBody: "package runrecord\ntype TrainingEvidencePublication struct{}\nfunc PublishTrainingEvidence() {}\n",
	},
	"sequential-control": {
		ownerPath:     "internal/sequentialcontrol/plan.go",
		ownerBody:     "package sequentialcontrol\ntype SequentialControlPlan struct{}\nfunc CalibrateSequentialControl() {}\nfunc EvaluateSequentialControl() {}\n",
		duplicatePath: "internal/trainingworkflow/sequential_control.go",
		duplicateBody: "package trainingworkflow\ntype SequentialControlPlan struct{}\nfunc CalibrateSequentialControl() {}\nfunc EvaluateSequentialControl() {}\n",
	},
}

func requireReservedArchitectureOwner(t *testing.T, family string) {
	t.Helper()
	test, found := reservedArchitectureOwnerCases[family]
	if !found {
		t.Fatalf("unknown reserved architecture family %q", family)
	}
	snapshot := architectureRSISnapshot(t)
	owned := snapshot
	if test.ownerPath != "" {
		owned = architectureRSIOverlay(t, snapshot, map[string][]byte{
			test.ownerPath: []byte(test.ownerBody),
		})
	}
	if report := architectureRSIAudit(t, owned); report.Error() != nil {
		t.Fatalf("reserved %s owner was refused: %v", family, report.Error())
	}
	duplicate := architectureRSIOverlay(t, owned, map[string][]byte{
		test.duplicatePath: []byte(test.duplicateBody),
	})
	requireArchitectureFinding(t, architectureRSIAudit(t, duplicate), family, "reserved-owner")
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
