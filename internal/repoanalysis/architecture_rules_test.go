package repoanalysis_test

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/closurescan"
	"overgo/internal/gosyntax"
	"overgo/internal/repoanalysis"
)

// TestEntryAuthoritiesHoldOverTheTree holds the whole tree to the entry
// authorities: storage, processes, capabilities, tools, triggers, promotion
// and the Go-only runtime each have one owner, and a script named as a
// runtime entry is refused.
func TestEntryAuthoritiesHoldOverTheTree(t *testing.T) {
	t.Parallel()
	var rules []closurescan.EntryAuthorityRule
	for _, domain := range []closurescan.EntryAuthorityDomain{
		closurescan.EntryAuthorityStorage, closurescan.EntryAuthorityProcess,
		closurescan.EntryAuthorityCapability, closurescan.EntryAuthorityTool,
		closurescan.EntryAuthorityTrigger, closurescan.EntryAuthorityPromotion,
		closurescan.EntryAuthorityGoOnly,
	} {
		rule, found := closurescan.EntryAuthorityRuleFor(domain)
		if !found {
			t.Fatalf("the entry authority table lost the %s domain", domain)
		}
		rules = append(rules, rule)
	}
	if _, err := closurescan.ValidateEntryAuthorities(architectureRSISnapshot(t), rules); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	rogue := filepath.Join(root, "internal", "rogue")
	if err := os.MkdirAll(rogue, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(rogue, "rogue.go"), []byte("package rogue\n\nconst deployHook = \"deploy.ps1\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	violation, err := gosyntax.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	goOnly, _ := closurescan.EntryAuthorityRuleFor(closurescan.EntryAuthorityGoOnly)
	if _, err := closurescan.ValidateEntryAuthorities(violation, []closurescan.EntryAuthorityRule{goOnly}); err == nil ||
		!strings.Contains(err.Error(), "internal/rogue/rogue.go bypasses the owner") {
		t.Fatalf("runtime script literal was not refused: %v", err)
	}
	architectureRSIFinding(t, "internal/loop/rogue_markdown.go", "package loop\nimport \"os\"\nfunc loadBehavior() { _, _ = os.ReadFile(\"skill.md\") }\n", "go-only", "non-go-behavior")
}

func TestProductionAuthorityBoundaries(t *testing.T) {
	t.Parallel()
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := gosyntax.DiscoverGo(repo, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	clean, err := repoanalysis.AuditProductionAuthorityBoundaries(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if err := clean.Error(); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name    string
		family  string
		overlay map[string][]byte
	}{
		{name: "storage", family: "storage", overlay: map[string][]byte{
			"internal/server/rogue_commit.go": []byte("package server\nfunc rogueCommit(value interface{ Commit() error }) { _ = value.Commit() }\n"),
		}},
		{name: "aliased process", family: "process", overlay: map[string][]byte{
			"internal/server/rogue_process.go": []byte("package server\nimport runner \"os/exec\"\nfunc rogueProcess() { _ = runner.Command(\"probe\") }\n"),
		}},
		{name: "capability", family: "capability", overlay: map[string][]byte{
			"internal/server/rogue_capability.go": []byte("package server\nimport capability \"overgo/internal/capabilityruntime\"\nfunc rogueCapability(c capability.ExecutorCatalog) { _ = c.Execute() }\n"),
		}},
		{name: "tool", family: "tool", overlay: map[string][]byte{
			"internal/server/rogue_tool.go": []byte("package server\nimport tools \"overgo/internal/agenttool\"\nfunc rogueTool(e *tools.Executor) { _, _ = e.Invoke(nil, tools.Manual{}, nil) }\n"),
		}},
		{name: "active catalog bound through the alias owner", family: "tool", overlay: map[string][]byte{
			"internal/server/rogue_catalog.go": []byte("package server\nimport (\"overgo/internal/agenttool\"; \"overgo/internal/artifact\")\nfunc rogueCatalog(target, held artifact.ID) artifact.AliasBinding { return artifact.AliasMove(agenttool.ActiveCatalogAlias, target, held) }\n"),
		}},
		{name: "trigger", family: "trigger", overlay: map[string][]byte{
			"internal/server/rogue_trigger.go": []byte("package server\nimport records \"overgo/internal/runrecord\"\nvar rogueTrigger = records.CausalContext{Trigger: records.TriggerWebhook}\n"),
		}},
		{name: "promotion", family: "promotion", overlay: map[string][]byte{
			"internal/server/rogue_promotion.go": []byte("package server\nimport comp \"overgo/internal/composition\"\nfunc roguePromotion() { _, _, _, _ = comp.ActiveCompositeGeneration(nil, nil, artifact.ID{}, artifact.ID{}, recipe.Task{}) }\n"),
		}},
		{name: "missing owner", family: "promotion", overlay: map[string][]byte{
			"internal/inference/composite_generation_surface.go": nil,
		}},
		{name: "stale allowance", family: "process", overlay: map[string][]byte{
			"cmd/eval-lane/main.go": nil,
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			candidate, err := snapshot.Overlay(test.overlay)
			if err != nil {
				t.Fatal(err)
			}
			report, err := repoanalysis.AuditProductionAuthorityBoundaries(candidate)
			if err != nil {
				t.Fatal(err)
			}
			if report.Error() == nil {
				t.Fatalf("%s bypass was admitted", test.family)
			}
			if !slices.ContainsFunc(report.Findings, func(finding repoanalysis.ProductionAuthorityFinding) bool {
				return finding.Family == test.family
			}) {
				t.Fatalf("%s bypass produced findings for the wrong family: %+v", test.family, report.Findings)
			}
		})
	}
}

func TestArchitectureRatchetClassifiesCapabilityInvocationBoundaries(t *testing.T) {
	t.Parallel()
	architectureRSIFinding(t, "internal/server/rogue_invocation.go", "package server\nimport tools \"overgo/internal/agenttool\"\nfunc invoke(e *tools.Executor) { _, _, _ = e.InvokeWithEffect(nil, tools.Manual{}, nil) }\n", "invocation-boundary", "unclassified")
}

func TestInternalOrchestrationDoesNotInvokeUTCPTransport(t *testing.T) {
	t.Parallel()
	architectureRSIFinding(t, "internal/loop/rogue_transport.go", "package loop\nimport tools \"overgo/internal/agenttool\"\nfunc routeInternally(e *tools.Executor) { _, _, _ = e.InvokeWithEffect(nil, tools.Manual{}, nil) }\n", "direct-go", "transport-bypass")
}

func TestCrossDomainCandidateHasSingleAdmissionOwner(t *testing.T) {
	t.Parallel()
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
	t.Parallel()
	architectureRSIFinding(t, "internal/composition/duplicate_transition.go", "package composition\nfunc Transition() {}\n", "promotion-lifecycle", "owner")
}

func TestEvidenceDriverHasSingleDecisionOwner(t *testing.T) {
	t.Parallel()
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
	t.Parallel()
	requireReservedArchitectureOwner(t, "training-evidence-publication")
}

func TestSequentialControlHasSingleOwner(t *testing.T) {
	t.Parallel()
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
	t.Parallel()
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

func architectureRSISnapshot(t *testing.T) gosyntax.Snapshot {
	t.Helper()
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := gosyntax.DiscoverGo(repo, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

func architectureRSIOverlay(t *testing.T, snapshot gosyntax.Snapshot, overlay map[string][]byte) gosyntax.Snapshot {
	t.Helper()
	candidate, err := snapshot.Overlay(overlay)
	if err != nil {
		t.Fatal(err)
	}
	return candidate
}

func architectureRSIAudit(t *testing.T, snapshot gosyntax.Snapshot) repoanalysis.ProductionAuthorityReport {
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
