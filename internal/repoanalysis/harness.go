package repoanalysis

import (
	"fmt"
	"go/ast"
	"go/token"
	"path/filepath"
	"sort"
	"strings"
)

// HarnessOwnerRule declares the sole package allowed to own one agent-harness
// authority. These are semantic ownership boundaries, not a file census.
type HarnessOwnerRule struct {
	Kind   string
	Symbol string
	Owner  string
}

// HarnessConsolidationFinding is stable, machine-readable audit output.
type HarnessConsolidationFinding struct {
	Kind     string
	Symbol   string
	Expected string
	Actual   string
	File     string
}

type AgentHarnessConsolidation struct {
	SourceIdentity string
	Rules          int
	Declarations   int
	Findings       []HarnessConsolidationFinding
}

var agentHarnessOwnerRules = []HarnessOwnerRule{
	{Kind: "type", Symbol: "Manual", Owner: "internal/agenttool"},
	{Kind: "type", Symbol: "InvocationEffect", Owner: "internal/agenttool"},
	{Kind: "type", Symbol: "CatalogSnapshot", Owner: "internal/agenttool"},
	{Kind: "func", Symbol: "DeriveInvocationEffect", Owner: "internal/agenttool"},
	{Kind: "type", Symbol: "AgentTaskContract", Owner: "internal/recipe"},
	{Kind: "type", Symbol: "DelegatedAgentInvocation", Owner: "internal/recipe"},
	{Kind: "type", Symbol: "SteeringProposal", Owner: "internal/recipe"},
	{Kind: "type", Symbol: "AgentObligation", Owner: "internal/runrecord"},
	{Kind: "type", Symbol: "AgentMutationCheckpoint", Owner: "internal/runrecord"},
	{Kind: "type", Symbol: "InteractionTrace", Owner: "internal/runrecord"},
	{Kind: "type", Symbol: "AttemptRecord", Owner: "internal/runrecord"},
	{Kind: "type", Symbol: "AttemptDirection", Owner: "internal/runrecord"},
	{Kind: "type", Symbol: "AutomationPolicyLifecycle", Owner: "internal/runrecord"},
	{Kind: "func", Symbol: "PublishAutomationPolicyTransition", Owner: "internal/runrecord"},
	{Kind: "type", Symbol: "WorkspaceClaims", Owner: "internal/plan"},
	{Kind: "func", Symbol: "AdmitSteeringProposal", Owner: "internal/plan"},
	{Kind: "type", Symbol: "WorkspaceClaimLifecycle", Owner: "internal/operation"},
	{Kind: "type", Symbol: "ContractState", Owner: "internal/agentloop"},
	{Kind: "type", Symbol: "MutationCheckpointRuntime", Owner: "internal/agentloop"},
	{Kind: "type", Symbol: "StrategyExperiment", Owner: "internal/loop"},
	{Kind: "type", Symbol: "StrategyComparison", Owner: "internal/loop"},
	{Kind: "func", Symbol: "CompareStrategyExperiment", Owner: "internal/loop"},
	{Kind: "type", Symbol: "ClosureConfig", Owner: "internal/loop"},
}

var supersededHarnessPackages = map[string]bool{
	"internal/automationpolicy": true,
}

// AuditAgentHarnessConsolidation proves that every harness authority still has
// one declaration in its designated common owner and that superseded parallel
// packages have not returned.
func AuditAgentHarnessConsolidation(snapshot SourceSnapshot) (AgentHarnessConsolidation, error) {
	report := AgentHarnessConsolidation{SourceIdentity: snapshot.Identity(), Rules: len(agentHarnessOwnerRules)}
	type declaration struct{ owner, file string }
	declarations := map[string][]declaration{}
	for _, source := range snapshot.Files {
		if source.Test {
			continue
		}
		generated, err := source.Generated()
		if err != nil {
			return AgentHarnessConsolidation{}, err
		}
		if generated {
			continue
		}
		owner := filepath.ToSlash(filepath.Dir(source.Path))
		if supersededHarnessPackages[owner] {
			report.Findings = append(report.Findings, HarnessConsolidationFinding{Kind: "package", Symbol: owner, Expected: "absent", Actual: owner, File: source.Path})
		}
		file, err := source.Syntax()
		if err != nil {
			return AgentHarnessConsolidation{}, err
		}
		for _, item := range file.Decls {
			switch value := item.(type) {
			case *ast.GenDecl:
				if value.Tok != token.TYPE {
					continue
				}
				for _, spec := range value.Specs {
					name := spec.(*ast.TypeSpec).Name.Name
					declarations["type\x00"+name] = append(declarations["type\x00"+name], declaration{owner: owner, file: source.Path})
				}
			case *ast.FuncDecl:
				if value.Recv == nil {
					declarations["func\x00"+value.Name.Name] = append(declarations["func\x00"+value.Name.Name], declaration{owner: owner, file: source.Path})
				}
			}
		}
	}
	for _, rule := range agentHarnessOwnerRules {
		found := declarations[rule.Kind+"\x00"+rule.Symbol]
		report.Declarations += len(found)
		if len(found) == 0 {
			report.Findings = append(report.Findings, HarnessConsolidationFinding{Kind: "missing", Symbol: rule.Symbol, Expected: rule.Owner})
			continue
		}
		for _, actual := range found {
			if actual.owner != rule.Owner || len(found) != 1 {
				report.Findings = append(report.Findings, HarnessConsolidationFinding{Kind: "owner", Symbol: rule.Symbol, Expected: rule.Owner, Actual: actual.owner, File: actual.file})
			}
		}
	}
	sort.Slice(report.Findings, func(i, j int) bool {
		left, right := report.Findings[i], report.Findings[j]
		return strings.Join([]string{left.Kind, left.Symbol, left.Actual, left.File}, "\x00") < strings.Join([]string{right.Kind, right.Symbol, right.Actual, right.File}, "\x00")
	})
	return report, nil
}

func (report AgentHarnessConsolidation) Error() error {
	if len(report.Findings) == 0 {
		return nil
	}
	first := report.Findings[0]
	return fmt.Errorf("agent harness consolidation: %d finding(s); first=%s %s expected=%s actual=%s", len(report.Findings), first.Kind, first.Symbol, first.Expected, first.Actual)
}
