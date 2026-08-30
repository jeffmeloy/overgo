// Package protection verifies the repository-owned part of automation
// protection. Host branch rules and parent-harness hook activation remain
// external facts and are reported without being inferred from configuration.
package protection

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/jsonfile"
)

const policyPath = ".github/protection.json"

// HarnessSettingsPath is the repository-relative harness hook configuration
// this guard verifies. The protection guard is the single production owner of
// that path; other components reference it only through this constant, so the
// go-only ratchet keeps harness-configuration discovery in one place.
const HarnessSettingsPath = ".claude/settings.json"

// HarnessConfigDirectory prefixes every harness-owned configuration path.
const HarnessConfigDirectory = ".claude/"

type policy struct {
	Version                 uint16            `json:"version"`
	ProtectedBranches       []string          `json:"protected_branches"`
	Workflow                string            `json:"workflow"`
	GPUWorkflow             string            `json:"gpu_workflow"`
	RequiredJobs            []string          `json:"required_jobs"`
	RequiredHooks           map[string]string `json:"required_hooks"`
	HostEnforcementRequired bool              `json:"host_enforcement_required"`
	SealedAuthority         sealedPolicy      `json:"sealed_authority"`
}

type sealedPolicy struct {
	Required  bool     `json:"required"`
	Principal string   `json:"principal"`
	Artifacts []string `json:"artifacts"`
}

type settings struct {
	Hooks map[string][]hookGroup `json:"hooks"`
}

type hookGroup struct {
	Matcher string        `json:"matcher,omitempty"`
	Hooks   []hookCommand `json:"hooks"`
}

type hookCommand struct {
	Type    string `json:"type"`
	Command string `json:"command"`
}

// Verify returns the protection facts a gate may safely record. Configured is
// repository evidence; Activated stays unobserved because a child cannot
// attest that its parent harness executed a hook.
func Verify(root string) (configured, activated string, err error) {
	var contract policy
	if err := jsonfile.DecodeStrict(filepath.Join(root, policyPath), &contract); err != nil {
		return "", "", fmt.Errorf("protection policy: %w", err)
	}
	if contract.Version != 1 || len(contract.ProtectedBranches) == 0 ||
		len(contract.RequiredJobs) == 0 || len(contract.RequiredHooks) == 0 ||
		!contract.HostEnforcementRequired || !safeRelative(contract.Workflow) || !safeRelative(contract.GPUWorkflow) ||
		!validSealedPolicy(contract.SealedAuthority) {
		return "", "", errors.New("protection policy: incomplete contract")
	}
	var configuredHooks settings
	if err := jsonfile.DecodeStrict(filepath.Join(root, HarnessSettingsPath), &configuredHooks); err != nil {
		return "", "", fmt.Errorf("hook settings: %w", err)
	}
	for event, command := range contract.RequiredHooks {
		groups := configuredHooks.Hooks[event]
		if !slices.ContainsFunc(groups, func(group hookGroup) bool {
			return slices.ContainsFunc(group.Hooks, func(hook hookCommand) bool {
				return hook.Type == "command" && hook.Command == command
			})
		}) {
			return "", "", fmt.Errorf("hook settings: %s does not configure %q", event, command)
		}
	}
	workflow, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(contract.Workflow)))
	if err != nil {
		return "", "", fmt.Errorf("protection workflow: %w", err)
	}
	workflowText := strings.ReplaceAll(string(workflow), "\r\n", "\n")
	if !strings.Contains(workflowText, "\n  pull_request:") {
		return "", "", errors.New("protection workflow: pull_request trigger missing")
	}
	for _, job := range contract.RequiredJobs {
		if !strings.Contains(workflowText, "\n  "+job+":") {
			return "", "", fmt.Errorf("protection workflow: required job %q missing", job)
		}
	}
	gpuWorkflow, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(contract.GPUWorkflow)))
	if err != nil {
		return "", "", fmt.Errorf("GPU workflow: %w", err)
	}
	gpuText := strings.ReplaceAll(string(gpuWorkflow), "\r\n", "\n")
	for _, required := range []string{"\n  workflow_dispatch:", "runs-on: [self-hosted, windows, x64, gpu]", "run: go run ./cmd/device-lane"} {
		if !strings.Contains(gpuText, required) {
			return "", "", fmt.Errorf("GPU workflow: required contract %q missing", required)
		}
	}
	return fmt.Sprintf("configured:branches=%s,jobs=%s,hooks=%d,sealed=%s;host_enforcement=external",
			strings.Join(contract.ProtectedBranches, "+"), strings.Join(contract.RequiredJobs, "+"), len(contract.RequiredHooks), contract.SealedAuthority.Principal),
		"unobserved:parent-harness-fact", nil
}

func validSealedPolicy(value sealedPolicy) bool {
	want := []string{string(SealedChampionAlias), string(SealedEvaluator), string(SealedGolden), string(SealedPromotionPolicy)}
	got := slices.Clone(value.Artifacts)
	slices.Sort(got)
	return value.Required && strings.HasPrefix(value.Principal, "service:") && slices.Equal(got, want)
}

func safeRelative(path string) bool {
	clean := filepath.ToSlash(filepath.Clean(path))
	return path != "" && !filepath.IsAbs(path) && clean == path && clean != ".." && !strings.HasPrefix(clean, "../")
}
