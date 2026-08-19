package protection

import (
	"path/filepath"
	"testing"

	"overgo/internal/testutil"
)

func TestProtectionContract(t *testing.T) {
	root := t.TempDir()
	testutil.WriteTextFile(t, root, policyPath, `{
  "version": 1,
  "protected_branches": ["master"],
  "workflow": ".github/workflows/test.yml",
  "gpu_workflow": ".github/workflows/gpu.yml",
  "required_jobs": ["unit"],
  "required_hooks": {"PreToolUse": "bash scripts/guard.sh"},
  "sealed_authority": {"required":true,"principal":"service:test","artifacts":["golden","evaluator","promotion-policy","champion-alias"]},
  "host_enforcement_required": true
}`, 0o600)
	testutil.WriteTextFile(t, root, ".claude/settings.json", `{
  "hooks": {"PreToolUse": [{"hooks": [{"type": "command", "command": "bash scripts/guard.sh"}]}]}
}`, 0o600)
	testutil.WriteTextFile(t, root, ".github/workflows/test.yml", "name: test\non:\n  pull_request:\njobs:\n  unit:\n    runs-on: ubuntu-latest\n", 0o600)
	testutil.WriteTextFile(t, root, ".github/workflows/gpu.yml", "name: gpu\non:\n  workflow_dispatch:\njobs:\n  device:\n    runs-on: [self-hosted, windows, x64, gpu]\n    steps:\n      - run: go run ./cmd/device-lane\n", 0o600)
	configured, activated, err := Verify(root)
	if err != nil || configured == "" || activated != "unobserved:parent-harness-fact" {
		t.Fatalf("Verify() = (%q, %q, %v)", configured, activated, err)
	}

	testutil.WriteTextFile(t, root, ".github/workflows/test.yml", "name: test\non: push\njobs:\n  unit:\n", 0o600)
	if _, _, err := Verify(root); err == nil {
		t.Fatal("Verify accepted a workflow without pull_request protection")
	}
}

func TestGPUWorkflowContract(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	configured, _, err := Verify(root)
	if err != nil || configured == "" {
		t.Fatalf("GPU workflow contract = (%q, %v)", configured, err)
	}
}
