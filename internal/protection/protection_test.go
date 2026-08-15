package protection

import (
	"os"
	"path/filepath"
	"testing"
)

func TestProtectionContract(t *testing.T) {
	root := t.TempDir()
	writeFixture(t, root, policyPath, `{
  "version": 1,
  "protected_branches": ["master"],
  "workflow": ".github/workflows/test.yml",
  "gpu_workflow": ".github/workflows/gpu.yml",
  "required_jobs": ["unit"],
  "required_hooks": {"PreToolUse": "bash scripts/guard.sh"},
  "host_enforcement_required": true
}`)
	writeFixture(t, root, ".claude/settings.json", `{
  "hooks": {"PreToolUse": [{"hooks": [{"type": "command", "command": "bash scripts/guard.sh"}]}]}
}`)
	writeFixture(t, root, ".github/workflows/test.yml", "name: test\non:\n  pull_request:\njobs:\n  unit:\n    runs-on: ubuntu-latest\n")
	writeFixture(t, root, ".github/workflows/gpu.yml", "name: gpu\non:\n  workflow_dispatch:\njobs:\n  device:\n    runs-on: [self-hosted, windows, x64, gpu]\n    steps:\n      - run: go run ./cmd/device-lane\n")
	configured, activated, err := Verify(root)
	if err != nil || configured == "" || activated != "unobserved:parent-harness-fact" {
		t.Fatalf("Verify() = (%q, %q, %v)", configured, activated, err)
	}

	writeFixture(t, root, ".github/workflows/test.yml", "name: test\non: push\njobs:\n  unit:\n")
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

func writeFixture(t *testing.T, root, name, content string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(name))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}
