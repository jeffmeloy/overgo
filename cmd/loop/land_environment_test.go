package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/processcontrol"
	"overgo/internal/testcheck"
)

// TestALandingTakesItsEnvironmentFromTheConfiguration holds a landing's tools to the environment the loop's
// configuration names (fail-early-runs/land-environment): every landing was started with a typed prefix, the role, the
// data root and variables unset, and on 2026-11-11 guidance-ops/moss-presets, landed with no data root, was refused after
// its full lanes for want of cuDNN. A session that sets neither the role nor the data root, and holds the card's and the
// live store's opt-ins, starts a landing's tool: the tool runs under the configured role and data root, and with neither
// opt-in.
func TestALandingTakesItsEnvironmentFromTheConfiguration(t *testing.T) {
	const child = "OVERGO_LANDING_ENVIRONMENT_CHILD"
	shown := []string{"OVERGO_AUTOMATION_ROLE", "OVERGO_DATA_ROOT", "OVERGO_CUDA_TEST", "OVERGO_STORE_ACCEPTANCE"}
	if os.Getenv(child) != "" {
		for _, name := range shown {
			fmt.Printf("%s=%s\n", name, os.Getenv(name))
		}
		return
	}
	root := t.TempDir()
	data := filepath.Join(root, "data")
	configuration := testcheck.Must(json.Marshal(map[string]any{
		"name": "fixture", "environment": map[string]string{"OVERGO_AUTOMATION_ROLE": "master-lead", "OVERGO_DATA_ROOT": data},
	}))(t)
	testcheck.NoError(t, os.MkdirAll(filepath.Join(root, ".overgo-runtime"), 0o755))
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, ".overgo-runtime", processcontrol.LoopConfigFile), configuration, 0o600))
	t.Chdir(root)
	t.Setenv(child, "1")
	t.Setenv("OVERGO_AUTOMATION_ROLE", "")
	t.Setenv("OVERGO_DATA_ROOT", "")
	t.Setenv("OVERGO_CUDA_TEST", "1")
	t.Setenv("OVERGO_STORE_ACCEPTANCE", "1")
	var printed bytes.Buffer
	testcheck.Must((&execRowWorld{}).tool(&printed, os.Args[0], "-test.run=^TestALandingTakesItsEnvironmentFromTheConfiguration$"))(t)
	got := map[string]string{}
	for line := range strings.Lines(printed.String()) {
		if name, value, found := strings.Cut(strings.TrimSpace(line), "="); found {
			got[name] = value
		}
	}
	want := map[string]string{"OVERGO_AUTOMATION_ROLE": "master-lead", "OVERGO_DATA_ROOT": data, "OVERGO_CUDA_TEST": "", "OVERGO_STORE_ACCEPTANCE": ""}
	for _, name := range shown {
		if got[name] != want[name] {
			t.Errorf("a landing's tool ran with %s=%q; want %q", name, got[name], want[name])
		}
	}
}
