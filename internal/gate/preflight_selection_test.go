package gate

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/runrecord"
)

// Preflight reports acceptance, failures and source-bound selection.
// Gate-only changes retain owner tests and exclude the contract's model lanes.
func TestPreflightRunsAcceptanceAndReportsSelection(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	if err := os.MkdirAll(filepath.Join(repo, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	document := `{"campaign":"t","doctrine":"d","items":[{"id":"r0","status":"open","steps":[` +
		`{"id":"do","status":"open","verify":"echo verified"},` +
		`{"id":"broken","status":"open","verify":"exit 3"},` +
		`{"id":"bare","status":"open"}]}]}`
	if err := os.WriteFile(filepath.Join(repo, "docs", "plan.json"), []byte(document), 0o644); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	g := &gateContext{repo: repo, planRef: "r0/do", preflight: true}
	if err := g.preflightAcceptance(t.Context(), &output); err != nil {
		t.Fatalf("acceptance = %v\n%s", err, output.String())
	}
	if !strings.Contains(output.String(), "preflight: acceptance ok ") || !strings.Contains(output.String(), "verdict=") || !strings.Contains(output.String(), ": echo verified") {
		t.Fatalf("acceptance output:\n%s", output.String())
	}
	output.Reset()
	broken := &gateContext{repo: repo, planRef: "r0/broken", preflight: true}
	if err := broken.preflightAcceptance(t.Context(), &output); err == nil || !strings.Contains(err.Error(), "r0/broken") || !strings.Contains(output.String(), "preflight: acceptance FAIL ") {
		t.Fatalf("failing verify = %v\n%s", err, output.String())
	}
	output.Reset()
	bare := &gateContext{repo: repo, planRef: "r0/bare", preflight: true}
	if err := bare.preflightAcceptance(t.Context(), &output); err != nil || !strings.Contains(output.String(), "r0/bare declares no verify") {
		t.Fatalf("verify-less step = %v\n%s", err, output.String())
	}
	if err := (&gateContext{repo: repo, planRef: "r0/absent"}).preflightAcceptance(t.Context(), &output); err == nil || !strings.Contains(err.Error(), "absent from the plan") {
		t.Fatalf("absent step = %v", err)
	}
	output.Reset()
	if err := (&gateContext{repo: repo}).preflightAcceptance(t.Context(), &output); err != nil || !strings.Contains(output.String(), "-plan names no row") {
		t.Fatalf("no row = %v\n%s", err, output.String())
	}

	output.Reset()
	report := gatePlanReport{
		Selected: []planDisposition{{Name: "vet", Phase: runrecord.PhaseValidate}, {Name: testOwnersCheckName}, {Name: "device"}, {Name: "commit"}},
		Excluded: []planDisposition{{Name: "sbom", Reason: "no dependency authority changed"}, {Name: "webui-lane", Reason: "no web UI changed"}},
	}
	groups := selectionGroups{
		owners: []string{"overgo/internal/gate"}, short: []string{"overgo/internal/gate", "overgo/cmd/gate"},
		complete: []string{"overgo/cmd/plan"}, devices: []string{"overgo/internal/gate"},
		shortReused: 3, completeReused: 40, excluded: 200,
	}
	writeSelectionReport(&output, report, groups)
	want := "preflight: selection: checks selected=4 excluded=2 unresolved=0; excluded=[sbom,webui-lane]\n" +
		"preflight: selection: lanes selected=[device] excluded=[test-device,webui-lane,model-journey]; a selected lane runs after the commit unless a failed obligation forces it inline\n" +
		"preflight: selection: test scope: short group 2 pending + 3 reused receipts, complete group 1 pending + 40 reused, 200 packages excluded, 1 under the device lease\n" +
		"preflight: selection: owners first=[overgo/internal/gate]\n" +
		"preflight: selection: short pending=[overgo/internal/gate,overgo/cmd/gate]\n" +
		"preflight: selection: complete pending=[overgo/cmd/plan]\n" +
		"preflight: selection: device lease=[overgo/internal/gate]\n"
	if output.String() != want {
		t.Fatalf("selection report:\n%s\nwant:\n%s", output.String(), want)
	}

	liveRepositoryFixture(t).plan(t, []string{"internal/gate/preflight.go"}, func(g *gateContext) {
		var live bytes.Buffer
		if err := g.preflightSelection(&live); err != nil {
			t.Fatal(err)
		}
		text := live.String()
		for _, want := range []string{
			"preflight: selection: checks selected=", "preflight: selection: lanes selected=[test-device]", "excluded=[" + strings.Join(testingReductionContract().Consumer, ",") + "]",
			"preflight: selection: owners first=[overgo/internal/gate]", "preflight: selection: test scope: short group ",
		} {
			if !strings.Contains(text, want) {
				t.Errorf("live selection lacks %q:\n%s", want, text)
			}
		}
	})
}
