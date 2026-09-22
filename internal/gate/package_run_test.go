package gate

import (
	"context"
	"errors"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testevidence"
)

// TestPackageRunRecordsReceipt holds a standalone package run to recording
// the gate's receipt for the package's obligation exactly as a landing does:
// a passing run leaves a passed receipt naming its tests, a failing run a
// failed one, and a run that ends without a terminal verdict is an error
// that leaves the obligation open.
func TestPackageRunRecordsReceipt(t *testing.T) {
	t.Parallel()
	const pkg = "fixture/good"
	for _, tc := range []struct {
		name    string
		passed  bool
		observe bool
		tests   map[string]string
	}{
		{name: "passed", passed: true, observe: true, tests: map[string]string{"TestImageVideoWanAcceptance": "pass"}},
		{name: "failed", passed: false, observe: true},
		{name: "no verdict", observe: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			g, inputs := terminalEvidenceFixture(t, root)
			run := func(_ context.Context, packages []string, short bool, observe func(string, bool, map[string]string) error, _ bool) (testevidence.GoTestReport, error) {
				if len(packages) != 1 || packages[0] != pkg || short {
					t.Fatalf("run received packages=%v short=%v", packages, short)
				}
				if !tc.observe {
					return testevidence.GoTestReport{}, nil
				}
				if err := observe(pkg, tc.passed, tc.tests); err != nil {
					return testevidence.GoTestReport{}, err
				}
				if !tc.passed {
					return testevidence.GoTestReport{}, errors.New("go test evidence: fixture/good failed")
				}
				return testevidence.GoTestReport{PassedTests: 1, PassedPackages: 1}, nil
			}
			err := g.runPackageReceipt(t.Context(), pkg, receiptModeComplete, inputs[pkg], run)
			if err := g.closeStore(); err != nil {
				t.Fatal(err)
			}
			if tc.observe == (err != nil) && tc.passed {
				t.Fatalf("run error = %v", err)
			}
			if !tc.observe {
				if err == nil {
					t.Fatal("a run without a terminal verdict was accepted")
				}
				return
			}
			store, err := overgodb.OpenReadOnly(filepath.Join(root, StorePath))
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			obligation := packageObligationFor(t, g, pkg, receiptModeComplete, inputs[pkg])
			receipt, found, err := packageReceiptCodec.Resolve(t.Context(), store, packageReceiptAlias+obligation.String())
			if err != nil || !found {
				t.Fatalf("receipt found=%v err=%v", found, err)
			}
			if receipt.Passed != tc.passed || tc.passed && receipt.Tests["TestImageVideoWanAcceptance"] != "pass" {
				t.Fatalf("receipt = %+v", receipt)
			}
		})
	}
}

// packageObligationFor computes the obligation identity a read of the receipt
// would key by, from the same pieces the ledger uses.
func packageObligationFor(t *testing.T, g *gateContext, pkg, mode string, input artifact.ID) artifact.ID {
	t.Helper()
	_, obligation, err := packageObligation(pkg, mode, input, g.environment.ID)
	if err != nil {
		t.Fatal(err)
	}
	return obligation.ID
}
