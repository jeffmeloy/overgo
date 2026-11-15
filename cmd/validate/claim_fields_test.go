package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAClaimHoldsNoStatusOrVerify holds the compatibility manifest's claims to neither a status nor a verify
// (single-source/claim-fields): every claim's status read "implemented" and its verify was the command its test
// artifact derives, which the manifest refused any other of, so both were copies. The matrix prints the derived
// command, a device claim's with the switch that enables the device.
func TestAClaimHoldsNoStatusOrVerify(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	var document struct {
		Claims []map[string]json.RawMessage `json:"claims"`
	}
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join(root, manifestPath)))(t), &document))
	for _, fields := range document.Claims {
		for _, copied := range []string{"status", "verify"} {
			if _, held := fields[copied]; held {
				t.Errorf("claim %s holds a %s", fields["id"], copied)
			}
		}
	}
	claims := testcheck.Must(loadManifest(root))(t).Claims
	matrix := string(testcheck.Must(generate(root))(t))
	for _, item := range claims {
		command := testcheck.Must(item.command(root))(t)
		if !strings.Contains(matrix, "verify: `"+item.shown(command)+"`") {
			t.Errorf("the matrix does not print claim %s's command %q", item.ID, item.shown(command))
		}
	}
}
