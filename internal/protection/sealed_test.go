package protection

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestCandidatePrincipalCannotWriteSealedAuthority is the live sealing
// verifier. It always runs against the real repository: a malformed manifest
// fails unconditionally, and a present manifest is a sealing claim that must
// hold -- every sealed file hashing to its provenance digest and carrying an
// explicit write-deny ACE for the candidate principal. Only under
// OVERGO_SEALED_AUTHORITY_TEST=1 does it additionally DEMAND that sealing is
// provisioned: that is the row verifier, and it fails with the provisioning
// runbook until the owner seals the host outside any candidate-reachable
// process.
func TestCandidatePrincipalCannotWriteSealedFiles(t *testing.T) {
	root := filepath.Join("..", "..")
	authority, sealed, err := LoadSealedFileAuthority(root)
	if err != nil {
		t.Fatal(err)
	}
	demanded := os.Getenv("OVERGO_SEALED_AUTHORITY_TEST") == "1"
	if !sealed {
		if demanded {
			t.Fatal("external-prereq: owner must author .github/sealed_authority.json naming the candidate principal, the sealed files (goldens, evaluator definitions, promotion policy, champion aliases) and per-file provenance, then provision the principal and write-deny ACLs outside any candidate-reachable process")
		}
		t.Log("sealing not provisioned; manifest absence verified (set OVERGO_SEALED_AUTHORITY_TEST=1 to demand provisioned sealing)")
		return
	}
	if err := authority.Verify(root); err != nil {
		t.Fatal(err)
	}
	t.Logf("sealed: principal=%s files=%d provenance verified against digests and deny ACEs",
		authority.CandidatePrincipal, len(authority.SealedPaths))
}

// TestSealedAuthorityManifestValidation pins the manifest contract without
// any provisioned host state: absence reports unsealed, a complete manifest
// loads, partial sealing claims are refused, and a sealed file that drifts
// from its provenance digest is refused before any ACL is consulted.
func TestSealedAuthorityManifestValidation(t *testing.T) {
	root := t.TempDir()
	if _, sealed, err := LoadSealedFileAuthority(root); err != nil || sealed {
		t.Fatalf("missing manifest = (sealed=%v, %v), want unsealed without error", sealed, err)
	}
	golden := "docs/golden.json"
	if err := os.MkdirAll(filepath.Join(root, ".github"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	content := []byte(`{"champion":"model-a"}`)
	if err := os.WriteFile(filepath.Join(root, "docs", "golden.json"), content, 0o644); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(content)
	manifest := SealedFileAuthority{
		Version: 1, CandidatePrincipal: "OVERGO-CANDIDATE",
		SealedPaths: []string{golden},
		Provenance: []GoldenProvenance{{
			Path: golden, Tool: "cmd/serving-exact", Version: "v1.0.0", SHA256: hex.EncodeToString(digest[:]),
		}},
	}
	write := func(value SealedFileAuthority) {
		data, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, ".github", "sealed_authority.json"), data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write(manifest)
	authority, sealed, err := LoadSealedFileAuthority(root)
	if err != nil || !sealed || authority.CandidatePrincipal != "OVERGO-CANDIDATE" {
		t.Fatalf("complete manifest = (%+v, %v, %v)", authority, sealed, err)
	}

	for name, mutate := range map[string]func(*SealedFileAuthority){
		"no principal":         func(a *SealedFileAuthority) { a.CandidatePrincipal = " " },
		"no sealed paths":      func(a *SealedFileAuthority) { a.SealedPaths = nil },
		"absolute sealed path": func(a *SealedFileAuthority) { a.SealedPaths = []string{"C:/evil"} },
		"escaping sealed path": func(a *SealedFileAuthority) { a.SealedPaths = []string{"../outside"} },
		"missing provenance":   func(a *SealedFileAuthority) { a.Provenance = nil },
		"unsealed provenance": func(a *SealedFileAuthority) {
			a.Provenance[0].Path = "docs/other.json"
		},
		"anonymous tool": func(a *SealedFileAuthority) { a.Provenance[0].Tool = "" },
		"no version":     func(a *SealedFileAuthority) { a.Provenance[0].Version = " " },
		"bad digest":     func(a *SealedFileAuthority) { a.Provenance[0].SHA256 = "zz" },
	} {
		broken := manifest
		broken.SealedPaths = append([]string(nil), manifest.SealedPaths...)
		broken.Provenance = append([]GoldenProvenance(nil), manifest.Provenance...)
		mutate(&broken)
		write(broken)
		if _, _, err := LoadSealedFileAuthority(root); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}

	write(manifest)
	if err := os.WriteFile(filepath.Join(root, "docs", "golden.json"), []byte(`{"champion":"model-b"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	authority, _, err = LoadSealedFileAuthority(root)
	if err != nil {
		t.Fatal(err)
	}
	if err := authority.Verify(root); err == nil {
		t.Fatal("golden drifted from its provenance digest yet verification passed")
	}
}

// TestDenyWriteListing pins the icacls deny parsing: a write deny for the
// candidate principal satisfies sealing; grants, denies for other
// principals, or read-only denies do not.
func TestDenyWriteListing(t *testing.T) {
	principal := "OVERGO-CANDIDATE"
	for listing, want := range map[string]bool{
		"docs\\golden.json HOST\\OVERGO-CANDIDATE:(DENY)(W)\n":     true,
		"docs\\golden.json HOST\\OVERGO-CANDIDATE:(DENY)(GW)\n":    true,
		"docs\\golden.json HOST\\OVERGO-CANDIDATE:(DENY)(WD,AD)\n": true,
		"docs\\golden.json HOST\\OVERGO-CANDIDATE:(F)\n":           false,
		"docs\\golden.json HOST\\OVERGO-CANDIDATE:(DENY)(RX)\n":    false,
		"docs\\golden.json HOST\\OTHER:(DENY)(W)\n":                false,
		"docs\\golden.json HOST\\OVERGO-CANDIDATE:(W)\nHOST:(F)\n": false,
		"HOST\\overgo-candidate:(deny)(w)\n":                       true,
	} {
		if got := denyWriteListed(listing, principal); got != want {
			t.Fatalf("denyWriteListed(%q) = %v, want %v", listing, got, want)
		}
	}
}
