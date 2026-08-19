// Sealed external authority: goldens, evaluator definitions, promotion
// policy and champion aliases held beyond candidate-worker reach.
package protection

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"overgo/internal/jsonfile"
)

const sealedManifestPath = ".github/sealed_authority.json"

// GoldenProvenance binds one sealed file to the external tool that produced
// it: any change to the file changes its digest and therefore demands a new
// provenance entry naming the producing tool and version. Provenance without
// a digest would be a label; the digest makes it a claim the gate can refute.
type GoldenProvenance struct {
	Path    string `json:"path"`
	Tool    string `json:"tool"`
	Version string `json:"version"`
	SHA256  string `json:"sha256"`
}

// SealedFileAuthority is the owner-authored file-sealing contract: the OS principal
// candidate workers run as, the repository files holding goldens, evaluator
// definitions, promotion policy and champion aliases, and the provenance
// ledger covering every sealed file. The manifest asserts sealing; Verify
// holds the assertion against the filesystem and the host ACLs.
type SealedFileAuthority struct {
	Version            uint16             `json:"version"`
	CandidatePrincipal string             `json:"candidate_principal"`
	SealedPaths        []string           `json:"sealed_paths"`
	Provenance         []GoldenProvenance `json:"provenance"`
}

// LoadSealedFileAuthority reads the sealing manifest. A missing manifest reports
// unsealed (false) without error -- sealing is provisioned by the owner, and
// its absence is a fact to report, not a failure to invent. A present
// manifest must be complete: partial sealing claims are refused.
func LoadSealedFileAuthority(root string) (SealedFileAuthority, bool, error) {
	path := filepath.Join(root, filepath.FromSlash(sealedManifestPath))
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		return SealedFileAuthority{}, false, nil
	}
	var authority SealedFileAuthority
	if err := jsonfile.DecodeStrict(path, &authority); err != nil {
		return SealedFileAuthority{}, false, fmt.Errorf("sealed authority manifest: %w", err)
	}
	if err := validateSealedFileAuthority(authority); err != nil {
		return SealedFileAuthority{}, false, err
	}
	return authority, true, nil
}

// Verify holds the sealing claim against reality: every sealed file exists
// and hashes to its provenance digest, and the host DACL carries an explicit
// write-deny for the candidate principal on every sealed file. Verification
// only reads -- it never provisions principals or edits ACLs; those are
// owner actions performed outside any candidate-reachable process.
func (a SealedFileAuthority) Verify(root string) error {
	if runtime.GOOS != "windows" {
		return errors.New("sealed authority: ACL verification requires the windows host")
	}
	provenance := make(map[string]GoldenProvenance, len(a.Provenance))
	for _, entry := range a.Provenance {
		provenance[entry.Path] = entry
	}
	for _, sealed := range a.SealedPaths {
		path := filepath.Join(root, filepath.FromSlash(sealed))
		data, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("sealed authority: sealed file %s: %w", sealed, err)
		}
		digest := sha256.Sum256(data)
		if entry := provenance[sealed]; entry.SHA256 != hex.EncodeToString(digest[:]) {
			return fmt.Errorf(
				"sealed authority: %s does not match its provenance digest -- a golden change requires a new provenance entry naming the producing tool and version", sealed)
		}
		listing, err := exec.Command("icacls", path).CombinedOutput()
		if err != nil {
			return fmt.Errorf("sealed authority: icacls %s: %w", sealed, err)
		}
		if !denyWriteListed(string(listing), a.CandidatePrincipal) {
			return fmt.Errorf(
				"sealed authority: %s carries no write-deny ACE for candidate principal %q -- owner must run icacls %s /deny %s:(W)",
				sealed, a.CandidatePrincipal, sealed, a.CandidatePrincipal)
		}
	}
	return nil
}

// denyWriteListed reports whether an icacls listing contains an explicit
// deny entry covering write access for the principal. Accepted deny shapes:
// full write (W), generic write (GW), or explicit write-data (WD).
func denyWriteListed(listing, principal string) bool {
	needle := strings.ToLower(principal)
	for _, line := range strings.Split(listing, "\n") {
		lowered := strings.ToLower(line)
		if !strings.Contains(lowered, needle) || !strings.Contains(lowered, "(deny)") {
			continue
		}
		grant := lowered[strings.Index(lowered, "(deny)")+len("(deny)"):]
		if strings.Contains(grant, "(w)") || strings.Contains(grant, "gw") || strings.Contains(grant, "wd") {
			return true
		}
	}
	return false
}

func validateSealedFileAuthority(authority SealedFileAuthority) error {
	if authority.Version != 1 {
		return errors.New("sealed authority: invalid manifest version")
	}
	principal := strings.TrimSpace(authority.CandidatePrincipal)
	if principal == "" || principal != authority.CandidatePrincipal || strings.ContainsAny(principal, "\x00\r\n\t") {
		return errors.New("sealed authority: manifest requires the candidate principal")
	}
	if len(authority.SealedPaths) == 0 {
		return errors.New("sealed authority: manifest seals no paths")
	}
	sealed := make(map[string]bool, len(authority.SealedPaths))
	for _, path := range authority.SealedPaths {
		if !safeRelative(path) {
			return fmt.Errorf("sealed authority: sealed path %q must be repository-relative", path)
		}
		if sealed[path] {
			return fmt.Errorf("sealed authority: sealed path %q listed twice", path)
		}
		sealed[path] = true
	}
	if len(authority.Provenance) != len(authority.SealedPaths) {
		return errors.New("sealed authority: every sealed file requires exactly one provenance entry")
	}
	covered := make(map[string]bool, len(authority.Provenance))
	for _, entry := range authority.Provenance {
		if !sealed[entry.Path] {
			return fmt.Errorf("sealed authority: provenance for %q which is not sealed", entry.Path)
		}
		if covered[entry.Path] {
			return fmt.Errorf("sealed authority: provenance for %q listed twice", entry.Path)
		}
		covered[entry.Path] = true
		if strings.TrimSpace(entry.Tool) == "" || strings.TrimSpace(entry.Version) == "" {
			return fmt.Errorf("sealed authority: provenance for %q requires the producing tool and version", entry.Path)
		}
		if len(entry.SHA256) != 64 || strings.ToLower(entry.SHA256) != entry.SHA256 {
			return fmt.Errorf("sealed authority: provenance for %q requires a lowercase sha256 digest", entry.Path)
		}
		if _, err := hex.DecodeString(entry.SHA256); err != nil {
			return fmt.Errorf("sealed authority: provenance for %q digest is not hex", entry.Path)
		}
	}
	return nil
}
