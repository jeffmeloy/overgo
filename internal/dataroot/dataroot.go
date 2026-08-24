// Package dataroot is the single owner of where the data roots live (floor
// component 9): the OvergoDB store and the three bulk roots — models (vendor
// inputs, irreplaceable), datasets (training data), checkpoints (trained
// outputs, lineage-bearing). Every command resolves through here so discovery
// cannot drift per-tool; explicit flags stay authoritative overrides.
//
// Resolution order:
//  1. OVERGO_DATA_ROOT env — one base directory holding overgodb-store/,
//     models/, datasets/, checkpoints/.
//  2. local-models.json in the working directory — explicit per-root paths,
//     which is how overgo points at data that lives in another repository's
//     home without moving a byte (data never moves; code comes to the data).
//  3. Working-directory defaults (overgodb-store, models, datasets,
//     checkpoints) — the pre-contract behavior, preserved.
package dataroot

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/strictjson"
)

const (
	// Env names the base directory for all roots (resolution step 1).
	Env = "OVERGO_DATA_ROOT"
	// ConfigFile is the per-checkout root override file (resolution step 2);
	// gitignored, machine-local by design.
	ConfigFile = "local-models.json"
)

// Roots: resolved data locations. Source names which resolution step won.
type Roots struct {
	Store       string `json:"store"`
	Models      string `json:"models"`
	Datasets    string `json:"datasets"`
	Checkpoints string `json:"checkpoints"`
	Source      string `json:"-"`
}

// ResolveCurrent returns roots for the process working directory.
func ResolveCurrent() (Roots, error) {
	working, err := os.Getwd()
	if err != nil {
		return Roots{}, err
	}
	return Resolve(working)
}

// Resolve returns the data roots for the given working directory.
func Resolve(workingDirectory string) (Roots, error) {
	if base := strings.TrimSpace(os.Getenv(Env)); base != "" {
		info, err := os.Stat(base)
		if err != nil || !info.IsDir() {
			return Roots{}, fmt.Errorf("dataroot: %s=%q is not a directory", Env, base)
		}
		return guardLegacyStore(Roots{
			Store:       filepath.Join(base, "overgodb-store"),
			Models:      filepath.Join(base, "models"),
			Datasets:    filepath.Join(base, "datasets"),
			Checkpoints: filepath.Join(base, "checkpoints"),
			Source:      Env,
		})
	}
	configPath := filepath.Join(workingDirectory, ConfigFile)
	raw, err := os.ReadFile(configPath)
	if os.IsNotExist(err) {
		return guardLegacyStore(fallback(workingDirectory))
	}
	if err != nil {
		return Roots{}, fmt.Errorf("dataroot: read %s: %w", ConfigFile, err)
	}
	var roots Roots
	if err := strictjson.DecodeBytes(raw, &roots); err != nil {
		return Roots{}, fmt.Errorf("dataroot: parse %s: %w", ConfigFile, err)
	}
	defaults := fallback(workingDirectory)
	if roots.Store == "" {
		roots.Store = defaults.Store
	}
	if roots.Models == "" {
		roots.Models = defaults.Models
	}
	if roots.Datasets == "" {
		roots.Datasets = defaults.Datasets
	}
	if roots.Checkpoints == "" {
		roots.Checkpoints = defaults.Checkpoints
	}
	for _, root := range []*string{&roots.Store, &roots.Models, &roots.Datasets, &roots.Checkpoints} {
		*root = configuredRoot(workingDirectory, *root)
	}
	roots.Source = ConfigFile
	return guardLegacyStore(roots)
}

// guardLegacyStore refuses to steer callers at a store root that would
// be created empty beside an unmigrated pre-rename store: opening the
// resolved root would silently mint a new authority while the real
// catalog sits in the legacy directory. Migration is explicit, never a
// side effect of resolution.
func guardLegacyStore(roots Roots) (Roots, error) {
	if _, err := os.Stat(filepath.Join(roots.Store, "overgodb.log")); err == nil {
		return roots, nil
	}
	if _, err := os.Stat(filepath.Join(roots.Store, "repodb.log")); err == nil {
		return roots, nil
	}
	legacy := filepath.Join(filepath.Dir(roots.Store), "repodb-store")
	if _, err := os.Stat(filepath.Join(legacy, "repodb.log")); err == nil {
		return Roots{}, fmt.Errorf(
			"dataroot: store root %q is empty while legacy store %q holds a catalog; migrate it (docs/OVERGODB_IMPORT.md) or point %s at it explicitly",
			roots.Store, legacy, Env)
	}
	return roots, nil
}

func configuredRoot(workingDirectory, root string) string {
	root = filepath.Clean(filepath.FromSlash(root))
	if filepath.IsAbs(root) {
		return root
	}
	return filepath.Join(workingDirectory, root)
}

func fallback(workingDirectory string) Roots {
	return Roots{
		Store:       filepath.Join(workingDirectory, "overgodb-store"),
		Models:      filepath.Join(workingDirectory, "models"),
		Datasets:    filepath.Join(workingDirectory, "datasets"),
		Checkpoints: filepath.Join(workingDirectory, "checkpoints"),
		Source:      "defaults",
	}
}

// ResolveModelPath maps a model reference to a filesystem path: an existing
// path (absolute or working-directory-relative) wins unchanged; otherwise a
// bare reference resolves under the models root, then the checkpoints root
// (trained outputs are servable artifacts too). The original reference
// returns unchanged when nothing resolves, so the caller's open error names
// what the user typed.
func (r Roots) ResolveModelPath(reference string) string {
	reference = strings.TrimSpace(reference)
	if reference == "" {
		return reference
	}
	if _, err := os.Stat(reference); err == nil {
		return reference
	}
	if filepath.IsAbs(reference) {
		return reference
	}
	for _, root := range []string{r.Models, r.Checkpoints} {
		candidate := filepath.Join(root, reference)
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
	}
	return reference
}
