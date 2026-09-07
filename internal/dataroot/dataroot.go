// Package dataroot is the single owner of where the data roots live (floor
// component 9): the OvergoDB store and the three bulk roots — models (vendor
// inputs, irreplaceable), datasets (training data), checkpoints (trained
// outputs, lineage-bearing). Every command resolves through here so discovery
// cannot drift per-tool; explicit flags stay authoritative overrides.
//
// Resolution order:
//  1. OVERGO_DATA_ROOT env — one base directory holding overgodb-store/,
//     models/, datasets/, checkpoints/, or a local-models.json of its own
//     that names where those live.
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

// Resolve returns the data roots for the given working directory. The
// OVERGO_DATA_ROOT base is itself resolved as a working directory: its own
// local-models.json, when present, redirects the bulk roots the way a
// checkout's does, so a base that keeps its store beside a config pointing
// at data in another home resolves the same from any tree.
func Resolve(workingDirectory string) (Roots, error) {
	if base := strings.TrimSpace(os.Getenv(Env)); base != "" {
		info, err := os.Stat(base)
		if err != nil || !info.IsDir() {
			return Roots{}, fmt.Errorf("dataroot: %s=%q is not a directory", Env, base)
		}
		roots, err := resolveIn(base)
		if err != nil {
			return Roots{}, err
		}
		roots.Source = Env
		return roots, nil
	}
	return resolveIn(workingDirectory)
}

// resolveIn resolves the roots a directory declares: its local-models.json
// when present, else the directory's own default layout.
func resolveIn(workingDirectory string) (Roots, error) {
	configPath := filepath.Join(workingDirectory, ConfigFile)
	raw, err := os.ReadFile(configPath)
	if os.IsNotExist(err) {
		return fallback(workingDirectory), nil
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
