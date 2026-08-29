package repoanalysis

import (
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/strictjson"
)

const (
	// ModernGoCatalogRepository is the upstream owner of the pinned guidance.
	ModernGoCatalogRepository = "https://github.com/JetBrains/go-modern-guidelines"
	// ModernGoCatalogCommit is the exact upstream revision audited by this campaign.
	ModernGoCatalogCommit = "40781f167719913666fe2a7dc1c77ea6f256df0a"
	// ModernGoCatalogPluginVersion is the wrapper version shipped by the pinned revision.
	ModernGoCatalogPluginVersion = "v0.1.1"
	// ModernGoCatalogSHA256 binds the embedded catalog bytes to the audited source.
	ModernGoCatalogSHA256 = "f452804a0d451f676701025989aff5223ffb63849e344e3714eb42842acb05a5"
	// ModernGoTargetVersion is Overgo's current language and standard-library floor.
	ModernGoTargetVersion = "1.26"
)

// ModernGoExample is one before-and-after example from the pinned catalog.
type ModernGoExample struct {
	Before []string `json:"before"`
	After  []string `json:"after"`
}

// ModernGoGuideline is one version-qualified rule in the pinned catalog.
type ModernGoGuideline struct {
	ID           string            `json:"id"`
	SinceVersion string            `json:"since_version"`
	Guideline    string            `json:"guideline"`
	Details      string            `json:"details"`
	Examples     []ModernGoExample `json:"examples"`
}

//go:embed modern_go_catalog.json
var modernGoCatalogJSON []byte

var modernGoCatalog = mustLoadModernGoCatalog()

func mustLoadModernGoCatalog() []ModernGoGuideline {
	actual := sha256.Sum256(modernGoCatalogJSON)
	if hex.EncodeToString(actual[:]) != ModernGoCatalogSHA256 {
		panic("embedded modern-Go catalog does not match its pinned identity")
	}
	var catalog []ModernGoGuideline
	if err := strictjson.DecodeBytes(modernGoCatalogJSON, &catalog); err != nil {
		panic(fmt.Sprintf("parse embedded modern-Go catalog: %v", err))
	}
	if err := validateModernGoCatalog(catalog); err != nil {
		panic(err)
	}
	return catalog
}

func validateModernGoCatalog(catalog []ModernGoGuideline) error {
	if len(catalog) == 0 {
		return errors.New("modern-Go catalog is empty")
	}
	seen := make(map[string]bool, len(catalog))
	for index, guideline := range catalog {
		if guideline.ID == "" || guideline.SinceVersion == "" ||
			guideline.Guideline == "" || guideline.Details == "" {
			return fmt.Errorf("modern-Go catalog entry %d is incomplete", index)
		}
		if seen[guideline.ID] {
			return fmt.Errorf("modern-Go catalog guideline %q is duplicated", guideline.ID)
		}
		seen[guideline.ID] = true
		if _, _, err := modernGoVersion(guideline.SinceVersion); err != nil {
			return fmt.Errorf("modern-Go catalog guideline %q: %w", guideline.ID, err)
		}
		for exampleIndex, example := range guideline.Examples {
			if len(example.Before) == 0 || len(example.After) == 0 {
				return fmt.Errorf("modern-Go catalog guideline %q example %d is incomplete", guideline.ID, exampleIndex)
			}
		}
	}
	return nil
}

// ModernGoCatalog returns a detached copy of the complete pinned catalog.
func ModernGoCatalog() []ModernGoGuideline {
	catalog := make([]ModernGoGuideline, len(modernGoCatalog))
	for index, guideline := range modernGoCatalog {
		catalog[index] = guideline
		catalog[index].Examples = make([]ModernGoExample, len(guideline.Examples))
		for exampleIndex, example := range guideline.Examples {
			catalog[index].Examples[exampleIndex] = ModernGoExample{
				Before: slices.Clone(example.Before),
				After:  slices.Clone(example.After),
			}
		}
	}
	return catalog
}

// ModernGoApplicableGuidelines selects rules available at targetVersion while
// preserving the upstream newest-first catalog order.
func ModernGoApplicableGuidelines(targetVersion string) ([]ModernGoGuideline, error) {
	targetMajor, targetMinor, err := modernGoVersion(targetVersion)
	if err != nil {
		return nil, err
	}
	var applicable []ModernGoGuideline
	for _, guideline := range ModernGoCatalog() {
		major, minor, err := modernGoVersion(guideline.SinceVersion)
		if err != nil {
			return nil, err
		}
		if major < targetMajor || major == targetMajor && minor <= targetMinor {
			applicable = append(applicable, guideline)
		}
	}
	return applicable, nil
}

func modernGoVersion(value string) (int, int, error) {
	value = strings.TrimSpace(value)
	value, _ = strings.CutPrefix(value, "go")
	majorText, minorText, ok := strings.Cut(value, ".")
	if !ok || strings.Contains(minorText, ".") {
		return 0, 0, fmt.Errorf("go version %q must be major.minor", value)
	}
	major, err := strconv.Atoi(majorText)
	if err != nil || major < 0 {
		return 0, 0, fmt.Errorf("go version %q has invalid major version", value)
	}
	minor, err := strconv.Atoi(minorText)
	if err != nil || minor < 0 {
		return 0, 0, fmt.Errorf("go version %q has invalid minor version", value)
	}
	return major, minor, nil
}
