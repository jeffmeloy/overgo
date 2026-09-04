// Package audioparity owns pinned behavioral evidence for external audio oracles.
package audioparity

import (
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"maps"
	"path"
	"slices"
	"strings"
	"testing"
)

//go:embed testdata/audio_cpp_census.json
var pinnedCensusJSON []byte

type oracleCensus struct {
	Version int `json:"version"`
	Source  struct {
		Repository string `json:"repository"`
		Commit     string `json:"commit"`
		License    struct {
			SPDX      string `json:"spdx"`
			Path      string `json:"path"`
			GitObject string `json:"git_object"`
			Copyright string `json:"copyright"`
		} `json:"license"`
		Trees map[string]string `json:"trees"`
	} `json:"source"`
	ModelSpecs struct {
		Path         string              `json:"path"`
		PackageCount int                 `json:"package_count"`
		Categories   map[string][]string `json:"categories"`
		Statuses     map[string][]string `json:"statuses"`
		SchemaV1     []string            `json:"schema_v1"`
		Streaming    []string            `json:"streaming"`
	} `json:"model_specs"`
	Loaders struct {
		Registry                  string   `json:"registry"`
		FactoriesMatchModelSpecs  bool     `json:"factories_match_model_specs"`
		Families                  []string `json:"families"`
		AdditionalBuiltinFamilies []string `json:"additional_builtin_families"`
		RuntimeRegistry           string   `json:"runtime_registry"`
	} `json:"loaders"`
	Surfaces []struct {
		Area         string   `json:"area"`
		Paths        []string `json:"paths"`
		Capabilities []string `json:"capabilities"`
		Transfer     string   `json:"transfer"`
	} `json:"surfaces"`
	Training struct {
		NativeTrainingLoop bool     `json:"native_training_loop"`
		ReusableEvidence   []string `json:"reusable_evidence"`
		MissingForOvergo   []string `json:"missing_for_overgo"`
		Decision           string   `json:"decision"`
	} `json:"training"`
	KnownIncomplete []struct {
		Path        string `json:"path"`
		Limitation  string `json:"limitation"`
		Disposition string `json:"disposition"`
	} `json:"known_incomplete"`
	Elections []struct {
		Capability string `json:"capability"`
		Family     string `json:"family"`
		Artifact   string `json:"artifact"`
		Reason     string `json:"reason"`
	} `json:"elections"`
}

func TestPinnedOracleCensus(t *testing.T) {
	var census oracleCensus
	decoder := json.NewDecoder(strings.NewReader(string(pinnedCensusJSON)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&census); err != nil {
		t.Fatalf("decode pinned census: %v", err)
	}
	if err := validateCensus(census); err != nil {
		t.Fatal(err)
	}
	canonical, err := json.Marshal(census)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(canonical)
	const expected = "0fc1913298bbc8cb2a604b4a92bc8d17e23ea085b783bff018537078f94c6a06"
	if got := hex.EncodeToString(digest[:]); got != expected {
		t.Fatalf("census digest = %s, want %s", got, expected)
	}
}

func validateCensus(census oracleCensus) error {
	if census.Version != 1 || census.Source.Repository != "https://github.com/0xShug0/audio.cpp" ||
		census.Source.Commit != "3497b7cc44753e2c141d8fe60ac42cec433e3281" || census.Source.License.SPDX != "Apache-2.0" ||
		census.Source.License.Path != "LICENSE" || len(census.Source.License.GitObject) != 40 {
		return fmt.Errorf("invalid pinned source identity or license: %+v", census.Source)
	}
	for name, object := range census.Source.Trees {
		if strings.TrimSpace(name) == "" || len(object) != 40 {
			return fmt.Errorf("invalid source tree identity %q=%q", name, object)
		}
	}
	categories, err := partition("category", census.ModelSpecs.Categories)
	if err != nil {
		return err
	}
	statuses, err := partition("status", census.ModelSpecs.Statuses)
	if err != nil {
		return err
	}
	if !slices.Equal(categories, statuses) || !slices.Equal(categories, census.Loaders.Families) ||
		!census.Loaders.FactoriesMatchModelSpecs {
		return fmt.Errorf("model-spec, status, and loader inventories differ")
	}
	if census.ModelSpecs.PackageCount <= len(categories) || len(census.ModelSpecs.SchemaV1) == 0 ||
		len(census.ModelSpecs.Streaming) == 0 {
		return fmt.Errorf("model-spec coverage is incomplete")
	}
	for label, subset := range map[string][]string{"schema_v1": census.ModelSpecs.SchemaV1, "streaming": census.ModelSpecs.Streaming} {
		if err := requireSortedSubset(label, subset, categories); err != nil {
			return err
		}
	}
	if census.Training.NativeTrainingLoop || len(census.Training.ReusableEvidence) == 0 ||
		len(census.Training.MissingForOvergo) == 0 || !strings.Contains(census.Training.Decision, "Overgo") {
		return fmt.Errorf("training relevance does not distinguish forward oracle evidence from Overgo training ownership")
	}
	areas := map[string]bool{}
	for _, surface := range census.Surfaces {
		if areas[surface.Area] || len(surface.Paths) == 0 || len(surface.Capabilities) == 0 || strings.TrimSpace(surface.Transfer) == "" {
			return fmt.Errorf("invalid or duplicate surface %q", surface.Area)
		}
		areas[surface.Area] = true
		for _, sourcePath := range surface.Paths {
			if err := validateSourcePath(sourcePath); err != nil {
				return fmt.Errorf("surface %s: %w", surface.Area, err)
			}
		}
	}
	for _, required := range []string{"artifact loading", "audio preprocessing", "execution graphs", "streaming", "testing"} {
		if !areas[required] {
			return fmt.Errorf("missing required census surface %q", required)
		}
	}
	incomplete := map[string]bool{}
	for _, entry := range census.KnownIncomplete {
		if err := validateSourcePath(entry.Path); err != nil {
			return fmt.Errorf("known incomplete: %w", err)
		}
		if incomplete[entry.Path] || strings.TrimSpace(entry.Limitation) == "" || strings.TrimSpace(entry.Disposition) == "" {
			return fmt.Errorf("invalid or duplicate incomplete implementation %q", entry.Path)
		}
		incomplete[entry.Path] = true
	}
	elections := map[string]bool{}
	for _, election := range census.Elections {
		if election.Capability == "" || election.Family == "" || election.Artifact == "" || election.Reason == "" || elections[election.Capability] {
			return fmt.Errorf("invalid or duplicate election %q", election.Capability)
		}
		elections[election.Capability] = true
	}
	for _, capability := range []string{"offline ASR", "streaming ASR", "speech activity segmentation"} {
		if !elections[capability] {
			return fmt.Errorf("missing election for %s", capability)
		}
	}
	return nil
}

func partition(label string, groups map[string][]string) ([]string, error) {
	seen := map[string]bool{}
	for group, members := range groups {
		if strings.TrimSpace(group) == "" || !slices.IsSorted(members) {
			return nil, fmt.Errorf("%s group %q is empty or unsorted", label, group)
		}
		for _, member := range members {
			if member == "" || seen[member] {
				return nil, fmt.Errorf("%s member %q is empty or duplicated", label, member)
			}
			seen[member] = true
		}
	}
	return slices.Sorted(maps.Keys(seen)), nil
}

func requireSortedSubset(label string, subset, complete []string) error {
	if !slices.IsSorted(subset) {
		return fmt.Errorf("%s inventory is unsorted", label)
	}
	for _, member := range subset {
		if _, found := slices.BinarySearch(complete, member); !found {
			return fmt.Errorf("%s member %q is absent from model specs", label, member)
		}
	}
	return nil
}

func validateSourcePath(value string) error {
	if value == "" || strings.Contains(value, "\\") || path.IsAbs(value) || path.Clean(value) != value || strings.HasPrefix(value, "../") {
		return fmt.Errorf("invalid source-relative path %q", value)
	}
	return nil
}
