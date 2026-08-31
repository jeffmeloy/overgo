package repoanalysis

import (
	"crypto/sha256"
	"encoding/hex"
	"slices"
	"testing"
)

func TestModernGoCatalogMatchesPinnedSource(t *testing.T) {
	if ModernGoCatalogRepository != "https://github.com/JetBrains/go-modern-guidelines" ||
		ModernGoCatalogCommit != "40781f167719913666fe2a7dc1c77ea6f256df0a" ||
		ModernGoCatalogPluginVersion != "v0.1.1" {
		t.Fatalf("catalog authority = %s %s %s",
			ModernGoCatalogRepository, ModernGoCatalogCommit, ModernGoCatalogPluginVersion)
	}
	digest := sha256.Sum256(modernGoCatalogJSON)
	if got := hex.EncodeToString(digest[:]); got != ModernGoCatalogSHA256 {
		t.Fatalf("catalog digest = %s, want %s", got, ModernGoCatalogSHA256)
	}
	catalog := ModernGoCatalog()
	if len(catalog) != 54 {
		t.Fatalf("catalog size = %d, want 54", len(catalog))
	}
	applicable, err := ModernGoApplicableGuidelines(ModernGoTargetVersion)
	if err != nil {
		t.Fatal(err)
	}
	if len(applicable) != 48 {
		t.Fatalf("Go %s applicable guidelines = %d, want 48", ModernGoTargetVersion, len(applicable))
	}
	var excluded []string
	for _, guideline := range catalog {
		if !slices.ContainsFunc(applicable, func(candidate ModernGoGuideline) bool {
			return candidate.ID == guideline.ID
		}) {
			excluded = append(excluded, guideline.ID)
		}
	}
	wantExcluded := []string{
		"generic_methods", "json_v2", "promoted_field_literals",
		"strings_bytes_cut_last", "stdlib_uuid", "url_clone",
	}
	if !slices.Equal(excluded, wantExcluded) {
		t.Fatalf("excluded guidelines = %v, want %v", excluded, wantExcluded)
	}
}

func TestModernGoPriorCampaignReconciliation(t *testing.T) {
	catalog := ModernGoCatalog()
	commits := ModernGoPriorCampaign()
	if len(commits) != 17 {
		t.Fatalf("prior campaign commits = %d, want 17", len(commits))
	}
	seen := make(map[string]bool, len(commits))
	for _, commit := range commits {
		if len(commit.Commit) != 40 || commit.Subject == "" || commit.Rationale == "" || commit.Disposition == "" {
			t.Errorf("incomplete prior campaign record: %+v", commit)
		}
		if seen[commit.Commit] {
			t.Errorf("duplicate prior campaign commit %s", commit.Commit)
		}
		seen[commit.Commit] = true
		for _, id := range commit.Guidelines {
			if !slices.ContainsFunc(catalog, func(guideline ModernGoGuideline) bool { return guideline.ID == id }) {
				t.Errorf("prior campaign commit %s names unknown guideline %q", commit.Commit, id)
			}
		}
	}
	for _, disposition := range []ModernGoPriorDisposition{
		ModernGoPriorSuperseded, ModernGoPriorReimplement, ModernGoPriorReplayCandidate,
		ModernGoPriorSupporting, ModernGoPriorExcluded,
	} {
		if !slices.ContainsFunc(commits, func(commit ModernGoPriorCommit) bool {
			return commit.Disposition == disposition
		}) {
			t.Errorf("prior campaign has no %q disposition", disposition)
		}
	}
}

func TestModernGoCatalogReturnsDetachedData(t *testing.T) {
	first := ModernGoCatalog()
	first[0].ID = "changed"
	first[0].Examples[0].Before[0] = "changed"
	second := ModernGoCatalog()
	if second[0].ID == "changed" || second[0].Examples[0].Before[0] == "changed" {
		t.Fatal("catalog caller mutated embedded authority")
	}
}
