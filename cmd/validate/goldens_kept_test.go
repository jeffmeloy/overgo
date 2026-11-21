package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/testcheck"
)

// goldensKept is docs/verification/goldens-kept.json: the goldens the plan's campaign still needs before parity (the
// golden scripts git ignores and the capture manifests its rows read), each kept under the data root's references in
// kept with the SHA-256 of its bytes.
type goldensKept struct {
	Campaign string `json:"campaign"`
	Kept     string `json:"kept"`
	Goldens  []struct {
		Name   string `json:"name"`
		SHA256 string `json:"sha256"`
	} `json:"goldens"`
}

// TestEveryGoldenStillNeededIsKept holds the goldens a campaign needs before parity to a place no cleanup reaches (the
// owner's review of 2026-11-20): each one docs/verification/goldens-kept.json names lies under the data root's
// references, in the record's kept folder, with the bytes its digest says, where the golden scripts' own folder is
// ignored by git and a clean checkout or a removed folder would lose them. The record is the plan's campaign's: a
// campaign that ends lets its goldens go, and the next writes its own.
func TestEveryGoldenStillNeededIsKept(t *testing.T) {
	t.Parallel()
	repo := filepath.Join("..", "..")
	var record goldensKept
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join(repo, "docs", "verification", "goldens-kept.json")))(t), &record))
	var plan struct {
		Campaign string `json:"campaign"`
	}
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join(repo, "docs", "plan.json")))(t), &plan))
	if record.Campaign != plan.Campaign {
		t.Fatalf("the goldens kept are the campaign %q's; the plan's is %q", record.Campaign, plan.Campaign)
	}
	if len(record.Goldens) == 0 || record.Kept == "" {
		t.Fatal("the record keeps no golden")
	}
	roots := testcheck.Must(dataroot.Resolve(repo))(t)
	for _, golden := range record.Goldens {
		path := filepath.Join(roots.References(), filepath.FromSlash(record.Kept), golden.Name)
		bytes, err := os.ReadFile(path)
		if err != nil {
			t.Errorf("%s is not kept: %v", golden.Name, err)
			continue
		}
		if sum := sha256.Sum256(bytes); hex.EncodeToString(sum[:]) != golden.SHA256 {
			t.Errorf("%s kept at %s hashes %x; the record says %s", golden.Name, path, sum, golden.SHA256)
		}
	}
}
