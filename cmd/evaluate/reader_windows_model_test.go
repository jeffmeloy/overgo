//go:build windows && modeltest

package main

import (
	"encoding/json"
	"image"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// TestUnlimitedOCRReadsAPictureForTheSuite holds the reader to the bounds TestUnlimitedOCRProductionParity holds
// Unlimited-OCR's own path to: read through the command's reader, its reference page gives the reference decode's 29
// rows, each of the same kind, its box within 1 of the reference's on each side, and its text within one edit. Then a
// SenseNova drawn-text picture's reading carries the reader's model, recipe and projection.
func TestUnlimitedOCRReadsAPictureForTheSuite(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	repository := testutil.RepoRoot(t)
	roots := testcheck.Must(dataroot.Resolve(repository))(t)
	fixtures := filepath.Join(roots.References(), "fixtures")
	// The reference decode's tokens, rendered as the reader renders its own, without the detection markers its stored
	// text keeps, as TestUnlimitedOCRProductionParity compares them.
	var golden struct {
		Generated []tokenizer.TokenID `json:"generated_token_ids"`
	}
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join(fixtures, "unlimitedocr_golden.json")))(t), &golden))
	model := filepath.Join(roots.ResolveModelPath("Unlimited-OCR-bf16"), "Unlimited-OCR-bf16.gguf")
	projectorPath := filepath.Join(filepath.Dir(model), "Unlimited-OCR-mmproj-bf16.gguf")
	store := testcheck.Must(overgodb.OpenReadOnly(roots.Store))(t)
	t.Cleanup(func() { testcheck.NoError(t, store.Close()) })
	reader := testcheck.Must(openTextReader(ctx, store, roots.Store, model, projectorPath, 0))(t)
	t.Cleanup(func() { testcheck.NoError(t, reader.Close()) })

	page := decodePicture(t, filepath.Join(fixtures, "unlimitedocr_image.png"))
	reading := testcheck.Must(reader.read(ctx, "unlimitedocr_image.png", page))(t)
	want := readRows(testcheck.Must(reader.runner.Detokenize(golden.Generated, inference.RenderText))(t))
	if len(want) != 29 || len(reading.Rows) != len(want) {
		t.Fatalf("the reader read %d rows of the reference page, want the reference decode's %d", len(reading.Rows), len(want))
	}
	worst := 0
	for index, row := range reading.Rows {
		distance := editDistance(row.Text, want[index].Text)
		worst = max(worst, distance)
		far := row.Kind != want[index].Kind || distance > 1
		for side := range row.Box {
			far = far || row.Box[side]-want[index].Box[side] > 1 || want[index].Box[side]-row.Box[side] > 1
		}
		if far {
			t.Errorf("row %d reads %q %v %q, want %q %v %q within a box of 1 and one edit", index, row.Kind, row.Box, row.Text, want[index].Kind, want[index].Box, want[index].Text)
		}
	}
	t.Logf("the reference page: %d rows, the farthest one edit distance %d", len(reading.Rows), worst)

	// The page Unlimited-OCR's projection recipe was verified on reads as its verification recorded it.
	var verified struct {
		Cases []struct {
			Name string `json:"name"`
			Text string `json:"text"`
		} `json:"cases"`
	}
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join(repository, "docs", "verification", "fixtures", "unlimited-ocr", "suite.json")))(t), &verified))
	top := testcheck.Must(reader.read(ctx, "page-top.png", decodePicture(t, filepath.Join(repository, "docs", "verification", "fixtures", "unlimited-ocr", "page-top.png"))))(t)
	if want := readRows(verified.Cases[0].Text); verified.Cases[0].Name != "page-top" || !slices.Equal(top.Rows, want) {
		t.Errorf("the verification page reads %+v, want its recorded %+v", top.Rows, want)
	}

	drawn := filepath.Join(roots.Datasets, "drawn-text", "sensenova")
	entries := testcheck.Must(os.ReadDir(drawn))(t)
	at := slices.IndexFunc(entries, func(entry os.DirEntry) bool { return strings.HasSuffix(entry.Name(), ".png") })
	if at < 0 {
		t.Fatalf("%s holds no drawn-text picture", drawn)
	}
	picture := testcheck.Must(reader.read(ctx, entries[at].Name(), decodePicture(t, filepath.Join(drawn, entries[at].Name()))))(t)
	if !picture.Reader.Model.Valid() || !picture.Reader.Recipe.Valid() || !picture.Reader.Projection.Valid() || len(picture.Rows) == 0 {
		t.Fatalf("the reading of %s is %+v, want rows and the reader's model, recipe and projection", entries[at].Name(), picture)
	}
	t.Logf("%s, read by %s under %s and %s: %+v", entries[at].Name(), picture.Reader.Model, picture.Reader.Recipe, picture.Reader.Projection, picture.Rows)
}

func decodePicture(t *testing.T, path string) image.Image {
	t.Helper()
	file := testcheck.Must(os.Open(path))(t)
	defer file.Close()
	decoded := func(file *os.File) (image.Image, error) {
		picture, _, err := image.Decode(file)
		return picture, err
	}
	return testcheck.Must(decoded(file))(t)
}

// editDistance is the fewest single-character insertions, deletions and substitutions that turn a into b.
func editDistance(a, b string) int {
	left, right := []rune(a), []rune(b)
	previous := make([]int, len(right)+1)
	for index := range previous {
		previous[index] = index
	}
	for i, x := range left {
		current := make([]int, len(right)+1)
		current[0] = i + 1
		for j, y := range right {
			cost := 1
			if x == y {
				cost = 0
			}
			current[j+1] = min(previous[j+1]+1, current[j]+1, previous[j]+cost)
		}
		previous = current
	}
	return previous[len(right)]
}
