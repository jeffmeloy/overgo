package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/hfrepo"
	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
)

func TestRecipeSpecSourceReceipt(t *testing.T) {
	original, want := registrationFixture(t)
	files := testcheck.Must(os.ReadDir(filepath.Join(original, want.Directory)))(t)
	source := hfrepo.SourceReceipt{Source: want.SourceRepository, Revision: want.SourceCommit}
	root := t.TempDir()
	directory := filepath.Join(root, want.Directory)
	testcheck.NoError(t, os.MkdirAll(directory, 0700))
	for _, file := range files {
		if file.IsDir() {
			continue
		}
		data := testcheck.Must(os.ReadFile(filepath.Join(original, want.Directory, file.Name())))(t)
		testcheck.NoError(t, os.WriteFile(filepath.Join(directory, file.Name()), data, 0600))
		source.Files = append(source.Files, hfrepo.SourceFile{Path: file.Name(), SHA256: fmt.Sprintf("%x", sha256.Sum256(data)), Bytes: int64(len(data))})
	}
	if _, err := os.Stat(filepath.Join(directory, ".git")); !os.IsNotExist(err) {
		t.Fatal("test source unexpectedly has Git metadata")
	}
	receiptPath := filepath.Join(t.TempDir(), "source.json")
	data := testcheck.Must(json.Marshal(source))(t)
	testcheck.NoError(t, os.WriteFile(receiptPath, data, 0600))
	output := filepath.Join(t.TempDir(), "registration.json")
	args := []string{"-root", root, "-output", output, "-spdx", want.Directory + "=" + want.License.SPDX, "-source-receipt", receiptPath, want.Directory}
	testcheck.NoError(t, assembleSpecification(args))
	declared := testcheck.Must(os.ReadFile(output))(t)
	var got []modelRegistration
	testcheck.NoError(t, json.Unmarshal(declared, &got))
	if len(got) != 1 || got[0].Model != want.Model || got[0].TensorInventory != want.TensorInventory || got[0].SourceCommit != want.SourceCommit || got[0].SourceRepository != want.SourceRepository {
		t.Fatalf("source-backed declaration differs: %+v", got)
	}
	storePath := t.TempDir()
	register := []string{"-repo", storePath, "-root", root, "-spec", output}
	testcheck.NoError(t, registerModels(register))
	head := func() artifact.CommitID {
		t.Helper()
		store := testcheck.Must(overgodb.OpenReadOnly(storePath))(t)
		defer store.Close()
		if _, found, err := store.Manifest(t.Context(), want.Model); err != nil || !found {
			t.Fatalf("model absent: %v", err)
		}
		recipes, err := store.Query(t.Context(), overgodb.Query{Kind: artifact.KindRecipe, MaxResults: 1, Projection: overgodb.ProjectArtifacts})
		if err != nil || recipes.Matched != 0 {
			t.Fatal("source registration created an execution recipe")
		}
		aliases, err := store.Query(t.Context(), overgodb.Query{MaxResults: 32, Projection: overgodb.ProjectAliases})
		if err != nil || len(aliases.Aliases) != 0 {
			t.Fatal("source registration changed routing or activation")
		}
		head, _ := store.Head()
		return head
	}
	first := head()
	testcheck.NoError(t, registerModels(register))
	if head() != first {
		t.Fatal("repeated source registration changed the journal")
	}
	for _, failure := range []string{"wrong revision", "wrong source", "wrong digest", "unlisted component", "unlisted license", "missing receipt"} {
		t.Run(failure, func(t *testing.T) {
			var values []map[string]json.RawMessage
			testcheck.NoError(t, json.Unmarshal(declared, &values))
			var receipt hfrepo.SourceReceipt
			testcheck.NoError(t, json.Unmarshal(values[0]["source_receipt"], &receipt))
			switch failure {
			case "wrong revision":
				receipt.Revision = strings.Repeat("0", len(receipt.Revision))
			case "wrong source":
				receipt.Source = "https://example.invalid/another"
			case "wrong digest":
				receipt.Files[0].SHA256 = strings.Repeat("0", 64)
			case "unlisted component":
				receipt.Files = slices.DeleteFunc(receipt.Files, func(file hfrepo.SourceFile) bool { return file.Path == "config.json" })
			case "unlisted license":
				receipt.Files = slices.DeleteFunc(receipt.Files, func(file hfrepo.SourceFile) bool { return file.Path == want.License.Path })
			}
			updated := testcheck.Must(json.Marshal(receipt))(t)
			values[0]["source_receipt"] = updated
			if failure == "missing receipt" {
				delete(values[0], "source_receipt")
			}
			updated = testcheck.Must(json.Marshal(values))(t)
			testcheck.NoError(t, os.WriteFile(output, updated, 0600))
			if err := registerModels(register); err == nil {
				t.Fatal("invalid source registration succeeded")
			}
			if head() != first {
				t.Fatal("failed source registration changed the journal")
			}
		})
	}
	if err := assembleSpecification(append(slices.Clone(args), want.Directory)); err == nil {
		t.Fatal("one source receipt was shared across multiple directories")
	}
	// A second physical location and reverse receipt enumeration preserve logical identities.
	secondRoot := t.TempDir()
	secondDir := filepath.Join(secondRoot, want.Directory)
	testcheck.NoError(t, os.MkdirAll(secondDir, 0700))
	slices.Reverse(source.Files)
	for _, file := range source.Files {
		data := testcheck.Must(os.ReadFile(filepath.Join(directory, file.Path)))(t)
		testcheck.NoError(t, os.WriteFile(filepath.Join(secondDir, file.Path), data, 0600))
	}
	data = testcheck.Must(json.Marshal(source))(t)
	testcheck.NoError(t, os.WriteFile(receiptPath, data, 0600))
	args[1] = secondRoot
	testcheck.NoError(t, assembleSpecification(args))
	repeated := testcheck.Must(os.ReadFile(output))(t)
	if !bytes.Equal(declared, repeated) {
		t.Fatal("relocation or enumeration changed source-backed declaration")
	}
}

func TestRecipeSpecSourceReceiptLicenseAuthority(t *testing.T) {
	root, want := registrationFixture(t)
	directory := filepath.Join(root, want.Directory)
	testcheck.NoError(t, os.Mkdir(filepath.Join(directory, "transformer"), 0700))
	for _, name := range []string{"config.json", "model.safetensors"} {
		data := testcheck.Must(os.ReadFile(filepath.Join(directory, name)))(t)
		testcheck.NoError(t, os.WriteFile(filepath.Join(directory, "transformer", name), data, 0600))
	}
	for name, body := range map[string]string{
		"model_index.json": `{"_class_name":"Fixture","transformer":["diffusers","Fixture"]}`,
		"README.md":        "---\nlicense: mit\n---\n",
		"LICENSE":          "Fixture license bytes",
	} {
		testcheck.NoError(t, os.WriteFile(filepath.Join(directory, name), []byte(body), 0600))
	}
	source := hfrepo.SourceReceipt{Source: want.SourceRepository, Revision: want.SourceCommit}
	err := filepath.WalkDir(directory, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			if entry.Name() == ".git" {
				return filepath.SkipDir
			}
			return nil
		}
		if entry.Name() == "README.md" {
			return nil
		}
		relative, err := filepath.Rel(directory, path)
		if err != nil {
			return err
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		source.Files = append(source.Files, hfrepo.SourceFile{Path: filepath.ToSlash(relative), SHA256: fmt.Sprintf("%x", sha256.Sum256(data)), Bytes: int64(len(data))})
		return nil
	})
	testcheck.NoError(t, err)
	if _, err := assembleRegistration(t.Context(), root, want.Directory, "", &source); err == nil {
		t.Fatal("unlisted model card supplied the license identifier")
	}
	if _, err := assembleRegistration(t.Context(), root, want.Directory, "MIT", &source); err != nil {
		t.Fatalf("explicitly reviewed license was refused: %v", err)
	}
	card := testcheck.Must(os.ReadFile(filepath.Join(directory, "README.md")))(t)
	source.Files = append(source.Files, hfrepo.SourceFile{Path: "README.md", SHA256: fmt.Sprintf("%x", sha256.Sum256(card)), Bytes: int64(len(card))})
	declared := testcheck.Must(assembleRegistration(t.Context(), root, want.Directory, "", &source))(t)
	var got registrationDeclaration
	testcheck.NoError(t, json.Unmarshal(declared, &got))
	if got.License.SPDX != "mit" {
		t.Fatalf("verified card license=%q", got.License.SPDX)
	}
}
