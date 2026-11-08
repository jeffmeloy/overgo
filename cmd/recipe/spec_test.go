package main

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/gitauthority"
	"overgo/internal/testcheck"
)

// fixtureGit runs git in a fixture model directory.
func fixtureGit(t *testing.T, directory string, arguments ...string) {
	t.Helper()
	command := exec.CommandContext(t.Context(), "git", append([]string{"-C", directory}, arguments...)...)
	command.Env = gitauthority.ReaderEnvironment()
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("fixture git %v: %v: %s", arguments, err, output)
	}
}

// TestRecipeSpecAssemblesARegistration holds recipe spec to writing exactly
// the declaration recipe register verifies, so a model directory is registered
// by two commands and no hand-written identities. It computes what the worked
// fixture states by hand: the model and tensor-inventory identities, the
// directory's own origin and commit, and the license file with its content
// identity. The license identifier is the model card's own declaration; a card
// that declares none, or "other", is no licence, and the model registers with
// none recorded, since no model is ever distributed. A directory whose tracked
// bytes differ from its commit is refused, as registration would refuse it.
func TestRecipeSpecAssemblesARegistration(t *testing.T) {
	root, want := registrationFixture(t)
	directory := filepath.Join(root, "fixture")
	output := filepath.Join(t.TempDir(), "registration.json")
	assemble := func(extra ...string) error {
		return assembleSpecification(append(append([]string{"-root", root, "-output", output}, extra...), "fixture"))
	}
	testcheck.NoError(t, assemble("-spdx", "fixture="+want.License.SPDX))
	data, err := os.ReadFile(output)
	testcheck.NoError(t, err)
	var got []modelRegistration
	testcheck.NoError(t, json.Unmarshal(data, &got))
	if len(got) != 1 || got[0].Model != want.Model || got[0].TensorInventory != want.TensorInventory ||
		got[0].SourceRepository != want.SourceRepository || got[0].SourceCommit != want.SourceCommit ||
		got[0].License != want.License || got[0].Directory != want.Directory {
		t.Fatalf("assembled %+v\nwant %+v", got, want)
	}
	if err := registerModels([]string{"-repo", t.TempDir(), "-root", root, "-spec", output}); err != nil {
		t.Fatalf("register refused what spec wrote: %v", err)
	}

	card := filepath.Join(directory, "README.md")
	commit := func(content string) {
		t.Helper()
		testcheck.NoError(t, os.WriteFile(card, []byte(content), 0o600))
		fixtureGit(t, directory, "add", ".")
		fixtureGit(t, directory, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--no-gpg-sign", "-qm", "card")
	}
	commit("---\nlicense: other\n---\n# Card\n")
	testcheck.NoError(t, assemble())
	got = nil
	if data, err = os.ReadFile(output); err != nil || json.Unmarshal(data, &got) != nil || got[0].License.Path != "" {
		t.Fatalf("a card that declares other was recorded as a licence: %+v (%v)", got, err)
	}
	commit("---\ntags:\n- fixture\nlicense: \"mit\"\n---\n# Card\n")
	testcheck.NoError(t, assemble())
	if data, err = os.ReadFile(output); err != nil || json.Unmarshal(data, &got) != nil || got[0].License.SPDX != "mit" {
		t.Fatalf("the card's own license was not declared: %+v (%v)", got, err)
	}
	testcheck.NoError(t, os.WriteFile(card, []byte("an edit that is not committed\n"), 0o600))
	if err := assemble(); err == nil || !strings.Contains(err.Error(), "unchanged tracked bytes") {
		t.Fatalf("a directory with uncommitted tracked changes was assembled: %v", err)
	}
}

// TestAModelWithoutALicenceIsServed holds registration to the owner's rule of 2026-11-08: no model is ever
// distributed, so a licence binds only a model's local use, recorded where the model ships one. Registration refused
// a model whose card declared no licence identifier until one was supplied. A model whose card declares none
// assembles with no licence recorded and registers, and its record names none.
func TestAModelWithoutALicenceIsServed(t *testing.T) {
	t.Parallel()
	root, want := registrationFixture(t)
	output := filepath.Join(t.TempDir(), "registration.json")
	testcheck.NoError(t, assembleSpecification([]string{"-root", root, "-output", output, "fixture"}))
	var got []modelRegistration
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(output))(t), &got))
	if len(got) != 1 || got[0].Model != want.Model || got[0].License.Path != "" || got[0].License.SPDX != "" || got[0].License.Artifact.Valid() {
		t.Fatalf("a model with no licence assembled as %+v, want its identities and no licence", got)
	}
	testcheck.NoError(t, registerModels([]string{"-repo", t.TempDir(), "-root", root, "-spec", output}))
}
