package modelartifact

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/gitauthority"
	"overgo/internal/pathidentity"
	"overgo/internal/processcontrol"
)

// licenseFiles are where a model directory keeps its license, in the order a
// declaration prefers them: the license itself before the card that names it.
var licenseFiles = []string{"LICENSE", "LICENSE.txt", "LICENSE.md", modelCardFile}

const (
	// modelCardFile carries the publisher's own declarations in its front matter.
	modelCardFile = "README.md"
	// cardFence opens and closes a model card's front matter.
	cardFence = "---"
	// cardLicenseField is the front matter field that declares the license.
	cardLicenseField = "license:"
	// hfConfigFile marks a directory the Hugging Face repository walker owns.
	hfConfigFile      = "config.json"
	markdownExtension = ".md"
	markdownMediaType = "text/markdown"
)

// undeclaredLicenses are card values that name no license.
var undeclaredLicenses = []string{"", "other", "unknown"}

// AssembleRegistration computes, for one model directory beneath root, the
// declaration that PrepareRegistration verifies: the identities of the model
// and its tensor inventory, the origin and commit of the directory's own
// repository, and the license file with its content identity. The license
// identifier is the one the model card itself declares unless spdx supplies
// one: a card that declares none, or "other", is refused, because a license is
// a publisher's statement and is never invented here. A directory whose
// tracked bytes differ from its commit is refused, as registration would.
func AssembleRegistration(ctx context.Context, root, directory, spdx string) (json.RawMessage, error) {
	path := filepath.Join(root, filepath.FromSlash(directory))
	contained, err := pathidentity.Contains(root, path)
	if err != nil || !contained || !filepath.IsLocal(directory) {
		return nil, fmt.Errorf("assemble %s: model directory escapes root", directory)
	}
	var declared modelRegistration
	declared.Directory = filepath.ToSlash(directory)
	for _, read := range []struct {
		arguments []string
		into      *string
	}{
		{[]string{"rev-parse", "HEAD"}, &declared.SourceCommit},
		{[]string{"remote", "get-url", "origin"}, &declared.SourceRepository},
		{[]string{"diff", "--no-ext-diff", "--no-textconv", "--quiet", "HEAD", "--"}, nil},
	} {
		var output strings.Builder
		receipt, err := processcontrol.Run(ctx, processcontrol.Command{
			Path: "git", Args: append([]string{"--no-replace-objects", "-C", path}, read.arguments...),
			Env: gitauthority.ReaderEnvironment(), Stdout: &output,
		})
		if err != nil {
			return nil, fmt.Errorf("assemble %s: git %s: %w", directory, read.arguments[0], err)
		}
		// The check reads the index as it stands, never refreshing it: a
		// repository copied into place has files newer than its index and
		// reads as changed until its owner runs git status in it once.
		if receipt.ExitCode != 0 {
			return nil, fmt.Errorf("assemble %s: git %s exit=%d: the directory needs its own repository with an origin and unchanged tracked bytes; if nothing was edited, run git status in it once to refresh an index older than its files", directory, read.arguments[0], receipt.ExitCode)
		}
		if read.into != nil {
			*read.into = strings.TrimSpace(output.String())
		}
	}
	// A directory no repository walker owns -- a bare checkpoint file, one
	// file per head -- is declared by its components, and which file plays
	// which role is a judgement this command does not make.
	if _, err := os.Stat(filepath.Join(path, hfConfigFile)); err != nil {
		return nil, fmt.Errorf("assemble %s: not a Hugging Face layout (no %s): declare its components by hand, as the registration of such a directory takes them", directory, hfConfigFile)
	}
	inventory, err := FromHFPath(path)
	if err != nil {
		return nil, fmt.Errorf("assemble %s: %w", directory, err)
	}
	declared.Model, declared.TensorInventory = inventory.Manifest.ID, inventory.TensorInventory.ID
	declared.License.SPDX = spdx
	if spdx == "" {
		declared.License.SPDX = cardLicense(path)
	}
	if slices.Contains(undeclaredLicenses, strings.ToLower(declared.License.SPDX)) {
		return nil, fmt.Errorf("assemble %s: the model card declares no license identifier (%q); review the license and pass one", directory, declared.License.SPDX)
	}
	for _, name := range licenseFiles {
		data, err := os.ReadFile(filepath.Join(path, name))
		if err != nil {
			continue
		}
		declared.License.Path, declared.License.MediaType = name, textMediaType
		if strings.EqualFold(filepath.Ext(name), markdownExtension) {
			declared.License.MediaType = markdownMediaType
		}
		if declared.License.Artifact, err = artifact.IdentifyBytes(artifact.KindFile, data); err != nil {
			return nil, err
		}
		return json.Marshal(declared)
	}
	return nil, fmt.Errorf("assemble %s: %w", directory, errors.New("no license file or model card to bind the license to"))
}

// cardLicense returns the license the model card's front matter declares.
func cardLicense(directory string) string {
	data, err := os.ReadFile(filepath.Join(directory, modelCardFile))
	if err != nil {
		return ""
	}
	lines := strings.Split(strings.ReplaceAll(string(data), "\r\n", "\n"), "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) != cardFence {
		return ""
	}
	for _, line := range lines[1:] {
		if strings.TrimSpace(line) == cardFence {
			break
		}
		if value, isLicense := strings.CutPrefix(strings.TrimSpace(line), cardLicenseField); isLicense {
			return strings.Trim(strings.TrimSpace(value), `"'`)
		}
	}
	return ""
}
