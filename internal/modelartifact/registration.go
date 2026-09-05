package modelartifact

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/gitauthority"
	"overgo/internal/pathidentity"
	"overgo/internal/processcontrol"
	"overgo/internal/strictjson"
)

type modelRegistration struct {
	Directory        string      `json:"directory"`
	Model            artifact.ID `json:"model"`
	TensorInventory  artifact.ID `json:"tensor_inventory"`
	SourceRepository string      `json:"source_repository"`
	SourceCommit     string      `json:"source_commit"`
	Components       []FileSpec  `json:"components,omitempty"`
	License          struct {
		Path      string      `json:"path"`
		Artifact  artifact.ID `json:"artifact"`
		SPDX      string      `json:"spdx"`
		MediaType string      `json:"media_type"`
	} `json:"license"`
}

// PrepareRegistration verifies a bounded source declaration and every physical
// model, tensor inventory and license before returning one publication batch.
// It references model bytes in place and does not activate or execute models.
func PrepareRegistration(ctx context.Context, root, declaration string) (artifact.Batch, error) {
	if ctx == nil {
		return artifact.Batch{}, errors.New("model registration: nil context")
	}
	data, err := artifact.ReadContentFile(declaration)
	if err != nil {
		return artifact.Batch{}, err
	}
	var specifications []modelRegistration
	if err := strictjson.DecodeBytes(data, &specifications); err != nil {
		return artifact.Batch{}, err
	}
	if len(specifications) == 0 {
		return artifact.Batch{}, errors.New("model registration: empty declaration")
	}
	specID, err := artifact.IdentifyBytes(artifact.KindFile, data)
	if err != nil {
		return artifact.Batch{}, err
	}
	batch := artifact.Batch{Contents: []artifact.Content{{
		Descriptor: artifact.Descriptor{ID: specID, Size: uint64(len(data)), MediaType: artifact.JSONMediaType}, Data: data,
	}}}
	seen := make(map[artifact.ID]bool, len(specifications))
	for _, specification := range specifications {
		if seen[specification.Model] {
			return artifact.Batch{}, errors.New("model registration: duplicate model identity")
		}
		seen[specification.Model] = true
		inventory, license, err := registrationInventory(ctx, root, specification)
		if err != nil {
			return artifact.Batch{}, fmt.Errorf("register %s: %w", specification.Directory, err)
		}
		part, err := inventory.Batch("registration")
		if err != nil {
			return artifact.Batch{}, err
		}
		batch.Artifacts = append(batch.Artifacts, part.Artifacts...)
		batch.Contents = append(batch.Contents, part.Contents...)
		if !slices.ContainsFunc(batch.Contents, func(content artifact.Content) bool { return content.Descriptor.ID == license.Descriptor.ID }) {
			batch.Contents = append(batch.Contents, license)
		}
		batch.Manifests = append(batch.Manifests, part.Manifests...)
		batch.Locations = append(batch.Locations, part.Locations...)
		batch.Lineage = append(batch.Lineage, part.Lineage...)
		batch.Lineage = append(batch.Lineage, artifact.DependencyLineage(inventory.Manifest.ID, specID, license.Descriptor.ID)...)
	}
	return batch, nil
}

func registrationInventory(ctx context.Context, root string, specification modelRegistration) (Inventory, artifact.Content, error) {
	var empty Inventory
	var noLicense artifact.Content
	if err := ctx.Err(); err != nil {
		return empty, noLicense, err
	}
	if specification.Model.Kind() != artifact.KindModel || specification.TensorInventory.Kind() != artifact.KindTensorInventory ||
		!gitauthority.ValidObjectID(specification.SourceCommit) || specification.SourceRepository == "" || specification.License.SPDX == "" {
		return empty, noLicense, errors.New("incomplete identity or source declaration")
	}
	directory := filepath.Join(root, filepath.FromSlash(specification.Directory))
	contained, err := pathidentity.Contains(root, directory)
	if err != nil || !contained || !filepath.IsLocal(specification.Directory) {
		return empty, noLicense, errors.New("model directory escapes root")
	}
	for _, check := range []struct {
		arguments []string
		expected  string
	}{
		{[]string{"rev-parse", "HEAD"}, specification.SourceCommit},
		{[]string{"remote", "get-url", "origin"}, specification.SourceRepository},
		{[]string{"diff", "--no-ext-diff", "--no-textconv", "--quiet", "HEAD", "--"}, ""},
	} {
		arguments := append([]string{"--no-replace-objects", "-C", directory}, check.arguments...)
		var output strings.Builder
		receipt, err := processcontrol.Run(ctx, processcontrol.Command{Path: "git", Args: arguments, Env: gitauthority.ReaderEnvironment(), Stdout: &output})
		if err != nil || receipt.ExitCode != 0 || strings.TrimSpace(output.String()) != check.expected {
			return empty, noLicense, errors.New("source repository, commit or tracked bytes differ")
		}
	}
	var inventory Inventory
	if len(specification.Components) == 0 {
		inventory, err = FromHFPath(directory)
	} else {
		specification.Components = slices.Clone(specification.Components)
		for index := range specification.Components {
			component := &specification.Components[index]
			if !filepath.IsLocal(component.Path) {
				return empty, noLicense, errors.New("component path escapes model directory")
			}
			component.Path = filepath.Join(directory, filepath.FromSlash(component.Path))
		}
		inventory, err = FromFiles(directory, specification.Components)
	}
	if err != nil {
		return empty, noLicense, err
	}
	if inventory.Manifest.ID != specification.Model || inventory.TensorInventory.ID != specification.TensorInventory {
		return empty, noLicense, errors.New("model or tensor inventory differs from declaration")
	}
	licensePath := filepath.Join(directory, filepath.FromSlash(specification.License.Path))
	contained, err = pathidentity.Contains(directory, licensePath)
	if err != nil || !contained || !filepath.IsLocal(specification.License.Path) {
		return empty, noLicense, errors.New("license path escapes model directory")
	}
	data, err := artifact.ReadContentFile(licensePath)
	if err != nil {
		return empty, noLicense, err
	}
	if specification.License.MediaType != "text/plain" && specification.License.MediaType != "text/markdown" {
		return empty, noLicense, errors.New("license declaration requires a text media type")
	}
	license := artifact.Content{Descriptor: artifact.Descriptor{ID: specification.License.Artifact, Size: uint64(len(data)), MediaType: specification.License.MediaType}, Data: data}
	if license.Descriptor.ID.Kind() != artifact.KindFile || license.Validate() != nil {
		return empty, noLicense, errors.New("license bytes differ from declaration")
	}
	return inventory, license, ctx.Err()
}
