package modelartifact

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

// LocalImport declares model files an install already holds, served from where they lie in place of a download: the
// imported model's folder name, and for each file its path in that folder, its role, where it lies in the install,
// where its maker publishes it, its licence, and the SHA-256 and size it must have. A declaration pins what a later
// import must find, so a file whose bytes changed is refused, never imported.
type LocalImport struct {
	Name  string            `json:"name"`
	Files []LocalImportFile `json:"files"`
}

// LocalImportFile is one file of a local import.
type LocalImportFile struct {
	Path    string `json:"path"`
	Role    string `json:"role"`
	Source  string `json:"source"`
	Origin  string `json:"origin"`
	License string `json:"license"`
	SHA256  string `json:"sha256"`
	Bytes   int64  `json:"bytes"`
}

// ReadLocalImport reads a local import's declaration.
func ReadLocalImport(path string) (LocalImport, error) {
	var declaration LocalImport
	data, err := os.ReadFile(path)
	if err == nil {
		err = strictjson.DecodeBytes(data, &declaration)
	}
	if err == nil && (declaration.Name == "" || !filepath.IsLocal(declaration.Name) || len(declaration.Files) == 0) {
		err = errors.New("model artifact: a local import names its folder and at least one file")
	}
	return declaration, err
}

// ImportLocal hard-links each declared file from the install at from into root/<name>/<path>, the models root's folder
// for the import, and returns the batch that publishes the model's inventory with the import's receipt beside it: the
// declaration with each file's source path in the install. No byte is copied, and a file whose bytes are not the
// declared ones is refused, its link removed; a name already standing for another file is refused too. A hard link
// needs the install and the models root on one volume.
func ImportLocal(ctx context.Context, declaration LocalImport, from, root string) (artifact.Batch, error) {
	folder := filepath.Join(root, declaration.Name)
	specs := make([]FileSpec, 0, len(declaration.Files))
	var linked []string
	refuse := func(err error) (artifact.Batch, error) {
		for _, name := range linked {
			err = errors.Join(err, os.Remove(name))
		}
		return artifact.Batch{}, err
	}
	for _, file := range declaration.Files {
		role, err := artifact.ParseComponentRole(file.Role)
		if err != nil || !filepath.IsLocal(file.Path) || !filepath.IsLocal(file.Source) {
			return refuse(fmt.Errorf("model artifact: the local import's file %s has an unknown role or a path outside its folder: %v", file.Path, err))
		}
		source, target := filepath.Join(from, filepath.FromSlash(file.Source)), filepath.Join(folder, filepath.FromSlash(file.Path))
		sourceInfo, err := os.Stat(source)
		if err != nil {
			return refuse(err)
		}
		if targetInfo, err := os.Stat(target); err == nil {
			if !os.SameFile(sourceInfo, targetInfo) {
				return refuse(fmt.Errorf("model artifact: %s already names another file than %s", target, source))
			}
		} else {
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return refuse(err)
			}
			if err := os.Link(source, target); err != nil {
				return refuse(fmt.Errorf("model artifact: link %s to %s: %w", target, source, err))
			}
			linked = append(linked, target)
		}
		specs = append(specs, FileSpec{Path: target, Name: strings.TrimSuffix(file.Path, filepath.Ext(file.Path)), Role: role})
	}
	inventory, err := fromFilesContext(ctx, folder, specs)
	if err != nil {
		return refuse(err)
	}
	for index, file := range declaration.Files {
		component := inventory.Components[index]
		if component.ID.DigestHex() != strings.ToLower(file.SHA256) || int64(component.Size) != file.Bytes {
			return refuse(fmt.Errorf("model artifact: %s is %s of %d bytes, not the declared %s of %d bytes", file.Source, component.ID.DigestHex(), component.Size, file.SHA256, file.Bytes))
		}
	}
	receipt := struct {
		LocalImport
		From string `json:"from"`
	}{declaration, from}
	data, err := json.Marshal(receipt)
	if err != nil {
		return refuse(err)
	}
	receiptID, err := artifact.IdentifyBytes(artifact.KindFile, data)
	if err != nil {
		return refuse(err)
	}
	batch, err := inventory.Batch("local-import/" + declaration.Name)
	if err != nil {
		return refuse(err)
	}
	batch.Contents = append(batch.Contents, artifact.Content{Descriptor: artifact.Descriptor{ID: receiptID, Size: uint64(len(data)), MediaType: artifact.JSONMediaType}, Data: data})
	batch.Lineage = append(batch.Lineage, artifact.DependencyLineage(inventory.Manifest.ID, receiptID)...)
	return batch, nil
}
