package modelartifact

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
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
// needs the install and the models root on one volume. Every declared file is admitted, its role, paths, digest and
// size, and its size held to the install's, before any file is linked or read (SQA-387): a declaration refused after
// its files were hashed had read gigabytes first.
func ImportLocal(ctx context.Context, declaration LocalImport, from, root string) (artifact.Batch, error) {
	folder := filepath.Join(root, declaration.Name)
	specs := make([]FileSpec, 0, len(declaration.Files))
	roles := make([]artifact.ComponentRole, len(declaration.Files))
	sources := make([]os.FileInfo, len(declaration.Files))
	for index, file := range declaration.Files {
		role, err := artifact.ParseComponentRole(file.Role)
		if err != nil || !filepath.IsLocal(file.Path) || !filepath.IsLocal(file.Source) {
			return artifact.Batch{}, fmt.Errorf("model artifact: the local import's file %s has an unknown role or a path outside its folder: %v", file.Path, err)
		}
		if digest, err := hex.DecodeString(file.SHA256); err != nil || len(digest) != sha256.Size || file.Bytes <= 0 {
			return artifact.Batch{}, fmt.Errorf("model artifact: the local import's file %s declares %q of %d bytes, which no file is", file.Path, file.SHA256, file.Bytes)
		}
		info, err := os.Stat(filepath.Join(from, filepath.FromSlash(file.Source)))
		if err != nil {
			return artifact.Batch{}, err
		}
		if info.Size() != file.Bytes {
			return artifact.Batch{}, fmt.Errorf("model artifact: %s is %d bytes in the install, not the declared %d", file.Source, info.Size(), file.Bytes)
		}
		roles[index], sources[index] = role, info
	}
	var linked []string
	refuse := func(err error) (artifact.Batch, error) {
		for _, name := range linked {
			err = errors.Join(err, os.Remove(name))
		}
		return artifact.Batch{}, err
	}
	for index, file := range declaration.Files {
		role, sourceInfo := roles[index], sources[index]
		source, target := filepath.Join(from, filepath.FromSlash(file.Source)), filepath.Join(folder, filepath.FromSlash(file.Path))
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
