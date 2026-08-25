package dataset

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
)

const (
	// InventoryMediaType identifies dataset inventory payloads.
	InventoryMediaType = "application/vnd.overgo.dataset-inventory+json"
	// InventorySchema identifies the inventory contract version.
	InventorySchema = "overgo/dataset-inventory/v1"
	// InventoryAssetName names the version-owned inventory asset.
	InventoryAssetName = "inventory"
)

var inventoryContract = artifact.DocumentContract{
	Kind: artifact.KindDatasetShard, MediaType: InventoryMediaType, Schema: InventorySchema,
}

var inventoryCodec = artifact.JSONDocumentCodec(
	"dataset inventory", inventoryContract.Kind, inventoryContract.MediaType, inventoryContract.Schema,
	canonicalizeInventory, func(value Inventory) artifact.ID { return value.ID },
	func(value *Inventory, id artifact.ID) { value.ID = id }, cloneInventory,
)

// InventoryFile records one external dataset file without owning its bytes.
type InventoryFile struct {
	Path         string `json:"path"`
	OriginalName string `json:"original_name"`
	Extension    string `json:"extension,omitempty"`
	Modality     string `json:"modality"`
	Format       string `json:"format"`
	Bytes        uint64 `json:"bytes"`
	ModifiedUnix int64  `json:"modified_unix"`
	// Digest is the hex sha256 of the file's bytes. Directory
	// registration records it so dataset identity follows content, not
	// path and metadata: changed bytes re-identify even when size and
	// modification time are unchanged. Legacy-compiled inventories omit
	// it and keep their metadata-derived identity.
	Digest     string `json:"digest,omitempty"`
	Structured bool   `json:"structured,omitempty"`
	Attributes string `json:"attributes,omitempty"`
}

// Inventory defines immutable file-level dataset facts.
type Inventory struct {
	Version uint16          `json:"version"`
	Files   []InventoryFile `json:"files"`
	ID      artifact.ID     `json:"-"`
}

// NewInventory validates and identifies one file inventory.
func NewInventory(files []InventoryFile) (Inventory, error) {
	return inventoryCodec.New(Inventory{Version: artifact.InitialDocumentVersion, Files: slices.Clone(files)})
}

func loadInventory(ctx context.Context, reader artifact.Reader, id artifact.ID) (Inventory, bool, error) {
	return inventoryCodec.Read(ctx, reader, id)
}

func inventoryContent(value Inventory) (artifact.Content, error) {
	return inventoryCodec.Content(value)
}

func canonicalizeInventory(inventory *Inventory) error {
	if inventory == nil || inventory.Version != artifact.InitialDocumentVersion {
		return errors.New("dataset: invalid inventory")
	}
	sort.Slice(inventory.Files, func(i, j int) bool { return inventory.Files[i].Path < inventory.Files[j].Path })
	for index, file := range inventory.Files {
		if !validInventoryText(file.Path) || !validInventoryText(file.OriginalName) ||
			!validInventoryText(file.Modality) || !validInventoryText(file.Format) ||
			index > 0 && inventory.Files[index-1].Path == file.Path {
			return fmt.Errorf("dataset: invalid inventory file %q", file.Path)
		}
		if file.Extension != "" && !validInventoryText(file.Extension) || strings.ContainsAny(file.Attributes, "\r\n") {
			return fmt.Errorf("dataset: invalid inventory metadata for %q", file.Path)
		}
	}
	return nil
}

func cloneInventory(inventory Inventory) Inventory {
	inventory.Files = slices.Clone(inventory.Files)
	return inventory
}

func validInventoryText(value string) bool {
	return value != "" && strings.TrimSpace(value) == value && !strings.ContainsAny(value, "\r\n\\")
}
