package main

import (
	"cmp"
	"context"
	"encoding/json"
	"io"
	"maps"
	"os"
	"slices"

	"overgo/internal/overgodb"
)

// schemaCensus is what the store holds of one schema, read from the catalog.
// Records and Bytes count descriptors, which a released record keeps; Held
// and HeldBytes count the records whose bytes are still on disk. Small counts
// the records smaller than one page, the unit a filesystem allocates, so each
// costs a whole allocation for a fraction of its bytes. Modal is the most
// common record size and how many records share it: records stamped from one
// template cluster at one size.
type schemaCensus struct {
	Schema       string `json:"schema"`
	Records      int    `json:"records"`
	Bytes        uint64 `json:"bytes"`
	Held         int    `json:"held"`
	HeldBytes    uint64 `json:"held_bytes"`
	Small        int    `json:"small"`
	ModalSize    uint64 `json:"modal_size"`
	ModalRecords int    `json:"modal_records"`
	sizes        map[uint64]int
}

// writeCensus walks every descriptor once, a page at a time, and prints the
// schemas by record count, the order in which they tax a walk, a rebind or a
// lookup.
func writeCensus(output io.Writer, repository string, page int) error {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	schemas := map[string]*schemaCensus{}
	query := overgodb.Query{Projection: overgodb.ProjectArtifacts, MaxResults: page}
	for {
		result, err := store.Query(ctx, query)
		if err != nil {
			return err
		}
		for _, descriptor := range result.Artifacts {
			census := schemas[descriptor.Schema]
			if census == nil {
				census = &schemaCensus{Schema: descriptor.Schema, sizes: map[uint64]int{}}
				schemas[descriptor.Schema] = census
			}
			census.Records++
			census.Bytes += descriptor.Size
			if descriptor.Size < uint64(os.Getpagesize()) {
				census.Small++
			}
			if census.sizes[descriptor.Size]++; census.sizes[descriptor.Size] > census.ModalRecords {
				census.ModalSize, census.ModalRecords = descriptor.Size, census.sizes[descriptor.Size]
			}
			if durable, err := store.HasContent(ctx, descriptor.ID); err != nil {
				return err
			} else if durable {
				census.Held++
				census.HeldBytes += descriptor.Size
			}
		}
		if query.Cursor = result.Next; query.Cursor == nil {
			break
		}
	}
	encoder := json.NewEncoder(output)
	encoder.SetIndent("", "  ")
	return encoder.Encode(slices.SortedFunc(maps.Values(schemas), func(left, right *schemaCensus) int {
		return cmp.Or(cmp.Compare(right.Records, left.Records), cmp.Compare(left.Schema, right.Schema))
	}))
}
