package main

import (
	"crypto/sha256"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/binaryschema"
)

type mediaLoaderAcquisition struct {
	Consumer     string      `json:"consumer"`
	Observations artifact.ID `json:"observations"`
	Evidence     artifact.ID `json:"evidence"`
	Command      string      `json:"command"`
}

type mediaLoaderObservation struct {
	Consumer  string `json:"consumer"`
	State     string `json:"load_state"`
	Wall      int64  `json:"wall_ns"`
	Allocated uint64 `json:"total_alloc_bytes"`
	Calls     uint64 `json:"allocation_calls"`
	Peak      uint64 `json:"process_lifetime_peak_host_bytes"`
	Live      uint64 `json:"live_heap_bytes"`
	Released  uint64 `json:"heap_bytes_after_release_gc"`
	Source    uint64 `json:"source_bytes"`
	Elements  uint64 `json:"elements"`
	Tensors   uint64 `json:"tensors"`
	Values    string `json:"values_sha256"`
	Upload    uint64 `json:"host_to_device_bytes"`
	Download  uint64 `json:"device_to_host_bytes"`
	Device    uint64 `json:"device_allocated_bytes"`
	Scope     string `json:"scope"`
}

func checkMediaLoaderObservations(consumer string, rows []mediaLoaderObservation) error {
	states := []string{"first-load", "repeat-load"}
	if len(rows) != len(states) {
		return errors.New("loader baseline omits a load state")
	}
	for index, row := range rows {
		if row.Consumer != consumer || row.State != states[index] || row.Wall <= 0 || row.Allocated == 0 || row.Calls == 0 || row.Peak == 0 || row.Live == 0 || row.Released == 0 || row.Released >= row.Live || row.Elements == 0 || row.Tensors == 0 || row.Source != row.Elements*binaryschema.Uint32Bytes || len(row.Values) != sha256.Size*2 || row.Scope == "" {
			return errors.New("loader baseline lacks observed cost, output or release data")
		}
		if row.Upload != 0 || row.Download != 0 || row.Device != 0 {
			return errors.New("CPU-only loader observation contains device work")
		}
		first := rows[0]
		if row.Values != first.Values || row.Source != first.Source || row.Elements != first.Elements || row.Tensors != first.Tensors || row.Peak < first.Peak {
			return errors.New("repeated loader output or cumulative peak differs")
		}
	}
	return nil
}
