package main

import (
	"overgo/internal/artifact"
)

type mediaSharedComponents struct {
	Version        uint16                   `json:"version"`
	Baseline       string                   `json:"baseline_sha256"`
	Source         string                   `json:"source_base"`
	Sources        map[string]string        `json:"source_sha256"`
	SourceEvidence map[string]artifact.ID   `json:"source_evidence"`
	Model          artifact.ID              `json:"model"`
	Recipe         artifact.ID              `json:"recipe"`
	Harness        artifact.ID              `json:"harness"`
	Acceptance     artifact.ID              `json:"acceptance"`
	Environment    artifact.ID              `json:"environment"`
	Quality        []mediaSharedCheck       `json:"quality"`
	Acquisitions   []mediaLoaderAcquisition `json:"acquisitions"`
	Prior          []artifact.ID            `json:"prior_attempts"`
	Decision       string                   `json:"decision"`
	Changes        string                   `json:"changes"`
	Limitations    string                   `json:"limitations"`
}

type mediaSharedCheck struct {
	Name     string      `json:"name"`
	Evidence artifact.ID `json:"evidence"`
	Command  string      `json:"command"`
}
