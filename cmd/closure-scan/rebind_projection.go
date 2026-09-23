package main

import (
	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
)

// closureReview is valid only for its exact snapshot and store head.
// It shares one command's parsed sources and history; it is never persisted.
type closureReview struct {
	source             string
	head               artifact.CommitID
	sequence           uint64
	candidates         []closurescan.Candidate
	index              closurescan.RebindIndex
	documents, history []closureledger.Document
	aliases            map[string]artifact.ID
	recovery           closureRecoveryAnalysis
	projection         closurescan.Projection
}
