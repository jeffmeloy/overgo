package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/plan"
	"overgo/internal/repodb"
)

func recordWorkLease(root, inputPath string, output io.Writer) error {
	data, err := os.ReadFile(inputPath)
	if err != nil {
		return fmt.Errorf("read work lease: %w", err)
	}
	lease, err := plan.NormalizeWorkLease(data)
	if err != nil {
		return err
	}
	store, err := repodb.Open(filepath.Join(root, "repodb-store"))
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	current, exists, err := store.ResolveAlias(ctx, plan.WorkLeaseAlias(lease.Worktree))
	if err != nil {
		return err
	}
	var previous *artifact.ID
	if exists {
		previous = &current
	}
	batch, err := plan.WorkLeaseBatch(lease, previous)
	if err != nil {
		return err
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		return fmt.Errorf("record work lease: %w", err)
	}
	_, err = fmt.Fprintf(output, "recorded advisory work lease %s for %s\n", lease.ID, lease.Task)
	return err
}

type leaseReport struct {
	GeneratedAt string                `json:"generated_at"`
	Leases      []plan.WorkLease      `json:"leases"`
	Advisory    plan.ResourceAdvisory `json:"advisory"`
}

func printLeaseReport(root string, capacity plan.ResourceCapacity, output io.Writer) error {
	store, err := repodb.OpenReadOnly(filepath.Join(root, "repodb-store"))
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	result, err := store.Query(ctx, repodb.Query{Kind: artifact.KindEvidence, MaxResults: repodb.MaxQueryResults})
	if err != nil {
		return err
	}
	if result.Truncated {
		return fmt.Errorf("lease report evidence query was truncated")
	}
	leases := make([]plan.WorkLease, 0)
	seen := map[artifact.ID]bool{}
	for _, alias := range result.Aliases {
		if seen[alias.Target] {
			continue
		}
		content, ok, err := store.Content(ctx, alias.Target)
		if err != nil {
			return err
		}
		if !ok || content.Descriptor.MediaType != plan.WorkLeaseMediaType || content.Descriptor.Schema != plan.WorkLeaseSchema {
			continue
		}
		lease, err := plan.ParseWorkLease(content.Data)
		if err != nil {
			return err
		}
		leases, seen[lease.ID] = append(leases, lease), true
	}
	sort.Slice(leases, func(i, j int) bool { return leases[i].Task < leases[j].Task })
	now := time.Now().UTC()
	payload := leaseReport{GeneratedAt: now.Format(time.RFC3339Nano), Leases: leases, Advisory: plan.AssessResources(now, capacity, leases)}
	encoder := json.NewEncoder(output)
	encoder.SetEscapeHTML(false)
	return encoder.Encode(payload)
}
