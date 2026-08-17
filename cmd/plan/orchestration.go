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
	"overgo/internal/jsonfile"
	"overgo/internal/plan"
	"overgo/internal/repodb"
)

// recordExplorationGrant commits an externally-issued GPU-minute budget
// token for autonomous proposal work.
func recordExplorationGrant(root, inputPath string, output io.Writer) error {
	var specification struct {
		Proposer   artifact.ID `json:"proposer"`
		Authority  artifact.ID `json:"authority"`
		GPUMinutes uint64      `json:"gpu_minutes"`
	}
	if err := jsonfile.Decode(inputPath, &specification); err != nil {
		return err
	}
	grant, err := plan.NewExplorationGrant(specification.Proposer, specification.Authority, specification.GPUMinutes)
	if err != nil {
		return err
	}
	return commitExplorationContent(root, grant.Content, "exploration/grant/", output,
		fmt.Sprintf("issued %d GPU-minutes to %s", grant.IssuedGPUMinutes, grant.Proposer))
}

// recordExplorationCharge commits one experiment's spend against a grant,
// refusing spend the derived balance cannot admit.
func recordExplorationCharge(root, inputPath string, output io.Writer) error {
	var specification struct {
		Grant      artifact.ID `json:"grant"`
		Experiment artifact.ID `json:"experiment"`
		GPUMinutes uint64      `json:"gpu_minutes"`
	}
	if err := jsonfile.Decode(inputPath, &specification); err != nil {
		return err
	}
	store, err := repodb.Open(filepath.Join(root, "repodb-store"))
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	grantContent, ok, err := store.Content(ctx, specification.Grant)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("exploration grant %s is not committed", specification.Grant)
	}
	grant, err := plan.ParseExplorationGrant(grantContent.Data)
	if err != nil {
		return err
	}
	result, err := store.Query(ctx, repodb.Query{Kind: artifact.KindEvidence, MaxResults: repodb.MaxQueryResults})
	if err != nil {
		return err
	}
	charges := make([]plan.ExplorationCharge, 0)
	for _, descriptor := range result.Artifacts {
		content, ok, err := store.Content(ctx, descriptor.ID)
		if err != nil || !ok {
			continue
		}
		if charge, err := plan.ParseExplorationCharge(content.Data); err == nil && charge.Grant == grant.ID {
			charges = append(charges, charge)
		}
	}
	admission, err := plan.AdmitExploration(grant, charges, specification.GPUMinutes, false)
	if err != nil {
		return err
	}
	if !admission.Admitted {
		return fmt.Errorf("exploration spend refused: %s", admission.Reason)
	}
	charge, err := plan.NewExplorationCharge(grant.ID, specification.Experiment, specification.GPUMinutes)
	if err != nil {
		return err
	}
	if err := commitExplorationContent(root, charge.Content, "exploration/charge/", output,
		fmt.Sprintf("charged %d GPU-minutes; %s", charge.GPUMinutes, admission.Reason)); err != nil {
		return err
	}
	return nil
}

func commitExplorationContent(root string, content func() (artifact.Content, error), keyPrefix string, output io.Writer, message string) error {
	document, err := content()
	if err != nil {
		return err
	}
	store, err := repodb.Open(filepath.Join(root, "repodb-store"))
	if err != nil {
		return err
	}
	defer store.Close()
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: keyPrefix + document.Descriptor.ID.String(), Contents: []artifact.Content{document},
	}); err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "%s: %s\n", message, document.Descriptor.ID)
	return err
}

func recordWorkLease(root, inputPath string, output io.Writer) error {
	data, err := os.ReadFile(inputPath)
	if err != nil {
		return fmt.Errorf("read work lease: %w", err)
	}
	store, err := repodb.Open(filepath.Join(root, "repodb-store"))
	if err != nil {
		return err
	}
	defer store.Close()
	lease, err := plan.RecordWorkLease(context.Background(), store, data)
	if err != nil {
		return fmt.Errorf("record work lease: %w", err)
	}
	_, err = fmt.Fprintf(output, "recorded advisory work lease %s for %s\n", lease.ID, lease.Task)
	return err
}

func recordWorkLeaseOutcome(root, inputPath string, output io.Writer) error {
	data, err := os.ReadFile(inputPath)
	if err != nil {
		return fmt.Errorf("read lease outcome: %w", err)
	}
	store, err := repodb.Open(filepath.Join(root, "repodb-store"))
	if err != nil {
		return err
	}
	defer store.Close()
	outcome, err := plan.RecordLeaseOutcome(context.Background(), store, data)
	if err != nil {
		return fmt.Errorf("record lease outcome: %w", err)
	}
	_, err = fmt.Fprintf(output, "recorded lease outcome %s for %s\n", outcome.ID, outcome.Lease)
	return err
}

type leaseReport struct {
	GeneratedAt string                `json:"generated_at"`
	Leases      []plan.WorkLease      `json:"leases"`
	Advisory    plan.ResourceAdvisory `json:"advisory"`
	Exploration []explorationBalance  `json:"exploration,omitempty"`
}

func printLeaseReport(root string, capacity plan.Resources, output io.Writer) error {
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
		lease, ok, err := plan.ReadWorkLease(ctx, store, alias.Target)
		if err != nil {
			return err
		}
		if !ok {
			continue
		}
		leases, seen[lease.ID] = append(leases, lease), true
	}
	sort.Slice(leases, func(i, j int) bool { return leases[i].Task < leases[j].Task })
	grants, charges := make([]plan.ExplorationGrant, 0), make([]plan.ExplorationCharge, 0)
	for _, descriptor := range result.Artifacts {
		content, ok, err := store.Content(ctx, descriptor.ID)
		if err != nil {
			return err
		}
		if !ok {
			continue
		}
		if grant, err := plan.ParseExplorationGrant(content.Data); err == nil {
			grants = append(grants, grant)
			continue
		}
		if charge, err := plan.ParseExplorationCharge(content.Data); err == nil {
			charges = append(charges, charge)
		}
	}
	exploration := make([]explorationBalance, 0, len(grants))
	for _, grant := range grants {
		owned := make([]plan.ExplorationCharge, 0)
		for _, charge := range charges {
			if charge.Grant == grant.ID {
				owned = append(owned, charge)
			}
		}
		remaining, err := plan.ExplorationBalance(grant, owned)
		balance := explorationBalance{
			Grant: grant.ID.String(), Proposer: grant.Proposer.String(),
			IssuedGPUMinutes: grant.IssuedGPUMinutes, RemainingGPUMinutes: remaining,
		}
		if err != nil {
			balance.Violation = err.Error()
		}
		exploration = append(exploration, balance)
	}
	sort.Slice(exploration, func(i, j int) bool { return exploration[i].Grant < exploration[j].Grant })
	now := time.Now().UTC()
	payload := leaseReport{
		GeneratedAt: now.Format(time.RFC3339Nano), Leases: leases,
		Advisory: plan.AssessResources(now, capacity, leases), Exploration: exploration,
	}
	encoder := json.NewEncoder(output)
	encoder.SetEscapeHTML(false)
	return encoder.Encode(payload)
}

// explorationBalance is one grant's derived budget state in the lease report.
type explorationBalance struct {
	Grant               string `json:"grant"`
	Proposer            string `json:"proposer"`
	IssuedGPUMinutes    uint64 `json:"issued_gpu_minutes"`
	RemainingGPUMinutes uint64 `json:"remaining_gpu_minutes"`
	Violation           string `json:"violation,omitempty"`
}
