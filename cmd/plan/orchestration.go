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
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
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
	store, err := overgodb.Open(filepath.Join(root, "overgodb-store"))
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	grantContent, ok, err := artifact.ReadContent(ctx, store, specification.Grant)
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
	charges := make([]plan.ExplorationCharge, 0)
	_, err = overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: plan.ExplorationChargeMediaType, Schema: plan.ExplorationChargeSchema,
		}}, Order: overgodb.DocumentOldestFirst,
	}, plan.ParseExplorationCharge, func(_ overgodb.DocumentView, charge plan.ExplorationCharge) error {
		if charge.Grant == grant.ID {
			charges = append(charges, charge)
		}
		return nil
	})
	if err != nil {
		return err
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
	// Commit through the store handle already held: a second open would
	// deadlock on the single-writer lock.
	document, err := charge.Content()
	if err != nil {
		return err
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "exploration/charge/" + document.Descriptor.ID.String(), Contents: []artifact.Content{document},
	}); err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "charged %d GPU-minutes; %s: %s\n",
		charge.GPUMinutes, admission.Reason, document.Descriptor.ID)
	return err
}

func commitExplorationContent(root string, content func() (artifact.Content, error), keyPrefix string, output io.Writer, message string) error {
	document, err := content()
	if err != nil {
		return err
	}
	store, err := overgodb.Open(filepath.Join(root, "overgodb-store"))
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
	store, err := overgodb.Open(filepath.Join(root, "overgodb-store"))
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
	store, err := overgodb.Open(filepath.Join(root, "overgodb-store"))
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
	// Reconciliation is recommendation-only: abandoned-lease retries and
	// measurement error, never an action the report takes itself.
	Reconciliation []plan.LeaseRecommendation `json:"reconciliation,omitempty"`
}

func printLocalitySchedule(root, inputPath string, output io.Writer) error {
	var request plan.LocalityScheduleRequest
	if err := jsonfile.DecodeStrict(inputPath, &request); err != nil {
		return err
	}
	store, err := overgodb.OpenReadOnly(filepath.Join(root, "overgodb-store"))
	if err != nil {
		return err
	}
	defer store.Close()
	schedule, err := plan.ScheduleByArtifactLocality(context.Background(), store, request)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(output)
	encoder.SetEscapeHTML(false)
	return encoder.Encode(schedule)
}

func printLeaseReport(root string, capacity plan.Resources, output io.Writer) error {
	store, err := overgodb.OpenReadOnly(filepath.Join(root, "overgodb-store"))
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	leases := make([]plan.WorkLease, 0)
	_, err = overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: plan.WorkLeaseMediaType, Schema: plan.WorkLeaseSchema,
		}}, AliasPrefixes: []string{plan.WorkLeaseAliasRoot}, Order: overgodb.DocumentOldestFirst,
	}, plan.ParseWorkLease, func(_ overgodb.DocumentView, lease plan.WorkLease) error {
		leases = append(leases, lease)
		return nil
	})
	if err != nil {
		return err
	}
	sort.Slice(leases, func(i, j int) bool { return leases[i].Task < leases[j].Task })
	grants, charges := make([]plan.ExplorationGrant, 0), make([]plan.ExplorationCharge, 0)
	_, err = overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{Kind: artifact.KindEvidence, MediaType: plan.ExplorationGrantMediaType, Schema: plan.ExplorationGrantSchema}},
		Order:     overgodb.DocumentOldestFirst,
	}, plan.ParseExplorationGrant, func(_ overgodb.DocumentView, grant plan.ExplorationGrant) error {
		grants = append(grants, grant)
		return nil
	})
	if err != nil {
		return err
	}
	_, err = overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{Kind: artifact.KindEvidence, MediaType: plan.ExplorationChargeMediaType, Schema: plan.ExplorationChargeSchema}},
		Order:     overgodb.DocumentOldestFirst,
	}, plan.ParseExplorationCharge, func(_ overgodb.DocumentView, charge plan.ExplorationCharge) error {
		charges = append(charges, charge)
		return nil
	})
	if err != nil {
		return err
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
	outcomes := make([]plan.LeaseOutcome, 0)
	_, err = overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{Kind: artifact.KindEvidence, MediaType: plan.LeaseOutcomeMediaType, Schema: plan.LeaseOutcomeSchema}},
		Order:     overgodb.DocumentOldestFirst,
	}, plan.ParseLeaseOutcome, func(_ overgodb.DocumentView, outcome plan.LeaseOutcome) error {
		outcomes = append(outcomes, outcome)
		return nil
	})
	if err != nil {
		return err
	}
	now := time.Now().UTC()
	payload := leaseReport{
		GeneratedAt: now.Format(time.RFC3339Nano), Leases: leases,
		Advisory: plan.AssessResources(now, capacity, leases), Exploration: exploration,
		Reconciliation: plan.ReconcileExperimentLeases(leases, outcomes, now),
	}
	encoder := json.NewEncoder(output)
	encoder.SetEscapeHTML(false)
	return encoder.Encode(payload)
}

// recordExperimentTransition commits one experiment lifecycle transition:
// the spec names the state, the experiment, the evidence justifying the
// transition, and -- for every record after the initial proposal -- the
// prior record it succeeds. The state machine in runrecord refuses illegal
// edges and retry drift; this tool only carries facts to it.
func recordExperimentTransition(root, inputPath string, output io.Writer) error {
	var specification struct {
		State           string       `json:"state"`
		Experiment      artifact.ID  `json:"experiment"`
		Evidence        artifact.ID  `json:"evidence"`
		Prior           *artifact.ID `json:"prior,omitempty"`
		Retry           uint32       `json:"retry,omitempty"`
		HeartbeatExpiry string       `json:"heartbeat_expiry,omitempty"`
		Checkpoint      *artifact.ID `json:"checkpoint,omitempty"`
	}
	if err := jsonfile.Decode(inputPath, &specification); err != nil {
		return err
	}
	store, err := overgodb.Open(filepath.Join(root, "overgodb-store"))
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	var prior *runrecord.ExperimentLifecycle
	if specification.Prior != nil {
		content, ok, err := artifact.ReadContent(ctx, store, *specification.Prior)
		if err != nil {
			return err
		}
		if !ok {
			return fmt.Errorf("prior lifecycle record %s is not committed", specification.Prior)
		}
		parsed, err := runrecord.ParseExperimentLifecycle(content.Data)
		if err != nil {
			return err
		}
		prior = &parsed
	}
	record, err := runrecord.NewExperimentLifecycle(runrecord.ExperimentLifecycle{
		State: runrecord.ExperimentState(specification.State), Experiment: specification.Experiment,
		Retry: specification.Retry, Evidence: specification.Evidence,
		HeartbeatExpiry: specification.HeartbeatExpiry, Checkpoint: specification.Checkpoint,
	}, prior)
	if err != nil {
		return err
	}
	content, err := record.Content()
	if err != nil {
		return err
	}
	lineage := []artifact.Lineage{
		{Child: record.ID, Parent: record.Experiment, Relation: artifact.RelationDependsOn},
	}
	if record.Evidence != record.Experiment {
		lineage = append(lineage, artifact.Lineage{Child: record.ID, Parent: record.Evidence, Relation: artifact.RelationDependsOn})
	}
	if record.Prior != nil {
		lineage = append(lineage, artifact.Lineage{Child: record.ID, Parent: *record.Prior, Relation: artifact.RelationDependsOn})
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "automation/experiment/" + record.ID.String(), Contents: []artifact.Content{content},
		Lineage: lineage,
	}); err != nil {
		return err
	}
	fmt.Fprintf(output, "experiment transition committed: %s state=%s experiment=%s retry=%d\n",
		record.ID, record.State, record.Experiment, record.Retry)
	fmt.Fprintln(output, "honesty: transitions carry evidence identities; the state machine refuses illegal edges")
	return nil
}

// explorationBalance is one grant's derived budget state in the lease report.
type explorationBalance struct {
	Grant               string `json:"grant"`
	Proposer            string `json:"proposer"`
	IssuedGPUMinutes    uint64 `json:"issued_gpu_minutes"`
	RemainingGPUMinutes uint64 `json:"remaining_gpu_minutes"`
	Violation           string `json:"violation,omitempty"`
}
