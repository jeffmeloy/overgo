package runrecord

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

func TestAutomationScheduleClaim(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	data := []byte("schedule-plan")
	plan, err := artifact.IdentifyBytes(artifact.KindProfile, data)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key: "automation/schedule/plan", Contents: []artifact.Content{{
			Descriptor: artifact.Descriptor{ID: plan, Size: uint64(len(data)), MediaType: "application/octet-stream"},
			Data:       data,
		}},
	}); err != nil {
		t.Fatal(err)
	}
	authority := AutomationScheduleAuthority{Repository: store}
	due := time.Date(2026, time.August, 24, 12, 0, 0, 0, time.UTC)
	const contenders = 8
	start := make(chan struct{})
	errorsSeen := make(chan error, contenders)
	var winners atomic.Uint32
	var wait sync.WaitGroup
	for range contenders {
		wait.Go(func() {
			<-start
			_, won, claimErr := authority.Claim(t.Context(), "daily-report", plan, due)
			if claimErr != nil {
				errorsSeen <- claimErr
				return
			}
			if won {
				winners.Add(1)
			}
		})
	}
	close(start)
	wait.Wait()
	close(errorsSeen)
	for claimErr := range errorsSeen {
		t.Fatal(claimErr)
	}
	if winners.Load() != 1 {
		t.Fatalf("schedule claim winners = %d", winners.Load())
	}
	claim, found, err := authority.Current(t.Context(), "daily-report")
	if err != nil || !found || claim.Plan != plan || claim.DueUnixNano != due.UnixNano() {
		t.Fatalf("durable schedule claim = (%+v, %v, %v)", claim, found, err)
	}
}
