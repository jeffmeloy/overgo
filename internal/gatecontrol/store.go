// Package gatecontrol owns persistent gate lifecycle coordination. Commands
// supply domain facts; this package owns files, strict decoding, replay binding,
// heartbeat scheduling, and RepoDB reconciliation.
package gatecontrol

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
)

const (
	DebtFile      = "bin/gate_debt.json"
	HeartbeatFile = "bin/gate_lifecycle.json"
	debtVersion   = 1
)

type Store struct {
	root       string
	repository string
}

func New(root, repository string) (Store, error) {
	root = filepath.Clean(root)
	if root == "." || root == "" || !filepath.IsAbs(root) {
		return Store{}, errors.New("gate control: root must be absolute")
	}
	repository = filepath.Clean(repository)
	if repository == "" || repository == "." || filepath.IsAbs(repository) ||
		repository == ".." || strings.HasPrefix(repository, ".."+string(filepath.Separator)) {
		return Store{}, errors.New("gate control: RepoDB path must be relative")
	}
	return Store{root: root, repository: repository}, nil
}

func (store Store) DebtExists() (bool, error) {
	_, err := os.Stat(filepath.Join(store.root, filepath.FromSlash(DebtFile)))
	switch {
	case err == nil:
		return true, nil
	case errors.Is(err, os.ErrNotExist):
		return false, nil
	default:
		return false, err
	}
}

type debtEnvelope struct {
	Version     uint16         `json:"version"`
	Preparation artifact.ID    `json:"preparation"`
	Batch       artifact.Batch `json:"batch"`
}

func (store Store) PersistDebt(preparation artifact.ID, batch artifact.Batch) error {
	envelope := debtEnvelope{Version: debtVersion, Preparation: preparation, Batch: batch}
	if err := validateDebt(envelope); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(envelope, "", "  ")
	if err != nil {
		return fmt.Errorf("gate control: encode record debt: %w", err)
	}
	path := filepath.Join(store.root, filepath.FromSlash(DebtFile))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(path, append(raw, '\n'), 0o600); err != nil {
		return fmt.Errorf("gate control: persist record debt: %w", err)
	}
	return nil
}

func (store Store) Reconcile(ctx context.Context) (artifact.ID, error) {
	path := filepath.Join(store.root, filepath.FromSlash(DebtFile))
	raw, err := os.ReadFile(path)
	if err != nil {
		return artifact.ID{}, fmt.Errorf("gate control: read record debt: %w", err)
	}
	var debt debtEnvelope
	if err := strictjson.DecodeBytes(raw, &debt); err != nil {
		return artifact.ID{}, fmt.Errorf("gate control: decode record debt: %w", err)
	}
	if err := validateDebt(debt); err != nil {
		return artifact.ID{}, err
	}
	repository, err := repodb.Open(filepath.Join(store.root, store.repository))
	if err != nil {
		return artifact.ID{}, err
	}
	defer repository.Close()
	if _, ok, err := repository.Content(ctx, debt.Preparation); err != nil {
		return artifact.ID{}, err
	} else if !ok {
		return artifact.ID{}, errors.New("gate control: debt preparation is absent from RepoDB")
	}
	if _, err := repository.Commit(ctx, debt.Batch); err != nil {
		return artifact.ID{}, err
	}
	if err := os.Remove(path); err != nil {
		return artifact.ID{}, err
	}
	return debt.Preparation, nil
}

func validateDebt(debt debtEnvelope) error {
	if debt.Version != debtVersion || debt.Preparation.Kind() != artifact.KindEvidence {
		return errors.New("gate control: invalid record debt envelope")
	}
	if err := debt.Batch.Validate(); err != nil {
		return fmt.Errorf("gate control: invalid record debt batch: %w", err)
	}
	if debt.Batch.Key != "gate/final/"+debt.Preparation.String() {
		return errors.New("gate control: record debt batch is not bound to its preparation")
	}
	matching := 0
	for _, content := range debt.Batch.Contents {
		if content.Descriptor.MediaType != runrecord.GateLifecycleMediaType || content.Descriptor.Schema != runrecord.GateLifecycleSchema {
			continue
		}
		lifecycle, err := runrecord.ParseGateLifecycle(content.Data)
		if err != nil {
			return err
		}
		if lifecycle.State == runrecord.GateFinalized && lifecycle.Preparation != nil && *lifecycle.Preparation == debt.Preparation {
			matching++
		}
	}
	if matching != 1 {
		return fmt.Errorf("gate control: record debt has %d matching finalizations, want 1", matching)
	}
	return nil
}

type WatchdogStatus struct {
	Version   uint16                       `json:"version"`
	State     runrecord.GateHeartbeatState `json:"state"`
	Heartbeat *runrecord.GateHeartbeat     `json:"heartbeat,omitempty"`
}

func (store Store) Watchdog(now time.Time, staleAfter time.Duration) (WatchdogStatus, error) {
	path := filepath.Join(store.root, filepath.FromSlash(HeartbeatFile))
	raw, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return WatchdogStatus{Version: runrecord.GateHeartbeatVersion, State: runrecord.HeartbeatAbsent}, nil
	}
	if err != nil {
		return WatchdogStatus{}, err
	}
	var heartbeat runrecord.GateHeartbeat
	if err := strictjson.DecodeBytes(raw, &heartbeat); err != nil {
		return WatchdogStatus{}, err
	}
	if err := heartbeat.Validate(); err != nil {
		return WatchdogStatus{}, err
	}
	return WatchdogStatus{
		Version: heartbeat.Version, State: heartbeat.Watchdog(now, staleAfter), Heartbeat: &heartbeat,
	}, nil
}

func (store Store) WriteHeartbeat(heartbeat runrecord.GateHeartbeat) error {
	if err := heartbeat.Validate(); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(heartbeat, "", "  ")
	if err != nil {
		return err
	}
	path := filepath.Join(store.root, filepath.FromSlash(HeartbeatFile))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, append(raw, '\n'), 0o644)
}

func (store Store) StartHeartbeat(heartbeat runrecord.GateHeartbeat, interval time.Duration, now func() time.Time) (func(), error) {
	if interval <= 0 || now == nil {
		return nil, errors.New("gate control: heartbeat interval and clock are required")
	}
	write := func() error {
		heartbeat.Updated = now().UTC()
		return store.WriteHeartbeat(heartbeat)
	}
	if err := write(); err != nil {
		return nil, err
	}
	done := make(chan struct{})
	finished := make(chan struct{})
	var once sync.Once
	go func() {
		defer close(finished)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				_ = write()
			case <-done:
				return
			}
		}
	}()
	return func() {
		once.Do(func() {
			close(done)
			<-finished
		})
	}, nil
}
