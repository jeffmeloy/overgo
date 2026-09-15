//overgo:runtime-inputs caller

package processcontrol

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"

	"overgo/internal/jsonfile"
	"overgo/internal/processlock"
)

// CampaignEnvironment carries an unpredictable session identity, not a claim
// that supervision exists. Admission also requires the live owner and OS lock.
const CampaignEnvironment = "OVERGO_CAMPAIGN_SESSION"

const campaignLocatorPath = "tmp/loop_supervisor.json"
const campaignLockPath = "tmp/loop_supervisor.lock"
const campaignFileMode = 0o600
const campaignDirectoryMode = 0o700
const campaignNonceBytes = 32

type campaignLocator struct {
	Root    string `json:"root"`
	Worker  string `json:"worker"`
	PID     int    `json:"pid"`
	Session string `json:"session"`
	Address string `json:"address"`
}

// Campaign owns one worktree's continuous worker lifetime. The locator remains
// after exit so an abandoned campaign cannot silently become manual execution.
type Campaign struct {
	lock   *processlock.Lock
	owner  campaignLocator
	server *http.Server
	done   chan struct{}
	stops  chan struct{}
}

// BeginCampaign acquires the process-lifetime owner before any dispatch.
func BeginCampaign(root, worker string) (*Campaign, error) {
	if worker == "" {
		return nil, errors.New("campaign: a stable worker identity is required")
	}
	root, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Join(root, "tmp"), campaignDirectoryMode); err != nil {
		return nil, err
	}
	lock, err := processlock.Acquire(filepath.Join(root, campaignLockPath), campaignFileMode)
	if err != nil {
		return nil, fmt.Errorf("campaign: another supervisor owns this worktree: %w", err)
	}
	nonce := make([]byte, campaignNonceBytes)
	if _, err := rand.Read(nonce); err != nil {
		_ = lock.Close()
		return nil, err
	}
	owner := campaignLocator{Root: root, Worker: worker, PID: os.Getpid(), Session: hex.EncodeToString(nonce)}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		_ = lock.Close()
		return nil, err
	}
	owner.Address = listener.Addr().String()
	if err := jsonfile.Write(filepath.Join(root, campaignLocatorPath), owner, campaignFileMode); err != nil {
		_ = listener.Close()
		_ = lock.Close()
		return nil, err
	}
	campaign := &Campaign{lock: lock, owner: owner, stops: make(chan struct{}, 1), done: make(chan struct{})}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /stop", func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("Authorization") != "Bearer "+owner.Session {
			response.WriteHeader(http.StatusForbidden)
			return
		}
		select {
		case campaign.stops <- struct{}{}:
		default:
		}
		response.WriteHeader(http.StatusAccepted)
	})
	campaign.server = &http.Server{Handler: mux}
	go func() { defer close(campaign.done); _ = campaign.server.Serve(listener) }()
	return campaign, nil
}

// Environment is inherited only by this owner's tools and workers.
func (campaign *Campaign) Environment() string {
	return CampaignEnvironment + "=" + campaign.owner.Session
}

// Close releases live authority; the retained locator still requires a restart.
func (campaign *Campaign) Close() error {
	err := campaign.server.Close()
	<-campaign.done
	return errors.Join(err, campaign.lock.Close())
}

// Stops receives a wakeup, not stop authority. The loop rereads the durable
// scoped stop before cancelling its worker.
func (campaign *Campaign) Stops() <-chan struct{} { return campaign.stops }

// NotifyCampaignStop wakes a live supervisor after a durable stop was recorded.
// It does not create a stop or infer one from the request itself.
func NotifyCampaignStop(ctx context.Context, root string) error {
	var owner campaignLocator
	if err := jsonfile.DecodeStrict(filepath.Join(root, campaignLocatorPath), &owner); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	if !ProcessAlive(owner.PID) {
		return nil
	}
	host, _, err := net.SplitHostPort(owner.Address)
	if err != nil || host != "127.0.0.1" {
		return errors.New("campaign: invalid local stop endpoint")
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://"+owner.Address+"/stop", nil)
	if err != nil {
		return err
	}
	request.Header.Set("Authorization", "Bearer "+owner.Session)
	request.Close = true
	transport := &http.Transport{}
	defer transport.CloseIdleConnections()
	response, err := (&http.Client{Transport: transport}).Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusAccepted {
		return fmt.Errorf("campaign: stop notification returned %s", response.Status)
	}
	return nil
}

// RequireCampaign refuses executable admission outside a configured campaign's
// live session. Unconfigured worktrees retain ordinary interactive operation.
func RequireCampaign(root, worker string) error {
	root, err := filepath.Abs(root)
	if err != nil {
		return err
	}
	var owner campaignLocator
	if err := jsonfile.DecodeStrict(filepath.Join(root, campaignLocatorPath), &owner); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			if _, configErr := os.Stat(filepath.Join(root, "docs/loop.json")); errors.Is(configErr, os.ErrNotExist) {
				return nil
			}
		}
		return fmt.Errorf("campaign: start cmd/loop before executable dispatch: %w", err)
	}
	if owner.Root != root || owner.Worker != worker || owner.Session == "" || owner.Session != os.Getenv(CampaignEnvironment) || !ProcessAlive(owner.PID) {
		return errors.New("campaign: executable dispatch requires this worktree's live cmd/loop session")
	}
	lock, err := processlock.Acquire(filepath.Join(root, campaignLockPath), campaignFileMode)
	if err == nil {
		_ = lock.Close()
		return errors.New("campaign: supervisor ownership ended; restart cmd/loop")
	}
	if !errors.Is(err, processlock.ErrBusy) {
		return err
	}
	return nil
}
