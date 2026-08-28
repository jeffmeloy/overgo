package modelswap

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"strconv"
	"time"

	"overgo/internal/processcontrol"
)

// healthPollInterval paces readiness probes against a starting child;
// model load dominates startup, so a coarse poll costs nothing.
const healthPollInterval = 250 * time.Millisecond

// ServerLauncher launches the overgo server binary as the child for a
// servable: one child at a time over the shared store, listening on a
// loopback port picked at launch.
type ServerLauncher struct {
	// Binary is the built overgo server executable.
	Binary string
	// Store is the OvergoDB root every child serves over.
	Store string
}

// Launch starts one child server for the servable and returns before
// readiness; Ready polls the child's health until it answers.
func (l ServerLauncher) Launch(ctx context.Context, servable Servable) (Process, error) {
	if l.Binary == "" || l.Store == "" {
		return nil, errors.New("model swap: launcher requires a server binary and a store")
	}
	if servable.Location == "" {
		return nil, fmt.Errorf("model swap: servable %q has no on-disk location", servable.Name)
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	port := listener.Addr().(*net.TCPAddr).Port
	if err := listener.Close(); err != nil {
		return nil, err
	}
	address := "127.0.0.1:" + strconv.Itoa(port)
	// The child self-reports the servable's name so the GUI's model pill
	// (and the proxy's self-identity short-circuit) track the swap.
	supervised, err := processcontrol.Start(ctx, processcontrol.Command{
		Path:   l.Binary,
		Args:   []string{"-listen", address, "-repo", l.Store, "-model-id", servable.Name, servable.Location},
		Stdout: os.Stderr,
		Stderr: os.Stderr,
	})
	if err != nil {
		return nil, err
	}
	exited := make(chan error, 1)
	go func() {
		_, waitErr := supervised.Wait(context.WithoutCancel(ctx))
		exited <- waitErr
		close(exited)
	}()
	return &serverProcess{supervised: supervised, exited: exited, url: "http://" + address}, nil
}

type serverProcess struct {
	supervised *processcontrol.Supervised
	// exited delivers the child's Wait result; a child that dies during
	// startup fails Ready immediately instead of polling forever.
	exited chan error
	url    string
}

// URL is the child's loopback base address.
func (p *serverProcess) URL() string { return p.url }

// Ready polls the child's health endpoint until it answers OK, the
// child exits, or the context ends -- model load time is the wait.
func (p *serverProcess) Ready(ctx context.Context) error {
	client := &http.Client{Timeout: healthPollInterval}
	for {
		response, err := client.Get(p.url + "/health")
		if err == nil {
			response.Body.Close()
			if response.StatusCode == http.StatusOK {
				return nil
			}
		}
		select {
		case exit := <-p.exited:
			return fmt.Errorf("child server exited before becoming ready: %v", exit)
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(healthPollInterval):
		}
	}
}

// Stop terminates the child process tree and reaps it.
func (p *serverProcess) Stop() error {
	if p.supervised == nil {
		return nil
	}
	if err := p.supervised.Terminate(); err != nil {
		return err
	}
	<-p.exited
	return nil
}
