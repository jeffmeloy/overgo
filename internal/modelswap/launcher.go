package modelswap

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"strconv"

	"overgo/internal/processcontrol"
)

// ServerLauncher launches the overgo server binary as the child for a
// servable: one child at a time over the shared store, listening on a
// loopback port picked at launch.
type ServerLauncher struct {
	// Binary is the built overgo server executable.
	Binary string
	// Store is the OvergoDB root every child serves over.
	Store string
	// Dir is the working directory the child runs in; empty inherits the
	// launcher's, which must be the repository root the server reads its
	// policy documents from.
	Dir string
	// Workspaces is the operator's launch declaration; every child enables
	// the same workspaces, so a swap never loses a tab.
	Workspaces Workspaces
}

// Workspaces names the server workspaces a launch enables beyond serving.
type Workspaces struct {
	// Training enables the recipe-bound training workspace (-training).
	Training bool
	// ModelBuilder enables the corpus-derived model builder (-model-builder).
	ModelBuilder bool
}

// arguments renders the declaration as the server flags it enables.
func (w Workspaces) arguments() []string {
	var flags []string
	if w.Training {
		flags = append(flags, "-training")
	}
	if w.ModelBuilder {
		flags = append(flags, "-model-builder")
	}
	return flags
}

// Launch starts one child server for the servable and returns before
// readiness; Ready waits for the child's own listening announcement.
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
	arguments := append([]string{"-listen", address, "-repo", l.Store, "-model-id", servable.Name}, l.Workspaces.arguments()...)
	// The child logs its bound listener; the announcement is the readiness signal.
	announced := processcontrol.WatchLine(os.Stderr, processcontrol.ListeningAnnouncement+address)
	supervised, err := processcontrol.Start(ctx, processcontrol.Command{
		Path:   l.Binary,
		Args:   append(arguments, servable.Location),
		Dir:    l.Dir,
		Stdout: os.Stderr,
		Stderr: announced,
	})
	if err != nil {
		return nil, err
	}
	exited := make(chan error, 1)
	go func() {
		receipt, waitErr := supervised.Wait(context.WithoutCancel(ctx))
		if receipt.ExitCode == processcontrol.ResourceBusyExitCode && !receipt.TreeTerminated {
			waitErr = processcontrol.ErrResourceBusy
		} else if receipt.ExitCode == processcontrol.DeviceMemoryExitCode && !receipt.TreeTerminated {
			waitErr = processcontrol.ErrDeviceMemory
		} else {
			waitErr = cmp.Or(waitErr, fmt.Errorf("server exited with status %d", receipt.ExitCode))
		}
		exited <- waitErr
		close(exited)
	}()
	return &serverProcess{supervised: supervised, exited: exited, announced: announced.Line(), url: "http://" + address}, nil
}

type serverProcess struct {
	supervised *processcontrol.Supervised
	// exited delivers the child's Wait result; a child that dies during
	// startup fails Ready at once.
	exited chan error
	// announced delivers the child's listening line once its socket is bound.
	announced <-chan string
	url       string
}

// URL is the child's loopback base address.
func (p *serverProcess) URL() string { return p.url }

// Ready waits for the child's listening announcement, its exit, or the
// context's end, then confirms the bound listener answers health once --
// model load time is the wait, and the child itself reports its end.
func (p *serverProcess) Ready(ctx context.Context) error {
	select {
	case <-p.announced:
	case exit := <-p.exited:
		exit = cmp.Or(exit, errors.New("child exited without serving"))
		return fmt.Errorf("child server exited before becoming ready: %w", exit)
	case <-ctx.Done():
		return ctx.Err()
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, p.url+"/health", nil)
	if err != nil {
		return err
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return fmt.Errorf("child server announced its listener but did not answer health: %w", err)
	}
	response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("child server health answered status %d", response.StatusCode)
	}
	return nil
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
