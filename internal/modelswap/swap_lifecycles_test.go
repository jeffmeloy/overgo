package modelswap

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"math/rand/v2"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"overgo/internal/testcheck"
)

// A generated lifecycle's shape: its seeds, its steps, the idle timeout, the loads a child may take, and the share of
// lifecycles that close in their midst (one in swapCloseOdds of the steps a close may fall at lie inside it).
const (
	swapSeeds     = 2000
	swapSteps     = 40
	swapIdle      = 4 * time.Second
	swapCloseOdds = 2
	swapTraceTail = 8
)

var swapLoads = []time.Duration{0, time.Second, 7 * time.Second, 40 * time.Second}

// swapChild is a fake child server: it lives until a Stop that is not refused, or until it exits by itself.
type swapChild struct {
	world        *swapWorld
	id           int
	name         string
	alive        bool
	exitedItself bool
	ended        chan struct{}
	// end closes ended, once.
	end func()
}

func (c *swapChild) URL() string             { return "http://child-" + strconv.Itoa(c.id) }
func (c *swapChild) Exited() <-chan struct{} { return c.ended }

// Ready waits out the world's load, and answers as a child server's does: a child that ended during it is refused.
func (c *swapChild) Ready(ctx context.Context) error {
	c.world.mu.Lock()
	load := c.world.load
	c.world.mu.Unlock()
	if load > 0 {
		select {
		case <-time.After(load):
		case <-c.ended:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	c.world.mu.Lock()
	defer c.world.mu.Unlock()
	if !c.alive {
		return errors.New("the child exited before serving")
	}
	if c.world.failReady {
		c.world.failReady = false
		return errors.New("injected: the child died before serving")
	}
	return nil
}

func (c *swapChild) Stop() error {
	c.world.mu.Lock()
	defer c.world.mu.Unlock()
	if c.world.failStop {
		c.world.failStop = false
		return errors.New("injected: the child could not be terminated")
	}
	c.alive = false
	c.end()
	return nil
}

// swapWorld launches fake children, and refuses the next launch, ready or stop when told to.
type swapWorld struct {
	mu                              sync.Mutex
	children                        []*swapChild
	failLaunch, failReady, failStop bool
	load                            time.Duration
}

func (w *swapWorld) Launch(_ context.Context, servable Servable) (Process, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.failLaunch {
		w.failLaunch = false
		return nil, errors.New("injected: launch refused")
	}
	ended := make(chan struct{})
	child := &swapChild{world: w, id: len(w.children), name: servable.Name, alive: true, ended: ended, end: sync.OnceFunc(func() { close(ended) })}
	w.children = append(w.children, child)
	return child, nil
}

func (w *swapWorld) live() []*swapChild {
	w.mu.Lock()
	defer w.mu.Unlock()
	return slices.DeleteFunc(slices.Clone(w.children), func(child *swapChild) bool { return !child.alive })
}

func (w *swapWorld) byURL(url string) *swapChild {
	w.mu.Lock()
	defer w.mu.Unlock()
	for _, child := range w.children {
		if child.URL() == url {
			return child
		}
	}
	return nil
}

// swapRequest is one acquire in flight, and what it returned.
type swapRequest struct {
	name         string
	cancel       context.CancelFunc
	done         bool
	lease        *Lease
	deadAtReturn bool
	granted      time.Time
}

// swapLifecycle is one seed's run: the supervisor, its world, the acquires in flight and the leases held.
type swapLifecycle struct {
	rng        *rand.Rand
	world      *swapWorld
	supervisor *Supervisor
	pending    []*swapRequest
	leases     []*Lease
	closeErr   error
	trace      []string
	failure    string
}

func (l *swapLifecycle) fail(format string, arguments ...any) {
	if l.failure == "" {
		l.failure = fmt.Sprintf(format, arguments...)
	}
}

func (l *swapLifecycle) acquire() {
	name := []string{"a", "b", "c"}[l.rng.IntN(3)]
	l.trace = append(l.trace, "acquire "+name)
	ctx, cancel := context.WithCancel(context.Background())
	request := &swapRequest{name: name, cancel: cancel}
	l.pending = append(l.pending, request)
	go func() {
		lease, _ := l.supervisor.Acquire(ctx, Servable{Name: name})
		if lease != nil {
			child := l.world.byURL(lease.Target)
			l.world.mu.Lock()
			request.deadAtReturn = child == nil || !child.alive
			l.world.mu.Unlock()
			request.granted = time.Now()
		}
		request.lease, request.done = lease, true
	}()
}

func (l *swapLifecycle) release() {
	if len(l.leases) != 0 {
		index := l.rng.IntN(len(l.leases))
		l.trace = append(l.trace, "release")
		l.leases[index].Release()
		l.leases = slices.Delete(l.leases, index, index+1)
	}
}

func (l *swapLifecycle) holdOpen() {
	if len(l.leases) != 0 {
		l.trace = append(l.trace, "hold open")
		l.leases[l.rng.IntN(len(l.leases))].HoldOpen()
	}
}

func (l *swapLifecycle) exitItself() {
	if alive := l.world.live(); len(alive) != 0 {
		l.trace = append(l.trace, "a child exits by itself")
		l.world.mu.Lock()
		alive[0].alive, alive[0].exitedItself = false, true
		alive[0].end()
		l.world.mu.Unlock()
	}
}

func (l *swapLifecycle) refuseNext() {
	which := l.rng.IntN(3)
	l.trace = append(l.trace, []string{"the next launch fails", "the next ready fails", "the next stop fails"}[which])
	l.world.mu.Lock()
	defer l.world.mu.Unlock()
	*[]*bool{&l.world.failLaunch, &l.world.failReady, &l.world.failStop}[which] = true
}

func (l *swapLifecycle) loadTakes() {
	load := swapLoads[l.rng.IntN(len(swapLoads))]
	l.trace = append(l.trace, "loads take "+load.String())
	l.world.mu.Lock()
	defer l.world.mu.Unlock()
	l.world.load = load
}

func (l *swapLifecycle) cancelOne() {
	if len(l.pending) != 0 {
		l.trace = append(l.trace, "cancel an acquire")
		l.pending[l.rng.IntN(len(l.pending))].cancel()
	}
}

func (l *swapLifecycle) wait() {
	wait := []time.Duration{swapIdle / 2, swapIdle * 2, drainGrace + time.Second}[l.rng.IntN(3)]
	l.trace = append(l.trace, "wait "+wait.String())
	time.Sleep(wait)
}

// swapActions are a lifecycle's steps, each drawn as often as it is listed.
var swapActions = []func(*swapLifecycle){
	(*swapLifecycle).acquire, (*swapLifecycle).acquire, (*swapLifecycle).acquire, (*swapLifecycle).acquire,
	(*swapLifecycle).release, (*swapLifecycle).release, (*swapLifecycle).release,
	(*swapLifecycle).holdOpen, (*swapLifecycle).exitItself, (*swapLifecycle).refuseNext, (*swapLifecycle).loadTakes,
	(*swapLifecycle).cancelOne, (*swapLifecycle).wait, (*swapLifecycle).wait, (*swapLifecycle).wait, (*swapLifecycle).wait,
}

// settle waits for the bubble to block, takes the leases acquires returned, and holds each to a child that was alive
// at its return and that no stop cut short of drainGrace.
func (l *swapLifecycle) settle() {
	synctest.Wait()
	for _, request := range l.pending {
		if !request.done || request.lease == nil {
			continue
		}
		l.leases = append(l.leases, request.lease)
		child := l.world.byURL(request.lease.Target)
		l.world.mu.Lock()
		stopped := child == nil || !child.alive && !child.exitedItself
		l.world.mu.Unlock()
		if request.deadAtReturn {
			l.fail("an acquire of %s was given child %s, which was dead at its return", request.name, request.lease.Target)
		} else if stopped && time.Since(request.granted) < drainGrace {
			l.fail("an acquire of %s was severed %v after its grant", request.name, time.Since(request.granted).Round(time.Second))
		}
		request.lease = nil
	}
	l.pending = slices.DeleteFunc(l.pending, func(request *swapRequest) bool { return request.done })
}

// check holds the supervisor, after every step, to one live child at most, and to naming a model only while a child
// lives or a load is pending.
func (l *swapLifecycle) check() {
	alive := l.world.live()
	if len(alive) > 1 {
		l.fail("%d children alive at once on an exclusive device", len(alive))
	}
	if running, ok := l.supervisor.Status(); ok && len(alive) == 0 && len(l.pending) == 0 {
		l.fail("Status names %s and no child lives", running.Name)
	}
}

// swapRun drives seed's lifecycle in a bubble of its own and says what it found, with the steps before it.
func swapRun(t *testing.T, seed uint64) (string, []string) {
	var lifecycle *swapLifecycle
	synctest.Test(t, func(t *testing.T) {
		world := &swapWorld{}
		supervisor, err := New(world, swapIdle)
		testcheck.NoError(t, err)
		rng := rand.New(rand.NewPCG(seed, seed))
		lifecycle = &swapLifecycle{rng: rng, world: world, supervisor: supervisor}
		closeAt := rng.IntN(swapSteps * swapCloseOdds)
		for step := range swapSteps {
			if step == closeAt {
				lifecycle.trace = append(lifecycle.trace, "close")
				lifecycle.closeErr = supervisor.Close()
			} else {
				swapActions[rng.IntN(len(swapActions))](lifecycle)
			}
			lifecycle.settle()
			lifecycle.check()
			if lifecycle.failure != "" {
				break
			}
		}
		// The end: every acquire cancelled, every lease released, the supervisor closed, and the last load over.
		for _, request := range lifecycle.pending {
			request.cancel()
		}
		lifecycle.settle()
		for _, lease := range lifecycle.leases {
			lease.Release()
		}
		lifecycle.closeErr = supervisor.Close()
		time.Sleep(slices.Max(swapLoads) + drainGrace)
		synctest.Wait()
		for _, child := range world.live() {
			if lifecycle.closeErr == nil || !strings.Contains(lifecycle.closeErr.Error(), strconv.Quote(child.name)) {
				lifecycle.fail("after Close child %d (%s) lives, and Close said %v", child.id, child.name, lifecycle.closeErr)
			}
		}
	})
	return lifecycle.failure, lifecycle.trace
}

// TestGeneratedSwapLifecycles drives swapSeeds generated lifecycles of acquires, releases, holds, cancels, children that
// exit by themselves, refused launches, readies and stops, loads of several lengths, waits and a close through the
// supervisor, and holds it to its promises: at most one live child, a model named only while a child lives or loads, a
// lease on a live child that lives drainGrace unless it exits by itself, and no child left after Close unless Close
// names it.
func TestGeneratedSwapLifecycles(t *testing.T) {
	t.Parallel()
	classes := map[string]int{}
	examples := map[string]string{}
	for seed := range uint64(swapSeeds) {
		failure, trace := swapRun(t, seed+1)
		if failure == "" {
			continue
		}
		class := strings.Map(func(r rune) rune {
			if r >= '0' && r <= '9' {
				return '#'
			}
			return r
		}, failure)
		classes[class]++
		if examples[class] == "" {
			examples[class] = fmt.Sprintf("seed %d after %s", seed+1, strings.Join(trace[max(0, len(trace)-swapTraceTail):], "; "))
		}
	}
	for _, class := range slices.Sorted(maps.Keys(classes)) {
		t.Errorf("%d of %d lifecycles: %s\n  %s", classes[class], swapSeeds, class, examples[class])
	}
}
