package executor

import (
	"cmp"
	"errors"
	"slices"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

const (
	// moePrefetchTokens is the fewest tokens a routed prompt node copies the next routed node's host experts ahead for:
	// Strata streams every non-resident expert of every layer from chunks of 1,024 tokens (src/prefill/prefill.cpp,
	// stream_all_min), where nearly every expert is routed, so the copy engine works through the attention between two
	// routed layers instead of waiting for each layer's routes. Flash-Next's 16K read waited 382 ms on its copies, each
	// started once its layer's routes were read, where Strata waits 96 ms.
	moePrefetchTokens = uint32(1024)
	// moePrefetchHalfBytes is each of the two halves a prefetch copies into in turn, the next node's host experts in id
	// order until it is full, the rest copied after the node's routes as before: a Flash-Next layer of an 8,192-token
	// chunk copied about 120 MB of host experts. It trades the experts copied ahead against the card memory the halves
	// take from the experts during a read.
	moePrefetchHalfBytes = uint64(192 << 20)
	// moePrefetchHalves: the next node's half fills while the node before reads the other.
	moePrefetchHalves = 2
	// moePrefetchBytes is what a run whose routed prompt nodes prefetch takes past its arena's plan.
	moePrefetchBytes = moePrefetchHalves * moePrefetchHalfBytes
)

// moePrefetch copies the next routed prompt node's experts in host memory into a half past the prompt arena's plan
// while the node before it computes, as Strata's prompt path streams them (prefill-16k/copy-waits).
//   - nodes are the run's routed prompt nodes in launch order, base the halves, zero where the run has none
//   - node is the node the last prefetch copied for, sources its experts' copies a bank and expert apart, zero where not
//     copied, and copied the copies' mark, cleared once the node's products wait for it
//   - read is each half's last readers' mark, which the next copies into it wait for
//   - routed is the slots each expert of the routed prompt node at each place in that order took at its last run, which
//     picks the experts copied ahead: those the copies after the routes would take, the CPU keeping its few-slot ones
//   - wants holds whether the routed prompt nodes of the last run that had them held experts in host memory, which
//     the halves are taken for, so a model whose experts all lie on the card gives none of its card to them
//   - experts counts the experts copied ahead since the executor began
type moePrefetch struct {
	nodes   []*tensor.Tensor
	base    driver.DevicePtr
	node    *tensor.Tensor
	sources []driver.DevicePtr
	hosted  [][]byte
	copied  driver.Event
	read    [moePrefetchHalves]driver.Event
	routed  [][]uint32
	order   []int
	wants   bool
	experts uint64
}

// observe keeps the slots each of node's experts took, from the offsets its routes set.
func (p *moePrefetch) observe(node *tensor.Tensor, offsets []uint32) {
	at := slices.Index(p.nodes, node)
	if at < 0 {
		return
	}
	for len(p.routed) <= at {
		p.routed = append(p.routed, nil)
	}
	slots := p.routed[at][:0]
	for expert := range len(offsets) - 1 {
		slots = append(slots, offsets[expert+1]-offsets[expert])
	}
	p.routed[at] = slots
}

// begin starts a run over nodes, the halves at the base its arena set, zero for none.
func (p *moePrefetch) begin(nodes []*tensor.Tensor) {
	p.nodes, p.node = nodes, nil
}

// prefetchArena is the prompt arena a run of arena bytes takes with its routed prompt nodes' halves past them, where
// the last such run found experts in host memory.
func (c *CompiledGraph) prefetchArena(arena uint64, prefetch *moePrefetch) uint64 {
	if c.prefetchBytes == 0 || prefetch == nil || !prefetch.wants {
		return arena
	}
	return moeAlign(arena) + c.prefetchBytes
}

// end lets the run's last copies land, so the halves go back to the experts with nothing in flight, and releases the
// marks no stream waited for.
func (p *moePrefetch) end(state *device.State, copies driver.Stream) error {
	var errs []error
	if p.copied != 0 {
		errs = append(errs, state.Driver.StreamSynchronize(copies), state.Driver.ReleaseEvent(p.copied))
		p.copied = 0
	}
	for half, mark := range p.read {
		if mark != 0 {
			errs = append(errs, state.Driver.ReleaseEvent(mark))
			p.read[half] = 0
		}
	}
	p.nodes, p.base, p.node = nil, 0, nil
	return errors.Join(errs...)
}

// take points node's experts the node before copied at their copies, which the compute stream waits for, and answers
// whether it did: a copied expert reads as a card expert does, so neither a batch nor the CPU takes it.
func (p *moePrefetch) take(state *device.State, node *tensor.Tensor, sources []driver.DevicePtr, hosted [][]byte) (bool, error) {
	if p.node != node || len(p.sources) != len(sources) {
		return false, nil
	}
	for at, place := range p.sources {
		if place != 0 {
			sources[at], hosted[at] = place, nil
		}
	}
	if p.copied != 0 {
		mark := p.copied
		p.copied = 0
		if err := state.Driver.WaitEvent(state.Stream, mark); err != nil {
			return false, err
		}
	}
	return true, nil
}

// ahead copies the host experts of the routed node after node into its half on copies, once the products queued before
// it, which read that half last, are done: the node's own half is marked read here, after its products. A node of
// fewer than moePrefetchTokens tokens, or of several groups, copies nothing ahead.
func (p *moePrefetch) ahead(state *device.State, copies driver.Stream, node *tensor.Tensor, count uint32, took bool, residency ExpertBanks) error {
	at := slices.Index(p.nodes, node)
	if residency == nil || at < 0 || count < moePrefetchTokens || uint64(count) != node.Shape.Dims[1] {
		return nil
	}
	if took {
		mark, err := state.Driver.RecordEvent(state.Stream)
		if err != nil {
			return err
		}
		p.read[at%moePrefetchHalves] = mark
	}
	if at+1 >= len(p.nodes) {
		return nil
	}
	next := p.nodes[at+1]
	// A prefetch no node took leaves its mark: its copies are ordered before these on copies.
	if p.copied != 0 {
		if err := state.Driver.ReleaseEvent(p.copied); err != nil {
			return err
		}
		p.copied = 0
	}
	g, _ := nativeMoEMatrix(next)
	a := next.Attrs.(tensor.MoEAttributes)
	banks := next.Inputs[moeFirstBankInput : moeFirstBankInput+int(moeBanks)]
	if a.FusedGateUp {
		banks = banks[:moeBanks-1]
	}
	experts := int(g.experts)
	p.sources = slices.Grow(p.sources[:0], len(banks)*experts)[:len(banks)*experts]
	p.hosted = slices.Grow(p.hosted[:0], len(banks)*experts)[:len(banks)*experts]
	clear(p.hosted)
	for index, bank := range banks {
		if err := residency.ExpertSources(state, bank, p.sources[index*experts:(index+1)*experts], p.hosted[index*experts:(index+1)*experts]); err != nil {
			return err
		}
	}
	clear(p.sources)
	p.wants = slices.ContainsFunc(p.hosted, func(host []byte) bool { return len(host) > 0 })
	if p.base == 0 {
		return nil
	}
	half := (at + 1) % moePrefetchHalves
	if mark := p.read[half]; mark != 0 {
		p.read[half] = 0
		if err := state.Driver.WaitEvent(copies, mark); err != nil {
			return err
		}
	}
	// The experts its place routed to more slots than the CPU takes last time, the most routed first.
	var routed []uint32
	if at+1 < len(p.routed) {
		routed = p.routed[at+1]
	}
	p.order = p.order[:0]
	for expert := range min(experts, len(routed)) {
		if routed[expert] > hostExpertSlotsMost {
			p.order = append(p.order, expert)
		}
	}
	slices.SortStableFunc(p.order, func(x, y int) int { return cmp.Compare(routed[y], routed[x]) })
	start := p.base + driver.DevicePtr(uint64(half)*moePrefetchHalfBytes)
	used := uint64(0)
	for _, expert := range p.order {
		var bytes uint64
		for index := range banks {
			bytes += moeAlign(uint64(len(p.hosted[index*experts+expert])))
		}
		if bytes == 0 || used+bytes > moePrefetchHalfBytes {
			continue
		}
		for index := range banks {
			if host := p.hosted[index*experts+expert]; len(host) > 0 {
				place := start + driver.DevicePtr(used)
				if err := state.Driver.MemcpyHtoDAsync(place, host, copies); err != nil {
					return err
				}
				p.sources[index*experts+expert] = place
				used += moeAlign(uint64(len(host)))
			}
		}
		p.experts++
	}
	mark, err := state.Driver.RecordEvent(copies)
	if err != nil {
		return err
	}
	p.node, p.copied = next, mark
	// Submitted at once: under Windows' display driver model the copies otherwise wait in the driver's buffer.
	_, err = state.Driver.StreamQuery(copies)
	return err
}
