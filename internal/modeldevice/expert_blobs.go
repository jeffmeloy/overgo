package modeldevice

import (
	"slices"
	"strings"

	"overgo/internal/cuda/driver"
	"overgo/internal/hostexpert"
	"overgo/internal/tensor/dtype"
)

// ExpertBlobPart is one bank's part of a layer's expert blob: its bank, its GGUF type, its offset in the blob and its
// bytes.
type ExpertBlobPart struct {
	Name          string
	Type          dtype.Type
	Offset, Bytes uint64
}

// ExpertBlobLayer is a routed layer's experts as Strata's decode round reads them (its verify window's VerifyHits and
// its pool's expert source): each expert's blob address - its card slot, or its pinned host copy's device alias - and
// whether the card holds it, a host expert's pinned bytes by part, and the blob's layout. Its answers change only
// between runs, when the residency advances.
type ExpertBlobLayer struct{ blob *expertBlob }

// ExpertBlobLayer answers the routed layer's experts, or false where no blob names the layer.
func (w *DeviceWeights) ExpertBlobLayer(layer int) (ExpertBlobLayer, bool) {
	if w == nil || w.experts == nil {
		return ExpertBlobLayer{}, false
	}
	for _, blob := range w.experts.blobs {
		if int(blob.layer) == layer {
			return ExpertBlobLayer{blob}, true
		}
	}
	return ExpertBlobLayer{}, false
}

// ExpertResidentTable answers the card's residency table (Strata's d_res: [layer][expert] int32, 0 on the card and -1
// in host memory) and its experts a layer, zero where the layers hold different expert counts.
func (w *DeviceWeights) ExpertResidentTable() (driver.DevicePtr, int) {
	if w == nil || w.experts == nil {
		return 0, 0
	}
	return w.experts.resident, w.experts.residentExperts
}

// Experts is the layer's expert count.
func (l ExpertBlobLayer) Experts() int { return len(l.blob.head().pointers) }

// Bytes is an expert's blob.
func (l ExpertBlobLayer) Bytes() uint64 { return l.blob.bytes }

// Parts is the blob's layout, its banks in order.
func (l ExpertBlobLayer) Parts() []ExpertBlobPart {
	parts := make([]ExpertBlobPart, len(l.blob.banks))
	for index, bank := range l.blob.banks {
		parts[index] = ExpertBlobPart{Name: bank.info.Name, Type: bank.info.Type, Offset: bank.offset, Bytes: bank.bytes}
	}
	return parts
}

// Blob answers an expert's blob address and whether the card holds it.
func (l ExpertBlobLayer) Blob(expert int) (driver.DevicePtr, bool) {
	head := l.blob.head()
	return head.pointers[expert], head.entries[expert] != nil
}

// Host appends a host expert's pinned bytes of each part, in the blob's order, to parts; nothing for a card expert.
func (l ExpertBlobLayer) Host(expert int, parts [][]byte) [][]byte {
	for _, bank := range l.blob.banks {
		host := bank.hosted[expert]
		if host == nil {
			return parts
		}
		parts = append(parts, host[:bank.bytes])
	}
	return parts
}

// gatedExpertBanks are a gated expert layer's bank names by their suffix, gate, up and down, which a host expert's
// sample reads.
var gatedExpertBanks = [...]string{"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}

// HostExpertSample answers one routed expert in host memory as the CPU computes one: its gate, up and down rows, their
// formats, and the layer's hidden and intermediate widths (the gate bank's rows and columns); false where no routed
// layer holds its gate, up and down as banks of their own with an expert in host memory. A load measures the
// machine's costs on it (executor.MeasureHostExpertCosts).
func (w *DeviceWeights) HostExpertSample() (hostexpert.Expert, hostexpert.Formats, int, int, bool) {
	if w == nil || w.experts == nil {
		return hostexpert.Expert{}, hostexpert.Formats{}, 0, 0, false
	}
	for _, blob := range w.experts.blobs {
		var banks [len(gatedExpertBanks)]*expertBank
		for _, bank := range blob.banks {
			for index, suffix := range gatedExpertBanks {
				if strings.HasSuffix(bank.info.Name, suffix) {
					banks[index] = bank
				}
			}
		}
		if slices.Contains(banks[:], nil) || len(blob.banks) != len(gatedExpertBanks) {
			continue
		}
		gate, up, down := banks[0], banks[1], banks[len(banks)-1]
		for expert, host := range gate.hosted {
			if host == nil || up.hosted[expert] == nil || down.hosted[expert] == nil {
				continue
			}
			formats := hostexpert.Formats{Gate: gate.info.Type, Up: up.info.Type, Down: down.info.Type}
			sample := hostexpert.Expert{Gate: host[:gate.bytes], Up: up.hosted[expert][:up.bytes], Down: down.hosted[expert][:down.bytes]}
			return sample, formats, int(gate.info.Shape[0]), int(gate.info.Shape[1]), true
		}
	}
	return hostexpert.Expert{}, hostexpert.Formats{}, 0, 0, false
}
