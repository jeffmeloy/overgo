package modeldevice

import (
	"overgo/internal/cuda/driver"
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
