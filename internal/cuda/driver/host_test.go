package driver

import (
	"encoding/binary"
	"testing"
)

func TestHostBytes(t *testing.T) {
	values := []uint32{1, 2}
	data := Bytes(values)
	if len(data) != 8 || binary.LittleEndian.Uint32(data[4:]) != 2 {
		t.Fatalf("bytes = %v", data)
	}
	binary.LittleEndian.PutUint32(data, 3)
	if values[0] != 3 {
		t.Fatalf("aliased value = %d", values[0])
	}
}
