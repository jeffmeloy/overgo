//go:build windows

package driver

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// cudaUUIDBytes is sizeof(CUuuid.bytes), the CUDA Driver API's 16-octet UUID.
const cudaUUIDBytes = 16

func (l *Library) deviceUUID(device Device) (string, error) {
	var uuid [cudaUUIDBytes]byte
	var pinned runtime.Pinner
	pinned.Pin(&uuid)
	defer pinned.Unpin()
	result, _, _ := l.cuDeviceGetUUID.Call(uintptr(unsafe.Pointer(&uuid[0])), uintptr(device))
	if err := l.result("cuDeviceGetUuid_v2", result); err != nil {
		return "", err
	}
	if uuid == [cudaUUIDBytes]byte{} {
		return "", errors.New("CUDA physical device UUID is empty")
	}
	return fmt.Sprintf("GPU-%x-%x-%x-%x-%x", uuid[:4], uuid[4:6], uuid[6:8], uuid[8:10], uuid[10:]), nil
}
