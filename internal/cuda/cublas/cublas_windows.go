//go:build windows

package cublas

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"unsafe"

	"llamacpp2go/internal/cuda/driver"
)

// Library is a dynamically loaded cuBLAS library.
type Library struct {
	dll *syscall.DLL

	create    *syscall.Proc
	destroy   *syscall.Proc
	setStream *syscall.Proc
	sgemm     *syscall.Proc
}

func Open() (*Library, error) {
	candidates := []string{"cublas64_12.dll"}
	if cudaPath := os.Getenv("CUDA_PATH"); cudaPath != "" {
		candidates = append(candidates, filepath.Join(cudaPath, "bin", "cublas64_12.dll"))
	}
	var dll *syscall.DLL
	var loadErr error
	for _, candidate := range candidates {
		dll, loadErr = syscall.LoadDLL(candidate)
		if loadErr == nil {
			break
		}
	}
	if dll == nil {
		return nil, errors.New("load cublas64_12.dll: " + loadErr.Error())
	}
	lib := &Library{dll: dll}
	required := []struct {
		name string
		dst  **syscall.Proc
	}{
		{"cublasCreate_v2", &lib.create},
		{"cublasDestroy_v2", &lib.destroy},
		{"cublasSetStream_v2", &lib.setStream},
		{"cublasSgemm_v2", &lib.sgemm},
	}
	for _, item := range required {
		proc, err := dll.FindProc(item.name)
		if err != nil {
			_ = dll.Release()
			return nil, errors.New("resolve " + item.name + ": " + err.Error())
		}
		*item.dst = proc
	}
	return lib, nil
}

func (l *Library) Close() error {
	if l == nil || l.dll == nil {
		return nil
	}
	err := l.dll.Release()
	l.dll = nil
	return err
}

func (l *Library) Create() (Handle, error) {
	var handle Handle
	status, _, _ := l.create.Call(uintptr(unsafe.Pointer(&handle)))
	if err := result("cublasCreate_v2", status); err != nil {
		return 0, err
	}
	return handle, nil
}

func (l *Library) Destroy(handle Handle) error {
	if handle == 0 {
		return nil
	}
	status, _, _ := l.destroy.Call(uintptr(handle))
	return result("cublasDestroy_v2", status)
}

func (l *Library) SetStream(handle Handle, stream driver.Stream) error {
	status, _, _ := l.setStream.Call(uintptr(handle), uintptr(stream))
	return result("cublasSetStream_v2", status)
}

// SGEMM performs column-major C = alpha*op(A)*op(B) + beta*C.
func (l *Library) SGEMM(
	handle Handle,
	operationA Operation,
	operationB Operation,
	m, n, k int32,
	alpha float32,
	a driver.DevicePtr,
	leadingA int32,
	b driver.DevicePtr,
	leadingB int32,
	beta float32,
	c driver.DevicePtr,
	leadingC int32,
) error {
	if m <= 0 || n <= 0 || k <= 0 {
		return errors.New("cublasSgemm_v2: matrix dimensions must be positive")
	}
	status, _, _ := l.sgemm.Call(
		uintptr(handle),
		uintptr(operationA),
		uintptr(operationB),
		uintptr(m),
		uintptr(n),
		uintptr(k),
		uintptr(unsafe.Pointer(&alpha)),
		uintptr(a),
		uintptr(leadingA),
		uintptr(b),
		uintptr(leadingB),
		uintptr(unsafe.Pointer(&beta)),
		uintptr(c),
		uintptr(leadingC),
	)
	runtime.KeepAlive(alpha)
	runtime.KeepAlive(beta)
	return result("cublasSgemm_v2", status)
}

func result(operation string, status uintptr) error {
	if int32(status) == 0 {
		return nil
	}
	return &StatusError{Operation: operation, Status: int32(status)}
}
