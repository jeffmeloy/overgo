package driver

import (
	"os"
	"path/filepath"
	"strconv"
)

// IdentityEnvironment names the environment variable a lane that claims the device sets to LibraryIdentity and
// cudatest.Require reads, so go test's cache keys every CUDA test on the driver it ran on: go test keys no file
// outside the module, and the driver library lies in the system directory.
const IdentityEnvironment = "OVERGO_CUDA_DRIVER"

// libraryName is the CUDA driver library Open loads, which Windows finds in the systemDirectory of the root
// systemRootEnvironment names; absentLibrary is the identity of a system that holds none.
const (
	libraryName           = "nvcuda.dll"
	systemDirectory       = "System32"
	systemRootEnvironment = "SystemRoot"
	absentLibrary         = "absent"
)

// identityBase is the base LibraryIdentity writes its numbers in.
const identityBase = 10

// LibraryIdentity is the identity of the CUDA driver library Open loads, its size and modification time where the
// system directory holds it, which a driver update changes; absentLibrary when it holds none.
func LibraryIdentity() string {
	info, err := os.Stat(filepath.Join(os.Getenv(systemRootEnvironment), systemDirectory, libraryName))
	if err != nil {
		return absentLibrary
	}
	return strconv.FormatInt(info.Size(), identityBase) + ":" + strconv.FormatInt(info.ModTime().UnixNano(), identityBase)
}
