package gpt2

// #cgo CFLAGS: -I${SRCDIR}/ggml.cpp/include/ -I${SRCDIR}/ggml.cpp/include/ggml/ -I${SRCDIR}/ggml.cpp/examples/ -I${SRCDIR}/ggml.cpp/src/
// #cgo CXXFLAGS: -I${SRCDIR}/ggml.cpp/include/ -I${SRCDIR}/ggml.cpp/include/ggml/ -I${SRCDIR}/ggml.cpp/examples/ -I${SRCDIR}/ggml.cpp/src/
// #cgo darwin LDFLAGS: -framework Accelerate
// #cgo darwin CXXFLAGS: -std=c++17
// #cgo LDFLAGS: -ltransformers -lm -lstdc++
// #include <falcon.h>
import "C"
import (
	"fmt"
	common "github.com/go-skynet/go-common"
	"strings"
	"unsafe"
)

type Falcon struct {
	state unsafe.Pointer
}

var FalconBackendInitializer common.BackendInitializer[Falcon] = common.BackendInitializer[Falcon]{
	DefaultInitializationOptions: common.InitializationOptions{},
	Constructor: func(modelPath string, initializationOptions common.InitializationOptions) (*Falcon, error) {
		state := C.falcon_allocate_state()
		cModelPath := C.CString(modelPath)
		result := C.falcon_bootstrap(cModelPath, state)
		if result != 0 {
			return nil, fmt.Errorf("failed loading model")
		}

		return &Falcon{state: state}, nil
	},
}

func (l Falcon) Name() string {
	return "falcon"
}

func (l Falcon) Close() error {
	C.falcon_free_model(l.state)
	return nil
}

func (l *Falcon) Predict(text string, opts ...common.PredictTextOptionSetter) (string, error) {
	return l.PredictWithOptions(text, *MergePredictOptionsWithDefaults(opts...))
}

func (l *Falcon) PredictWithOptions(text string, po common.PredictTextOptions) (string, error) {
	input := C.CString(text)
	if po.Tokens == 0 {
		po.Tokens = 99999999
	}
	out := make([]byte, po.Tokens)

	params := C.falcon_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.Temperature), C.int(po.Batch))
	ret := C.falcon_predict(params, l.state, (*C.char)(unsafe.Pointer(&out[0])))
	if ret != 0 {
		return "", fmt.Errorf("inference failed")
	}
	res := C.GoString((*C.char)(unsafe.Pointer(&out[0])))

	res = strings.TrimPrefix(res, " ")
	res = strings.TrimPrefix(res, text)
	res = strings.TrimPrefix(res, "\n")
	res = strings.TrimSuffix(res, "<|endoftext|>")
	C.falcon_free_params(params)

	return res, nil
}
