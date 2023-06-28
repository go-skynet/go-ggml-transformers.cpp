package gpt2

// #cgo CFLAGS: -I./ggml.cpp/include/ggml/ -I./ggml.cpp/examples/ -I./ggml.cpp/src/
// #cgo CXXFLAGS: -I./ggml.cpp/include/ggml/ -I./ggml.cpp/examples/ -I./ggml.cpp/src/
// #cgo darwin LDFLAGS: -framework Accelerate
// #cgo darwin CXXFLAGS: -std=c++17
// #cgo LDFLAGS: -ltransformers -lm -lstdc++
// #include <mpt.h>
import "C"
import (
	"fmt"
	common "github.com/go-skynet/go-common"
	"strings"
	"unsafe"
)

type MPT struct {
	state unsafe.Pointer
}

var MPTBackendInitializer common.BackendInitializer[MPT] = common.BackendInitializer[MPT]{
	DefaultInitializationOptions: common.InitializationOptions{},
	Constructor: func(modelPath string, initializationOptions common.InitializationOptions) (*MPT, error) {
		state := C.mpt_allocate_state()
		cModelPath := C.CString(modelPath)
		result := C.mpt_bootstrap(cModelPath, state)
		if result != 0 {
			return nil, fmt.Errorf("failed loading model")
		}

		return &MPT{state: state}, nil
	},
}

func (l MPT) Name() string {
	return "mpt"
}

func (l MPT) Close() error {
	C.mpt_free_model(l.state)
	return nil
}

func (l *MPT) Predict(text string, opts ...common.PredictTextOptionSetter) (string, error) {
	return l.PredictWithOptions(text, *MergePredictOptionsWithDefaults(opts...))
}

func (l *MPT) PredictWithOptions(text string, po common.PredictTextOptions) (string, error) {
	input := C.CString(text)
	if po.Tokens == 0 {
		po.Tokens = 99999999
	}
	out := make([]byte, po.Tokens)

	params := C.mpt_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.Temperature), C.int(po.Batch))
	ret := C.mpt_predict(params, l.state, (*C.char)(unsafe.Pointer(&out[0])))
	if ret != 0 {
		return "", fmt.Errorf("inference failed")
	}
	res := C.GoString((*C.char)(unsafe.Pointer(&out[0])))

	res = strings.TrimPrefix(res, " ")
	res = strings.TrimPrefix(res, text)
	res = strings.TrimPrefix(res, "\n")
	res = strings.TrimSuffix(res, "<|endoftext|>")
	C.mpt_free_params(params)

	return res, nil
}
