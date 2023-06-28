package gpt2

// #cgo CFLAGS: -I./ggml.cpp/include/ggml/ -I./ggml.cpp/examples/ -I./ggml.cpp/src/
// #cgo CXXFLAGS: -I./ggml.cpp/include/ggml/ -I./ggml.cpp/examples/ -I./ggml.cpp/src/
// #cgo darwin LDFLAGS: -framework Accelerate
// #cgo darwin CXXFLAGS: -std=c++17
// #cgo LDFLAGS: -ltransformers -lm -lstdc++
// #include <starcoder.h>
import "C"
import (
	"fmt"
	common "github.com/go-skynet/go-common"
	"strings"
	"unsafe"
)

type Starcoder struct {
	state unsafe.Pointer
}

var StarcoderBackendInitializer common.BackendInitializer[Starcoder] = common.BackendInitializer[Starcoder]{
	DefaultInitializationOptions: common.InitializationOptions{},
	Constructor: func(modelPath string, initializationOptions common.InitializationOptions) (*Starcoder, error) {
		state := C.starcoder_allocate_state()
		cModelPath := C.CString(modelPath)
		result := C.starcoder_bootstrap(cModelPath, state)
		if result != 0 {
			return nil, fmt.Errorf("failed loading model")
		}

		return &Starcoder{state: state}, nil
	},
}

func (l Starcoder) Name() string {
	return "starcoder"
}

func (l Starcoder) Close() error {
	C.starcoder_free_model(l.state)
	return nil
}

func (l *Starcoder) Predict(text string, opts ...common.PredictTextOptionSetter) (string, error) {
	return l.PredictWithOptions(text, *MergePredictOptionsWithDefaults(opts...))
}

func (l *Starcoder) PredictWithOptions(text string, po common.PredictTextOptions) (string, error) {
	input := C.CString(text)
	if po.Tokens == 0 {
		po.Tokens = 99999999
	}
	out := make([]byte, po.Tokens)

	params := C.starcoder_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.Temperature), C.int(po.Batch))
	ret := C.starcoder_predict(params, l.state, (*C.char)(unsafe.Pointer(&out[0])))
	if ret != 0 {
		return "", fmt.Errorf("inference failed")
	}
	res := C.GoString((*C.char)(unsafe.Pointer(&out[0])))

	res = strings.TrimPrefix(res, " ")
	res = strings.TrimPrefix(res, text)
	res = strings.TrimPrefix(res, "\n")
	res = strings.TrimSuffix(res, "<|endoftext|>")
	C.starcoder_free_params(params)

	return res, nil
}
