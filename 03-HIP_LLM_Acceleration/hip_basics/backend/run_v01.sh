# /bin/bash

hipcc --offload-arch=gfx90a demo_gemv_v0.cpp -o gemv_v0
hipcc --offload-arch=gfx90a demo_gemv_v1.cpp -o gemv_v1
