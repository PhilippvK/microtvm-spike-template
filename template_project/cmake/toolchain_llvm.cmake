SET(CMAKE_SYSTEM_NAME Generic)

SET(CMAKE_SYSTEM_PROCESSOR Spike)

SET(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")

INCLUDE(LookupClang)

SET(CMAKE_C_COMPILER ${CLANG_EXECUTABLE})
SET(CMAKE_CXX_COMPILER ${CLANG++_EXECUTABLE})
SET(CMAKE_ASM_COMPILER ${CLANG_EXECUTABLE})

SET(XLEN 32)  # TODO

SET(CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS} --target=riscv${XLEN} -march=${RISCV_ARCH} -mabi=${RISCV_ABI}"
)
SET(CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS} --gcc-toolchain=${RISCV_ELF_GCC_PREFIX} --sysroot=${RISCV_ELF_GCC_PREFIX}/${RISCV_ELF_GCC_BASENAME}"
)

SET(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} --target=riscv${XLEN} -march=${RISCV_ARCH} -mabi=${RISCV_ABI}"
)
SET(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} --gcc-toolchain=${RISCV_ELF_GCC_PREFIX} --sysroot=${RISCV_ELF_GCC_PREFIX}/${RISCV_ELF_GCC_BASENAME}"
)

SET(CMAKE_ASM_FLAGS
    "${CMAKE_ASM_FLAGS} --target=riscv${XLEN} -march=${RISCV_ARCH} -mabi=${RISCV_ABI}"
)
SET(CMAKE_ASM_FLAGS
    "${CMAKE_ASM_FLAGS} --gcc-toolchain=${RISCV_ELF_GCC_PREFIX} --sysroot=${RISCV_ELF_GCC_PREFIX}/${RISCV_ELF_GCC_BASENAME}"
)

# SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
